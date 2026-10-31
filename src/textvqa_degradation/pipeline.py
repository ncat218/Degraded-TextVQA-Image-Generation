"""All automatic stages; run_all.py is the supported HPC entry point."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import html
import importlib.metadata
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time

import cv2
import numpy as np
from PIL import Image

from . import operators
from .common import (require,sha_file,json_bytes,read_json,atomic_bytes,
                     save_json,lock_bytes,csv_bytes,stable_seed)
from .data import discover, select_images, IMAGE_COLUMNS, QA_COLUMNS

PROTOCOL='TEXTVQA_REALISTIC_DEGRADATION_V2_FULL_R1'
REPO=Path(__file__).resolve().parents[2]
RECORD_COLUMNS=['dataset','split','image_id','source_image_id','condition','level',
    'degradation_type','severity','seed','params','output_path','output_sha256',
    'source_image_path','source_image_sha256','source_manifest_sha256','config_sha256',
    'clean_width','clean_height','degraded_width','degraded_height',
    'median_text_height_px','severity_basis','file_size_bytes','psnr','ssim',
    'mean_luminance','std_luminance','laplacian_variance']


def init_worker():
    cv2.setNumThreads(1)
    # The parent handles Slurm termination; each output is atomic and resume-safe.
    signal.signal(signal.SIGINT,signal.SIG_IGN)


def generation_task(task):
    image,group,condition,level,config_sha,source_sha,config=task
    started=time.perf_counter()
    split=image['split']; ident=image['image_id']; root=Path(config['output_root'])
    base=root/group
    if group=='preflight': base=base/config_sha
    path=base/'images'/condition/level/split/(ident+'.png')
    side=Path(str(path)+'.json')
    seed=stable_seed(config['seed'],source_sha,config_sha,split,ident,condition,level)
    source=Path(image['clean_image_path'])
    require(sha_file(source)==image['source_image_sha256'],f'Clean bytes changed: {source}')
    if side.exists() or path.exists():
        if side.exists() and path.exists():
            old=read_json(side)
            require(old.get('source_image_sha256')==image['source_image_sha256'] and
                    old.get('source_image_path')==str(source) and old.get('output_path')==str(path) and
                    old.get('source_manifest_sha256')==source_sha and old.get('config_sha256')==config_sha and
                    old.get('seed')==seed and old.get('output_sha256')==sha_file(path),
                    f'Existing output/metadata mismatch: {path}; do not silently overwrite verified output')
            return old,True,time.perf_counter()-started
        # An orphaned PNG/JSON after interruption can be recreated deterministically.
    with Image.open(source) as im:
        clean=np.asarray(im.convert('RGB'),dtype=np.uint8)
    require(clean.shape[:2]==(image['clean_height'],image['clean_width']),f'Source size changed: {source}')
    # Official v0.5.1 QA files do not include OCR boxes. This is the original
    # notebook image-relative fallback, not an inferred text-size measurement.
    out,params=operators.degrade(clean,condition,level,None,seed)
    require(out.shape==clean.shape and out.dtype==np.uint8,'Invalid degraded shape/type')
    ok,encoded=cv2.imencode('.png',cv2.cvtColor(out,cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_PNG_COMPRESSION,config['png_compression']])
    require(ok,'PNG encoding failed')
    data=encoded.tobytes()
    record=dict(dataset='textvqa',split=split,image_id=ident,source_image_id=ident,
        condition=condition,level=level,degradation_type=condition,severity=level,
        seed=seed,params=params,output_path=str(path),output_sha256=hashlib.sha256(data).hexdigest(),
        source_image_path=str(source),source_image_sha256=image['source_image_sha256'],
        source_manifest_sha256=source_sha,config_sha256=config_sha,
        clean_width=clean.shape[1],clean_height=clean.shape[0],
        degraded_width=out.shape[1],degraded_height=out.shape[0],
        median_text_height_px=None,severity_basis='image_relative_fallback',
        file_size_bytes=len(data),**operators.diagnostics(clean,out))
    atomic_bytes(path,data)
    save_json(side,record)
    return record,False,time.perf_counter()-started


def table_records(records):
    return [dict(r,params=json.dumps(r['params'],ensure_ascii=False,sort_keys=True)) for r in records]


def run_group(group,jobs,config,config_sha,source_sha,root):
    started=time.perf_counter(); records=[]; reused=0
    tasks=[(image,group,condition,level,config_sha,source_sha,config)
           for condition,level,images in jobs for image in images]
    pool=None
    if config['workers']>1:
        pool=ProcessPoolExecutor(max_workers=config['workers'],initializer=init_worker,
                                 mp_context=multiprocessing.get_context('spawn'))
        iterator=pool.map(generation_task,tasks,chunksize=1)
    else:
        cv2.setNumThreads(1)
        iterator=map(generation_task,tasks)
    try:
        for i,(record,was_reused,seconds) in enumerate(iterator,1):
            records.append(record); reused+=int(was_reused)
            if i==1 or i%100==0 or i==len(tasks):
                print(f'[{group}] {i}/{len(tasks)} | elapsed {time.perf_counter()-started:.1f}s | reused {reused}',flush=True)
    except BaseException as exc:
        save_json(root/group/'failure.json',dict(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
            completed_this_attempt=len(records),error_type=type(exc).__name__,message=str(exc)))
        if pool: pool.shutdown(wait=True,cancel_futures=True)
        raise
    else:
        if pool: pool.shutdown()
    require(len({(r['split'],r['image_id'],r['condition'],r['level']) for r in records})==len(tasks), 'Duplicate generated key')
    # Lock record order/content; a resume revalidates every PNG and source hash.
    if group!='preflight':
        lock_bytes(root/group/'image_manifest.csv',csv_bytes(table_records(records),RECORD_COLUMNS))
    summary=dict(status='complete',output_count=len(records),failure_count=0,
        reused_count=reused,runtime_seconds=time.perf_counter()-started,
        disk_bytes=sum(r['file_size_bytes'] for r in records),
        by_condition={f'{c}/{l}':sum(r['condition']==c and r['level']==l for r in records)
                      for c,l,_ in jobs})
    save_json(root/group/'diagnostics_summary.json',summary)
    save_json(root/group/'failures.json',[])
    print(group,'outputs:',len(records),'failures: 0','disk GB:',round(summary['disk_bytes']/1e9,3),flush=True)
    return records,summary


def preflight(calibration,config,sha,cal_sha,root):
    measurements=[]
    for i,image in enumerate(calibration[:config['preflight_images']]):
        condition=(('realistic_mix',)+operators.SINGLES)[i%8]
        record,reused,seconds=generation_task((image,'preflight',condition,'L2',sha,cal_sha,config))
        measurements.append((condition,record['file_size_bytes'],seconds))
    costs={c:dict(size_mb=float(np.mean([size/1e6 for name,size,_ in measurements if name==c])),
                  seconds=float(np.mean([seconds for name,_,seconds in measurements if name==c])))
           for c in ('realistic_mix',)+operators.SINGLES}
    counts=dict(main=sum(config['expected_images'].values()),
                calibration=config['calibration_images']*3+config['single_calibration_images']*21,
                ablation=config['ablation_images']*7)
    disk=dict(main=counts['main']*costs['realistic_mix']['size_mb']/1000,
        calibration=(3*config['calibration_images']*costs['realistic_mix']['size_mb']+
                     3*config['single_calibration_images']*sum(costs[c]['size_mb'] for c in operators.SINGLES))/1000,
        ablation=config['ablation_images']*sum(costs[c]['size_mb'] for c in operators.SINGLES)/1000)
    sequential=dict(main=counts['main']*costs['realistic_mix']['seconds'],
        calibration=3*config['calibration_images']*costs['realistic_mix']['seconds']+
                    3*config['single_calibration_images']*sum(costs[c]['seconds'] for c in operators.SINGLES),
        ablation=config['ablation_images']*sum(costs[c]['seconds'] for c in operators.SINGLES))
    report=dict(config_sha256=sha,preflight_outputs=len(measurements),expected_outputs=counts,
                estimated_disk_gb=disk,per_condition=costs,
                estimated_sequential_runtime_seconds=sequential,
                note='25 calibration samples; rough estimates, not a performance/disk guarantee. Parallel runtime is not measured here.')
    save_json(root/'preflight'/('estimate_'+sha+'.json'),report)
    print('[PREFLIGHT]',json.dumps(report,ensure_ascii=False),flush=True)
    return report


def disk_gate(root,config,report,groups):
    actual=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())/1e9
    selected=sum(p.stat().st_size for g in groups for p in (root/g/'images').rglob('*.png'))/1e9
    additional=max(0,sum(report['estimated_disk_gb'][g] for g in groups)*1.25-selected)
    # Room for manifests, previews and sampling uncertainty beyond the 25% margin.
    reserve=0.25
    free=shutil.disk_usage(root).free/1e9
    require(actual+additional+reserve<=config['max_output_disk_gb'],
            f'Projected {actual+additional+reserve:.2f} GB > configured maximum {config["max_output_disk_gb"]} GB')
    require(additional+reserve<free,f'Need about {additional+reserve:.2f} GB more; only {free:.2f} GB free')
    print(f'[DISK] existing={actual:.2f} GB, additional+reserve={additional+reserve:.2f} GB, free={free:.2f} GB',flush=True)


def make_review(records,questions,config,root):
    samples=[]
    for condition in ('realistic_mix',)+operators.SINGLES:
        for level in ('L1','L2','L3'):
            part=sorted((r for r in records if r['condition']==condition and r['level']==level),key=lambda r:r['image_id'])
            n=min(100 if condition=='realistic_mix' else 15,len(part))
            # Use the original notebook's stable review seeds.
            seed=stable_seed(config['seed'],'review',level) if condition=='realistic_mix' else stable_seed(config['seed'],'review',condition,level)
            idx=np.random.default_rng(seed).choice(len(part),size=n,replace=False)
            chosen={int(i) for i in idx}
            samples.extend(r for i,r in enumerate(part) if i in chosen)
    lookup={}
    for q in questions: lookup.setdefault((q['split'],q['image_id']),q)
    n=len(samples); easy_count=int(math.floor(n*0.7+0.5))
    easy=set(map(int,np.random.default_rng(config['seed']).permutation(n)[:easy_count]))
    rows=[]; labeled=[]; cards=[]
    directory=root/'calibration/manual_review'
    for i,r in enumerate(samples):
        q=lookup[r['split'],r['image_id']]
        key=f'{r["split"]}|{r["image_id"]}|{r["condition"]}|{r["level"]}'
        row=dict(review_key=key,image_id=r['image_id'],condition=r['condition'],level=r['level'],
                 question=q['question'],answers=q['answers'],params=json.dumps(r['params'],ensure_ascii=False,sort_keys=True),label='')
        rows.append(row)
        labeled.append(dict(row,label='EASY' if i in easy else 'DIFFICULT_BUT_ANSWERABLE'))
        preview_paths=[]
        for source,name in ((r['source_image_path'],r['image_id']+'_clean.jpg'),
                            (r['output_path'],r['image_id']+'_'+r['condition']+'_'+r['level']+'.jpg')):
            target=directory/'previews'/name
            if not target.is_file():
                with Image.open(source) as image:
                    image=image.convert('RGB'); image.thumbnail((600,600))
                    buffer=io.BytesIO(); image.save(buffer,format='JPEG',quality=82)
                    atomic_bytes(target,buffer.getvalue())
            preview_paths.append('previews/'+name)
        cards.append('<article><h3>'+html.escape(key)+'</h3>'+''.join('<img src="'+html.escape(p,quote=True)+'">' for p in preview_paths)+
                     '<p>'+html.escape(q['question'])+'</p><p>'+html.escape(q['answers'])+'</p></article>')
    columns=['review_key','image_id','condition','level','question','answers','params','label']
    lock_bytes(directory/'review_samples.csv',csv_bytes(rows,columns))
    label_sha=lock_bytes(directory/'reviewed_samples.csv',csv_bytes(labeled,columns))
    provenance=dict(labeling_mode='simulated_random',human_review_performed=False,seed=config['seed'],
        method='NumPy default_rng permutation; round half up(70% * number of samples)',
        counts=dict(Counter(r['label'] for r in labeled)),row_count=n,
        reviewed_samples_sha256=label_sha,
        meaning='Random placeholders for compatibility; not evidence of image answerability or manual review')
    lock_bytes(directory/'reviewed_samples.random_label_provenance.json',json_bytes(provenance))
    page='<!doctype html><meta charset="utf-8"><title>TextVQA calibration previews</title><style>body{font:16px sans-serif;max-width:1100px;margin:auto}img{max-width:46%;max-height:400px}article{border-bottom:1px solid #aaa}</style><h1>Calibration previews — automatic random labels</h1><p>No human review was performed. Labels are simulated.</p>'+''.join(cards)
    atomic_bytes(directory/'review_gallery.html',page.encode())
    print('[REVIEW] Simulated labels:',provenance['counts'],'human_review_performed=False',flush=True)
    return label_sha,provenance


def export_questions(group,records,questions,root):
    by_image={}
    for record in records:
        by_image.setdefault((record['split'],record['image_id']),[]).append(record)
    rows=[]
    for q in questions:
        for record in by_image.get((q['split'],q['image_id']),[]):
            r=dict(q,condition=record['condition'],level=record['level'],
                degradation_type=record['condition'],severity=record['level'],seed=record['seed'],
                params=json.dumps(record['params'],ensure_ascii=False,sort_keys=True),
                clean_image_path=record['source_image_path'],degraded_image_path=record['output_path'],
                clean_path=record['source_image_path'],degraded_path=record['output_path'],
                clean_width=record['clean_width'],clean_height=record['clean_height'],
                degraded_width=record['degraded_width'],degraded_height=record['degraded_height'],
                source_image_sha256=record['source_image_sha256'],degraded_sha256=record['output_sha256'],
                config_sha256=record['config_sha256'],source_manifest_sha256=record['source_manifest_sha256'])
            rows.append(r)
    columns=list(rows[0])
    lock_bytes(root/group/'question_manifest.csv',csv_bytes(rows,columns))
    if group=='main':
        lock_bytes(root/group/'final_main_manifest.csv',csv_bytes(rows,columns))
        for split in ('train','val','test'):
            part=[r for r in rows if r['split']==split]
            lock_bytes(root/group/'manifests'/(split+'.csv'),csv_bytes(part,columns))
    require(len({(r['split'],r['question_id'],r['condition'],r['level']) for r in rows})==len(rows),'Duplicate QA/variant')
    return rows


class RunLock:
    def __init__(self,path): self.path=path
    def __enter__(self):
        self.file=self.path.open('a+')
        if os.name=='nt':
            import msvcrt
            self.file.seek(0); self.file.write(' '); self.file.flush(); self.file.seek(0)
            msvcrt.locking(self.file.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            try: fcntl.flock(self.file,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError: raise RuntimeError('Another job is writing this output; wait or cancel it first')
        return self
    def __exit__(self,*args): self.file.close()


def execute(config):
    root=Path(config['output_root']); root.mkdir(parents=True,exist_ok=True)
    with RunLock(root/'.run.lock'):
        # refuse a folder with unrelated files on its first adoption
        allowed={'.run.lock','config','metadata','calibration','preflight','main','ablation','environment.json',
                 'environment.sha256','environment.json.sha256',
                 'generation_summary.json','pipeline_failure.json'}
        require(all(p.name in allowed for p in root.iterdir()),'Output folder contains unrelated files; choose a fresh output root')
        try:
            return run_pipeline(config,root)
        except BaseException as exc:
            save_json(root/'pipeline_failure.json',dict(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                error_type=type(exc).__name__,message=str(exc)))
            raise


def run_pipeline(config,root):
    cv2.setNumThreads(1)
    started=time.perf_counter()
    for folder in ('config','metadata','calibration','preflight','main','ablation'):
        (root/folder).mkdir(exist_ok=True)
    old_summary=read_json(root/'generation_summary.json') if (root/'generation_summary.json').is_file() else None
    if old_summary:
        save_json(root/'generation_summary.json',dict(status='revalidating',previous_status=old_summary['status']))
    versions={name:importlib.metadata.version(name) for name in ('numpy','pandas','Pillow','opencv-python-headless','scikit-image','scipy')}
    env=dict(python=platform.python_version(),packages=versions,platform=platform.platform(),
             execution_backend='CPU OpenCV/NumPy; no GPU transformations')
    envpath=root/'environment.json'
    if envpath.exists():
        prior=read_json(envpath)
        require(prior['packages']==versions and prior['python']==env['python'],'Environment changed; restore versions before resume')
    else: lock_bytes(envpath,json_bytes(env))
    images,questions,provenance=discover(config,root)
    train=[r for r in images if r['split']=='train']
    calibration=select_images(train,config['calibration_images'],config['seed'])
    singles=select_images(calibration,config['single_calibration_images'],config['seed']+1)
    ablation=select_images(train,config['ablation_images'],config['seed']+2)
    cal_sha=lock_bytes(root/'calibration/calibration_manifest.csv',csv_bytes(calibration,IMAGE_COLUMNS))
    abl_sha=lock_bytes(root/'ablation/ablation_manifest.csv',csv_bytes(ablation,IMAGE_COLUMNS))
    policy=dict(official_splits_preserved=True,all_source_images_in_main=True,all_questions_preserved=True,
        calibration_role='Automatic diagnostic subset of official train; overlaps main intentionally; not an independent holdout',
        ablation_role='Train-only subset; not selected by downstream model score',
        calibration_images=len(calibration),ablation_images=len(ablation),seed=config['seed'])
    lock_bytes(root/'metadata/split_policy.json',json_bytes(policy))
    draft=dict(protocol=PROTOCOL,seed=config['seed'],png_compression=config['png_compression'],
        params=operators.DEFAULT_PARAMS,operators_file_sha256=sha_file(Path(operators.__file__)),
        pipeline_file_sha256=sha_file(Path(__file__)),data_file_sha256=sha_file(Path(__file__).with_name('data.py')),
        common_file_sha256=sha_file(Path(__file__).with_name('common.py')),
        source_provenance=provenance,split_policy=policy,source_manifest_sha256=provenance['qa_manifest_sha256'],
        calibration_manifest_sha256=cal_sha,ablation_manifest_sha256=abl_sha,
        review_mode='simulated_random',human_review_performed=False,
        severity_basis='image_relative_fallback',clean_exif_transpose=False,
        source_root=config['source_root'],output_root=config['output_root'])
    draft_sha=lock_bytes(root/'config/draft_degradation_config.json',json_bytes(draft))
    initial=preflight(calibration,config,draft_sha,cal_sha,root)
    disk_gate(root,config,initial,['calibration','main','ablation'])
    cal_jobs=[('realistic_mix',level,calibration) for level in ('L1','L2','L3')]+[
             (condition,level,singles) for condition in operators.SINGLES for level in ('L1','L2','L3')]
    cal_records,cal_summary=run_group('calibration',cal_jobs,config,draft_sha,cal_sha,root)
    label_sha,review=make_review(cal_records,questions,config,root)
    frozen=dict(draft,draft_sha256=draft_sha,reviewed_samples_sha256=label_sha,review_provenance=review,
                freeze_meaning='Automatic immutable configuration lock; not human answerability approval')
    frozen_sha=lock_bytes(root/'config/frozen_degradation_config.json',json_bytes(frozen))
    print('[FREEZE] Frozen config SHA:',frozen_sha,flush=True)
    estimate=preflight(calibration,config,frozen_sha,cal_sha,root)
    disk_gate(root,config,estimate,['main','ablation'])
    main_records,main_summary=run_group('main',[('realistic_mix','L2',images)],config,frozen_sha,provenance['qa_manifest_sha256'],root)
    main_qa=export_questions('main',main_records,questions,root)
    require(len(main_qa)==sum(config['expected_questions'].values()),'Main QA count incorrect')
    require(len(main_records)==sum(config['expected_images'].values()),'Main image count incorrect')
    # Recheck free disk after actual main sizes are known, not only the sample estimate.
    disk_gate(root,config,estimate,['ablation'])
    abl_records,abl_summary=run_group('ablation',[(c,'L2',ablation) for c in operators.SINGLES],config,frozen_sha,abl_sha,root)
    export_questions('ablation',abl_records,questions,root)
    all_records=cal_records+main_records+abl_records
    print('[VERIFY] Verify every PNG hash and consolidate metadata',flush=True)
    for r in all_records:
        require(sha_file(r['output_path'])==r['output_sha256'],f'Output verification failed: {r["output_path"]}')
    jsonl=b''.join((json.dumps(r,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode() for r in all_records)
    lock_bytes(root/'metadata/degradation_params.jsonl',jsonl)
    total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    require(total/1e9<=config['max_output_disk_gb'],'Actual output exceeds configured maximum')
    summary=dict(status='complete',protocol=PROTOCOL,counts=provenance['counts'],
        main_images=len(main_records),main_questions=len(main_qa),
        stages=dict(calibration=cal_summary,main=main_summary,ablation=abl_summary),
        source_manifest_sha256=provenance['qa_manifest_sha256'],frozen_config_sha256=frozen_sha,
        review_mode='simulated_random',human_review_performed=False,
        output_root=str(root),total_output_bytes=total,runtime_seconds=time.perf_counter()-started)
    save_json(root/'generation_summary.json',summary)
    save_json(root/'pipeline_failure.json',dict(status='resolved_or_no_failure'))
    print('READY FOR DOWNSTREAM TEXTVQA / VQA EVALUATION: main, ablation',flush=True)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    require(bool(os.environ.get('SLURM_JOB_ID')),'Use python run_all.py; real execution requires a Slurm allocation')
    def terminate(signum,frame): raise KeyboardInterrupt('Slurm/job termination; rerun the same command to resume')
    signal.signal(signal.SIGTERM,terminate)
    signal.signal(signal.SIGINT,terminate)
    execute(read_json(args.config))


if __name__=='__main__': main()
