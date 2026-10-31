from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import re

import numpy as np
from PIL import Image

from .common import require, sha_file, lock_bytes, json_bytes, csv_bytes, read_json

IMAGE_COLUMNS=['dataset','split','image_id','source_image_id','clean_image_path',
               'source_image_sha256','clean_width','clean_height']
QA_COLUMNS=['dataset','split','image_id','source_image_id','question_id','question','answers','gt_available']


def audit_image(item):
    split,ident,path=item
    with Image.open(path) as image:
        width,height=image.size
        image.verify()
    require(min(width,height)>=7,f'Image too small for original SSIM: {path}')
    return dict(dataset='textvqa',split=split,image_id=ident,source_image_id=ident,
                clean_image_path=str(path),source_image_sha256=sha_file(path),
                clean_width=width,clean_height=height)


def select_images(images,n,seed):
    ordered=sorted(images,key=lambda r:(r['split'],r['image_id']))
    require(len(ordered)>=n,f'Need {n} source images; only {len(ordered)} available')
    indices=np.random.default_rng(seed).choice(len(ordered),size=n,replace=False)
    return sorted((ordered[int(i)] for i in indices),key=lambda r:(r['split'],r['image_id']))


def discover(config,root):
    source=Path(config['source_root'])
    questions=[]; tasks=[]; annotation_hashes={}; split_counts={}
    indexed={}
    for folder in ('train_images','test_images'):
        index={}
        for path in sorted((source/folder).rglob('*')):
            if path.is_file() and path.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}:
                key=path.stem.lower()
                require(key not in index, f'Ambiguous source image ID {key}: {index.get(key)}, {path}')
                index[key]=path
        indexed[folder]=index
    for split in ('train','val','test'):
        path=source/('TextVQA_0.5.1_'+split+'.json')
        annotation_hashes[split]=sha_file(path)
        obj=read_json(path)
        rows=obj['data']
        require(len(rows)==config['expected_questions'][split],f'Unexpected {split} question count: {len(rows)}')
        ids=set(); qids=set()
        for r in rows:
            ident=str(r['image_id']).strip(); qid=str(r['question_id'])
            require(bool(re.fullmatch('[A-Za-z0-9_-]+',ident)),f'Unsafe image ID: {ident}')
            require(qid not in qids,f'Duplicate question ID in {split}: {qid}')
            qids.add(qid); ids.add(ident)
            require(r.get('set_name',split)==split,f'Annotation split mismatch for {qid}')
            answers=r.get('answers',[])
            require(isinstance(answers,list),f'answers is not a list: {qid}')
            gt=bool(answers) and any(str(a).strip() for a in answers)
            require(split=='test' or gt,f'Missing reference answers: {split}/{qid}')
            questions.append(dict(dataset='textvqa',split=split,image_id=ident,source_image_id=ident,
                                  question_id=qid,question=r['question'],
                                  answers=json.dumps(answers,ensure_ascii=False),gt_available=gt))
        require(len(ids)==config['expected_images'][split],f'Unexpected {split} image count: {len(ids)}')
        index=indexed['test_images' if split=='test' else 'train_images']
        missing=sorted(ident for ident in ids if ident.lower() not in index)
        require(not missing,f'Missing {len(missing)} source images in {split}: {missing[:10]}')
        tasks.extend((split,ident,index[ident.lower()]) for ident in sorted(ids))
        split_counts[split]=dict(images=len(ids),questions=len(rows))
    print('[AUDIT] Read/verify/hash',len(tasks),'clean images; source files are read-only',flush=True)
    images=[]
    if config['workers']==1:
        iterator=map(audit_image,tasks)
        for i,item in enumerate(iterator,1):
            images.append(item)
            if i%1000==0 or i==len(tasks): print('[AUDIT]',i,'/',len(tasks),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=config['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            for i,item in enumerate(pool.map(audit_image,tasks,chunksize=16),1):
                images.append(item)
                if i%1000==0 or i==len(tasks): print('[AUDIT]',i,'/',len(tasks),flush=True)
    images.sort(key=lambda r:(r['split'],r['image_id']))
    questions.sort(key=lambda r:(r['split'],r['image_id'],int(r['question_id'])))
    image_sha=lock_bytes(root/'metadata/source_image_manifest.csv',csv_bytes(images,IMAGE_COLUMNS))
    qa_sha=lock_bytes(root/'metadata/locked_eval_manifest.csv',csv_bytes(questions,QA_COLUMNS))
    duplicate_hashes=Counter(r['source_image_sha256'] for r in images)
    provenance=dict(dataset='textvqa',version='0.5.1',source_root=str(source),
        annotation_sha256=annotation_hashes,source_image_manifest_sha256=image_sha,
        qa_manifest_sha256=qa_sha,counts=split_counts,
        exact_duplicate_hash_groups=sum(n>1 for n in duplicate_hashes.values()),
        duplicate_policy='Keep all official image IDs; preserve official splits; report exact byte duplicates',
        test_answers='Use only provided values; never fabricate hidden test answers')
    lock_bytes(root/'metadata/source_provenance.json',json_bytes(provenance))
    print('[AUDIT] Counts:',split_counts,flush=True)
    return images,questions,provenance
