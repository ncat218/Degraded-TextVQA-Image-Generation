#!/usr/bin/env python3
"""One command: submit Slurm, create an isolated environment, generate full TextVQA."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parent


def validate(config):
    for key in ('source_root', 'output_root', 'workspace_root'):
        if not PurePosixPath(config[key]).is_absolute():
            raise ValueError(key + ' must be an absolute server path')
    roots = [PurePosixPath(posixpath.normpath(config[k])) for k in ('source_root','output_root','workspace_root')]
    for i,a in enumerate(roots):
        for b in roots[i+1:]:
            if a == b or a in b.parents or b in a.parents:
                raise ValueError('Source, output and workspace must be separate non-nested folders')
    s = config['scheduler']
    limits = {'small': (12,40,'gpu:1g.18gb:1'), 'normal': (16,72,'gpu:2g.35gb:1'), 'large': (24,112,'gpu:3g.71gb:1')}
    if s['partition'] not in limits:
        raise ValueError('Configure a documented UIT partition: small/normal/large')
    cpus,ram,gres=limits[s['partition']]
    if not (1 <= s['cpus'] <= cpus and 1 <= s['ram_gb'] <= ram and 1 <= config['workers'] <= s['cpus']):
        raise ValueError('CPU/RAM/workers exceed the configured allocation')
    if s['gres'] and s['gres'] != gres:
        raise ValueError('GPU GRES does not match the partition')
    h,m,sec=map(int,s['walltime'].split(':'))
    if not (0 <= m < 60 and 0 <= sec < 60 and 0 < h*3600+m*60+sec <= 72*3600):
        raise ValueError('Job walltime must be between 1 second and 72 hours')
    if not 0 <= config['png_compression'] <= 9 or config['max_output_disk_gb'] <= 0:
        raise ValueError('Invalid PNG compression/disk limit')
    if not (20 <= config['preflight_images'] <= 30 and
            config['calibration_images'] >= config['preflight_images'] and
            0 < config['single_calibration_images'] <= config['calibration_images'] and
            config['ablation_images'] > 0):
        raise ValueError('Invalid sample counts')
    if set(config['expected_images']) != {'train','val','test'} or set(config['expected_questions']) != {'train','val','test'}:
        raise ValueError('All three official splits must be configured')


def atomic_json(path, obj):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    os.replace(temporary,path)


def submit(config):
    # Only small configuration/path checks occur on login. No images are read here.
    source=Path(config['source_root'])
    for split in ('train','val','test'):
        if not (source / ('TextVQA_0.5.1_'+split+'.json')).is_file():
            raise FileNotFoundError('Missing '+str(source / ('TextVQA_0.5.1_'+split+'.json')))
    for folder in ('train_images','test_images'):
        if not (source/folder).is_dir():
            raise FileNotFoundError(str(source/folder))
    log_dir=Path(config['workspace_root'])/'logs'
    log_dir.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    snapshot=log_dir/(stamp+'_config.json')
    atomic_json(snapshot,config)
    script=log_dir/(stamp+'_submit_full_textvqa.sh')
    command=[sys.executable,'-u',str(REPO/'run_all.py'),'--in-allocation','--config',str(snapshot)]
    script.write_text('#!/bin/bash\nset -euo pipefail\ncd '+shlex.quote(str(REPO))+'\nexec '+
                      ' '.join(shlex.quote(x) for x in command)+'\n',encoding='utf-8')
    s=config['scheduler']
    args=['sbatch','--parsable','--job-name=full_textvqa_degraded','--ntasks=1',
          '--account='+s['account'],'--partition='+s['partition'],
          '--cpus-per-task='+str(s['cpus']),'--mem='+str(s['ram_gb'])+'G',
          '--time='+s['walltime'],'--signal=B:TERM@120',
          '--output='+str(log_dir/(stamp+'_%j.out')),
          '--error='+str(log_dir/(stamp+'_%j.err'))]
    if s['gres']: args+=['--gres='+s['gres']]
    if s.get('qos'): args+=['--qos='+s['qos']]
    args.append(str(script))
    result=subprocess.check_output(args,text=True).strip()
    print('Submitted ONE Slurm job:',result,flush=True)
    print('Logs:',log_dir,flush=True)
    print('Output:',config['output_root'],flush=True)
    print('Follow: tail -f '+shlex.quote(str(log_dir/(stamp+'_'+result.split(';')[0]+'.out'))),flush=True)


def bootstrap(config, config_path):
    workspace=Path(config['workspace_root'])
    workspace.mkdir(parents=True,exist_ok=True)
    # fcntl locks are released by the OS when a job exits or is killed.
    import fcntl
    lock=(workspace/'environment.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    env_root=workspace/'venv'
    python=env_root/'bin/python'
    requirements=REPO/'requirements.txt'
    wanted=hashlib.sha256(requirements.read_bytes()).hexdigest()
    stamp=workspace/'environment_requirements.sha256'
    if not python.is_file():
        selected=shutil.which('python3.11')
        if not selected:
            if not (3,10) <= sys.version_info[:2] <= (3,13):
                raise RuntimeError('Install/load Python 3.11 (supported: 3.10–3.13) before running')
            selected=sys.executable
        subprocess.check_call([selected,'-m','venv',str(env_root)])
    if not stamp.is_file() or stamp.read_text().strip() != wanted:
        cache=workspace/'pip_cache'
        subprocess.check_call([str(python),'-m','pip','install','--cache-dir',str(cache),'-r',str(requirements)])
        subprocess.check_call([str(python),'-m','pip','check'])
        stamp.write_text(wanted+'\n')
    fcntl.flock(lock,fcntl.LOCK_UN)
    lock.close()
    environment=os.environ.copy()
    environment.update(PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    environment['PYTHONPATH']=str(REPO/'src')
    os.execve(str(python),[str(python),'-u','-m','textvqa_degradation.pipeline','--config',str(config_path)],environment)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(REPO/'configs/full_textvqa.json'))
    parser.add_argument('--plan-only',action='store_true',help='Print settings; do not access data, install or submit')
    parser.add_argument('--in-allocation',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    config_path=Path(args.config).resolve()
    config=json.loads(config_path.read_text(encoding='utf-8'))
    validate(config)
    if args.plan_only:
        print(json.dumps(config,ensure_ascii=False,indent=2)); return
    if sys.platform != 'linux':
        raise RuntimeError('Actual execution requires UIT Linux/Slurm; use --plan-only on Windows')
    if not os.environ.get('SLURM_JOB_ID'):
        if args.in_allocation:
            raise RuntimeError('Compute requires SLURM_JOB_ID; refusing to run on login')
        if not shutil.which('sbatch'):
            raise RuntimeError('sbatch not found. Connect to the UIT HPC login node.')
        submit(config)
    else:
        allocated=int(os.environ.get('SLURM_CPUS_PER_TASK',config['scheduler']['cpus']))
        if config['workers'] > allocated:
            raise RuntimeError('workers exceeds SLURM_CPUS_PER_TASK; correct allocation/config')
        bootstrap(config,config_path)


if __name__=='__main__':
    main()
