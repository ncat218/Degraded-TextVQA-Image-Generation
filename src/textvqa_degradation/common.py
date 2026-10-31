import csv
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def json_bytes(obj):
    return (json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_bytes(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.tmp_',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)


def save_json(path,obj):
    atomic_bytes(path,json_bytes(obj))


def lock_bytes(path,data):
    path=Path(path)
    if path.exists():
        require(path.read_bytes()==data, f'Locked artifact changed: {path}. Use a new output folder for a changed protocol/source.')
    else:
        atomic_bytes(path,data)
    checksum=hashlib.sha256(data).hexdigest()
    side=path.with_suffix(path.suffix+'.sha256')
    if side.exists():
        require(side.read_text().strip()==checksum,f'Hash sidecar mismatch: {side}')
    else:
        atomic_bytes(side,(checksum+'\n').encode())
    # Original notebook convention, for .csv/.json consumers.
    if path.suffix in {'.csv','.json'}:
        legacy=path.with_suffix('.sha256')
        if legacy != side:
            if legacy.exists(): require(legacy.read_text().strip()==checksum,f'Hash mismatch: {legacy}')
            else: atomic_bytes(legacy,(checksum+'\n').encode())
    return checksum


def csv_bytes(rows,columns):
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n',extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:8],'big')
