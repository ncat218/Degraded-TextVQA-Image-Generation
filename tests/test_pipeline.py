import contextlib
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src'))
import run_all
import numpy as np
from PIL import Image
from textvqa_degradation import operators
from textvqa_degradation.common import read_json,sha_file,lock_bytes
from textvqa_degradation.pipeline import execute,generation_task


def read_csv(path):
    with path.open(encoding='utf-8',newline='') as f: return list(csv.DictReader(f))


class OperatorTests(unittest.TestCase):
    def test_original_notebook_cell_and_params(self):
        provenance=read_json(ROOT/'operator_provenance.json')
        module=Path(operators.__file__).read_text(encoding='utf-8')
        cell=module.split('# Original notebook cell below, byte-for-byte unchanged.\n',1)[1]
        self.assertEqual(hashlib.sha256(cell.encode()).hexdigest(),provenance['original_cell_sha256'])
        params=json.dumps(operators.DEFAULT_PARAMS,sort_keys=True).encode()
        self.assertEqual(hashlib.sha256(params).hexdigest(),provenance['default_params_sha256'])

    def test_repeatability_mix_constraints_and_noise(self):
        image=np.tile(np.arange(128,dtype=np.uint8)[None,:,None],(96,1,3))
        image=np.clip(image+60,0,255).astype(np.uint8)
        for condition in ('realistic_mix',)+operators.SINGLES:
            a,p=operators.degrade(image,condition,'L2',None,1234)
            b,q=operators.degrade(image,condition,'L2',None,1234)
            np.testing.assert_array_equal(a,b)
            self.assertEqual(p,q)
            self.assertEqual(a.shape,image.shape)
            self.assertEqual(a.dtype,np.uint8)
        for seed in range(100):
            _,params=operators.degrade(image,'realistic_mix','L2',None,seed)
            ops=[r['operation'] for r in params]
            self.assertIn(len(ops),(2,3))
            self.assertLessEqual(sum(c in ops for c in ('motion_camera_shake','defocus_blur')),1)
            self.assertLessEqual(sum(c in ops for c in ('lowlight_sensor_noise','overexposure_glare')),1)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.base=Path(cls.temp.name)
        source=cls.base/'clean'
        (source/'train_images').mkdir(parents=True)
        (source/'test_images').mkdir()
        cls.config=read_json(ROOT/'configs/full_textvqa.json')
        cls.config.update(source_root=str(source),output_root=str(cls.base/'output'),
                          workspace_root=str(cls.base/'workspace'),workers=1,
                          calibration_images=25,single_calibration_images=2,ablation_images=4,
                          expected_images=dict(train=28,val=7,test=5),
                          expected_questions=dict(train=56,val=14,test=10))
        rng=np.random.default_rng(19)
        serial=0
        for split,n in cls.config['expected_images'].items():
            rows=[]
            for i in range(n):
                ident=f'{split}_{i:04d}'
                folder=source/('test_images' if split=='test' else 'train_images')
                image=rng.integers(25,235,(80,112,3),dtype=np.uint8)
                Image.fromarray(image).save(folder/(ident+'.jpg'))
                for q in range(2):
                    row=dict(image_id=ident,question_id=serial,question='What does the sign say?',set_name=split)
                    if split!='test': row['answers']=['example']*10
                    rows.append(row); serial+=1
            (source/f'TextVQA_0.5.1_{split}.json').write_text(json.dumps(dict(data=rows)),encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            cls.summary=execute(cls.config)
        cls.root=Path(cls.config['output_root'])

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_complete_all_splits_all_questions_and_metadata(self):
        self.assertEqual(self.summary['status'],'complete')
        self.assertEqual(self.summary['main_images'],40)
        self.assertEqual(self.summary['main_questions'],80)
        self.assertEqual(self.summary['stages']['calibration']['output_count'],117)
        self.assertEqual(self.summary['stages']['ablation']['output_count'],28)
        rows=read_csv(self.root/'main/final_main_manifest.csv')
        for split in ('train','val','test'):
            part=[r for r in rows if r['split']==split]
            self.assertEqual(len(part),self.config['expected_questions'][split])
            self.assertEqual(len({r['image_id'] for r in part}),self.config['expected_images'][split])
        self.assertTrue(all(json.loads(r['answers'])==[] and r['gt_available']=='False' for r in rows if r['split']=='test'))
        for r in rows:
            self.assertEqual(r['clean_width'],r['degraded_width'])
            self.assertEqual(r['clean_height'],r['degraded_height'])
            self.assertEqual(sha_file(r['degraded_image_path']),r['degraded_sha256'])
            self.assertEqual(sha_file(r['clean_image_path']),r['source_image_sha256'])
            self.assertIn(len(json.loads(r['params'])),(2,3))
        provenance=read_json(self.root/'calibration/manual_review/reviewed_samples.random_label_provenance.json')
        self.assertFalse(provenance['human_review_performed'])
        self.assertEqual(sum(provenance['counts'].values()),117)
        self.assertEqual(provenance['counts']['EASY'],82)
        self.assertEqual(provenance['counts']['DIFFICULT_BUT_ANSWERABLE'],35)

    def test_resume_in_parallel_same_pngs_and_manifests(self):
        root=self.root
        old_manifest=(root/'main/final_main_manifest.csv').read_bytes()
        before={str(p):sha_file(p) for p in (root/'main/images').rglob('*.png')}
        config=dict(self.config,workers=2)
        with contextlib.redirect_stdout(io.StringIO()): result=execute(config)
        self.assertEqual(result['stages']['main']['reused_count'],40)
        self.assertEqual(result['stages']['calibration']['reused_count'],117)
        self.assertEqual(old_manifest,(root/'main/final_main_manifest.csv').read_bytes())
        self.assertEqual(before,{str(p):sha_file(p) for p in (root/'main/images').rglob('*.png')})

    def test_orphan_resume_and_corrupt_sidecar_rejected(self):
        r=read_csv(self.root/'main/image_manifest.csv')[0]
        image=dict(dataset='textvqa',split=r['split'],image_id=r['image_id'],
                   clean_image_path=r['source_image_path'],source_image_sha256=r['source_image_sha256'],
                   clean_width=int(r['clean_width']),clean_height=int(r['clean_height']))
        side=Path(r['output_path']+'.json')
        old=side.read_bytes(); side.unlink()
        task=(image,'main',r['condition'],r['level'],r['config_sha256'],r['source_manifest_sha256'],self.config)
        regenerated,reused,_=generation_task(task)
        self.assertFalse(reused)
        self.assertEqual(regenerated['output_sha256'],r['output_sha256'])
        broken=json.loads(old); broken['seed']=0
        side.write_text(json.dumps(broken),encoding='utf-8')
        try:
            with self.assertRaisesRegex(RuntimeError,'mismatch'): generation_task(task)
        finally: side.write_bytes(old)

    def test_new_parallel_generation_matches_original_pngs(self):
        tasks=[]; expected=[]
        for r in read_csv(self.root/'main/image_manifest.csv')[:2]:
            image=dict(dataset='textvqa',split=r['split'],image_id=r['image_id'],
                clean_image_path=r['source_image_path'],source_image_sha256=r['source_image_sha256'],
                clean_width=int(r['clean_width']),clean_height=int(r['clean_height']))
            tasks.append((image,'main',r['condition'],r['level'],r['config_sha256'],r['source_manifest_sha256'],self.config))
            expected.append(r['output_sha256'])
            Path(r['output_path']).unlink()
            Path(r['output_path']+'.json').unlink()
        with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
            results=list(pool.map(generation_task,tasks))
        self.assertEqual([r[0]['output_sha256'] for r in results],expected)
        self.assertTrue(all(not r[1] for r in results))

    def test_modified_annotation_refused(self):
        p=Path(self.config['source_root'])/'TextVQA_0.5.1_val.json'
        old=p.read_bytes(); obj=json.loads(old); obj['data'][0]['question']='Changed question'
        p.write_text(json.dumps(obj),encoding='utf-8')
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError,'Locked artifact changed'): execute(self.config)
        finally: p.write_bytes(old)


class ControllerTests(unittest.TestCase):
    def test_default_configuration_and_nested_paths(self):
        config=read_json(ROOT/'configs/full_textvqa.json')
        run_all.validate(config)
        config['output_root']=config['source_root']+'/bad'
        with self.assertRaisesRegex(ValueError,'separate'): run_all.validate(config)

    def test_submit_one_script_and_job_no_compute(self):
        config=read_json(ROOT/'configs/full_textvqa.json')
        with tempfile.TemporaryDirectory(prefix='test space ') as d:
            root=Path(d)
            config['source_root']=str(root/'clean'); config['workspace_root']=str(root/'workspace')
            source=Path(config['source_root']); source.mkdir()
            for split in ('train','val','test'): (source/f'TextVQA_0.5.1_{split}.json').write_text('{}')
            for folder in ('train_images','test_images'): (source/folder).mkdir()
            with patch('subprocess.check_output',return_value='12345\n') as submit:
                with contextlib.redirect_stdout(io.StringIO()): run_all.submit(config)
            submit.assert_called_once()
            args=submit.call_args.args[0]
            self.assertIn('--partition=normal',args)
            self.assertIn('--gres=gpu:2g.35gb:1',args)
            self.assertIn('--qos=mig2-35',args)
            self.assertIn('--cpus-per-task=12',args)
            self.assertIn('--mem=40G',args)
            scripts=list((root/'workspace/logs').glob('*.sh'))
            self.assertEqual(len(scripts),1)
            text=scripts[0].read_text()
            self.assertIn('--in-allocation',text)
            self.assertIn('exec ',text)
            snapshots=list((root/'workspace/logs').glob('*_config.json'))
            self.assertEqual(read_json(snapshots[0]),config)

    def test_no_login_compute_fallback(self):
        with patch.object(sys,'argv',['run_all.py','--in-allocation']),patch.object(sys,'platform','linux'),patch.dict('os.environ',{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'refusing to run on login'): run_all.main()


if __name__=='__main__': unittest.main()
