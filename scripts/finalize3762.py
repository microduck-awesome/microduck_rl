"""Wait for the named run, then validate and evaluate its actual final artifacts."""
from pathlib import Path
import hashlib,json,subprocess,time,sys,shutil
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT.parent/'formal-resumed'
OUT=ROOT.parent/'final-evaluation'
TASK='Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline'
OLD=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/baseline/paired-feedback-allocation-001/checkpoint_003762.pt')

def write(d):
 p=OUT/'status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)

def equal(a,b):
 import torch
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:equal(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):equal(x,y)
 else:assert a==b

def main():
 OUT.mkdir(exist_ok=False)
 write(dict(status='WAITING_FOR_TRAINING'))
 while True:
  active=subprocess.check_output(['systemctl','--user','show','microduck-rise3762-collision-resumed','--property=ActiveState','--value'],text=True).strip()
  if active not in ('active','activating','deactivating'):break
  time.sleep(30)
 for rank in (0,1):
  d=json.loads((RUN/f'rank{rank}/status.json').read_text())
  assert d['status']=='ALLOCATION_COMPLETE' and d['completed_updates']==3762,d
 import torch,numpy as np,onnx,onnxruntime as ort
 from dataclasses import asdict
 from tensordict import TensorDict
 import mjlab.tasks
 from mjlab.tasks.registry import load_rl_cfg
 from rsl_rl.utils import resolve_callable,resolve_obs_groups
 from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS
 torch.set_num_threads(2)
 cps=[RUN/f'rank{rank}/resume_003762.pt' for rank in (0,1)]
 states=[torch.load(p,map_location='cpu',weights_only=False) for p in cps]
 for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict'):equal(states[0][key],states[1][key])
 for field in ('stage','polish'):equal(states[0]['infos']['v2']['recovery'][field],states[1]['infos']['v2']['recovery'][field])
 assert hashlib.sha256(OLD.read_bytes()).hexdigest()=='e753856776bb43e44d9465b4067a0b9cd2d10fe03fc8cb88ee4177ec786f93e1'
 exported=OUT/'rise3762-collision.onnx';shutil.copy2(RUN/'rank0/rank0.onnx',exported)
 model=onnx.load(str(exported));onnx.checker.check_model(model)
 metadata={p.key:p.value for p in model.metadata_props}
 assert tuple(metadata['joint_names'].split(','))==POLICY_JOINTS
 config=asdict(load_rl_cfg(TASK));obs=TensorDict({'actor':torch.zeros(1,61),'critic':torch.zeros(1,74)},batch_size=[1])
 groups=resolve_obs_groups(obs,config['obs_groups'],['actor','critic'])
 for option in ('cnn_cfg','distribution_cfg'):
  if config['actor'].get(option) is None:config['actor'].pop(option,None)
 if config['actor'].get('rnn_type') is None:
  for option in ('rnn_type','rnn_hidden_dim','rnn_num_layers'):config['actor'].pop(option,None)
 policy=resolve_callable(config['actor'].pop('class_name'))(obs,groups,'actor',14,**config['actor'])
 policy.load_state_dict(states[0]['actor_state_dict']);policy.eval()
 session=ort.InferenceSession(str(exported),providers=['CPUExecutionProvider'])
 rng=np.random.default_rng(20261023);error=0.
 samples=[np.zeros((1,61),np.float32),np.linspace(-.1,.1,61,dtype=np.float32)[None]]
 mean=policy.obs_normalizer.mean.detach().numpy();std=policy.obs_normalizer.std.detach().numpy()
 samples += [(mean+rng.normal(size=(1,61))*std*.5).astype(np.float32) for _ in range(64)]
 for x in samples:
  result=session.run(None,{session.get_inputs()[0].name:x})[0]
  with torch.inference_mode():expected=policy({'actor':torch.as_tensor(x)}).numpy()
  assert result.shape==(1,14) and np.isfinite(result).all()
  np.testing.assert_allclose(result,expected,atol=2e-5,rtol=2e-5)
  error=max(error,float(np.abs(result-expected).max()))
 report=dict(status='PASS',updates=3762,rank_weights_normalizers_optimizer_equal=True,onnx_normalizer_parity_max_error=error,parity_samples=len(samples),checkpoint_sha256=hashlib.sha256(cps[0].read_bytes()).hexdigest(),onnx_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),collision_xml_sha256=hashlib.sha256((ROOT/'src/mjlab_microduck/custom_v2/assets/robot.xml').read_bytes()).hexdigest())
 (OUT/'artifact-verification.json').write_text(json.dumps(report,indent=2)+'\n')
 jobs=[('recovery-new',['scripts/evaluate3762_recovery.py','--checkpoint',str(cps[0]),'--stage','5','--samples','128']),('recovery-old',['scripts/evaluate3762_recovery.py','--checkpoint',str(OLD),'--stage','5','--samples','128']),('head-new',['scripts/evaluate_v2_head.py','--task',TASK,'--checkpoint',str(cps[0]),'--repeats','8'])]
 for name,args in jobs:
  write(dict(status='EVALUATING',current=name,artifact_verification=report))
  with (OUT/f'{name}.log').open('w') as log:
   subprocess.run([sys.executable,*args,'--device','cuda:0','--output',str(OUT/name)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 write(dict(status='EVALUATION_COMPLETE_REQUIRES_REVIEW',artifact_verification=report))

if __name__=='__main__':
 try:main()
 except BaseException as e:
  if OUT.exists():write(dict(status='FAILED',error=repr(e)))
  raise
