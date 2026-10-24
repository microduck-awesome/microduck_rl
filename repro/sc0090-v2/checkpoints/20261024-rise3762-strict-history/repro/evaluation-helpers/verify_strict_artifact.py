"""Verify completed strict-history artifacts on CPU; never train or run before 3762."""
from pathlib import Path
import hashlib,json,shutil
BASE=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261024-strict3762')
ROOT=BASE/'publication/repro/sc0090-v2/strict3762-20261024'
SOURCE=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 chain=read(BASE/'chain-status.json')
 assert chain['status']=='TRAINING_CHAIN_FINISHED_PENDING_FINAL_AUDIT' and chain['completed_updates']==3762
 plan=read(ROOT/'execution-plan.json');history=read(ROOT/'history-inventory.json')
 phase_proofs=[]
 for row,old in zip(plan['phases'],history['phases']):
  output=Path(row['output']);record=read(output.parent/f'{row["name"]}-process.json')
  assert record['status']=='PASS' and record['returncode']==0 and record['completed_updates']==row['expected_completed_updates']
  assert sha(output/'run.json')==record['run_sha256']
  for filename,h in record['checkpoint_hashes'].items():assert sha(output/filename)==h
  assert read(output/'run.json')['source_sha256']==old['source_files']
  phase_proofs.append(dict(name=row['name'],completed_updates=record['completed_updates'],receipt_sha256=sha(output.parent/f'{row["name"]}-process.json')))
 for name,h in history['phases'][-1]['source_files'].items():assert sha(SOURCE/name)==h,name
 import torch,numpy as np,onnx,onnxruntime as ort
 from dataclasses import asdict
 from tensordict import TensorDict
 import mjlab.tasks
 from mjlab.tasks.registry import load_rl_cfg
 from rsl_rl.utils import resolve_callable,resolve_obs_groups
 from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS
 torch.set_num_threads(2)
 final=BASE/'formal/paired-feedback-allocation-001';checkpoint=final/'checkpoint_003762.pt';resume=final/'resume_003762.pt'
 before=torch.load(checkpoint,map_location='cpu',weights_only=False);state=torch.load(resume,map_location='cpu',weights_only=False)
 def equal(a,b):
  if isinstance(a,torch.Tensor):assert torch.equal(a,b)
  elif isinstance(a,dict):
   assert a.keys()==b.keys()
   for k in a:equal(a[k],b[k])
  elif isinstance(a,(list,tuple)):
   assert len(a)==len(b)
   for x,y in zip(a,b):equal(x,y)
  else:assert a==b
 for name in ('actor_state_dict','critic_state_dict','optimizer_state_dict'):equal(before[name],state[name])
 assert state['infos']['v2']['completed_updates']==3762 and state['infos']['env_state']['common_step_counter']==90288
 for name,tensor in state['actor_state_dict'].items():assert torch.isfinite(tensor).all(),name
 exported=final/'paired-feedback-allocation-001.onnx'
 model=onnx.load(str(exported));onnx.checker.check_model(model);metadata={p.key:p.value for p in model.metadata_props}
 assert tuple(metadata['joint_names'].split(','))==POLICY_JOINTS
 assert float(metadata['action_scale'])==1.
 task='Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline'
 config=asdict(load_rl_cfg(task));obs=TensorDict({'actor':torch.zeros(1,61),'critic':torch.zeros(1,74)},batch_size=[1]);groups=resolve_obs_groups(obs,config['obs_groups'],['actor','critic'])
 for option in ('cnn_cfg','distribution_cfg'):
  if config['actor'].get(option) is None:config['actor'].pop(option,None)
 if config['actor'].get('rnn_type') is None:
  for option in ('rnn_type','rnn_hidden_dim','rnn_num_layers'):config['actor'].pop(option,None)
 policy=resolve_callable(config['actor'].pop('class_name'))(obs,groups,'actor',14,**config['actor'])
 policy.load_state_dict(state['actor_state_dict'],strict=True);policy.eval()
 session=ort.InferenceSession(str(exported),providers=['CPUExecutionProvider'])
 rng=np.random.default_rng(20261023);error=0.
 samples=[np.zeros((1,61),np.float32),np.linspace(-.1,.1,61,dtype=np.float32)[None]]
 mean=policy.obs_normalizer.mean.detach().numpy();std=policy.obs_normalizer.std.detach().numpy()
 samples += [(mean+rng.normal(size=(1,61))*std*.5).astype(np.float32) for _ in range(64)]
 for x in samples:
  result=session.run(None,{session.get_inputs()[0].name:x})[0]
  with torch.inference_mode():expected=policy({'actor':torch.as_tensor(x)}).numpy()
  assert result.shape==(1,14) and np.isfinite(result).all()
  np.testing.assert_allclose(result,expected,atol=2e-5,rtol=2e-5);error=max(error,float(np.abs(result-expected).max()))
 output=BASE/'final-evaluation';output.mkdir(exist_ok=False)
 shutil.copyfile(exported,output/'policy.onnx')
 report=dict(status='PASS_ARTIFACTS_ONLY',completed_updates=3762,phases=phase_proofs,checkpoint_and_resume_actor_critic_optimizer_equal=True,checkpoint_sha256=sha(checkpoint),resume_sha256=sha(resume),onnx_sha256=sha(exported),normalizer_parity_max_error=error,parity_samples=66,robot_xml_sha256=sha(SOURCE/'src/mjlab_microduck/custom_v2/assets/robot.xml'),verifier_sha256=sha(Path(__file__)),behavior_evaluation_pending=True)
 (output/'artifact-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
