"""Verify a completed head-phase checkpoint and its baked-normalizer export."""
import argparse,json,hashlib,shutil
from pathlib import Path
from finalize3762 import ROOT,TASK,OLD,equal

def main():
 p=argparse.ArgumentParser();p.add_argument('--run-root',type=Path,required=True);p.add_argument('--update',type=int,required=True);p.add_argument('--head-stage',type=int,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 RUN=args.run_root.resolve();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
 for rank in (0,1):
  status=json.loads((RUN/f'rank{rank}/status.json').read_text())
  assert status['status']=='ALLOCATION_COMPLETE' and status['completed_updates']==args.update,status
 import torch,numpy as np,onnx,onnxruntime as ort
 from dataclasses import asdict
 from tensordict import TensorDict
 import mjlab.tasks
 from mjlab.tasks.registry import load_rl_cfg
 from rsl_rl.utils import resolve_callable,resolve_obs_groups
 from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS
 torch.set_num_threads(2)
 cps=[RUN/f'rank{rank}/resume_{args.update:06d}.pt' for rank in (0,1)]
 states=[torch.load(p,map_location='cpu',weights_only=False) for p in cps]
 for state in states:
  assert state['infos']['v2']['completed_updates']==args.update
  assert state['infos']['v2']['head_precision_recipe']['stage']==args.head_stage
  assert state['infos']['v2']['recovery']['polish']
 for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict'):equal(states[0][key],states[1][key])
 for field in ('stage','polish'):equal(states[0]['infos']['v2']['recovery'][field],states[1]['infos']['v2']['recovery'][field])
 assert hashlib.sha256(OLD.read_bytes()).hexdigest()=='e753856776bb43e44d9465b4067a0b9cd2d10fe03fc8cb88ee4177ec786f93e1'
 exported=OUT/'policy.onnx';shutil.copy2(RUN/'rank0/rank0.onnx',exported)
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
 report=dict(status='PASS',updates=args.update,rank_weights_normalizers_optimizer_equal=True,onnx_normalizer_parity_max_error=error,parity_samples=len(samples),checkpoint_sha256=hashlib.sha256(cps[0].read_bytes()).hexdigest(),onnx_sha256=hashlib.sha256(exported.read_bytes()).hexdigest(),collision_xml_sha256=hashlib.sha256((ROOT/'src/mjlab_microduck/custom_v2/assets/robot.xml').read_bytes()).hexdigest())
 (OUT/'artifact-verification.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report))

if __name__=='__main__':main()
