"""Diagnostic-only cross-checkpoint normalizer swap; never save or train the hybrid policy."""
from pathlib import Path
import argparse,fcntl,hashlib,json
BASE=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261024-strict3762')
ROOT=BASE/'publication/repro/sc0090-v2/strict3762-20261024'
SOURCE=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--normalizer-from',type=Path,required=True);p.add_argument('--device',choices=['cuda:1'],default='cuda:1');args=p.parse_args()
 assert read(BASE/'chain-status.json')['status']=='TRAINING_CHAIN_FINISHED_PENDING_FINAL_AUDIT','Do not interfere with formal history replay'
 lock=open(BASE/'qualification/process.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 permitted={str((BASE/'formal'/name/f'checkpoint_{n:06d}.pt').resolve()):n for name,n in [('head-precision-allocation-001',2250),('head-precision-allocation-002',2500),('head-precision-allocation-003',2750)]}
 assert str(args.checkpoint.resolve()) in permitted
 assert str(args.normalizer_from.resolve()) in permitted
 assert permitted[str(args.checkpoint.resolve())] in (2250,2500) and permitted[str(args.normalizer_from.resolve())] in (2250,2500)
 assert args.checkpoint.resolve()!=args.normalizer_from.resolve()
 parent=read(args.checkpoint.parent/'run.json');history=read(ROOT/'history-inventory.json')['phases'][4]
 assert parent['source_sha256']==history['source_files']
 materialized=read(ROOT/'materialized-sources.json');built=materialized['phases'][4]
 expected={**materialized['static_file_hashes'],**history['source_files'],**built['added_meshes']}
 expected[built['only_modified_static_file']]=built['corrected_xml_sha256']
 for name,h in expected.items():assert sha(SOURCE/name)==h,name
 args.output.mkdir(parents=True,exist_ok=False)
 import numpy as np,torch,mujoco
 from dataclasses import asdict
 from tensordict import TensorDict
 import mjlab.tasks
 from mjlab.tasks.registry import load_rl_cfg
 from rsl_rl.utils import resolve_callable,resolve_obs_groups
 from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS
 import train_v2_standup as native
 torch.set_num_threads(2)
 task=parent['task'];assert task=='Mjlab-StandUp-Flat-MicroDuck-V2-Official-Baseline'
 state=torch.load(args.checkpoint,map_location=args.device,weights_only=False);update=permitted[str(args.checkpoint.resolve())]
 assert state['infos']['v2']['completed_updates']==update and state['infos']['env_state']['common_step_counter']==update*24
 donor=torch.load(args.normalizer_from,map_location=args.device,weights_only=False)
 norm_keys=[k for k in state['actor_state_dict'] if k.startswith('obs_normalizer.')]
 assert set(norm_keys)=={'obs_normalizer._mean','obs_normalizer._var','obs_normalizer._std','obs_normalizer.count'}
 for key in norm_keys:state['actor_state_dict'][key]=donor['actor_state_dict'][key].clone()
 del donor
 config=asdict(load_rl_cfg(task));obs=TensorDict({'actor':torch.zeros(1,61,device=args.device),'critic':torch.zeros(1,74,device=args.device)},batch_size=[1]);groups=resolve_obs_groups(obs,config['obs_groups'],['actor','critic'])
 for option in ('cnn_cfg','distribution_cfg'):
  if config['actor'].get(option) is None:config['actor'].pop(option,None)
 if config['actor'].get('rnn_type') is None:
  for option in ('rnn_type','rnn_hidden_dim','rnn_num_layers'):config['actor'].pop(option,None)
 policy=resolve_callable(config['actor'].pop('class_name'))(obs,groups,'actor',14,**config['actor']).to(args.device);policy.load_state_dict(state['actor_state_dict'],strict=True);policy.eval()
 environment_class=native.ManagerBasedRlEnv;traces=[]
 def observed_environment(cfg,device):
  env=environment_class(cfg,device=device);seed=cfg.seed;model=env.sim.mj_model;n=env.num_envs;assert n==896
  (args.output/f'eval-config-{seed}.json').write_text(json.dumps(asdict(cfg),indent=2,default=str)+'\n')
  env.scene.write(args.output/f'scene-{seed}')
  joints=[]
  for name in POLICY_JOINTS:
   matches=[j for j in range(model.njnt) if (mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or '').split('/')[-1]==name];assert len(matches)==1
   j=matches[0];joints.append(dict(name=name,joint_id=j,qpos_address=int(model.jnt_qposadr[j]),dof_address=int(model.jnt_dofadr[j]),limits_rad=model.jnt_range[j].tolist(),limited=bool(model.jnt_limited[j])))
  data=dict(qpos=torch.empty(400,n,model.nq,device=device),qvel=torch.empty(400,n,model.nv,device=device),actions=torch.empty(400,n,14,device=device),stable_hold=torch.empty(400,n,dtype=torch.long,device=device))
  count=0;initial=[];old_reset=env.reset;old_step=env.step;old_close=env.close
  def reset(*a,**kw):
   result=old_reset(*a,**kw);initial.append(env.sim.data.qpos.clone());return result
  def step(actions):
   nonlocal count
   result=old_step(actions);assert count<400
   data['qpos'][count].copy_(env.sim.data.qpos);data['qvel'][count].copy_(env.sim.data.qvel);data['actions'][count].copy_(actions);data['stable_hold'][count].copy_(native.recovery._sc0090_recovery_state(env).hold);count+=1
   return result
  def close():
   try:
    torch.cuda.synchronize(device)
    arrays={name:value[:count].cpu().numpy() for name,value in data.items()}
    arrays['env_origins']=env.scene.env_origins.cpu().numpy()
    if initial:arrays['initial_qpos']=initial[-1].cpu().numpy()
    path=args.output/f'traces-{seed}.npz';np.savez_compressed(path,**arrays)
    meta=dict(seed=seed,steps=count,reset_calls=len(initial),num_envs=n,nq=int(model.nq),nv=int(model.nv),policy_joints=joints,trace_sha256=sha(path),dt=.02,bucket_order=list(native.recovery.SC0090_RECOVERY_BUCKETS),samples_per_bucket=128,scene_scope='Exported scene is for kinematic replay of saved qpos, not a new physics integration')
    (args.output/f'metadata-{seed}.json').write_text(json.dumps(meta,indent=2)+'\n');traces.append(meta)
   finally:old_close()
  env.reset=reset;env.step=step;env.close=close;return env
 native.ManagerBasedRlEnv=observed_environment
 try:
  with torch.inference_mode():cases=native.evaluate(task,policy,args.device,5,128)
 finally:native.ManagerBasedRlEnv=environment_class
 assert len(traces)==2 and all(x['steps']==400 and x['reset_calls']==1 for x in traces)
 for name,value in state['actor_state_dict'].items():assert torch.equal(policy.state_dict()[name],value),name
 reference=args.checkpoint.parent/f'evaluation_{update:06d}.json';old=read(reference)
 def counts(rows):return [(c['seed'],{name:{k:g[k] for k in ('trials','wins','any_stable','faults')} for name,g in c['groups'].items()}) for c in rows]
 result=dict(status='DIAGNOSTIC_NORMALIZER_ABLATION_NOT_A_TRAINED_MODEL',normalizer_checkpoint=str(args.normalizer_from),normalizer_checkpoint_sha256=sha(args.normalizer_from),normalizer_override_keys=norm_keys,hybrid_policy_not_saved_or_trained=True,checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),completed_updates=update,task=task,device=args.device,native_evaluate_source_sha256=sha(SOURCE/'scripts/train_v2_standup.py'),observer_sha256=sha(Path(__file__)),reference_evaluation_sha256=sha(reference),same_counts_as_native_reference=counts(cases)==counts(old['cases']),actor_and_normalizer_unchanged=True,cases=cases,traces=traces,limitations=['Recording only adds tensor copies and output; it is not a policy/reward modification','Fresh-process replay may expose numerical sensitivity; mismatched counts must be investigated rather than forced','Control-rate traces do not capture all 200 Hz substep extremes'])
 (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(status=result['status'],update=update,same_counts_as_reference=result['same_counts_as_native_reference'],right_side=[c['groups']['right_side'] for c in cases])),flush=True)
if __name__=='__main__':main()
