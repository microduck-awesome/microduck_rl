"""CPU-only B0 contract against original official observation/action configuration."""
from pathlib import Path
from dataclasses import asdict,is_dataclass
from copy import deepcopy
import hashlib,json,inspect,functools
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab_microduck.custom_v2.ground_pick_baseline import BASELINE_TASK,SENSOR_CHAIN,make_baseline_ground_pick_env_cfg,mark_baseline_contract
from mjlab_microduck.custom_v2.ground_pick_history import HISTORY_TASK
from mjlab_microduck.custom_v2 import microduck_v2 as v2
from mjlab_microduck.tasks.microduck_ground_pick_env_cfg import make_microduck_ground_pick_env_cfg
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/baseline');W=R/'groundpick-route-worktree';P=R.parent/'p04/groundpick-health-history-worktree'
def canonical(x):
 if isinstance(x,slice):return {'__slice__':[x.start,x.stop,x.step]}
 if is_dataclass(x)and not isinstance(x,type):return canonical(asdict(x))
 if isinstance(x,dict):return {str(k):canonical(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [canonical(v)for v in x]
 if isinstance(x,functools.partial):return ['partial',canonical(x.func),canonical(x.args),canonical(x.keywords)]
 if callable(x):return [x.__module__,x.__qualname__,inspect.getsource(x),[canonical(c.cell_contents)for c in(getattr(x,'__closure__',None)or())]]
 return x

def main():
 reports=[]
 for play in(False,True):
  base=load_env_cfg(HISTORY_TASK,play=play);candidate=make_baseline_ground_pick_env_cfg(play=play);reference=make_microduck_ground_pick_env_cfg(play=play,rough=False);v2._policy_selectors(reference)
  assert type(candidate.actions['joint_pos'])is JointPositionActionCfg
  a=deepcopy(reference.actions['joint_pos']);a.actuator_names=v2.POLICY_JOINTS;a.preserve_order=True
  assert canonical(candidate.actions['joint_pos'])==canonical(a)
  for group in('actor','critic'):
   for name in('joint_pos','joint_vel'):assert canonical(candidate.observations[group].terms[name])==canonical(reference.observations[group].terms[name])
  assert canonical(candidate.events['encoder_bias'])==canonical(reference.events['encoder_bias'])
  assert candidate.events['mouth_hold'].func is v2.hold_mouth and candidate.events['mouth_hold'].mode=='step'
  assert candidate.events['baseline_contract'].func is mark_baseline_contract
  # Revert only declared interface differences; everything else must be exact.
  restored=deepcopy(candidate);restored.actions['joint_pos']=base.actions['joint_pos']
  for group in('actor','critic'):
   for name in('joint_pos','joint_vel'):restored.observations[group].terms[name]=base.observations[group].terms[name]
  for name in('encoder_bias','mouth_hold','baseline_contract'):restored.events.pop(name)
  restored.terminations['sc0090_feedback_fault']=base.terminations['sc0090_feedback_fault']
  assert canonical(restored)==canonical(base),'Unapproved physical/task/curriculum difference'
  assert candidate.scene.entities['robot'].init_state.joint_pos=={'.*':0.}
  assert candidate.sim.mujoco.timestep==.005 and candidate.decimation==4 and candidate.episode_length_s==20.
  assert 'sc0090_feedback_fault'not in candidate.terminations
  ar=candidate.observations['actor'].terms;cr=candidate.observations['critic'].terms
  assert(ar['joint_pos'].noise.n_min,ar['joint_pos'].noise.n_max)==(-.001,.001)
  assert(ar['joint_vel'].noise.n_min,ar['joint_vel'].noise.n_max)==(-.25,.25)
  assert ar['joint_pos'].delay_min_lag==ar['joint_pos'].delay_max_lag==0
  assert ar['joint_vel'].delay_min_lag==ar['joint_vel'].delay_max_lag==1
  assert ar['joint_pos'].params['biased'] is True and cr['joint_pos'].params['biased'] is False
  assert candidate.observations['critic'].enable_corruption is False
  assert all(cr[n].delay_max_lag==0 for n in('joint_pos','joint_vel'))
  # Native ObservationManager deepcopies each term and sets noise=None for
  # this group; raw shared-template term noise is not the runtime noise.
  reports.append(dict(play=play,action=canonical(candidate.actions['joint_pos']),actor_joint_pos=canonical(ar['joint_pos']),actor_joint_vel=canonical(ar['joint_vel']),encoder_bias=canonical(candidate.events['encoder_bias']),approved_only_interface_delta=True))
 old=asdict(load_rl_cfg(HISTORY_TASK));new=asdict(load_rl_cfg(BASELINE_TASK));new['experiment_name']=old['experiment_name'];new['run_name']=old['run_name'];assert canonical(old)==canonical(new)
 assert old['num_steps_per_env']==24 and old['algorithm']['num_learning_epochs']==5 and old['algorithm']['num_mini_batches']==24
 # Keep actual assets/model/actuator sources exact; no XML/coefficients changed.
 assets=[p for p in(P/'src').rglob('*')if p.is_file()and p.suffix not in('.py','.cpp','.pyc')and '__pycache__'not in p.parts]
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 assert all(sha(p)==sha(W/p.relative_to(P))for p in assets)
 for name in('src/mjlab_microduck/actuator/sc0090_v5.py','src/mjlab_microduck/custom_v2/runtime.py','src/mjlab_microduck/custom_v2/terminal.py','src/mjlab_microduck/custom_v2/ground_pick_history.py'):assert sha(W/name)==sha(P/name)
 import train_v2_ground_pick as trainer
 report=dict(status='PASS',task=BASELINE_TASK,sensor_chain=SENSOR_CHAIN,feedback_fault_metric_applicable=False,source_sha256=trainer.source_hashes(W),train_play=reports,physics_and_rewards_curriculum_exact=True,agent_exact_except_names=True,assets_exact=len(assets),B0='Continuous official JointPositionAction; officialjointnoise/delay/encoderbias, no P04stream/count/residual; sameV2v5fullcontactphysics',budget=dict(N=24576,T=24,E=5,M=24,fresh_updates=512,new_formal_controls=301989888),fresh_only=True,script_sha256=sha(Path(__file__)))
 (R/'groundpick-baseline-cpu-001.json').write_text(json.dumps(report,indent=2)+'\n');print('BASELINE_GROUNDPICK_CPU_PASS',flush=True);return report
if __name__=='__main__':main()
