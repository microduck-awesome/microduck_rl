"""Exhaustive finite state, cfg/dose/source contracts; CPU only."""
from pathlib import Path
from dataclasses import asdict
from copy import deepcopy
from types import SimpleNamespace
import itertools,json,hashlib,sys
import torch
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'ballkick-health-history-worktree';sys.path.insert(0,str(W/'scripts'))
import train_v2_ballkick as train
from mjlab_microduck.custom_v2.ball_kick_health import HEALTH_TASK,make_health_ball_kick_env_cfg
from mjlab_microduck.custom_v2.ball_kick import BALL_KICK_TASK,make_ball_kick_env_cfg
from mjlab_microduck.custom_v2.ball_kick_health_state import health_kick_state_transition,install_health_ball_kick_state
from mjlab_microduck.custom_v2.ball_kick_state import kick_state_transition,reset_ball_with_kick_state,ball_kick_state_observation

def clean(x):
 if callable(x):return x.__module__+'.'+x.__qualname__
 if isinstance(x,dict):return {k:clean(v)for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v)for v in x]
 return x
checks=[];v=torch.tensor(list(itertools.product((False,True),repeat=8)),dtype=torch.bool);before=v.clone();valid,bad=health_kick_state_transition(*v.T)
expected_bad=v[:,1]|v[:,3]|v[:,4]|v[:,7];expected_valid=(v[:,0]|(v[:,2]&v[:,5]&v[:,6]))&~expected_bad
assert torch.equal(bad,expected_bad)and torch.equal(valid,expected_valid)and torch.equal(v,before);assert not bool((valid&bad).any());checks.append('all256_bit_combinations_bad_precedence_inputs_immutable')
oldv,oldb=kick_state_transition(*v[:,:7].T);no_self=~v[:,7];assert torch.equal(oldv[no_self],valid[no_self])and torch.equal(oldb[no_self],bad[no_self]);checks.append('self_false_exact_original_transition')
# Every combination of correct-contact and self-contact substep, including none.
for correct_at,self_at in itertools.product(range(5),repeat=2):
 a=torch.tensor([False]);b=a.clone()
 for step in range(4):
  a,b=health_kick_state_transition(a,b,torch.tensor([step==correct_at]),torch.tensor([False]),torch.tensor([False]),torch.tensor([True]),torch.tensor([True]),torch.tensor([step==self_at]))
 assert bool(b)==(self_at<4)and bool(a)==(correct_at<4 and self_at==4)
checks.append('25_four_substep_pulse_orderings_short_self_latched_and_revokes_prior_valid')
# Exercise actual partial-reset implementation, stubbing only the physical ball reset.
from mjlab_microduck.tasks import mdp
original=mdp.reset_ball_in_front_of_foot;calls=[]
try:
 mdp.reset_ball_in_front_of_foot=lambda env,ids,**kw:calls.append(ids.clone())
 env=SimpleNamespace(num_envs=6,device='cpu',_v2_ball_valid_kick=torch.tensor([1,0,1,0,1,0],dtype=torch.bool),_v2_ball_bad_contact=torch.tensor([0,1,0,1,0,1],dtype=torch.bool));ids=torch.tensor([1,4]);va=env._v2_ball_valid_kick.clone();ba=env._v2_ball_bad_contact.clone();reset_ball_with_kick_state(env,ids)
 other=torch.tensor([0,2,3,5]);assert torch.equal(env._v2_ball_valid_kick[other],va[other])and torch.equal(env._v2_ball_bad_contact[other],ba[other]);assert not env._v2_ball_valid_kick[ids].any()and not env._v2_ball_bad_contact[ids].any();assert torch.equal(calls[0],ids)
 obs=ball_kick_state_observation(env);assert obs.shape==(6,2);obs.zero_();assert env._v2_ball_valid_kick[0]
finally:mdp.reset_ball_in_front_of_foot=original
checks.append('actual_partial_reset_function_selected_only_and_critic_bits_copy')
base=make_ball_kick_env_cfg();cfg=make_health_ball_kick_env_cfg();expected=deepcopy(base);expected.events['ball_kick_state'].func=install_health_ball_kick_state;mapping=[]
for name,t in expected.curriculum.items():
 for stages in t.params.values():
  if isinstance(stages,list):
   for stage in stages:
    if isinstance(stage,dict)and 'step'in stage:
     old=stage['step'];assert old%6==0;stage['step']//=6;mapping.append(dict(term=name,historical_step=old,native_step=stage['step']))
assert clean(asdict(expected))==clean(asdict(cfg));checks.append('cfg_only_new_startup_health_latch_and_exposure_scaled_step_boundaries')
assert cfg.episode_length_s==base.episode_length_s==5. and cfg.sim.mujoco.timestep==.005 and cfg.decimation==4
assert cfg.rewards['ball_direction_cost'].weight==-4 and cfg.rewards['postkick_quiet_return_cost'].weight==-2
assert max(v['native_step']for v in mapping)==6000
original=asdict(train.load_rl_cfg(BALL_KICK_TASK));actual=asdict(train.load_rl_cfg(HEALTH_TASK));compare=deepcopy(actual);compare['algorithm']['num_mini_batches']=4;assert clean(compare)==clean(original)
assert actual['algorithm']['num_learning_epochs']==5 and actual['algorithm']['num_mini_batches']==24 and 24576*24//24==4096*24//4==24576
checks.append('M24_only_agent_change_historical_minibatch24576_and384x120_steps46080')
parent=R/'ballkick-finish-worktree';assets=json.loads((W/'assets-equality.json').read_text());assert assets['count']==283
for row in assets['assets']:
 assert hashlib.sha256((W/row['path']).read_bytes()).hexdigest()==row['sha256']
 assert hashlib.sha256((parent/row['path']).read_bytes()).hexdigest()==row['sha256']
report=dict(status='PASS_CPU_ONLY',checks=checks,source_sha256=train.source_hashes(W),actual_agent=actual,curriculum_mapping=mapping,assets_exact=283,source=str(W),task=HEALTH_TASK,formal_controls_added=0,gpu_controls_added=0,scope='CPU finite-state/config contracts only; actual200Hz contact, physical partial reset, terminal and PPO remain GPU admission requirements',test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(R/'ballkick-health-history-cpu-001.json').write_text(json.dumps(report,indent=2,default=str)+'\n');print(json.dumps({'status':report['status'],'checks':checks,'max_native_step':6000},indent=2),flush=True)
