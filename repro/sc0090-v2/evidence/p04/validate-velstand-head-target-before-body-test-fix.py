"""Actual checkpoint migration, unchanged physics, and isolated BC label checks."""
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict
import importlib.util,json,hashlib,math,time
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab_microduck.custom_v2.velstand import TASK
from mjlab_microduck.custom_v2.velstand_ppo import VelStandPPO,teacher_masks,walking_head_target,head_target_transition_only
r=Path(__file__).resolve().parent;w=r/'velstand-head-target-worktree';torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('train_head_target',w/'scripts/train_v2_velstand.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
checkpoint=r/'velstand-v8-allocation-001/checkpoint_000128.pt';parent=torch.load(checkpoint,map_location='cpu',weights_only=False);meta=parent['infos']['v2']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=dict(schema=1,kind='walking_BC_head_target_only',parent_checkpoint_sha256=sha(checkpoint),parent_task=meta['task'],target_task=TASK,parent_updates=128,parent_source_sha256=meta['source_sha256'],target_source_sha256=m.source_hashes(w),changed_files=sorted(m.TRANSITION_FILES),parent_run_sha256=sha(checkpoint.parent/'run.json'),equivalent_source_roots=[str(w),str(r/'velstand-v8-repaired-worktree')],phase_total_maximum_controls=150994944,remaining_formal_controls=75497472)
p=r/'velstand-head-target-transition.json';assert not p.exists();p.write_text(json.dumps(manifest,indent=2)+'\n')
cfg=load_env_cfg(TASK);cfg.seed=2026101618;cfg.scene.num_envs=64
m.validate_source_transition(p,checkpoint,parent,m.source_hashes(w),TASK,cfg)
checks=['parent_SHA_and_exact_three_source_changes_and_physical_cfg_identity']
action=cfg.actions['joint_pos'];assert action.scale==1. and action.offset==0. and action.use_default_offset
assert tuple(action.actuator_names[5:9])==('neck_pitch','head_pitch','head_yaw','head_roll')
from mjlab_microduck.custom_v2.training import get_full_spec
model=get_full_spec().compile()
assert all(abs(float(model.qpos0[model.joint(name).qposadr[0]]))<1e-8 for name in action.actuator_names[5:9])
checks.append('head_action_order_scale_and_physical_HOME_zero')
agent=load_rl_cfg(TASK);bc=deepcopy(agent.algorithm.bc_cfg)
bc.update(checkpoint_path=str(r/'head2-micro-exposure-allocation-001/checkpoint_001856.pt'),anchor_checkpoint_path=str(r/'walking-yaw-fine-allocation-002/checkpoint_000512.pt'))
raw=torch.zeros(64,61);raw[:,5]=-1;raw[:,48]=.2;raw[:,51:55]=torch.linspace(-.1,.1,256).reshape(64,4)
raw[0,48]=0;raw[1,4:6]=torch.tensor([math.sin(math.radians(40)),-math.cos(math.radians(40))]);raw[2,58]=.1;raw[3,4:6]=torch.tensor([.5,-math.sqrt(.75)])
original=raw.clone();s,v,_,_,_=teacher_masks(raw,bc);assert s[:4].tolist()==[True,True,False,False] and not bool(v[:4].any())
labels=torch.randn(64,14);before=labels.clone();changed=walking_head_target(labels[v],raw[v]);assert torch.equal(changed[:,5:9],raw[v,51:55]);leg=list(range(5))+list(range(9,14));assert torch.equal(changed[:,leg],labels[v][:,leg]);assert torch.equal(raw,original) and torch.equal(labels,before)
checks.append('only_walk_routed_four_labels_changed_and_all_inputs_immutable')
obs=TensorDict({'actor':raw,'critic':torch.zeros(64,76)},batch_size=[64]);groups={'actor':['actor'],'critic':['critic']}
a=MLPModel(obs,groups,'actor',14,hidden_dims=(512,256,128),activation='elu',obs_normalization=True,distribution_cfg=dict(class_name='GaussianDistribution',init_std=1.,std_type='scalar'))
c=MLPModel(obs,groups,'critic',1,hidden_dims=(512,256,128),activation='elu',obs_normalization=True)
storage=RolloutStorage('rl',64,2,obs,(14,),'cpu');storage.observations['actor'][:]=raw
learner=VelStandPPO(a,c,storage,bc_cfg=bc,device='cpu')
oldcontract=parent['velstand_teacher_contract'];head_target_transition_only(oldcontract,learner.teacher_contract)
try:learner.load(parent,None,True)
except AssertionError:pass
else:raise AssertionError('unapproved contract transition accepted')
checks.append('ordinary_load_rejects_changed_contract')
learner.authorize_head_target_transition(oldcontract);learner.load(parent,None,True)
actual=learner.save()
for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict','expert_bc_optimizer_state_dict','expert_bc_update_count'):
 m.exact(actual[key],parent[key])
assert learner.learning_rate==parent['optimizer_state_dict']['param_groups'][0]['lr']
assert '_head_target_transition' not in learner.__dict__
checks.append('actual128_actor_critic_both_normalizers_two_Adam_count_LR_exact')
for mutation in ('teacher','coef'):
 bad=deepcopy(oldcontract)
 if mutation=='teacher':bad['files']['checkpoint_path']['sha256']='wrong'
 else:bad['parameters']['coef']=2.
 try:learner.authorize_head_target_transition(bad)
 except AssertionError:pass
 else:raise AssertionError('Unrelated change accepted: '+mutation)
checks.append('teacher_hash_and_other_BC_parameter_changes_rejected')
# Prove one actual new-label BC update keeps teacher/N/rollout/std state fixed.
state=deepcopy(actual);teacher=[deepcopy(t.state_dict()) for t in (learner.anchor,learner.expert)];observations=storage.observations.clone();std=learner.actor.distribution.std_param.clone();stats=learner._bc_update()
assert all(math.isfinite(v) for v in stats.values()) and stats['expert_bc_steps']>0
for old,new in zip(teacher,(learner.anchor,learner.expert)):m.exact(old,new.state_dict())
for k in observations.keys():m.exact(observations[k],storage.observations[k])
m.exact(std,learner.actor.distribution.std_param)
for key,value in state['actor_state_dict'].items():
 if 'normalizer' in key:m.exact(value,learner.actor.state_dict()[key])
checks.append('actual_new_label_BC_finite_frozen_teachers_N_std_and_rollout')
# Ordinary same-task restore still exact with the new contract.
saved=deepcopy(learner.save());learner.load(saved,None,True);m.exact(learner.save(),saved)
m.exact(parent['velstand_teacher_contract'],oldcontract)
checks.append('same_task_resume_exact_after_transition')
report=dict(status='PASS',checks=checks,BC_diagnostic=stats,manifest_sha256=sha(p),source_sha256=m.source_hashes(w),scope='CPU actual checkpoint migration and BC labels; no GPU/behavior acceptance')
(r/'velstand-head-target-cpu-001.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='PASS',checks=len(checks),BC=stats)),flush=True)
