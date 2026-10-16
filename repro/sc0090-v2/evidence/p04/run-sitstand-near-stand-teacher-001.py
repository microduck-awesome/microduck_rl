"""One4s local teacher handoff probe; no training or runtime modification."""
import os,sys,json,re,hashlib,time,traceback,gc
from pathlib import Path
from dataclasses import asdict
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'sitstand-history-hold320-worktree'
sys.path.insert(0,str(W/'scripts'))
MARK=R/'sitstand-near-stand-teacher-status-001.json';assert not MARK.exists()
S=R/'sitstand-history-hold320-allocation-001/checkpoint_000640.pt'
T=R/'head2-micro-exposure-allocation-001/checkpoint_001856.pt'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def norm(x):
    if isinstance(x,dict):return {k:norm(v)for k,v in x.items()}
    if isinstance(x,list):return [norm(v)for v in x]
    return re.sub(r' at 0x[0-9a-fA-F]+','',x)if isinstance(x,str)else x
sa,ta=read(S.parent/'run.json'),read(T.parent/'run.json')
assert sha(S)=='90b00e9867030d72bd97bf7a91490761c18306b754e2fc8aaeb80cdb3dc85ec5'
assert sha(T)=='e4772c33149410d62446180a9214b26695cb51ba1d5b7b26fac31d3ea273122b'
at,bt=sa['environment']['observations']['actor']['terms'],ta['environment']['observations']['actor']['terms']
assert list(at)==list(bt)
assert all(norm(at[k])==norm(bt[k])for k in at if k!='body_command')
actions=[norm(x['environment']['actions'])for x in [sa,ta]]
archive_roots=[Path(x['joint_pos']['archive_path'])for x in actions]
asset_manifests=[{str(p.relative_to(root)):sha(p)for p in root.rglob('*')if p.is_file()}for root in archive_roots]
assert asset_manifests[0] and asset_manifests[0]==asset_manifests[1]
for x in actions:x['joint_pos']['archive_path']='<verified-identical-archive>'
assert actions[0]==actions[1]
core=[k for k in ta['source_sha256']if k.startswith(('src/mjlab_microduck/sc0090_v2/','src/mjlab_microduck/actuator/'))or k in ('src/mjlab_microduck/custom_v2/fast_native.py','src/mjlab_microduck/custom_v2/head_exploration.py')]
assert all(sa['source_sha256'][k]==ta['source_sha256'][k]==sha(W/k)for k in core)
receipt=dict(status='PASS_INPUT_ACTION_CORE_SOURCE_CONTRACT',matched_core_files=len(core),student=str(S),student_sha256=sha(S),teacher=str(T),teacher_sha256=sha(T),
    teacher_adapter='Clone actual raw61; zero48:51 and55:61 only; keep actual previous actions/feedback and head command; teacher own frozen normalizer.',
    scenario='Same64world original four initial groups,4s fixedfirstposture/headHOME. At2s only initialHOME/STAND withheight>.1m,tilt<30deg,legMAE<.2,bothfeet,finite/ready becomes eligible for fullteacher through4s. Do not reject initialheadcontact because it is the target failure.',
    arms=['policy','near_stand_teacher'],conditions=['nominal','measured'],diagnostic_controls_cap=51200,formal_controls=0,
    scope='Local recovery diagnostic only; no lowSIT transition teacher labels and no actionfilter/state reset atswitch.')
(R/'sitstand-near-stand-teacher-plan-001.json').write_text(json.dumps(receipt,indent=2)+'\n')
completed=0;started=time.time()
def marker(**kw):
    p=MARK.with_suffix('.tmp');p.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),completed_diagnostic_controls=completed,diagnostic_cap=51200,new_formal_controls=0,**kw),indent=2)+'\n');p.replace(MARK)
marker(status='IMPORTING_AFTER_CPU_CONTRACT')
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.sensor import ContactSensorCfg,ContactMatch
from rsl_rl.utils import resolve_callable,resolve_obs_groups
from evaluate_v2_sitstand_clock_fixed import ground_cases,camera_summary
from mjlab_microduck.custom_v2.sitstand_metrics import SitStandMeasurements,BOUNDARY
from train_v2_sitstand import source_hashes
torch.set_num_threads(2);torch.cuda.set_device('cuda:0')
student=torch.load(S,map_location='cuda:0',weights_only=False);teacher=torch.load(T,map_location='cuda:0',weights_only=False)
assert source_hashes(W)==student['infos']['v2']['source_sha256']
def build_actor(cp,run,obs):
    ac=dict(run['agent']['actor']);td=TensorDict(obs,batch_size=[64]);groups=resolve_obs_groups(td,run['agent']['obs_groups'],['actor','critic'])
    for k in ('cnn_cfg','distribution_cfg'):
        if ac.get(k)is None:ac.pop(k,None)
    if ac.get('rnn_type')is None:
        for k in ('rnn_type','rnn_hidden_dim','rnn_num_layers'):ac.pop(k,None)
    actor=resolve_callable(ac.pop('class_name'))(td,groups,'actor',14,**ac).to('cuda:0');actor.load_state_dict(cp['actor_state_dict'],strict=True);actor.eval();return actor
try:
    for arm in receipt['arms']:
        for stage in receipt['conditions']:
            out=R/f'sitstand-near-stand-teacher-{arm}-{stage}-001';out.mkdir(exist_ok=False)
            marker(status='RUNNING_DIAGNOSTIC',arm=arm,stage=stage,output=str(out),ground_update_before=read(R/'groundpick-health-history-allocation-001/status.json')['completed_updates'])
            cfg=load_env_cfg(student['infos']['v2']['task'],play=True);cfg.seed=2603;cfg.scene.num_envs=64;cfg.actions['joint_pos'].randomization_stage=stage
            cfg.curriculum={};cfg.terminations={};cfg.auto_reset=False;cfg.observations['actor'].enable_corruption=False
            cfg.events={n:t for n,t in cfg.events.items()if t.mode=='startup'or n in {'reset_base','reset_robot_joints','set_ground_state','v2_reset_joint_guards','mouth_reset','sensor_error_reset'}}
            cfg.events['set_ground_state'].func=ground_cases
            cfg.scene.sensors=(*cfg.scene.sensors,ContactSensorCfg(name='diagnostic_head_ground',primary=ContactMatch(mode='geom',pattern='^ground_head_shell_collision$',entity='robot'),secondary=ContactMatch(mode='body',pattern='terrain'),fields=('found','force'),reduce='netforce',num_slots=1))
            (out/'config.json').write_text(json.dumps(asdict(cfg),indent=2,default=str)+'\n');(out/'executed-script.py').write_bytes(Path(__file__).read_bytes())
            env=ManagerBasedRlEnv(cfg,device='cuda:0');steps=0
            try:
                mode=torch.arange(64,device='cuda:0')%4;command=(mode%2).float();native=env.command_manager.compute
                def commands(dt):
                    t=env.command_manager.get_term('twist');h=env.command_manager.get_term('head_pose');t.time_left.fill_(1e9);h.time_left.fill_(1e9)
                    t.vel_command_b.zero_();t.vel_command_b[:,0]=command;h._command.zero_();native(dt)
                env.command_manager.compute=commands;obs,_=env.reset();robot=env.scene['robot'];reader=SitStandMeasurements(env,oracle=True)
                a=build_actor(student,sa,obs);b=build_actor(teacher,ta,obs);frozen=[{k:v.clone()for k,v in actor.state_dict().items()}for actor in [a,b]]
                initial=dict(qpos=env.sim.data.qpos.clone().cpu(),qvel=env.sim.data.qvel.clone().cpu(),actor_obs=obs['actor'].clone().cpu(),origins=env.scene.env_origins.clone().cpu())
                data={};idx=[0];native_reward=env.reward_manager.compute
                def found(name):return env.scene[name].data.found.reshape(64,-1).any(-1)
                def record(dt):
                    reward=native_reward(dt);x=reader.read();x['physical_q']=robot.data.joint_pos[:,reader.ids]-robot.data.default_joint_pos[:,reader.ids]
                    x['feet']=env.scene['feet_ground_contact'].data.found.reshape(64,2,-1).any(-1);x['head_contact']=found('diagnostic_head_ground');x['self_contact']=found('self_collision')
                    if not data:data.update({k:torch.empty((200,*v.shape),device=v.device,dtype=v.dtype)for k,v in x.items()})
                    for k,v in x.items():data[k][idx[0]].copy_(v)
                    return reward
                env.reward_manager.compute=record;raw=torch.empty(200,64,61,device='cuda:0');teacher_raw=torch.empty_like(raw)
                actions=torch.empty(200,64,14,device='cuda:0');student_actions=torch.empty_like(actions);teacher_actions=torch.empty_like(actions)
                qpos=torch.empty(201,*env.sim.data.qpos.shape,device='cuda:0');qvel=torch.empty(201,*env.sim.data.qvel.shape,device='cuda:0')
                camera=torch.empty(801,64,4,device='cuda:0');contact=torch.empty(800,64,4,device='cuda:0',dtype=torch.bool);force=torch.empty(800,64,device='cuda:0')
                sid=robot.find_sites('head_camera')[0][0];sub=[0];native_sub=env.metrics_manager.compute_substep
                def capture():
                    native_sub();i=sub[0];camera[i].copy_(robot.data.site_quat_w[:,sid]);feet=env.scene['feet_ground_contact'].data.found.reshape(64,2,-1).any(-1)
                    contact[i].copy_(torch.cat((feet,found('diagnostic_head_ground')[:,None],found('self_collision')[:,None]),-1));force[i].copy_(env.scene['diagnostic_head_ground'].data.force.sum(1).norm(dim=-1));sub[0]+=1
                env.metrics_manager.compute_substep=capture;eligible=torch.zeros(64,device='cuda:0',dtype=torch.bool)
                with torch.inference_mode():
                    for step in range(200):
                        idx[0]=step;x=obs['actor'];torch._assert_async((x[:,48]==command).all()&(x[:,49:]==0).all(),'Fixed posture/headHOME command mismatch')
                        raw[step].copy_(x);qpos[step].copy_(env.sim.data.qpos);qvel[step].copy_(env.sim.data.qvel)
                        tx=x.clone();tx[:,48:51]=0;tx[:,55:61]=0;teacher_raw[step].copy_(tx)
                        torch._assert_async((x[:,:48]==tx[:,:48]).all()&(x[:,51:55]==tx[:,51:55]).all(),'Teacher must retain realfeedback/previousaction/headcommand')
                        student_action=a(TensorDict(obs,batch_size=[64]),stochastic_output=False);td=dict(obs);td['actor']=tx;teacher_action=b(TensorDict(td,batch_size=[64]),stochastic_output=False)
                        if step==100:
                            q=robot.data.joint_pos[:,reader.ids]-robot.data.default_joint_pos[:,reader.ids];quat=robot.data.root_link_quat_w
                            z=robot.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2];cos=1-2*(quat[:,1].square()+quat[:,2].square());feet=env.scene['feet_ground_contact'].data.found.reshape(64,2,-1).any(-1)
                            eligible=(mode==0)&(z>.1)&(cos>0.8660254037844386)&(q[:,[0,1,2,3,4,9,10,11,12,13]].abs().mean(-1)<.2)&feet.all(-1)&env._v2_joint_feedback.ready&torch.isfinite(env.sim.data.qpos).all(-1)&torch.isfinite(env.sim.data.qvel).all(-1)
                        action=torch.where(eligible[:,None],teacher_action,student_action)if arm=='near_stand_teacher'and step>=100 else student_action
                        student_actions[step].copy_(student_action);teacher_actions[step].copy_(teacher_action);actions[step].copy_(action)
                        obs,_,_,_,_=env.step(action);steps+=1
                    qpos[-1].copy_(env.sim.data.qpos);qvel[-1].copy_(env.sim.data.qvel);camera[-1].copy_(robot.data.site_quat_w[:,sid])
                assert sub[0]==800
                for actor,before in zip([a,b],frozen):
                    for k,v in before.items():assert torch.equal(actor.state_dict()[k],v)
                torch.save(dict(initial=initial,mode=mode.cpu(),eligible_at2s=eligible.cpu(),measurements={k:v.cpu()for k,v in data.items()},actor_obs_preaction=raw.cpu(),teacher_obs_preaction=teacher_raw.cpu(),actions=actions.cpu(),student_actions=student_actions.cpu(),teacher_actions=teacher_actions.cpu(),qpos_boundary=qpos.cpu(),qvel_boundary=qvel.cpu(),camera_quat_w=camera.cpu(),contacts200=contact.cpu(),head_force200=force.cpu()),out/'traces.pt')
                selected=(mode==0);body=data['stable'][-25:].all(0);clean=~contact[-100:,:,2:].any((0,2))&contact[-100:,:,:2].all((0,2));home=data['physical_q'][-25:,:,5:9].abs().amax(-1).lt(.1).all(0)
                rep=dict(status='LOCAL_EXTERNAL_TEACHER_DIAGNOSTIC_ONLY',arm=arm,stage=stage,student_sha256=sha(S),teacher_sha256=sha(T),source_sha256=source_hashes(W),boundary=BOUNDARY,eligible_at2s=int(eligible.sum()),HOME_STAND_trials=int(selected.sum()),tail_body=int(body[selected].sum()),tail_contactscreen_body=int((body&clean)[selected].sum()),tail_body_head_HOME=int((body&clean&home)[selected].sum()),after_switch_tilt_over45=int((data['tilt'][100:,selected]>torch.pi/4).any(0).sum()),after_switch_min_height=float(data['height'][100:,selected].min()),tail_tilt_deg=float(torch.rad2deg(data['tilt'][-25:,selected]).mean()),tail_height_m=float(data['height'][-25:,selected].mean()),tail_head_mae_deg=torch.rad2deg(data['physical_q'][-25:,selected,5:9].abs().mean((0,1))).tolist(),feedback_fault_trials=int((~data['feedback_ready']).any(0).sum()),nonfinite_trials=int((~data['finite']).any(0).sum()),tail_head_force_mean_N=float(force[-100:,selected].mean()),oracle_max_abs=reader.oracle_max.cpu().tolist(),actors_and_normalizers_unchanged=True,diagnostic_controls=12800,new_formal_controls=0)
                (out/'report.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep),flush=True)
            finally:
                (out/'diagnostic-count.json').write_text(json.dumps(dict(completed_control_steps=steps,completed_controls=steps*64,new_formal_controls=0))+'\n');env.close()
            completed+=steps*64;gc.collect();torch.cuda.empty_cache()
    assert completed==51200;marker(status='FOUR_LOCAL_TEACHER_DIAGNOSTICS_COMPLETE',wall_seconds=time.time()-started)
except BaseException:
    marker(status='STOPPED_REQUIRES_REVIEW',exception=traceback.format_exc(),wall_seconds=time.time()-started);raise
