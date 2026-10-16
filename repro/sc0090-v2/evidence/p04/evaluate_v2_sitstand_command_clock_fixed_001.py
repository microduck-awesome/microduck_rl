"""Frozen command/reversal/hold battery in actual repaired P04 feedback."""
import argparse,json,hashlib,sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,'/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/sitstand-short-lesson-worktree/scripts')
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_rl_cfg,load_env_cfg
from rsl_rl.utils import resolve_callable,resolve_obs_groups
from mjlab_microduck.custom_v2.sitstand import TASK,make_sitstand_env_cfg
from mjlab_microduck.custom_v2.sitstand_clear import CLEAR_TASK
from mjlab_microduck.custom_v2.sitstand_lesson import LESSON_TASK
from mjlab_microduck.custom_v2.sitstand_metrics import SitStandMeasurements,BOUNDARY
from mjlab_microduck.tasks import mdp
from train_v2_sitstand import source_hashes
from mjlab_microduck.custom_v2.head_metrics import camera_metrics

def ground_cases(env,env_ids,**params):
    for mode in range(4):
        ids=env_ids[env_ids%4==mode];p=dict(params)
        p.update(face_down_prob=0.,face_up_prob=0.,sitting_prob=float(mode>=2),standing_prob=float(mode<2),sitting_joint_noise_std=.01,sitting_tilt_max=.035)
        mdp.set_random_ground_state(env,ids,**p)

def camera_summary(camera, body_stable):
    """Report all trials and the explicitly counted continuously stable subset."""
    values=camera_metrics(camera,dt=.005,exposure_s=.02)
    healthy=body_stable.bool().all(0)
    def aggregate(mask):
        if not bool(mask.any()):return None
        return {key:{'mean_over_trials':float(v[mask].mean()),
                     'p95_over_trials':float(torch.quantile(v[mask],.95)),
                     'max_over_trials':float(v[mask].max())} for key,v in values.items()}
    return dict(trials=camera.shape[1],body_stable_entire_window_trials=int(healthy.sum()),
                all_trials=aggregate(torch.ones_like(healthy)),
                body_stable_entire_window=aggregate(healthy))

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--stage',choices=('nominal','measured'),default='nominal');p.add_argument('--seed',type=int,default=2603);p.add_argument('--num-envs',type=int,default=256);a=p.parse_args()
    assert a.num_envs%32==0;a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_device(a.device)
    metadata=torch.load(a.checkpoint,map_location='cpu',weights_only=False)['infos']['v2'];trained_task=metadata['task']
    assert trained_task in (TASK,CLEAR_TASK,LESSON_TASK)
    cfg=load_env_cfg(trained_task,play=True);assert cfg.episode_length_s==12.;cfg.actions['joint_pos'].randomization_stage=a.stage;cfg.seed=a.seed;cfg.scene.num_envs=a.num_envs
    cfg.curriculum={};cfg.terminations={};cfg.auto_reset=False
    cfg.observations['actor'].enable_corruption=False
    cfg.events={n:t for n,t in cfg.events.items() if t.mode=='startup' or n in {'reset_base','reset_robot_joints','set_ground_state','v2_reset_joint_guards','mouth_reset','sensor_error_reset'}}
    cfg.events['set_ground_state'].func=ground_cases
    (a.output/'executed-script.py').write_bytes(Path(__file__).read_bytes())
    (a.output/'eval-config.json').write_text(json.dumps(asdict(cfg),indent=2,default=str)+'\n')
    env=ManagerBasedRlEnv(cfg,device=a.device)
    try:
        idx=torch.arange(a.num_envs,device=a.device);mode=idx%4;first=(mode%2).float()
        command=first.clone();head=torch.zeros(a.num_envs,4,device=a.device)
        targets=torch.tensor([[0,0,0,0],[.05,0,0,0],[-.05,0,0,0],[0,.05,0,0],[0,0,.07,0],[0,0,-.07,0],[0,0,0,.015],[0,0,0,-.015]],device=a.device)[idx//4%8]
        native_compute=env.command_manager.compute
        def held_compute(dt):
            twist=env.command_manager.get_term('twist');h=env.command_manager.get_term('head_pose')
            twist.time_left.fill_(1e9);h.time_left.fill_(1e9)
            twist.vel_command_b.zero_();twist.vel_command_b[:,0]=command;h._command.copy_(head)
            native_compute(dt)
        env.command_manager.compute=held_compute
        obs,_=env.reset();reader=SitStandMeasurements(env,oracle=True);rows=[]
        assert env.physics_dt==.005 and env.step_dt==.02 and env.cfg.decimation==4
        robot=env.scene['robot'];sites,_=robot.find_sites('head_camera');assert len(sites)==1
        # Each mj_step-derived pose is pre-integration: samples 0..2399 are
        # t=0,.005,...,11.995; the final native forward supplies t=12 exactly.
        camera=torch.empty(2401,a.num_envs,4,device=a.device);substep=[0]
        native_substep=env.metrics_manager.compute_substep
        def camera_sample():
            native_substep()
            camera[substep[0]].copy_(robot.data.site_quat_w[:,sites[0]])
            substep[0]+=1
        env.metrics_manager.compute_substep=camera_sample
        native_reward=env.reward_manager.compute
        def measured(dt):
            reward=native_reward(dt);x=reader.read()
            rows.append(torch.stack((x['height'],x['tilt'],x['linear'][:,2],x['omega'],x['stable'].float(),
              (~x['feedback_ready']).float(),x['head_error'].mean(-1),x['head_error'].amax(-1),x['alpha'],x['flag'],
              *x['head_error'].unbind(-1)),-1))
            return reward
        env.reward_manager.compute=measured
        checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
        assert checkpoint['infos']['v2']['task']==trained_task
        hashes=source_hashes(Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04/sitstand-short-lesson-worktree'))
        assert hashes==checkpoint['infos']['v2']['source_sha256'], 'Evaluation numerical source differs from checkpoint'
        agent=asdict(load_rl_cfg(trained_task));td=TensorDict(obs,batch_size=[a.num_envs]);groups=resolve_obs_groups(td,agent['obs_groups'],['actor','critic'])
        actor_cfg=agent['actor']
        for key in ('cnn_cfg','distribution_cfg'):
            if actor_cfg.get(key) is None:actor_cfg.pop(key,None)
        if actor_cfg.get('rnn_type') is None:
            for key in ('rnn_type','rnn_hidden_dim','rnn_num_layers'):actor_cfg.pop(key,None)
        ctor=resolve_callable(actor_cfg.pop('class_name'));actor=ctor(td,groups,'actor',14,**actor_cfg).to(a.device)
        actor.load_state_dict(checkpoint['actor_state_dict'],strict=True);actor.eval()
        frozen={k:v.clone() for k,v in actor.state_dict().items()}
        initial=dict(qpos=env.sim.data.qpos.clone(),qvel=env.sim.data.qvel.clone(),actor=obs['actor'].clone(),origins=env.scene.env_origins.clone())
        actual_inputs=torch.empty(600,a.num_envs,61,device=a.device)
        actual_actions=torch.empty(600,a.num_envs,14,device=a.device)
        qpos=torch.empty(601,*env.sim.data.qpos.shape,device=a.device)
        qvel=torch.empty(601,*env.sim.data.qvel.shape,device=a.device)
        with torch.inference_mode():
            for step in range(600):
                expected=1-first if step//200==1 else first
                expected_head=targets if step%200>=150 else torch.zeros_like(targets)
                torch._assert_async((obs['actor'][:,48]==expected).all()&(obs['actor'][:,49:51]==0).all(),'Current policy posture observation must match executed command')
                torch._assert_async((obs['actor'][:,51:55]==expected_head).all()&(obs['actor'][:,55:61]==0).all(),'Current policy head/body slots must match executed command')
                torch._assert_async((env.command_manager.get_command('twist')[:,0]==expected).all(),'Reward and actor posture command mismatch')
                actual_inputs[step].copy_(obs['actor']);qpos[step].copy_(env.sim.data.qpos);qvel[step].copy_(env.sim.data.qvel)
                action=actor(TensorDict(obs,batch_size=[a.num_envs]),stochastic_output=False)
                actual_actions[step].copy_(action)
                # Native command update at the end of this control prepares
                # exactly the next action's observation and internal alpha.
                command.copy_(1-first if (step+1)//200==1 else first)
                head.copy_(targets if (step+1)%200>=150 else torch.zeros_like(targets))
                obs,_,_,_,_=env.step(action)
            camera[-1].copy_(robot.data.site_quat_w[:,sites[0]])
            qpos[-1].copy_(env.sim.data.qpos);qvel[-1].copy_(env.sim.data.qvel)
        assert substep[0]==2400 and len(rows)==600
        assert bool(torch.isfinite(camera).all())
        assert bool(torch.isclose(camera.norm(dim=-1),torch.ones_like(camera[...,0]),atol=2e-5,rtol=0).all())
        for k,v in frozen.items():assert torch.equal(actor.state_dict()[k],v)
        traces=torch.stack(rows).cpu();torch.save({'initial':{k:v.cpu() for k,v in initial.items()},'actor_obs_preaction':actual_inputs.cpu(),'actions_preprocess':actual_actions.cpu(),'qpos_boundary':qpos.cpu(),'qvel_boundary':qvel.cpu(),'traces':traces,'mode':mode.cpu(),'head_targets':targets.cpu(),'camera_quat_w':camera.cpu(),'camera_time_s':torch.arange(2401,dtype=torch.float64)*.005,
             'columns':['height','tilt_rad','vz','omega','stable','feedback_fault','head_mae','head_max_error','alpha','flag','neck_error','pitch_error','yaw_error','roll_error']},a.output/'traces.pt')
        report=dict(status='EVALUATED_NOT_SKILL_ACCEPTANCE',checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),task=trained_task,evaluation_target='checkpoint SitStand target, unchanged full 12s command battery',stage=a.stage,seed=a.seed,num_envs=a.num_envs,
          boundary=BOUNDARY,protocol='4s first command / 4s reverse / 4s return; last second initial-range head step; last .5s hold',command_timing='Action consumes native current observation; desired next command applied only in native end-of-control command compute. No cache clearing or extra sense/forward. Current actor/reward/expected flags asserted at all600 controls.',
          hold_gate={'height_abs_m':.012,'tilt_deg':15,'linear_speed_mps':.05,'angular_speed_radps':.5,'final_leg_pose_mae_rad':.2,'continuous_seconds':.5,'finite_and_feedback_ready':True},
          head_step_gate='physical all-axis error <= max(10% abs step command, .002rad), with simultaneous body hold; 1s window, initial ranges only',
          actor_normalizer_unchanged=True,numerical_source_verified=True,source_sha256=hashes,
          camera_protocol={'site':'head_camera CAD placeholder; angular motion, not calibrated pixel blur',
            'sampling_hz':200,'exposure_seconds':.02,'samples':2401,
            'boundary':'native substep post-step/pre-integration derived pose, plus final native forward; no extra forward/state writes',
            'phase_windows_seconds':{'posture_transition_or_initial_hold':[0,2], 'zero_head_posture_hold':[2,3], 'active_head_or_idle_control':[3,4]},
            'scope':'Raw world-camera motion includes intended head movement; do not call active-command motion stationary jitter. No acceptance threshold invented.'},
          oracle_max_abs=reader.oracle_max.cpu().tolist(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),groups={})
        for m,name in enumerate(('stand_to_stand','stand_to_sit','sit_to_stand','sit_to_sit')):
            select=mode.cpu()==m;x=traces[:,select];result=[]
            head_targets=targets.cpu()[select];active=head_targets.abs().amax(-1)>0
            head_band=torch.maximum(head_targets.abs()*.1,torch.tensor(.002))
            for phase in range(3):
                t=x[phase*200:(phase+1)*200];tail=t[-25:]
                arrived=(t[150:,:,10:14]<=head_band).all(-1)&t[150:,:,4].bool()
                reached=arrived.any(0)&active
                arrival_s=(arrived.float().argmax(0)[reached].float()+1)*.02
                camera_windows={}
                for window,lo,hi in (('posture_transition_or_initial_hold',0,100),('zero_head_posture_hold',100,150),('active_head_or_idle_control',150,200)):
                    q=camera[(phase*200+lo)*4:(phase*200+hi)*4+1,select.to(a.device)]
                    stable=t[lo:hi,:,4].to(a.device)
                    camera_windows[window]=camera_summary(q,stable)
                    if lo==150:
                        for label,subset in (('active_head',active),('idle_control',~active)):
                            if bool(subset.any()):camera_windows[window][label]=camera_summary(q[:,subset.to(a.device)],stable[:,subset.to(a.device)])
                result.append(dict(trials=t.shape[1],tail_hold=int(tail[:,:,4].bool().all(0).sum()),faults=int(t[:,:,5].bool().any(0).sum()),
                  camera=camera_windows,head_step_trials=int(active.sum()),head_step_reached=int(reached.sum()),
                  head_step_arrival_p90_s=float(torch.quantile(arrival_s,.9)) if len(arrival_s) else None,
                  head_step_tail_band_hold=int((arrived[-25:].all(0)&active).sum()),
                  tilt_over45=int((t[:,:,1]>torch.pi/4).any(0).sum()),head_tail_mae_deg=float(torch.rad2deg(tail[:,:,6]).mean()),
                  head_tail_all_axes_1deg=int(((tail[:,:,7]<torch.pi/180)&tail[:,:,4].bool()).all(0).sum()),
                  vz_abs_p95=float(torch.quantile(t[:,:,2].abs().flatten(),.95)),vz_abs_max=float(t[:,:,2].abs().max()),
                  control_vertical_accel_abs_p95=float(torch.quantile(torch.diff(t[:,:,2],dim=0).abs().flatten()/.02,.95))))
            report['groups'][name]=result
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    finally:env.close()
if __name__=='__main__':main()
