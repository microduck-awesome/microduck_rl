"""Frozen command/reversal/hold battery in actual repaired P04 feedback."""
import argparse,json,hashlib
from dataclasses import asdict
from pathlib import Path
import torch
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_rl_cfg
from rsl_rl.utils import resolve_callable,resolve_obs_groups
from mjlab_microduck.custom_v2.sitstand import TASK,make_sitstand_env_cfg
from mjlab_microduck.custom_v2.sitstand_metrics import SitStandMeasurements,BOUNDARY
from mjlab_microduck.tasks import mdp
from train_v2_sitstand import source_hashes

def ground_cases(env,env_ids,**params):
    for mode in range(4):
        ids=env_ids[env_ids%4==mode];p=dict(params)
        p.update(face_down_prob=0.,face_up_prob=0.,sitting_prob=float(mode>=2),standing_prob=float(mode<2),sitting_joint_noise_std=.01,sitting_tilt_max=.035)
        mdp.set_random_ground_state(env,ids,**p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--policy-mode',choices=('mean','endpoints'),default='mean');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--stage',choices=('nominal','measured'),default='nominal');p.add_argument('--seed',type=int,default=2603);p.add_argument('--num-envs',type=int,default=256);a=p.parse_args()
    assert a.num_envs%32==0;a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_device(a.device)
    cfg=make_sitstand_env_cfg(play=True,stage=a.stage);cfg.seed=a.seed;cfg.scene.num_envs=a.num_envs
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
        reward_rows=[]
        native_reward=env.reward_manager.compute
        def measured(dt):
            reward=native_reward(dt);x=reader.read();reward_rows.append(env.reward_manager._step_reward.clone())
            rows.append(torch.stack((x['height'],x['tilt'],x['linear'][:,2],x['omega'],x['stable'].float(),
              (~x['feedback_ready']).float(),x['head_error'].mean(-1),x['head_error'].amax(-1),x['alpha'],x['flag'],
              *x['head_error'].unbind(-1)),-1))
            return reward
        env.reward_manager.compute=measured
        checkpoint=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
        assert checkpoint['infos']['v2']['task']==TASK
        hashes=source_hashes(Path(__file__).resolve().parents[1])
        assert hashes==checkpoint['infos']['v2']['source_sha256'], 'Evaluation numerical source differs from checkpoint'
        agent=asdict(load_rl_cfg(TASK));td=TensorDict(obs,batch_size=[a.num_envs]);groups=resolve_obs_groups(td,agent['obs_groups'],['actor','critic'])
        actor_cfg=agent['actor']
        for key in ('cnn_cfg','distribution_cfg'):
            if actor_cfg.get(key) is None:actor_cfg.pop(key,None)
        if actor_cfg.get('rnn_type') is None:
            for key in ('rnn_type','rnn_hidden_dim','rnn_num_layers'):actor_cfg.pop(key,None)
        ctor=resolve_callable(actor_cfg.pop('class_name'));actor=ctor(td,groups,'actor',14,**actor_cfg).to(a.device)
        actor.load_state_dict(checkpoint['actor_state_dict'],strict=True);actor.eval()
        frozen={k:v.clone() for k,v in actor.state_dict().items()}
        with torch.inference_mode():
            for step in range(600):
                phase=step//200;command.copy_(1-first if phase==1 else first)
                # Head movement once settled, then zero on each posture reversal.
                head.copy_(targets if step%200>=150 else torch.zeros_like(targets))
                twist=env.command_manager.get_term('twist');twist.vel_command_b[:,0]=command
                env.command_manager.get_term('head_pose')._command.copy_(head)
                obs=env.observation_manager.compute(update_history=False)
                action=actor(TensorDict(obs,batch_size=[a.num_envs]),stochastic_output=False)
                if a.policy_mode=='endpoints':action=reader.sit[None]*command[:,None]
                env.step(action)
        for k,v in frozen.items():assert torch.equal(actor.state_dict()[k],v)
        torch.save({'rewards':torch.stack(reward_rows).cpu(),'terms':list(env.reward_manager._term_names),'scale_by_dt':env.cfg.scale_rewards_by_dt,'step_dt':env.step_dt},a.output/'full-rewards.pt')
        traces=torch.stack(rows).cpu();torch.save({'traces':traces,'mode':mode.cpu(),'head_targets':targets.cpu(),
             'columns':['height','tilt_rad','vz','omega','stable','feedback_fault','head_mae','head_max_error','alpha','flag','neck_error','pitch_error','yaw_error','roll_error']},a.output/'traces.pt')
        report=dict(status='FULL_REWARD_DIAGNOSTIC_NOT_SKILL_ACCEPTANCE',policy_mode=a.policy_mode,checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),task=TASK,stage=a.stage,seed=a.seed,num_envs=a.num_envs,
          boundary=BOUNDARY,protocol='4s first command / 4s reverse / 4s return; last second initial-range head step; last .5s hold',
          hold_gate={'height_abs_m':.012,'tilt_deg':15,'linear_speed_mps':.05,'angular_speed_radps':.5,'final_leg_pose_mae_rad':.2,'continuous_seconds':.5,'finite_and_feedback_ready':True},
          head_step_gate='physical all-axis error <= max(10% abs step command, .002rad), with simultaneous body hold; 1s window, initial ranges only',
          actor_normalizer_unchanged=True,numerical_source_verified=True,source_sha256=hashes,
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
                result.append(dict(trials=t.shape[1],tail_hold=int(tail[:,:,4].bool().all(0).sum()),faults=int(t[:,:,5].bool().any(0).sum()),
                  head_step_trials=int(active.sum()),head_step_reached=int(reached.sum()),
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
