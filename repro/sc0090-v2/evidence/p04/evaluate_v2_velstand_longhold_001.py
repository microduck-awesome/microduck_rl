"""Deterministic walking command battery, raw physical evidence and latched falls.

Retains reset/startup training DR at the checkpoint clock. Removes interval
pushes and all automatic resets for this command-tracking battery. Measured
R1 is tested separately; this does not constitute hardware acceptance.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab_microduck.custom_v2.velstand import TASK as REPAIRED_WALK_TASK_ID
from mjlab_microduck.custom_v2.microduck_v2 import YAW_FINE_WALK_TASK_ID
from mjlab_microduck.custom_v2.walking_metrics import WalkingMeasurements, BOUNDARY
from mjlab_microduck.custom_v2.head_metrics import camera_metrics, response_metrics, summarize_response

PHASES = [('idle', 0, 50), ('move', 50, 550), ('stop', 550, 700),
          ('reverse', 700, 1200), ('return', 1200, 1400)]


def cases_for():
    cases = [dict(name='idle', twist=[0., 0., 0.], head=[0.]*4)]
    for axis, name, values in ((0,'forward',(.1,.3,.4)), (1,'lateral',(.1,.3)), (2,'yaw',(.5,1.))):
        for v in values:
            for sign in (-1,1):
                target = [0.,0.,0.]; target[axis] = sign*v
                cases.append(dict(name=f'{name}_{sign*v:+.2f}',twist=target,head=[0.]*4))
    for sign in (-1,1):
        cases.append(dict(name=f'curve_{sign:+}',twist=[.2,0.,sign*.7],head=[0.]*4))
        cases.append(dict(name=f'diagonal_{sign:+}',twist=[.15,sign*.15,0.],head=[0.]*4))
        cases.append(dict(name=f'walking_head_{sign:+}',twist=[.2,0.,0.],head=[sign*.05,-sign*.05,sign*.07,sign*.015]))
    return [c for c in cases if c['name'] in ('idle','forward_+0.10','forward_+0.30','yaw_+0.50','yaw_-0.50','walking_head_+1','walking_head_-1')]


def command_for(step, target):
    if 50 <= step < 550: return target
    if 700 <= step < 1200: return -target
    return torch.zeros_like(target)


def sustained_arrival(error, target, failure, dt=.02):
    band = torch.maximum(target.abs()*.2, target.new_tensor([.05,.05,.1]))
    good = (error.abs() <= band[None]).all(-1) & ~failure
    holds = good.unfold(0,10,1).all(-1)
    times = torch.arange(1,len(holds)+1,device=error.device)*dt
    return torch.where(holds,times[:,None],float('inf')).amin(0)


def stride_lengths(positions, touch):
    """Same-foot consecutive touchdowns, time-major order; no per-frame sync."""
    event=torch.where(touch,torch.arange(len(touch),device=touch.device)[:,None,None],-1)
    last=event.cummax(0).values
    previous=torch.cat((torch.full_like(last[:1],-1),last[:-1]),0)
    previous_position=positions.gather(0,previous.clamp_min(0)[...,None].expand(-1,-1,-1,2))
    valid=touch & (previous>=0)
    return (positions-previous_position).norm(dim=-1)[valid]


def finite_json(value):
    if isinstance(value, dict): return {k:finite_json(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def evaluate_seed(args, seed, cases):
    cfg = load_env_cfg(REPAIRED_WALK_TASK_ID)
    cfg.actions['joint_pos'].randomization_stage = args.stage
    cfg.seed, cfg.scene.num_envs = seed, len(cases)*args.repeats
    cfg.auto_reset, cfg.terminations = False, {}
    cfg.events = {k:v for k,v in cfg.events.items() if v.mode != 'interval' and k != 'random_prone_init'}
    cfg.curriculum = {k:v for k,v in cfg.curriculum.items()
        if not v.params.get('event_name') or v.params['event_name'] in cfg.events}
    cfg.observations['actor'].enable_corruption = False
    # Keep every other inherited reset/startup DR; disclose full config.
    (args.output/f'config-{seed}.json').write_text(json.dumps(asdict(cfg),default=str,indent=2))
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        agent = load_rl_cfg(YAW_FINE_WALK_TASK_ID)  # actor/critic76 identical; no teacher updates in evaluation
        agent.logger, agent.upload_model = 'tensorboard', False
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls(REPAIRED_WALK_TASK_ID)(wrapped,asdict(agent),None,args.device)
        infos = runner.load(str(args.checkpoint), map_location=args.device)
        policy = runner.alg.get_policy().eval()
        frozen = {k:v.clone() for k,v in policy.state_dict().items()}
        target = torch.tensor([c['twist'] for c in cases],device=args.device).repeat_interleave(args.repeats,0)
        head_target = torch.tensor([c['head'] for c in cases],device=args.device).repeat_interleave(args.repeats,0)
        desired, desired_head = torch.zeros_like(target), torch.zeros_like(head_target)
        native_command = env.command_manager.compute
        def commands(dt):
            native_command(dt)
            env.command_manager.get_command('twist').copy_(desired)
            env.command_manager.get_term('twist').is_standing_env.copy_(desired.abs().amax(-1)<=1e-6)
            env.command_manager.get_term('body_pose')._raw.zero_()
            env.command_manager.get_command('head_pose').copy_(desired_head)
            env.command_manager.get_command('body_pose').zero_()
        env.command_manager.compute = commands
        obs,_ = env.reset(seed=seed)
        initial = dict(qpos=env.sim.data.qpos.clone(),qvel=env.sim.data.qvel.clone(),
                       actor=obs['actor'].clone(),origins=env.scene.env_origins.clone())
        reader = WalkingMeasurements(env,oracle=True)
        storage, control, substep = {}, [0], [0]
        sites,_ = env.scene['robot'].find_sites('head_camera')
        assert len(sites)==1
        camera = torch.empty(5601,env.num_envs,4,device=args.device)
        native_substep = env.metrics_manager.compute_substep
        def camera_sample():
            native_substep()
            camera[substep[0]].copy_(env.scene['robot'].data.site_quat_w[:,sites[0]])
            substep[0]+=1
        env.metrics_manager.compute_substep = camera_sample
        native_reward = env.reward_manager.compute
        def record(dt):
            reward = native_reward(dt)
            values = reader.read()
            if not storage:
                storage.update({k:torch.empty((1400,*v.shape),dtype=v.dtype,device=v.device) for k,v in values.items()})
            for k,v in values.items(): storage[k][control[0]].copy_(v)
            return reward
        env.reward_manager.compute = record
        with torch.inference_mode():
            for step in range(1400):
                control[0] = step
                torch._assert_async((obs['actor'][:,48:51]==command_for(step,target)).all(),'Actor twist misaligned')
                torch._assert_async((obs['actor'][:,51:55]==command_for(step,head_target)).all(),'Actor head misaligned')
                torch._assert_async((obs['actor'][:,55:61]==0).all(),'Actor body misaligned')
                desired.copy_(command_for(step+1,target))
                desired_head.copy_(command_for(step+1,head_target))
                obs,_,_,_ = wrapped.step(policy(obs))
            camera[-1].copy_(env.scene['robot'].data.site_quat_w[:,sites[0]])
        assert reader.calls==1400 and substep[0]==5600
        for k,v in policy.state_dict().items(): assert torch.equal(v,frozen[k])
        for step in (0,49,50,549,550,699,700,1199,1200,1399):
            assert torch.equal(storage['command'][step],command_for(step,target))
        failure = storage['failed'].cummax(0).values
        rows = []
        for index,case in enumerate(cases):
            sl = slice(index*args.repeats,(index+1)*args.repeats)
            entry = dict(**case,seed=seed,trials=args.repeats,
                         survived_trials=int((~failure[-1,sl]).sum()),
                         feedback_invalid_trials=int((~storage['feedback_ready'][:,sl]).any(0).sum()),phases={})
            for name,a,b in PHASES:
                s = {k:v[a:b,sl] for k,v in storage.items()}
                velocity = torch.cat((s['linear'][...,:2],s['angular'][...,2:3]),-1)
                err = velocity-s['command']
                arrival = sustained_arrival(err,s['command'][0],failure[a:b,sl])
                contact = s['contact']
                previous_contact = storage['contact'][max(0,a-1):b-1,sl] if a else torch.cat((contact[:1],contact[:-1]))
                touch = contact & ~previous_contact
                lengths=stride_lengths(s['feet_position'][...,:2],touch)
                head_err=s['head_position']-s['head_command']
                camera_result=camera_metrics(camera[a*4:b*4+1,sl])
                survivors = ~failure[b-1,sl]
                metrics=dict(
                    survival_trials=int(survivors.sum()),
                    velocity_tail_mean=velocity[-25:].mean((0,1)).tolist(),
                    velocity_tail_rmse=err[-25:].square().mean((0,1)).sqrt().tolist(),
                    arrival_diagnostic_reached_trials=int(torch.isfinite(arrival).sum()),
                    arrival_diagnostic_median_all_trials_s=float(arrival.sort().values[(len(arrival)-1)//2]),
                    displacement_world_mean_m=(s['root_position'][-1]-s['root_position'][0]).mean(0).tolist(),
                    tilt_p95_deg=float(torch.quantile(s['tilt'],.95)*180/math.pi),
                    height_min_m=float(s['height'].min()),
                    double_support_fraction=float(contact.all(-1).float().mean()),
                    no_support_fraction=float((~contact.any(-1)).float().mean()),
                    foot_contact_speed_mps=float((s['feet_velocity'][...,:2].norm(dim=-1)*contact).sum()/contact.sum().clamp_min(1)),
                    touchdown_frequency_per_foot_hz=touch.sum((0,1)).float().div(args.repeats*(b-a)*.02).tolist(),
                    same_foot_stride_count=len(lengths),
                    same_foot_stride_mean_m=float(lengths.mean()) if len(lengths) else None,
                    touchdown_airtime_mean_s=float(s['last_air_time'][touch].mean()) if bool(touch.any()) else None,
                    foot_force_p95_N=float(torch.quantile(s['force'].norm(dim=-1),.95)),
                    head_tail_mae_deg=(head_err[-25:].abs().mean((0,1))*180/math.pi).tolist(),
                    head_tail_mae_survivors_deg=(head_err[-25:,survivors].abs().mean((0,1))*180/math.pi).tolist(),
                    head_all_axes_one_degree_tail_and_survived_trials=int(
                        ((head_err[-25:].abs()<=math.pi/180).all(0).all(-1)&survivors).sum()),
                    camera={k:float(v.mean()) for k,v in camera_result.items()})
                if name in ('move','reverse'):
                    initial_head=storage['head_position'][a-1,sl]
                    target_head=s['head_command'][0]
                    metrics['head_response']=summarize_response(response_metrics(s['head_position'],initial_head,target_head,target_head))
                entry['phases'][name]=metrics
            rows.append(entry)
        trace = args.output/f'trace-{seed}.pt'
        torch.save(dict(initial={k:v.cpu() for k,v in initial.items()},
                        measurements={k:v.cpu() for k,v in storage.items()},camera=camera.cpu()),trace)
        return rows,dict(seed=seed,oracle_calls=reader.calls,oracle_max_abs=reader.oracle_max.cpu().tolist(),
                         checkpoint_clock=infos['env_state']['common_step_counter'],trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--stage',choices=('nominal','measured'),default='nominal')
    p.add_argument('--repeats',type=int,default=16)
    p.add_argument('--seeds',type=int,nargs='+',default=[2026101612,2026101613])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    cases=cases_for();records=[];checks=[]
    for seed in args.seeds:
        rows,check=evaluate_seed(args,seed,cases);records.extend(rows);checks.append(check)
    report=dict(status='EVALUATION_COMPLETE',checkpoint=str(args.checkpoint),
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                stage=args.stage,boundary=BOUNDARY,records=records,checks=checks,
                command_protocol=PHASES,dt=.02,
                failure='latched from first control: height<.08m OR tilt>45deg OR nonfinite OR feedback not ready',
                velocity_arrival='diagnostic only: all axes within max(20% command, .05/.05mps/.1radps) for .2s; no user acceptance threshold',
                foot_timing='control-rate contacts at reward boundary; sub-20ms contact events may be missed',
                camera_timing='200Hz forward geometry t=0...27.995 plus final forward at28s',
                scope='Bounded28s longhold diagnostic:1s idle/10s move/3s stop/10s reverse/4s return. Seven retained cases,16 repeats/two seeds per stage. Original reset/startup DR; pushes and actor corruption disabled, no auto reset. Not training or hardware acceptance.',
                source_sha256={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (args.output/'report.json').write_text(json.dumps(finite_json(report),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
