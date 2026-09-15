#!/usr/bin/env python3
"""Diagnose SC0090 walking with fixed commands, matched seeds and optional action sampling.

This is an eight-second diagnostic sweep, not the staged-program acceptance test.
Randomized cases retain DR/observation noise but disable external pushes.
The first two seconds settle; reported velocities are robot body-frame values.
Motor statistics are sampled at 50 Hz control boundaries, not every physics step.
"""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import argparse, hashlib, json, math, subprocess

import torch
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab_microduck.tasks import SC0090FineTuneRunner, mdp
from mjlab_microduck.program_evaluation import nominal_walk_config

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint',type=Path,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--samples',type=int,default=32)
parser.add_argument('--device',default='cuda:0')
parser.add_argument('--seed',type=int,default=2026091507)
parser.add_argument('--save-traces',action='store_true')
parser.add_argument('--stochastic',action='store_true')
args=parser.parse_args()
checkpoint=args.checkpoint.resolve(strict=True)
out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
if args.samples<1 or not 0<=args.seed<2**32 or torch.device(args.device).type!='cuda':
    parser.error('Require positive samples, a uint32 seed and a CUDA device')
checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
task='Mjlab-Velocity-Flat-MicroDuck-SC0090-V4'
configure_torch_backends(); torch.set_num_threads(2); torch.cuda.set_device(args.device)
commands=[(v,0.,0.) for v in (0.,.03,mdp.SC0090_PROGRAM_WALK_MIN_SPEED,.1,.2)]+[(0.,0.,s*v) for s in (1.,-1.) for v in (.2,.4,.6,.8,1.)]
samples=args.samples
values=torch.tensor(commands,device=args.device).repeat_interleave(samples,dim=0)
def fixed(self,ids):
    self.vel_command_b[ids]=values[ids]; self.vel_command_w[ids]=values[ids]
    self.bucket[ids]=0
    self.is_standing_env[ids]=(values[ids]==0).all(-1)
    for key in ('is_heading_env','is_world_env','is_forward_env'):
        getattr(self,key)[ids]=False
mdp.SC0090ProgramVelocityCommand._resample_command=fixed
reports=[]
for condition in ('nominal','randomized'):
    cfg=deepcopy(load_env_cfg(task))
    if condition=='nominal': cfg=nominal_walk_config(cfg)
    cfg.events.pop('push_robot',None)
    cfg.scene.num_envs=len(values); cfg.auto_reset=True; cfg.episode_length_s=10.
    cfg.seed=args.seed
    for command in cfg.commands.values(): command.resampling_time_range=(1000.,1000.)
    env=ManagerBasedRlEnv(cfg,device=args.device); vec=RslRlVecEnvWrapper(env)
    runner=SC0090FineTuneRunner(vec,asdict(load_rl_cfg(task)),device=args.device)
    infos=runner.load(str(checkpoint),load_cfg={'actor':True},map_location=args.device)
    mdp.sc0090_program_restore(env,infos['sc0090_program'])
    policy=runner.get_inference_policy(device=args.device)
    actuator=env.scene['robot'].actuators[0]
    names=env.scene['robot'].joint_names
    data=env.scene['robot'].data
    frames=[]
    with torch.inference_mode():
        obs,_=vec.reset()
        failed=torch.zeros(len(values),dtype=torch.bool,device=args.device)
        for step in range(round(8/env.step_dt)):
            actions=policy(obs,stochastic_output=args.stochastic)
            if not torch.isfinite(actions).all():
                raise ValueError('Nonfinite policy actions')
            obs,reward,done,_=vec.step(actions)
            if env.termination_manager.get_term('nan_state').any():
                raise ValueError('Nonfinite physics terminated a diagnostic world')
            failed |= done.bool()
            assert all(torch.isfinite(v).all() for v in (*obs.values(),reward,data.actuator_force))
            if step>=round(2/env.step_dt):
                velocity=torch.cat((data.root_link_lin_vel_b[:,:2],data.root_link_ang_vel_b[:,2:3]),dim=-1)
                frames.append({'velocity':velocity.cpu(), 'rpm':data.joint_vel.abs().cpu()*30/math.pi,
                    'torque':data.actuator_force.abs().cpu(),'duty':actuator._bam_model.actuator.duty_cycle.abs().cpu(),
                    'rewards':env.reward_manager._step_reward.cpu().clone()})
        tensors={key:torch.stack([frame[key] for frame in frames]).reshape(len(frames),len(commands),samples,-1) for key in frames[0]}
        rewards=env.reward_manager.active_terms
        for i,command in enumerate(commands):
            good=(~failed.reshape(-1,samples)[i]).cpu()
            if not good.any():
                reports.append({'condition':condition,'command':command,'failed':samples})
                continue
            current={k:v[:,i,good] for k,v in tensors.items()}
            row={'condition':condition,'command':command,'failed':int(failed.reshape(-1,samples)[i].sum()),
                'velocity_mean':current['velocity'].mean((0,1)).tolist(),
                'rpm_max':current['rpm'].amax((0,1)).tolist(),
                'torque_max':current['torque'].amax((0,1)).tolist(),
                'duty_saturation_fraction':float((current['duty']>.99).float().mean()),
                'reward_mean':dict(zip(rewards,current['rewards'].mean((0,1)).tolist()))}
            reports.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('reward_mean','rpm_max','torque_max')},allow_nan=False),flush=True)
        if args.save_traces: torch.save(tensors,out/f'{condition}_{"stochastic" if args.stochastic else "deterministic"}_traces.pt')
    env.close()
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==checkpoint_hash, 'Checkpoint changed during diagnosis'
(out/('probe_stochastic.json' if args.stochastic else 'probe.json')).write_text(json.dumps({'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_hash,'source_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'seed':args.seed,'task':task,'motor_sampling_hz':50,'stochastic':args.stochastic,'samples':samples,'measurement_s':[2,8],'joint_names':names,'reports':reports},indent=2,allow_nan=False)+'\n')
