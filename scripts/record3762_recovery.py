"""Illustrative seven-pose replay using the unchanged recovery evaluator."""
import argparse,runpy
from pathlib import Path
import numpy as np
import train_v2_standup as evaluator
p=argparse.ArgumentParser(add_help=False);p.add_argument('--output',type=Path,required=True);args,_=p.parse_known_args()
Real=evaluator.ManagerBasedRlEnv

def recorded_env(cfg,device):
 env=Real(cfg,device=device);frames=[];origins=env.scene.env_origins.detach().cpu().numpy().copy()
 reset,step,close=env.reset,env.step,env.close
 def capture():frames.append(env.sim.data.qpos.detach().cpu().numpy().copy())
 def reset_record(*a,**kw):
  result=reset(*a,**kw);capture();return result
 def step_record(*a,**kw):
  result=step(*a,**kw);capture();return result
 def close_record():
  try:np.savez_compressed(args.output/f'replay-{cfg.seed}.npz',qpos=np.stack(frames),env_origins=origins,dt=.02)
  finally:close()
 env.reset=reset_record;env.step=step_record;env.close=close_record
 return env

evaluator.ManagerBasedRlEnv=recorded_env
runpy.run_path(str(Path(__file__).with_name('evaluate3762_recovery.py')),run_name='__main__')
