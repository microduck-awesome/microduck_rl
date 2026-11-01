"""Record 50 Hz physics qpos and raw actor action during recovery evaluation.

This runs the existing seven-bucket evaluator unchanged. It is diagnostic only;
the compressed traces expose whether 200 Hz near-limit events persist and what
the actor commanded, rather than inferring a cause from aggregate counts.
"""
import argparse
from pathlib import Path
import runpy

import numpy as np
import torch
import train_v2_standup as evaluator


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--output',type=Path,required=True)
    args,_=parser.parse_known_args()
    original=evaluator.ManagerBasedRlEnv

    def record_env(cfg,device):
        env=original(cfg,device=device)
        qpos=[];actions=[]
        origins=env.scene.env_origins.detach().cpu().numpy().copy()
        reset,step,close=env.reset,env.step,env.close
        def capture():qpos.append(env.sim.data.qpos.detach().clone())
        def reset_record(*a,**kw):
            result=reset(*a,**kw);capture();return result
        def step_record(action,*a,**kw):
            actions.append(action.detach().clone())
            result=step(action,*a,**kw);capture();return result
        def close_record():
            try:
                if qpos:
                    np.savez_compressed(args.output/f'diagnosis-trace-{cfg.seed}.npz',
                        qpos=torch.stack(qpos).cpu().numpy(),
                        actions=torch.stack(actions).cpu().numpy() if actions else np.empty((0,)),
                        env_origins=origins,dt=.02)
            finally:close()
        env.reset=reset_record;env.step=step_record;env.close=close_record
        return env

    evaluator.ManagerBasedRlEnv=record_env
    runpy.run_path(str(Path(__file__).with_name('evaluate3762_recovery.py')),run_name='__main__')


if __name__=='__main__':main()
