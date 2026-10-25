"""Five official PPO updates plus runtime checks for the restored motor DR."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import torch
import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab_microduck.custom_v2.microduck_v2 import TASK_ID, POLICY_JOINTS

OUT=Path('docs/evidence/official-walk/corrected/smoke64');OUT.mkdir(parents=True,exist_ok=True)
cfg=load_env_cfg(TASK_ID);cfg.scene.num_envs=64;cfg.seed=42
agent=load_rl_cfg(TASK_ID);agent.logger='tensorboard';agent.upload_model=False
torch.set_num_threads(2)
env=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
    runner=load_runner_cls(TASK_ID)(RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions),asdict(agent),str(OUT),'cuda:0')
    stats=[]
    for actuator in env.scene['robot'].actuators:
        scale=actuator.friction_scale
        assert (scale >= .9).all() and (scale <= 1.1).all() and scale.std()>0
        assert actuator.has_delay and actuator._delay_buffer is not None
        assert (actuator.cfg.delay_min_lag,actuator.cfg.delay_max_lag)==(3,6)
        assert (actuator.vin_tensor >= 10.2).all() and (actuator.vin_tensor <= 12.4).all()
        assert actuator.vin_tensor.std()>0
        assert (actuator.vin_drop_gain >= 0).all() and (actuator.vin_drop_gain <= .2).all()
        ids=actuator._dof_ids
        actual=env.sim.model.dof_armature[:,ids]
        nominal=env.sim.get_default_field('dof_armature')[ids]
        ratios=actual/nominal
        assert (ratios >= .9-1e-6).all() and (ratios <= 1.1+1e-6).all(), (actuator._target_names, ratios.min(), ratios.max())
        # V2 mouth is outside the official 14 policy-joint selector.
        if set(actuator._target_names).issubset(POLICY_JOINTS):
            assert ratios.std()>0, actuator._target_names
        stats.append(dict(friction_min=scale.min().item(),friction_max=scale.max().item(),
            armature_ratio_min=ratios.min().item(),armature_ratio_max=ratios.max().item(),
            vin_min=actuator.vin_tensor.min().item(),vin_max=actuator.vin_tensor.max().item(),
            delay_steps=[3,6],official_delay_buffer=True))
    updates=[]
    native=runner.alg.update
    def checked_update():
        losses=native()
        assert all(math.isfinite(float(v)) for v in losses.values()),losses
        assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())
        updates.append({k:float(v) for k,v in losses.items()})
        return losses
    runner.alg.update=checked_update
    runner.learn(num_learning_iterations=5,init_at_random_ep_len=True)
    assert len(updates)==5
    assert (OUT/'model_4.pt').exists()
    onnx=list(OUT.glob('*.onnx'));assert onnx,'Official ONNX export must complete'
    import onnxruntime as ort
    session=ort.InferenceSession(str(onnx[0]),providers=['CPUExecutionProvider'])
    assert session.get_inputs()[0].shape[-1]==61
    assert session.get_outputs()[0].shape[-1]==14
    for a in env.scene['robot'].actuators:
        assert (a._bam_model.actuator.vin >= 10.2).all()
        assert (a._bam_model.actuator.vin <= 12.4).all()
    report=dict(status='PASS',device='cuda:0',envs=64,updates=5,runtime_randomization=stats,
                finite_losses=updates,checkpoint='model_4.pt',onnx_actor_input=61,onnx_actions=14,
                physics_cuda_graph=env.sim.use_cuda_graph)
    (OUT.parent/'smoke64.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
finally:
    env.close()
