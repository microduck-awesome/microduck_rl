"""Finite CPU sampler checks and actual CUDA reset/voltage checks for this recipe."""
import json
from pathlib import Path
import torch
from mjlab_microduck.custom_v2.error_proxy import ErrorBlocks

def check_sampler():
    q=torch.arange(4*16,dtype=torch.float32).reshape(4,16)
    bank=torch.stack((q,2*q+3),-1)
    s=ErrorBlocks(bank,8,'cpu',17);ids=torch.arange(8);s.reset(ids,0)
    for step in range(40):
        value=s.sample(step).clone();torch.testing.assert_close(value[...,1],2*value[...,0]+3)
        torch.testing.assert_close(s.sample(step),value,rtol=0,atol=0)
    old=s.value.clone();s.reset(torch.tensor([1,6]),39)
    untouched=torch.tensor([0,2,3,4,5,7]);torch.testing.assert_close(s.value[untouched],old[untouched],rtol=0,atol=0)
    assert s.phase.min()>=0 and s.phase.max()<16

check_sampler()
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.custom_v2.training import configure_head_precision_learning
cfg=load_env_cfg('Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline');configure_head_precision_learning(cfg,4)
cfg.scene.num_envs=64;cfg.seed=2026101521
env=ManagerBasedRlEnv(cfg,device='cuda:1')
try:
    obs,_=env.reset();s=env._v2_error_blocks
    def verify():
        for act in env.scene['robot'].actuators:
            torch.testing.assert_close(act.vin_tensor[:,0],s.robot_voltage,rtol=0,atol=0)
    verify()
    with torch.inference_mode():
        for _ in range(6):obs,*_=env.step(torch.zeros(64,14,device='cuda:1'));verify()
        old_v=s.robot_voltage.clone();old_values=s.value.clone();env._reset_idx(torch.tensor([2,8],device='cuda:1'));verify()
        mask=torch.ones(64,dtype=torch.bool,device='cuda:1');mask[[2,8]]=False
        torch.testing.assert_close(s.robot_voltage[mask],old_v[mask],rtol=0,atol=0)
        torch.testing.assert_close(s.value[mask],old_values[mask],rtol=0,atol=0)
    result=dict(status='PASS',sampler_pairing=True,once_per_control=True,partial_reset_unaffected=True,
                actual_reset_and_poststep_voltage_matches=True,voltages=sorted(s.robot_voltage.unique().tolist()),
                bank_sha256=env._v2_sensor_proxy['bank_sha256'],scope='Approximate paired error DR and shared BAM voltage; no full P04 equivalence claim')
    Path('runs-v2/paired-feedback-checks.json').write_text(json.dumps(result,indent=2));print(result)
finally:env.close()
