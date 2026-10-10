"""CPU R1 operating-point, causal donor, reset and frozen R0 parity checks."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_microduck.sc0090.backend import GpuM6Batch
from mjlab_microduck.sc0090.registers import RegisterPipeline
from mjlab_microduck.tasks.sc0090_env_cfg import STUDY,M6_PATH,action_contract,make_sc0090_env_cfg


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);start=time.perf_counter()
    randomization=STUDY/'measurement/R1-frozen-v1'
    conditions=json.loads((randomization/'model.json').read_text())['conditions']
    frozen_source=STUDY/'runs/core-frozen500-v1/P03-seed42-n1024/source/src/mjlab_microduck/sc0090/registers.py'
    dump(out/'pre-run.json',{'R1_sha256':digest(randomization/'model.json'),'M6_sha256':digest(M6_PATH),
        'source_sha256':digest(__file__),'frozen_R0_source_sha256':digest(frozen_source),'device':'cpu',
        'conditions':len(conditions),'stationary_q_tolerance_rad':1e-6,'stationary_v_tolerance_rad_s':1e-5,
        'reset_geometry_tolerance':1e-7,'actor_dimensions':4,'action_dimensions':1,
        'formal_training':False,'hardware_executed':False})
    module_name='mjlab_microduck.sc0090._frozen_R0_registers'
    spec=importlib.util.spec_from_file_location(module_name,frozen_source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fit=STUDY/'measurement/fit-joint-residual';weights=STUDY/'observer/qualification/O1-native100-seed42.npz'
    parity=[]
    for S,O in [('S0','O0'),('S1','O1')]:
        old=module.RegisterPipeline(fit,weights,16,S,O,seed=42);new=RegisterPipeline(fit,weights,16,S,O,seed=42)
        good=True
        for tick in range(201):
            clock=1000000+tick*5000;t=tick*.005
            q=torch.full((16,),.2*np.sin(t));v=torch.full((16,),.2*np.cos(t))
            if tick and tick%4==0:
                count=503+round(8*np.sin(t));old.command(clock,count);new.command(clock,count)
            old.advance(clock,q,v);new.advance(clock,q,v)
            good &= torch.equal(old.last_raw,new.last_raw) and torch.equal(old.last_read_t,new.last_read_t)
            a,av=old.query(clock);b,bv=new.query(clock)
            good &= torch.equal(av,bv) and torch.equal(torch.nan_to_num(a),torch.nan_to_num(b))
        parity.append({'S':S,'O':O,'passed':bool(good)})
    settings=np.array([r['settings'] for r in conditions]);masses=np.array([r['mass_kg'] for r in conditions])
    nominal=(action_contract().count_from_physical(0.)-506.)*settings[0,6]
    entries=np.tile(np.array([nominal,0.,0.,1.]),(600*len(conditions),1))
    offsets=np.arange(len(conditions)+1)*600
    with torch.inference_mode():
        body=GpuM6Batch(M6_PATH,settings,masses,device='cpu');stationary=body.rollout(offsets,entries)
    stationary_maximum=abs(stationary[:,:2]).max(0)
    routes=[];first_conditions=None;first_phases=None
    with np.load(fit/'residual-library.npz',allow_pickle=False) as archive:
        donor_metadata=json.loads(str(archive['metadata']));donor_offsets=archive['offsets']
    for S,O in [('S0','O0'),('S1','O0'),('S0','O1'),('S1','O1')]:
        cfg=make_sc0090_env_cfg(S,O,R='R1');cfg.scene.num_envs=64;cfg.seed=42
        with torch.inference_mode():
            env=ManagerBasedRlEnv(cfg,device='cpu');observations,_=env.reset()
            action=env.action_manager.get_term('position');pipeline=action.pipeline;actuator=env.scene['bench'].actuators[0]
            initial=pipeline.condition_index.clone();phase=pipeline.next_start.clone()%10000
            if first_conditions is None:first_conditions=initial.clone();first_phases=phase.clone()
            matched=torch.equal(initial,first_conditions) and torch.equal(phase,first_phases)
            checks={'4D_actor':observations['actor'].shape==(64,4),'1D_action':env.action_manager.total_action_dim==1,
                'paired_condition_phase_seed':matched,'frozen_M6_parameters':digest(M6_PATH)=='7c35a732575bfad135d5bf7c7eeb44f268ffd96862064f04e9e7a5105bbf6fb4'}
            faults=0;donor_ok=True
            for tick in range(60):
                obs,reward,done,timeout,_=env.step(torch.zeros((64,1)))
                checks['finite_actor_and_physical_reward']=checks.get('finite_actor_and_physical_reward',True) and bool(torch.isfinite(obs['actor']).all() and torch.isfinite(reward).all())
                faults+=int(done.sum())
                for world,donor in enumerate(pipeline.last_raw[:,5].long().tolist()):
                    record=int(np.searchsorted(donor_offsets,donor,side='right')-1)
                    actual=donor_metadata[record];expected=conditions[int(pipeline.condition_index[world])]
                    donor_ok &= (actual['group'],actual['voltage_v'],actual['kp'])==(expected['group'],expected['voltage_v'],15)
            checks['matched_condition_donors']=donor_ok
            checks['zero_voltage_preserved']=bool(torch.all(pipeline.last_raw[:,3]==0))
            selected=torch.tensor([1,5,17,39]);others=torch.tensor([i for i in range(64) if i not in selected])
            before={name:getattr(pipeline,name)[others].clone() for name in ['clock','physics','physics_t','last_raw','condition_index']}
            positions=env.scene['bench'].data.joint_pos[others].clone();velocities=env.scene['bench'].data.joint_vel[others].clone()
            body_id=env.sim.mj_model.body('bench/arm').id;other_mass=env.sim.model.body_mass[others,body_id].clone()
            env.reset(env_ids=selected)
            checks['partial_reset_preserves_other_streams']=all(torch.equal(value,getattr(pipeline,name)[others]) for name,value in before.items())
            checks['partial_reset_preserves_other_physics']=torch.equal(positions,env.scene['bench'].data.joint_pos[others]) and torch.equal(velocities,env.scene['bench'].data.joint_vel[others]) and torch.equal(other_mass,env.sim.model.body_mass[others,body_id])
            geometry_error=0.
            for _ in range(12):
                env.reset();expected=pipeline.condition_settings[pipeline.condition_index]
                geometry_error=max(geometry_error,float(abs(actuator.settings[:,0]-expected).max()),
                    float(abs(env.sim.model.body_mass[:,body_id]-pipeline.condition_mass[pipeline.condition_index]).max()))
            checks['reset_never_accumulates']=geometry_error<1e-7
            routes.append({'S':S,'O':O,'passed':all(checks.values()),'checks':checks,
                'fault_resets':faults,'initial_unique_conditions':len(set(initial.tolist())),'maximum_geometry_settings_error':geometry_error})
            env.close()
        print(json.dumps({'route':S+'/'+O,'passed':routes[-1]['passed']}),flush=True)
    passed=all(r['passed'] for r in parity+routes) and bool(np.all(stationary_maximum<[1e-6,1e-5]))
    np.savez_compressed(out/'stationary36.npz',trace=stationary,offsets=offsets,settings=settings,masses=masses)
    report={'passed':passed,'R0_frozen_source_parity':parity,'stationary36_maximum_q_v':stationary_maximum.tolist(),
        'routes':routes,'elapsed_s':time.perf_counter()-start,'device':'cpu','hardware_executed':False,
        'scope':'Actual mjlab/Warp CPU R1 measured operating-point/reset/donor contract, common3s zero equilibrium and unchanged R0 output parity; distribution quality remains diagnostic'}
    dump(out/'report.json',report);print(json.dumps({k:v for k,v in report.items() if k!='routes'}),flush=True)
    if not passed:raise SystemExit('R1 task contract failed; formal training not admitted')


if __name__=='__main__':main()
