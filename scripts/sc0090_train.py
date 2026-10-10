"""Local SC0090 smoke, throughput and matched production training/export."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml

from mjlab_microduck.tasks.sc0090_env_cfg import STUDY, make_sc0090_env_cfg, make_sc0090_rl_cfg


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',choices=[f'P{i:02d}' for i in range(8)],default='P03')
    parser.add_argument('--stage',choices=['smoke','benchmark','train'],default='smoke')
    parser.add_argument('--num-envs',type=int,default=64)
    parser.add_argument('--iterations',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training-protocol',type=Path)
    parser.add_argument('--resume-checkpoint',type=Path)
    args=parser.parse_args()
    number=int(args.policy[1:]);S=f'S{number%2}';O=f'O{(number//2)%2}';R=f'R{number//4}'
    protocol=None;completed_before=0;resume_data=None
    if args.training_protocol:
        protocol=json.loads(args.training_protocol.read_text())
    if args.resume_checkpoint:
        if not args.training_protocol:raise SystemExit('Resume needs the explicit revised training protocol')
        resume_data=torch.load(args.resume_checkpoint,weights_only=False,map_location='cpu')
        completed_before=int(resume_data['iter'])+1
        previous_protocol=resume_data['infos']['training_protocol_sha256']
        if previous_protocol!=digest(args.training_protocol):
            migration=protocol.get('checkpoint_migration',{})
            allowed=migration.get('checkpoints',{}).get(str(args.resume_checkpoint.resolve()))
            checkpoint_sha=digest(args.resume_checkpoint)
            if isinstance(allowed,dict):
                authorized=(allowed.get('predecessor_protocol_sha256')==previous_protocol and
                            allowed.get('checkpoint_sha256')==checkpoint_sha)
            else:
                authorized=(previous_protocol==migration.get('predecessor_protocol_sha256') and
                            allowed==checkpoint_sha)
            if not authorized:
                raise SystemExit('Resume checkpoint lacks an explicit protocol migration and matching hash')
        if resume_data['infos']['policy']!=args.policy or resume_data['infos']['seed']!=args.seed:
            raise SystemExit('Resume checkpoint policy/seed identity mismatch')
    if args.training_protocol:
        if args.device!=protocol['device'] or args.num_envs!=protocol['num_envs']:
            raise SystemExit('Execution differs from the revised GPU protocol')
        if args.stage=='train':
            if not protocol.get('formal_training_admitted'):
                raise SystemExit('Revised protocol lacks actual GPU/task/smoke admission')
            interval=protocol['convergence']['evaluation_interval']
            if args.iterations%interval or not completed_before<args.iterations<=protocol['convergence']['maximum_iterations']:
                raise SystemExit('Target budget is not a declared evaluation boundary')
            for path,sha in protocol['source_sha256'].items():
                if digest(path)!=sha:raise SystemExit('Frozen production source changed: '+path)
            for path,sha in protocol['input_sha256'].items():
                if digest(path)!=sha:raise SystemExit('Frozen model/evidence changed: '+path)
    if args.stage=='train' and protocol is None:
        admission=json.loads((STUDY/'reports/training-admission.json').read_text())
        if not all(r['formal_training_admitted'] for r in admission['routes'].values()):
            raise SystemExit('Formal training needs recorded runtime/task/smoke evidence')
        if R=='R1':
            path=STUDY/'reports/R1-training-admission.json'
            if not path.is_file() or not json.loads(path.read_text()).get('formal_training_admitted'):
                raise SystemExit('R1 formal training requires separate operating-point/runtime/task/smoke admission')
            if json.loads(path.read_text())['training_device']!=args.device:
                raise SystemExit('R1 execution device differs from its qualified training path')
        budget=json.loads((STUDY/'reports/frozen-training-budget.json').read_text())
        if (args.num_envs,args.iterations)!=(budget['num_envs'],budget['iterations']):
            raise SystemExit('Production run differs from the frozen matched budget')
        if args.device=='cpu':
            path=STUDY/'integration/warp-development-attempt2/report.json'
            pre=STUDY/'integration/warp-development-attempt2/pre-run.json'
            if not path.is_file() or not json.loads(path.read_text())['passed'] or json.loads(pre.read_text())['environment']['device']!='cpu':
                raise SystemExit('CPU production requires the actual mjlab/Warp CPU full844 qualification')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    cfg=make_sc0090_env_cfg(S,O,R=R);cfg.scene.num_envs=args.num_envs
    # Block resumes restore PPO/optimizer/RNG; environment restarts use a
    # predeclared common seed schedule instead of replaying the first block.
    cfg.seed=args.seed+completed_before*104729
    agent=make_sc0090_rl_cfg();agent.seed=args.seed;agent.max_iterations=args.iterations
    agent.save_interval=max(1,min(100,args.iterations));agent.run_name=f'{args.stage}-{args.policy}-seed{args.seed}'
    sources=[*Path('src/mjlab_microduck/sc0090').glob('*.py'),Path(__file__),
             Path('src/mjlab_microduck/tasks/sc0090_env_cfg.py'),Path('src/mjlab_microduck/tasks/mdp.py')]
    if protocol:sources.append(Path(__file__).with_name('sc0090_validate_checkpoint.py'))
    for source in sources:
        destination=output/'source'/source.resolve().relative_to(Path.cwd())
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
    if protocol:shutil.copyfile(args.training_protocol,output/'training-protocol.json')
    if R=='R1':
        randomization=Path(cfg.actions['position'].randomization_path)
        shutil.copytree(randomization,output/'source/R1-frozen')
    write(output/'pre-run.json',{'policy':args.policy,'S':S,'O':O,'R':R,'stage':args.stage,
        'seed':args.seed,'device':args.device,'num_envs':args.num_envs,'iterations':args.iterations,
        'completed_iterations_before':completed_before,'environment_seed':cfg.seed,
        'training_protocol_sha256':digest(args.training_protocol) if protocol else None,
        'resume_checkpoint_sha256':digest(args.resume_checkpoint) if args.resume_checkpoint else None,
        'resume_predecessor_protocol_sha256':resume_data['infos']['training_protocol_sha256'] if resume_data else None,
        'source_sha256':{str(p):digest(p) for p in sources},
        'warmup_us':280000,'query_delay_us':20000,'normalization':False,
        'tasks_seed':91011,'source_frozen':True,'hardware_executed':False})
    torch.set_num_threads(1)
    with torch.inference_mode():
        env=ManagerBasedRlEnv(cfg,device=args.device)
        wrapped=RslRlVecEnvWrapper(env)
    dump_yaml(output/'params/env.yaml',asdict(cfg));dump_yaml(output/'params/agent.yaml',asdict(agent))
    if args.stage=='benchmark':
        # Uncontended real stream/physics throughput, including policy MLP.
        model=torch.nn.Sequential(torch.nn.Linear(4,256),torch.nn.ELU(),torch.nn.Linear(256,128),
            torch.nn.ELU(),torch.nn.Linear(128,64),torch.nn.ELU(),torch.nn.Linear(64,1)).to(args.device)
        synchronize=(lambda:torch.cuda.synchronize()) if args.device=='cuda:0' else (lambda:None)
        samples=[];faults=0
        with torch.inference_mode():
            for trial in range(4):
                obs=env.observation_manager.compute()['actor'];synchronize();start=time.perf_counter()
                for _ in range(24):
                    obs,reward,done,timeout,extras=env.step(model(obs))
                    faults+=int(done.sum());obs=obs['actor']
                synchronize();elapsed=time.perf_counter()-start
                if trial>0:samples.append(elapsed)
        write(output/'report.json',{'passed':True,'stage':'benchmark','num_envs':args.num_envs,
            'elapsed_24steps_s':samples,'median_interactions_per_s':args.num_envs*24/float(np.median(samples)),
            'fault_resets':faults,'device':args.device,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda:0' else None,
            'scope':'Single-process physics/register/observer/actor throughput; PPO timing from smoke separate'})
        env.close();print(json.dumps({'benchmark':args.num_envs,'seconds':samples}),flush=True);return
    runner=MjlabOnPolicyRunner(wrapped,asdict(agent),str(output),args.device)
    if args.resume_checkpoint:
        runner.load(str(args.resume_checkpoint),map_location=args.device)
        runner.current_learning_iteration=completed_before
        # The upstream loader restores optimizer groups but leaves the PPO
        # adaptive schedule attribute at its constructor value.
        runner.alg.learning_rate=float(runner.alg.optimizer.param_groups[0]['lr'])
        rng=resume_data['infos'].get('torch_rng')
        if rng is not None:torch.set_rng_state(rng)
        cuda_rng=resume_data['infos'].get('cuda_rng')
        if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
        del resume_data
    elapsed=0.;evaluation_reports=[]
    interval=protocol['convergence']['evaluation_interval'] if protocol else args.iterations
    for completed in range(completed_before+interval,args.iterations+1,interval):
        start=time.perf_counter();runner.learn(interval,init_at_random_ep_len=False)
        elapsed+=time.perf_counter()-start
        if protocol:
            checkpoint_path=output/f'model_{completed-1}.pt'
            runner.save(str(checkpoint_path),infos={'training_protocol_sha256':digest(args.training_protocol),
                'policy':args.policy,'seed':args.seed,'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all(),'learning_rate':runner.alg.learning_rate})
            snapshot=output/'evaluations'/f'iteration{completed:04}'
            snapshot.mkdir(parents=True,exist_ok=False)
            runner.export_policy_to_onnx(str(snapshot),'policy.onnx')
            cmd=[sys.executable,str(Path(__file__).with_name('sc0090_validate_checkpoint.py')),
                 '--policy',args.policy,'--checkpoint',str(checkpoint_path),
                 '--onnx',str(snapshot/'policy.onnx'),'--protocol',str(args.training_protocol.resolve()),
                 '--output',str(snapshot/'validation')]
            with (snapshot/'validation.log').open('w') as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            report_path=snapshot/'validation/report.json'
            if result.returncode or not report_path.is_file():
                raise RuntimeError('Checkpoint validation execution failed; checkpoint/log retained')
            validation=json.loads(report_path.read_text())
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            events=EventAccumulator(str(output),size_guidance={'scalars':0});events.Reload()
            rewards=events.Scalars('Train/mean_reward') if 'Train/mean_reward' in events.Tags()['scalars'] else []
            values=[float(e.value) for e in rewards if completed-interval<=e.step<completed]
            validation.update(completed_iterations=completed,
                reward_mean=float(np.mean(values)) if len(values)>=int(interval*.8) else None,
                reward_logged_iterations=len(values),validation_report_sha256=digest(report_path),
                validation_report=str(report_path.relative_to(output)))
            evaluation_reports.append(validation);write(output/'convergence-evaluations.json',evaluation_reports)
            # rsl_rl stores the last completed index. Advance once between
            # learn() calls to avoid duplicated iteration99/199/... logs.
            runner.current_learning_iteration=completed
    runner.export_policy_to_onnx(str(output),'policy.onnx')
    model=runner.alg.get_policy().as_onnx(verbose=False).cpu().eval()
    session=ort.InferenceSession(str(output/'policy.onnx'),providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(7109);maximum=0.
    with torch.inference_mode():
        for i in range(512):
            x=(rng.uniform(-1,1,(1,4))*[.8,10.,5.,.7]).astype(np.float32)
            expected=model(torch.from_numpy(x)).numpy()
            actual=session.run(None,{session.get_inputs()[0].name:x})[0]
            maximum=max(maximum,float(abs(expected-actual).max()))
    assert maximum<1e-5, maximum
    assert session.get_inputs()[0].shape[-1]==4 and session.get_outputs()[0].shape[-1]==1
    checkpoint=output/f'model_{args.iterations-1}.pt'
    manifest={'schema':'sc0090-policy-v1','policy':args.policy,'stage':args.stage,'seed':args.seed,
        'S':S,'O':O,'R':R,'D':'sc0090-v5-m6-seed1','training_device':args.device,
        'observer_weights_sha256':digest(cfg.actions['position'].weight_path),
        'M6_sha256':digest(env.scene['bench'].actuators[0].cfg.parameter_path),
        'S_model_sha256':digest(Path(cfg.actions['position'].fit_path)/'model.json'),
        'S_residual_library_sha256':digest(Path(cfg.actions['position'].fit_path)/'residual-library.npz'),
        'source_pre_run_sha256':digest(output/'pre-run.json'),
        'action_contract':asdict(cfg.actions['position'].contract),
        'observation':['physical_q_hat_rad','v_hat_rad_s','last_raw_action','physical_task_target_rad'],
        'network':[4,256,128,64,1],'activation':'elu','normalization':False,
        'main_condition':{'group':'L10_M070','P':15,'D':30,'voltage_v':11.1},
        'timing':{'native_hz':100,'feature_hz':100,'policy_hz':50,'physics_dt_s':.005,'query_delay_s':.02,'warmup_s':.28},
        'train_interactions':args.num_envs*24*args.iterations,'iterations':args.iterations,
        'checkpoint_sha256':digest(checkpoint),'onnx_sha256':digest(output/'policy.onnx'),
        'eligible_for_hardware_matrix':False,'sim2real_benefit_verified':False}
    if protocol:
        manifest.update(training_protocol_sha256=digest(args.training_protocol),
            num_envs=args.num_envs,steps_per_iteration=24,completed_iterations_before=completed_before,
            block_interactions=args.num_envs*24*(args.iterations-completed_before),
            resume_checkpoint_sha256=digest(args.resume_checkpoint) if args.resume_checkpoint else None,
            convergence_status='awaiting_common24_matrix_stopping_decision',
            checkpoint_selection='final at common matrix stopping budget; no hardware reselection')
    if R=='R1':
        manifest['randomization_sha256']=digest(randomization/'model.json')
        manifest['randomization_manifest_sha256']=digest(randomization/'manifest.json')
        manifest['training_conditions']='36 measured training-only P15 conditions,12 loads x exact10.2/11.1/12.4V, empirical read phases and matched residual blocks'
    write(output/'manifest.json',manifest)
    write(output/'report.json',{'passed':True,'stage':args.stage,'policy':args.policy,'iterations':args.iterations,
        'num_envs':args.num_envs,'elapsed_training_s':elapsed,'onnx_test_nodes':512,
        'completed_iterations_before':completed_before,
        'training_protocol_sha256':digest(args.training_protocol) if protocol else None,
        'maximum_torch_onnx_difference':maximum,'checkpoint':checkpoint.name,
        'scope':'Training and FP32 PyTorch/ONNX numeric check; full policy fault/long-run/hardware qualification separate'})
    env.close();print(json.dumps({'passed':True,'policy':args.policy,'stage':args.stage,'seconds':elapsed,'onnx_maximum':maximum}),flush=True)


if __name__=='__main__':main()
