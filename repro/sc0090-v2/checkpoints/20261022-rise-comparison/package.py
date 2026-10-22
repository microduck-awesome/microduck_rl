"""Package existing rise exports, with checkpoint/ONNX/V2 coefficient verification.

CPU only. Original exports and training directories are read-only. This script
requires the historical experiment root; colleagues can reconvert the included
ONNX files with tools/convert_policy.py without that root.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import struct
import sys
import zlib

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent
EXPERIMENT = Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015')
# id, original allocation, update, priority, comparison purpose
CANDIDATES = [
 ('rise_baseline_paired3762','baseline/paired-feedback-allocation-001',3762,1,'现有实机参照；head4＋配对误差块＋三档电压'),
 ('rise_baseline_foundation3506','baseline/head-foundation4-allocation-001',3506,1,'paired3762加入配对随机化前的父模型'),
 ('rise_baseline_foundation4018','baseline/head-foundation4-allocation-003',4018,1,'基础head4继续训练；不是paired3762续训'),
 ('rise_baseline_position_dr2_4012','baseline/position-robustness2-allocation-002',4012,1,'独立位置噪声±0.04rad＋20ms；head2'),
 ('rise_baseline_axis3_4268','baseline/head-axis3-allocation-002',4268,1,'位置DR2；逐轴收窄头部精度奖励'),
 ('rise_baseline_head_noise3_4012','baseline/head-noise3-allocation-001',4012,2,'腿位置噪声±0.04rad，头±0.015rad；head2'),
 ('rise_baseline_foundation_dr1_3762','baseline/head-foundation4-dr1-allocation-001',3762,2,'head4＋位置白噪声±0.02rad和20ms；不是paired3762'),
 ('rise_baseline_foundation_dr2_4018','baseline/head-foundation4-dr2-allocation-001',4018,2,'上一DR1支路继续，位置噪声扩大至±0.04rad'),
 ('rise_baseline_position_dr2_3756','baseline/position-robustness2-allocation-001',3756,2,'position4012、axis3和head-noise3的共同父节点'),
 ('rise_baseline_axis3_4012','baseline/head-axis3-allocation-001',4012,2,'axis3较早节点，与4268比较继续训练收益'),
 ('rise_p04_micro1856','p04/head2-micro-exposure-allocation-001',1856,2,'既有P04从零参照；实机曾被评价弱于paired3762'),
 ('rise_baseline_to_p04_4466','p04/paired-head-accuracy-allocation-002',4466,2,'既有基线到P04参照；实机曾被评价更差'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    torch.set_num_threads(2)
    previous = ROOT.parent/'20261022-current/tools'
    tools = ROOT/'tools'
    tools.mkdir(parents=True, exist_ok=True)
    (tools/'contracts').mkdir(exist_ok=True)
    for name in ('convert_policy.py','v2_contract.py'):
        shutil.copyfile(previous/name, tools/name)
    registry = json.loads((previous/'contracts/v2-rise-candidates.json').read_text())
    registry['supplement'] = '20261022 historical rise comparison; original deployment identity retained'
    for model_id, allocation, update, priority, purpose in CANDIDATES:
        source = EXPERIMENT/allocation
        graph = onnx.load(source/(source.name+'.onnx'))
        metadata = {p.key:p.value for p in graph.metadata_props}
        assert metadata['microduck_robot']=='microduck-v2-sc0090'
        assert metadata['action_scale']=='1.0'
        assert metadata['default_joint_pos'].split(',')==['0.000']*14
        assert metadata['command_names']=='twist,head_pose,body_pose'
        digest = sha(source/(source.name+'.onnx'))
        existing = next((c for c in registry['candidates'] if c['onnx_sha256']==digest), None)
        if existing:
            assert existing['id']==model_id and existing['source_metadata']==metadata
        else:
            registry['candidates'].append(dict(id=model_id,onnx_sha256=digest,
                source_metadata=metadata,experiment='baseline_transfer_to_native_o0'))
    write_json(tools/'contracts/v2-rise-candidates.json', registry)
    shutil.copyfile(previous/'contracts/v2-walking-candidates.json',tools/'contracts/v2-walking-candidates.json')
    sys.path.insert(0,str(tools))
    from convert_policy import export
    from v2_contract import info, RUNTIME, JOINTS
    rows=[]
    for model_id, allocation, update, priority, purpose in CANDIDATES:
        source=EXPERIMENT/allocation
        folder=ROOT/model_id
        folder.mkdir(exist_ok=True)
        checkpoint=source/f'checkpoint_{update:06d}.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        state=saved['actor_state_dict']
        shutil.copyfile(source/(source.name+'.onnx'),folder/'policy.onnx')
        shutil.copyfile(source/'run.json',folder/'training-run.json')
        graph=onnx.load(folder/'policy.onnx')
        metadata={p.key:p.value for p in graph.metadata_props}
        assert metadata['joint_names']==JOINTS
        arrays={x.name:numpy_helper.to_array(x) for x in graph.graph.initializer}
        for key in ('obs_normalizer._mean',*[f'mlp.{i}.{s}'for i in (0,2,4,6)for s in ('weight','bias')]):
            np.testing.assert_array_equal(arrays[key],state[key].numpy())
        denominator=arrays[next(n.input[1]for n in graph.graph.node if n.op_type=='Div')]
        np.testing.assert_array_equal(denominator,(state['obs_normalizer._std']+.01).numpy())
        rng=np.random.default_rng(20261022)
        standardized=rng.uniform(-2,2,(512,61)).astype(np.float32)
        inputs=np.concatenate((np.zeros((1,61),np.float32),
            state['obs_normalizer._mean'].numpy()+denominator*standardized))
        x=(torch.from_numpy(inputs)-state['obs_normalizer._mean'])/(state['obs_normalizer._std']+.01)
        for i in (0,2,4,6):
            x=torch.nn.functional.linear(x,state[f'mlp.{i}.weight'],state[f'mlp.{i}.bias'])
            if i!=6:x=torch.nn.functional.elu(x)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(folder/'policy.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:a[None]})[0]for a in inputs])
        error=float(np.max(np.abs(actual-x.numpy())))
        assert np.isfinite(actual).all() and error<2e-5,(model_id,error)
        manifest=export(folder/'policy.onnx',folder/(model_id+'.mdp'))
        binary=(folder/(model_id+'.mdp')).read_bytes()
        decoded=info(binary)
        assert decoded['runtime']==RUNTIME
        header=struct.unpack('<8s6I5I32s44s',binary[:128])
        assert header[1:5]==(2,61,14,4)
        assert header[6]==zlib.crc32(binary[128:])
        assert header[-2]==hashlib.sha256((folder/'policy.onnx').read_bytes()).digest()
        expected=[state['obs_normalizer._mean'].numpy().ravel(),denominator.ravel()]
        for i in (0,2,4,6):
            expected.extend([state[f'mlp.{i}.bias'].numpy().ravel(),state[f'mlp.{i}.weight'].numpy().ravel()])
        np.testing.assert_array_equal(np.frombuffer(binary[128:],dtype='<f4'),np.concatenate(expected))
        run=json.loads((source/'run.json').read_text())
        provenance=dict(model_id=model_id,allocation=allocation,checkpoint_filename=checkpoint.name,
            checkpoint_sha256=sha(checkpoint),checkpoint_local_source=str(checkpoint),
            checkpoint_included=False,source_archive_sha256=sha(source/'source.tar.gz') if (source/'source.tar.gz').exists() else None,
            original_onnx_sha256=sha(folder/'policy.onnx'),training_task=run['task'],parent=run.get('resume'),
            update=update,priority=priority,purpose=purpose,original_training_metadata=metadata,
            position_robustness_recipe=run.get('position_robustness_recipe'),head_precision_recipe=run.get('head_precision_recipe'),
            training_mass_note='Historical approximately600.94g CAD; not retrained on624.6g/corrected shell contacts',
            hardware_status='Historical operator comparison exists; no new test in this delivery' if model_id in ('rise_baseline_paired3762','rise_p04_micro1856','rise_baseline_to_p04_4466') else 'NOT_TESTED_IN_THIS_DELIVERY')
        write_json(folder/'provenance.json',provenance)
        validation=dict(model_id=model_id,checkpoint_onnx_coefficients_exact=True,normalizer_epsilon=.01,
            cpu_input_count=len(inputs),cpu_onnx_max_absolute_error=error,mdp_version=2,
            runtime_id=hex(decoded['policy_runtime_id']),mdp_coefficients_exact=True,crc_and_source_sha_pass=True,
            new_gpu_steps=0,new_training_updates=0,new_hardware_trials=0)
        write_json(folder/'validation.json',validation)
        report=source/f'evaluation_{update:06d}.json'
        if report.exists():shutil.copyfile(report,folder/'historical-training-evaluation.json')
        rows.append(dict(id=model_id,priority=priority,purpose=purpose,update=update,
            upload=f'{model_id}/{model_id}.mdp',mdp_sha256=manifest['binary_sha256'],
            onnx=f'{model_id}/policy.onnx',checkpoint_sha256=provenance['checkpoint_sha256'],validation=validation))
        print(model_id,'V2 PASS',error,flush=True)
    write_json(ROOT/'models.json',rows)
    fields='trial_id,model_id,mdp_sha256,operator,date_time,device_id,firmware_commit,config_sha256,initial_pose,test_kind,command,repeat_index,battery_start_v,battery_end_v,servo_max_temp_c,head_ema,leg_ema,voltage_compensation,rise_success,rise_to_stable_s,hold_duration_s,fell_before_stable,fell_after_stable,push_direction,push_method,maximum_tilt_deg,correction_latency_s,return_within_4deg_s,head_axis,head_command_deg,head_tail_error_deg,head_within_2deg,visible_jitter,feedback_faults,video_uri,telemetry_uri,notes'.split(',')
    with (ROOT/'hardware-results-template.csv').open('w',newline='') as f:csv.writer(f,lineterminator='\n').writerow(fields)
    files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(ROOT.rglob('*'))if p.is_file()and p.name!='manifest.json'and '__pycache__'not in p.parts]
    write_json(ROOT/'manifest.json',dict(schema=1,purpose='Existing historical rise candidates for hardware comparison; not causal ablation',models=rows,files=files))


if __name__=='__main__':
    main()
