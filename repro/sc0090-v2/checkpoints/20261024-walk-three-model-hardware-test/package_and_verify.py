"""Build and verify three frozen walking candidates for user-run hardware comparison."""
import argparse,hashlib,json,shutil,struct,sys,tarfile,zlib
from pathlib import Path
import numpy as np
import torch
import onnx
from onnx import numpy_helper
import onnxruntime as ort
ROOT=Path(__file__).resolve().parent
BASE=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261024/walk3762-anticipation')
SOURCE=Path('/home/l40-vm01/Documents/ai-research/microduck/microduck-v2-walk-3762-anticipation')
SPECS=[('A_walk3500',3500,'formal-0002'),('B_walk3000',3000,'formal-0001'),('C_walk3750',3750,'formal-0003')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def build():
    profile=json.loads((ROOT.parent/'20261022-current/tools/contracts/v2-walking-candidates.json').read_text())['deployment']
    assert profile['runtime']=='microduck-v2-sc0090-native-o0-walking-v1' and profile['head_limit_rad']==[0,0,0,0]
    registry=dict(schema=1,source_repository='microduck-awesome/microduck_rl',release=ROOT.name,deployment=profile,candidates=[])
    first=None
    for label,update,formal in SPECS:
        d=ROOT/label;d.mkdir(exist_ok=False);src=BASE/formal
        shutil.copy2(src/f'checkpoint_{update:06d}.pt',d/'checkpoint.pt');shutil.copy2(src/f'{formal}.onnx',d/'policy.onnx');shutil.copy2(d/'policy.onnx',ROOT/f'{label}_mass600p94g.onnx')
        saved=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False);record=saved['infos']['walk3762']
        assert record['completed_updates']==update and not record['validation_only']
        if first is None:first=record['source_sha256']
        assert first==record['source_sha256']
        write(d/'checkpoint-source-sha256.json',record['source_sha256'])
        metadata={v.key:v.value for v in onnx.load(d/'policy.onnx').metadata_props}
        registry['candidates'].append(dict(id=f'v2_walk3762_{label}_20261024',onnx_sha256=sha(d/'policy.onnx'),checkpoint_sha256=sha(d/'checkpoint.pt'),source_metadata=metadata,experiment='new_collision_paired3762_walking_zero_head_hardware_comparison_not_qualified'))
        shutil.copy2(src/'run.json',d/'training-run.json')
        evidence=d/'evaluation';evidence.mkdir()
        for section in (f'evaluation-{update:06d}',f'head-composition-{update:06d}',f'handoff-{update:06d}'):
            folder=BASE/section
            for name in ('report.json','adjudication.json','quality-audit.json'):
                p=folder/name
                if p.exists():shutil.copy2(p,evidence/f'{section}-{name}')
        for name in (f'artifact-verification-{update:06d}.json',f'all14-clearance-{update:06d}.json'):
            p=BASE/name
            if p.exists():shutil.copy2(p,evidence/name)
        with np.load(BASE/f'evaluation-{update:06d}/limit-context-2026102401.npz') as trace:
            obs=trace['actor'];assert len(obs)>0;indices=np.linspace(0,len(obs)-1,min(512,len(obs))).astype(int);np.save(d/'verification-observations.npy',obs[indices])
        write(d/'provenance.json',dict(label=label,completed_updates=update,checkpoint_source=str(src/f'checkpoint_{update:06d}.pt'),checkpoint_sha256=sha(d/'checkpoint.pt'),onnx_sha256=sha(d/'policy.onnx'),training_source=str(SOURCE),source_archive='../training-code.tar.gz',hardware_tested=False,simulation_acceptance_passed=False,hardware_experiment_head_command=[0,0,0,0],native_onnx_requires_effective_head_commands=True,head_projection_note='Zero head request equals effective zero; MDP contains native MLP only, not the nonzero-head geometry projection. This comparison uses zero head/body commands.'))
    write(ROOT/'tools/contracts/v2-walking-candidates.json',registry)
    write(ROOT/'tools/contracts/v2-rise-candidates.json',dict(schema=1,candidates=[]))
    archived=[]
    with tarfile.open(ROOT/'training-code.tar.gz','w:gz') as tar:
        for name,digest in first.items():
            p=SOURCE/name;assert sha(p)==digest,name
            if p.suffix in ('.py','.cpp','.xml','.json'):
                tar.add(p,arcname=name);archived.append(name)
        for name in ('pyproject.toml','uv.lock','AGENTS.md','LICENSE','LICENSE-MODELS.md'):
            p=SOURCE/name
            if p.exists():tar.add(p,arcname=name)
    write(ROOT/'source-archive.json',dict(sha256=sha(ROOT/'training-code.tar.gz'),verified_complete_source_identity=first,archived_source_paths=archived,scope='Exact hashed code,XML,JSON and dependency locks; large mesh and paired-error datasets are not duplicated. Frozen model geometry is the published6804 collision package, not firmware limits.'))

def verify(build_mode):
    sys.path.insert(0,str(ROOT/'tools'));import convert_policy,v2_contract
    torch.set_num_threads(2)
    for label,update,formal in SPECS:
        d=ROOT/label;saved=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False);state=saved['actor_state_dict']
        assert sha(ROOT/f'{label}_mass600p94g.onnx')==sha(d/'policy.onnx')
        graph=onnx.load(d/'policy.onnx');onnx.checker.check_model(graph);arrays={v.name:numpy_helper.to_array(v) for v in graph.graph.initializer};meta={v.key:v.value for v in graph.metadata_props}
        assert meta['joint_names']==v2_contract.JOINTS and meta['action_scale']=='1.0' and meta['default_joint_pos'].split(',')==['0.000']*14
        for key in ('obs_normalizer._mean',*[f'mlp.{i}.{s}' for i in (0,2,4,6) for s in ('weight','bias')]):np.testing.assert_array_equal(arrays[key],state[key].numpy())
        denominator=arrays[next(n.input[1] for n in graph.graph.node if n.op_type=='Div')];np.testing.assert_array_equal(denominator,(state['obs_normalizer._std']+.01).numpy())
        rng=np.random.default_rng(20261024);inputs=np.concatenate((np.zeros((1,61),np.float32),rng.standard_normal((128,61)).astype(np.float32),np.load(d/'verification-observations.npy')))
        y=(torch.from_numpy(inputs)-state['obs_normalizer._mean'])/(state['obs_normalizer._std']+.01)
        for i in (0,2,4,6):
            y=torch.nn.functional.linear(y,state[f'mlp.{i}.weight'],state[f'mlp.{i}.bias'])
            if i!=6:y=torch.nn.functional.elu(y)
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(d/'policy.onnx'),sess_options=options,providers=['CPUExecutionProvider']);out=np.concatenate([session.run(None,{session.get_inputs()[0].name:x[None]})[0] for x in inputs]);error=float(abs(out-y.numpy()).max());assert np.isfinite(out).all() and error<2e-5
        target=d/f'{label}.mdp'
        if build_mode:converted=convert_policy.export(d/'policy.onnx',target);write(d/'converter.json',converted)
        else:converted=json.loads((d/'converter.json').read_text())
        blob=target.read_bytes();header=struct.unpack('<8s6I5I32s44x',blob[:128]);assert header[0]==b'MDPOL001' and header[1]==2 and header[6]==zlib.crc32(blob[128:]) and header[-1]==bytes.fromhex(sha(d/'policy.onnx'))
        info=v2_contract.info(blob);assert info['runtime']==v2_contract.WALKING_RUNTIME
        expected=[arrays['obs_normalizer._mean'].ravel(),denominator.ravel()]
        for i in (0,2,4,6):expected += [state[f'mlp.{i}.bias'].numpy().ravel(),state[f'mlp.{i}.weight'].numpy().ravel()]
        payload=np.frombuffer(blob[128:],dtype='<f4');np.testing.assert_array_equal(payload,np.concatenate(expected));assert blob[84:128]==v2_contract.extension(meta,sha(d/'policy.onnx'))
        result=dict(status='PASS',inputs=len(inputs),onnx_checkpoint_max_absolute_error=error,weights_and_normalizer_exact=True,mdp_payload_exact=True,crc_and_source_sha_verified=True,runtime=info['runtime'],runtime_id_hex=f"0x{info['policy_runtime_id']:08X}",hardware_tested=False)
        if build_mode:
            write(d/'validation.json',result)
            write(d/'runtime-contract.json',dict(model_id=f'v2_walk3762_{label}_20261024',observation_dim=61,action_dim=14,policy_hz=50,home_rad=[0]*14,action_scale=1.,normalizer_included=True,normalize_again=False,joint_names=v2_contract.JOINTS.split(','),mouth='independent hold, excluded from policy',robot_mass_kg=.600941825,training_action_ema=False,motor='SC0090 v5 BAM P15/D30/I0',training_feedback='paired q/v error blocks, one control-step observation delay',deployment_contract=converted['deployment_contract'],deployment_runtime=info['runtime'],runtime_id_hex=result['runtime_id_hex'],head_projection_in_mdp=False,experiment_head_command=[0]*4,experiment_body_command=[0]*6,hardware_tested=False,simulation_acceptance_passed=False))
        print(label,json.dumps(result),flush=True)
    source=json.loads((ROOT/'source-archive.json').read_text());assert sha(ROOT/'training-code.tar.gz')==source['sha256']
    with tarfile.open(ROOT/'training-code.tar.gz') as tar:
        for name in source['archived_source_paths']:assert hashlib.sha256(tar.extractfile(name).read()).hexdigest()==source['verified_complete_source_identity'][name]
    if not build_mode:
        for row in json.loads((ROOT/'manifest.json').read_text())['files']:assert sha(ROOT/row['path'])==row['sha256']
def manifest():
    files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts]
    write(ROOT/'manifest.json',dict(schema=1,purpose='User-authorized three walking candidates for hardware comparison; not simulation or hardware qualification',models=[dict(id=n,completed_updates=u) for n,u,_ in SPECS],files=files))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');p.add_argument('--manifest-only',action='store_true');a=p.parse_args()
    if a.manifest_only:manifest()
    else:
        if not a.verify_only:build()
        verify(not a.verify_only)
        if not a.verify_only:manifest()
