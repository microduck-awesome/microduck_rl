"""Correct-routing1400 hardware experiment delivery. CPU only; no training/device writes.

Build requires this machine's immutable experiments; --verify-only uses the bundle.
"""
from pathlib import Path
import argparse,ast,copy,hashlib,importlib.util,json,shutil,struct,subprocess,sys,tarfile,zlib
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import torch
ROOT=Path(__file__).resolve().parent
EXP=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261022')
WORK=Path('/home/l40-vm01/Documents/ai-research/microduck')
KEY='sim2real_baseline_20261021'
SPECS=[dict(skill='rise',update=1400,folder=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261023/correct-routing-retrain-008/rise/formal-001'),repo=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261023/rise1400-hardware-delivery/frozen-source'),onnx='../unused',trace='tilt-assessment-1400',protocols=['tilt','four_way_push','head'])]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(x):
 if torch.is_tensor(x):return x.cpu().tolist()
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {k:serial(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [serial(v)for v in x]
 return x
def write(p,x):p.write_text(json.dumps(serial(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def recipe_paths(repo):
 tree=ast.parse((repo/'src/mjlab_microduck/custom_v2/sim2real_learning.py').read_text())
 fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='recipe_sha256'))
 body=[]
 for n in fn.body:
  if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='digest'for t in n.targets):break
  body.append(n)
 fn.name='recipe_paths';fn.body=body+[ast.Return(value=ast.Name(id='paths',ctx=ast.Load()))]
 env={'ROOT':repo,'Path':Path};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<recipe paths>','exec'),env)
 return env['recipe_paths']()
def build():
 previous=ROOT.parent/'20261022-current';tools=ROOT/'tools';(tools/'contracts').mkdir(parents=True,exist_ok=True)
 for name in ('convert_policy.py','v2_contract.py'):shutil.copy2(previous/'tools'/name,tools/name)
 for spec in SPECS:
  skill,n=spec['skill'],spec['update'];d=ROOT/f'{skill}-{n}';d.mkdir(exist_ok=False)
  source=spec['folder']/f'assessed-{n}.pt';shutil.copy2(source,d/'checkpoint.pt');shutil.copy2(Path('/home/l40-vm01/data2/experiments/microduck-v2-20261023/rise1400-hardware-delivery/policy.onnx'),d/'policy.onnx')
  c=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False);r=c['infos'][KEY]
  assert r['completed_updates']==n and r['kind']==skill and r['allocation_kind']=='formal'
  metadata={p.key:p.value for p in onnx.load(d/'policy.onnx').metadata_props}
  assert metadata['microduck_runtime']=='microduck-v2-sc0090-official-baseline-v1'
  old=json.loads((previous/f'tools/contracts/v2-{"rise"if skill=="rise"else"walking"}-candidates.json').read_text())
  registry=dict(schema=1,source_repository='microduck-awesome/microduck_rl',release='20261023-rise1400',profile_provenance='Pinned firmware13eea5de converter and existing V2 hardware profile; this package adds exact original candidate identities',deployment=old['deployment'],candidates=[dict(id=f'v2_sim2real_{skill}_{n}_20261023_correct_routing',onnx_sha256=sha(d/'policy.onnx'),source_metadata=metadata,experiment='fresh_6246g_baseline_hardware_observation_not_full_sim_pass')])
  write(tools/f'contracts/v2-{"rise"if skill=="rise"else"walking"}-candidates.json',registry)
  allocation=json.loads((spec['folder']/'allocation.json').read_text());write(d/'training-allocation.json',allocation);write(d/'checkpoint-info.json',r);write(d/'onnx-metadata.json',metadata)
  evidence=d/'evaluation';evidence.mkdir();summary=[]
  for protocol in spec['protocols']:
   for suffix in (('', '-seed2026102103')if protocol=='walking'else('',)):
    p=spec['folder']/f'{protocol}-assessment-{n}{suffix}/report.json'
    if not p.exists():continue
    report=json.loads(p.read_text());assert report['checkpoint_sha256']==sha(d/'checkpoint.pt')
    shutil.copy2(p,evidence/f'{protocol}{suffix}.json');worlds=[w for row in report['records']for w in row['worlds']]
    summary.append(dict(protocol=protocol,seed=report['seed'],trials=len(worlds),qualified=sum(row['wins']for row in report['records']),**{k:sum(bool(w.get(k))for w in worlds)for k in ('never_fell','body_stable_tail','quiet_stop_tail','head_tail_pass','self_contact')},hip_yaw_clearance=report.get('hip_yaw_clearance'),hip_roll_parking=report.get('hip_roll_parking')))
  p=spec['folder']/f'assessment-{n}.json'
  if p.exists():shutil.copy2(p,evidence/'recovery.json')
  p=spec['folder']/f'primary-motion-{n}.json'
  if p.exists():shutil.copy2(p,evidence/'primary-motion.json')
  write(d/'evaluation-summary.json',summary)
  paths=recipe_paths(spec['repo']);digest=hashlib.sha256()
  for p in paths:digest.update(str(p.relative_to(spec['repo'])).encode());digest.update(p.read_bytes())
  assert digest.hexdigest()==r['recipe_sha256'],f'{skill} frozen source recipe mismatch'
  extra=[spec['repo']/n for n in ('AGENTS.md','pyproject.toml','uv.lock')]
  assets=spec['repo']/'src/mjlab_microduck/custom_v2/assets'
  extra+=list(assets.glob('*.json'))+list(assets.glob('*.yaml'))+list(assets.glob('*.yml'))
  with tarfile.open(d/'training-code.tar.gz','w:gz')as tar:
   for p in dict.fromkeys(paths+extra):
    if p.exists():tar.add(p,arcname=str(p.relative_to(spec['repo'])))
  source_info=dict(source_repository='https://github.com/microduck-awesome/microduck_rl',source_worktree=str(spec['repo']),source_commit='fe2cc15205583df3508ec86ea9fcc2f9ebeee720',recipe_sha256=r['recipe_sha256'],recipe_paths=[str(p.relative_to(spec['repo']))for p in paths],checkpoint_source=str(source),onnx_source='/home/l40-vm01/data2/experiments/microduck-v2-20261023/rise1400-hardware-delivery/policy.onnx',trace_source=str(spec['folder']/spec['trace']/'traces.pt'),archive_scope='Exact recipe-hashed code/assets plus root dependency locks and model JSON/YAML; external mesh/measurement libraries/dependencies not fully bundled')
  write(d/'source-info.json',source_info)
def converter():
 sys.path.insert(0,str(ROOT/'tools'));import convert_policy,v2_contract
 return convert_policy,v2_contract
def verify_model(d,build_mode):
 saved=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False);state=saved['actor_state_dict'];record=saved['infos'][KEY]
 graph=onnx.load(d/'policy.onnx');onnx.checker.check_model(graph);arrays={v.name:numpy_helper.to_array(v)for v in graph.graph.initializer};metadata={v.key:v.value for v in graph.metadata_props}
 names=['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle','neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle']
 assert metadata['joint_names'].split(',')==names and metadata['action_scale']=='1.0'and metadata['default_joint_pos'].split(',')==['0.000']*14
 for key in ('obs_normalizer._mean',*[f'mlp.{i}.{s}'for i in(0,2,4,6)for s in('weight','bias')]):np.testing.assert_array_equal(arrays[key],state[key].numpy())
 denominator=arrays[next(node.input[1]for node in graph.graph.node if node.op_type=='Div')];np.testing.assert_array_equal(denominator,(state['obs_normalizer._std']+.01).numpy())
 rng=np.random.default_rng(20261022);inputs=np.concatenate((np.zeros((1,61),np.float32),rng.standard_normal((258,61)).astype(np.float32)))
 sample=d/'verification-observations.npy'
 if build_mode:
  source=json.loads((d/'source-info.json').read_text());data=torch.load(source['trace_source'],map_location='cpu',weights_only=False,mmap=True);obs=data['actor_observation'].reshape(-1,61);indices=torch.linspace(0,len(obs)-1,512).long();np.save(sample,obs[indices].numpy())
 inputs=np.concatenate((inputs,np.load(sample)));x=torch.from_numpy(inputs);y=(x-state['obs_normalizer._mean'])/(state['obs_normalizer._std']+.01)
 for i in (0,2,4,6):
  y=torch.nn.functional.linear(y,state[f'mlp.{i}.weight'],state[f'mlp.{i}.bias'])
  if i!=6:y=torch.nn.functional.elu(y)
 options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
 session=ort.InferenceSession(str(d/'policy.onnx'),sess_options=options,providers=['CPUExecutionProvider']);actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:v[None]})[0]for v in inputs]);error=float(np.abs(actual-y.numpy()).max());assert np.isfinite(actual).all()and error<2e-5,error
 mod,v2=converter();target=d/(d.name+'.mdp')
 if build_mode:converted=mod.export(d/'policy.onnx',target);write(d/'converter.json',converted)
 else:converted=json.loads((d/'converter.json').read_text())
 data=target.read_bytes();header=struct.unpack('<8s6I5I32s44x',data[:128]);assert header[0]==b'MDPOL001'and header[1]==2 and header[6]==zlib.crc32(data[128:])and header[-1]==bytes.fromhex(sha(d/'policy.onnx'))
 decoded=v2.info(data);assert decoded['runtime']==(v2.RUNTIME if record['kind']=='rise'else v2.WALKING_RUNTIME)
 expected=[arrays['obs_normalizer._mean'].ravel(),denominator.ravel()]
 for i in(0,2,4,6):expected += [state[f'mlp.{i}.bias'].numpy().ravel(),state[f'mlp.{i}.weight'].numpy().ravel()]
 np.testing.assert_array_equal(np.frombuffer(data[128:],dtype='<f4'),np.concatenate(expected));assert data[84:128]==v2.extension(metadata,sha(d/'policy.onnx'))
 bad={**metadata,'action_scale':'0.5'}
 try:v2.extension(bad,sha(d/'policy.onnx'))
 except ValueError:pass
 else:raise AssertionError('Exact candidate metadata guard was bypassed')
 source=json.loads((d/'source-info.json').read_text());digest=hashlib.sha256()
 with tarfile.open(d/'training-code.tar.gz')as tar:
  for path in source['recipe_paths']:digest.update(path.encode());digest.update(tar.extractfile(path).read())
 assert digest.hexdigest()==record['recipe_sha256']
 result=dict(status='PASS',checkpoint_and_onnx_weights_and_normalizer_exact=True,cpu_inputs=len(inputs),cpu_onnx_max_absolute_error=error,normalizer_epsilon=.01,fp32_mdp_payload_exact=True,crc_and_source_sha_verified=True,mdp_header_version=2,deployment_runtime=decoded['runtime'],policy_runtime_id_hex=f"0x{decoded['policy_runtime_id']:08X}",candidate_metadata_guard_verified=True,source_archive_recipe_sha_verified=True,hardware_tested=False,new_GPU_simulation=False)
 if build_mode:
  write(d/'validation.json',result)
  old=json.loads((ROOT.parent/'20261022-current'/('rise-4000'if record['kind']=='rise'else'walk-4600')/'runtime-contract.json').read_text());allocation=json.loads((d/'training-allocation.json').read_text())
  old.update(model_id=f"v2_sim2real_{record['kind']}_{record['completed_updates']}_20261023_correct_routing",training_runtime_family=metadata['microduck_runtime'],deployment_runtime_family=decoded['runtime'],policy_runtime_id_hex=result['policy_runtime_id_hex'],deployment_contract=converted['deployment_contract'],deployment_contract_sha256=decoded['deployment_contract_sha256'],hardware_tested=False,simulation_acceptance_passed=False,training_phase=record['training_phase'],training_randomization_contract=record['randomization_contract'],source_checkpoint_sha256=sha(d/'checkpoint.pt'),source_onnx_sha256=sha(d/'policy.onnx'),source_recipe_sha256=record['recipe_sha256'],observation_term_scales={k:v.get('scale')for k,v in allocation['env']['observations']['actor']['terms'].items()},action_target_clip_by_joint=allocation['env']['actions']['joint_pos']['clip'])
  reference=1000 if record['kind']=='rise'else 1500;factor=record['batch_contract']['reference_learning_step_factor']
  terms=[v for v in allocation['env']['curriculum'].values()if v['params'].get('command_name')=='head_pose']
  stage=max((v for v in terms[0]['params']['range_stages']if v['step']<=reference*24/factor),key=lambda v:v['step'])
  old.update(metadata_corrections={},experiment_head_command_rad=[0.,0.,0.,0.],training_head_ranges_rad_at_snapshot=None,training_head_reference_stage=None,deployment_head_command_limits_rad=old['deployment_contract']['head_limit_rad'],range_note='Snapshot head range not reconstructed; hardware experiment uses zero head command. Existing V2 interface caps unchanged; zero request is not zero actual head posture.')
  write(d/'runtime-contract.json',old)
 return result
def manifest():
 files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(ROOT.rglob('*'))if p.is_file()and p.name!='manifest.json'and '__pycache__'not in p.parts and p.suffix!='.pyc']
 write(ROOT/'manifest.json',dict(schema=1,as_of_utc='2026-10-23',purpose='User requested latest completed rise/walk for on-robot observation; not full simulation or hardware acceptance',models=[dict(directory=f"{s['skill']}-{s['update']}",completed_updates=s['update'])for s in SPECS],files=files))
def main():
 p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');p.add_argument('--manifest-only',action='store_true');a=p.parse_args();torch.set_num_threads(2)
 if a.manifest_only:manifest();return
 if not a.verify_only:build()
 for s in SPECS:
  d=ROOT/f"{s['skill']}-{s['update']}";print(d.name,json.dumps(verify_model(d,not a.verify_only)),flush=True)
 if a.verify_only:
  for f in json.loads((ROOT/'manifest.json').read_text())['files']:assert sha(ROOT/f['path'])==f['sha256'],f['path']
 else:manifest()
if __name__=='__main__':main()
