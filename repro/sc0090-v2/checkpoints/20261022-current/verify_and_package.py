"""Package saved training exports and verify checkpoint/ONNX/FP32-MDP parity on CPU."""
from pathlib import Path
import json,hashlib,importlib.util,subprocess,struct,zlib,tarfile,sys
import numpy as np, torch, onnx, onnxruntime as ort
from onnx import numpy_helper

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def serial(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if torch.is_tensor(x):return x.detach().cpu().tolist()
 if isinstance(x,dict):return {k:serial(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [serial(v)for v in x]
 return x

def main():
 torch.set_num_threads(2);root=Path(__file__).resolve().parent
 experiment=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261021/formal-four-way-20261021-001')
 converter=root/'tools/convert_policy.py'
 sys.path.insert(0,str(converter.parent))
 from v2_contract import info, RUNTIME, WALKING_RUNTIME
 spec=importlib.util.spec_from_file_location('firmware_converter',converter);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 models=[]
 for skill,n,source in [('rise',4000,Path('/home/l40-vm01/Documents/ai-research/microduck/microduck-v2-sim2real')),('walk',4600,experiment/'walk-smoothing-repair')]:
  d=root/f'{skill}-{n}';saved=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False);state=saved['actor_state_dict'];record=saved['infos']['sim2real_baseline_20261021'];assert record['completed_updates']==n
  graph=onnx.load(d/'policy.onnx');onnx.checker.check_model(graph);arrays={v.name:numpy_helper.to_array(v)for v in graph.graph.initializer}
  for key in ('obs_normalizer._mean',*[f'mlp.{i}.{s}'for i in (0,2,4,6)for s in ('weight','bias')]):np.testing.assert_array_equal(arrays[key],state[key].numpy())
  denominator=arrays[next(node.input[1]for node in graph.graph.node if node.op_type=='Div')];np.testing.assert_array_equal(denominator,(state['obs_normalizer._std']+.01).numpy())
  metadata={v.key:v.value for v in graph.metadata_props};assert len(metadata['joint_names'].split(','))==14;assert metadata['action_scale']=='1.0';assert metadata['default_joint_pos'].split(',')==['0.000']*14
  rng=np.random.default_rng(20261022);inputs=np.concatenate((np.zeros((1,61),np.float32),rng.standard_normal((258,61)).astype(np.float32)))
  protocol='tilt'if skill=='rise'else'walking';trace=experiment/f'{skill}-formal-005/{protocol}-assessment-{n}/traces.pt'
  if trace.exists():
   data=torch.load(trace,map_location='cpu',weights_only=False,mmap=True);obs=data['actor_observation'].reshape(-1,61);indices=torch.linspace(0,len(obs)-1,512).long();inputs=np.concatenate((inputs,obs[indices].numpy()))
  x=torch.from_numpy(inputs);y=(x-state['obs_normalizer._mean'])/(state['obs_normalizer._std']+.01)
  for i in (0,2,4,6):
   y=torch.nn.functional.linear(y,state[f'mlp.{i}.weight'],state[f'mlp.{i}.bias'])
   if i!=6:y=torch.nn.functional.elu(y)
  options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
  session=ort.InferenceSession(str(d/'policy.onnx'),sess_options=options,providers=['CPUExecutionProvider']);actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:row[None]})[0]for row in inputs])
  error=float(np.max(np.abs(actual-y.numpy())));assert np.isfinite(actual).all()and error<2e-5,error
  original=module.export(d/'policy.onnx',d/f'{skill}-{n}.mdp');(d/'converter-original.json').write_text(json.dumps(original,indent=2)+'\n')
  binary=(d/f'{skill}-{n}.mdp').read_bytes();header=struct.unpack('<8s6I5I32s44x',binary[:128]);assert header[0]==b'MDPOL001'and header[1]==2 and header[6]==zlib.crc32(binary[128:]);assert header[-1]==hashlib.sha256((d/'policy.onnx').read_bytes()).digest()
  decoded=info(binary);assert decoded['runtime']==(RUNTIME if skill=='rise'else WALKING_RUNTIME)
  expected=[arrays['obs_normalizer._mean'].ravel(),denominator.ravel()]
  for i in (0,2,4,6):expected += [state[f'mlp.{i}.bias'].numpy().ravel(),state[f'mlp.{i}.weight'].numpy().ravel()]
  np.testing.assert_array_equal(np.frombuffer(binary[128:],dtype='<f4'),np.concatenate(expected))
  contract=dict(model_id=f'v2_sim2real_{skill}_{n}_20261022',training_runtime_family='microduck-v2-sc0090-official-baseline-v1',deployment_runtime_family=decoded['runtime'],policy_runtime_id_hex=hex(decoded['policy_runtime_id']),mdp_header_version=2,deployment_contract=original['deployment_contract'],deployment_contract_sha256=decoded['deployment_contract_sha256'],observation_dim=61,action_dim=14,policy_hz=50,normalizer_included=True,normalize_again=False,joint_names=metadata['joint_names'].split(','),home_rad=[0.]*14,action_scale=1.,action_semantics='continuous CAD-radian target; apply existing V2 per-joint device target bounds; no P04 native-count policy semantics',observation_layout=['gyro_body_xyz:0:3','projected_gravity:3:6','joint_position_HOME_relative:6:20','joint_velocity:20:34','previous_raw_policy_action:34:48','twist_vx_vy_yaw_rate:48:51','head_neckpitch_pitch_yaw_roll:51:55','body_pose:55:61'],mouth='excluded; independent hold',training_action_ema=False,operating_point='SC0090 P15/D30/I0; training v5 derivative_seconds=0',robot_mass_kg=.6246,hardware_tested=False,simulation_acceptance_passed=False,metadata_corrections={'microduck_motor_operating_point':'Legacy short ONNX metadata says0..0.2V/Nm; actual training uses shared bus resistance0.05..0.15ohm and PWM-weighted current. See checkpoint randomization_contract.'})
  allocation=json.loads((d/'training-allocation.json').read_text());contract['observation_term_scales']={k:v.get('scale')for k,v in allocation['env']['observations']['actor']['terms'].items()};contract['action_target_clip_by_joint']=allocation['env']['actions']['joint_pos']['clip']
  (d/'runtime-contract.json').write_text(json.dumps(contract,indent=2)+'\n');manifest={**original,'source_repository':'https://github.com/microduck-awesome/microduck_rl','observation':'microduck-v2-61d-policy-joint-order','model_id':contract['model_id'],'runtime_contract':'runtime-contract.json','checkpoint_sha256':sha(d/'checkpoint.pt'),'hardware_tested':False};(d/f'{skill}-{n}.json').write_text(json.dumps(manifest,indent=2)+'\n')
  (d/'onnx-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');(d/'checkpoint-info.json').write_text(json.dumps(serial(record),indent=2)+'\n')
  paths=sorted((source/'src').rglob('*.py'))+[source/'scripts'/x for x in ('train_v2_sim2real.py','evaluate_v2_sim2real.py','validate_v2_sim2real.py')]+[source/'src/mjlab_microduck/custom_v2/assets/sim2real_balance_poses.json']
  digest=hashlib.sha256()
  for p in paths:digest.update(str(p.relative_to(source)).encode());digest.update(p.read_bytes())
  assert digest.hexdigest()==record['recipe_sha256'],'Source recipe changed'
  with tarfile.open(d/'training-code.tar.gz','w:gz')as tar:
   for p in paths:tar.add(p,arcname=str(p.relative_to(source)))
   for p in [source/f'src/mjlab_microduck/custom_v2/assets/robot_sim2real_{skill}_backlash.xml',source/'docs/SIM2REAL_LEARNING_RECIPE.md']:tar.add(p,arcname=str(p.relative_to(source)))
  evidence=d/'evaluation';evidence.mkdir(exist_ok=True);summaries=[]
  for protocol in (('tilt','four_way_push','head')if skill=='rise'else('walking','straight','four_way_push')):
   p=experiment/f'{skill}-formal-005/{protocol}-assessment-{n}/report.json'
   if p.exists():
    raw=p.read_bytes();(evidence/f'{protocol}.json').write_bytes(raw);report=json.loads(raw);worlds=[w for row in report['records']for w in row['worlds']];assert report['checkpoint_sha256']==sha(d/'checkpoint.pt')
    summaries.append(dict(protocol=protocol,trials=len(worlds),qualified=sum(a['wins']for a in report['records']),never_fell=sum(bool(a.get('never_fell'))for a in worlds),body_stable_tail=sum(bool(a.get('body_stable_tail'))for a in worlds),quiet_stop_tail=sum(bool(a.get('quiet_stop_tail'))for a in worlds),self_contact=sum(bool(a.get('self_contact'))for a in worlds)))
  p=experiment/f'{skill}-formal-005/assessment-{n}.json'
  if p.exists():(evidence/'recovery.json').write_bytes(p.read_bytes())
  validation=dict(mdp_header_version=2,v2_runtime_verified=True,deployment_runtime=decoded['runtime'],checkpoint_matches_onnx_weights_and_normalizer_exactly=True,normalizer_epsilon=.01,cpu_onnx_test_inputs=len(inputs),cpu_onnx_max_absolute_error=error,fp32_mdp_coefficients_exact=True,fp32_mdp_crc_and_source_sha_verified=True,converter_source_sha256=sha(converter),gpu_simulation_performed=False,hardware_tested=False)
  (d/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
  models.append(dict(id=contract['model_id'],directory=d.name,completed_updates=n,recipe_sha256=record['recipe_sha256'],source_commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),source_worktree=str(source),source_archive_scope='Recipe-hashed Python sources, three entry scripts, balance poses, task MJCF and recipe document; external mesh/measurement assets and dependencies not bundled',validation=validation,evaluation=summaries))
 files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p))for p in sorted(root.rglob('*'))if p.is_file()and p!=root/'manifest.json'and '__pycache__'not in p.parts and p.suffix!='.pyc']
 (root/'manifest.json').write_text(json.dumps(dict(schema=1,snapshot_date='2026-10-22',purpose='User-requested current-model hardware observation; not a passed-policy release',models=models,files=files),indent=2)+'\n');print(json.dumps(models,indent=2))
if __name__=='__main__':main()
