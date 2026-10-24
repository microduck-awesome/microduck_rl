from pathlib import Path
import json,hashlib,re,time,subprocess
import torch,numpy as np,mujoco
BASE=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261024-strict3762');ROOT=BASE/'publication/repro/sc0090-v2/strict3762-20261024';PUB=BASE/'publication'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):
 if isinstance(x,dict):return {k:canonical(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [canonical(v) for v in x]
 if isinstance(x,str):return re.sub(r' at 0x[0-9a-fA-F]+(?=>)','',x)
 return x
chain=read(BASE/'chain-status.json');assert chain['status']=='TRAINING_CHAIN_FINISHED_PENDING_FINAL_AUDIT' and chain['completed_updates']==3762 and len(chain['completed_phases'])==10
review=read(ROOT/'document-review.json')
for name,h in review['locked_files'].items():assert sha(ROOT/name)==h,name
plan=read(ROOT/'execution-plan.json');inv=read(ROOT/'history-inventory.json');mat=read(ROOT/'materialized-sources.json');oldstates=read(ROOT/'checkpoint-history.json')['checkpoints'];rows=[];previous_end=None;native_evals=[]
for row,hist,built,oldstate in zip(plan['phases'],inv['phases'],mat['phases'],oldstates):
 i=row['index'];end=row['expected_completed_updates'];p=Path(row['output']);source=Path(row['source']);rec=read(p.parent/(p.name+'-process.json'));run=read(p/'run.json');status=read(p/'status.json');cfg=read(p.parent/(p.name+'-config.json'));effective=read(p.parent/(p.name+'-config.config.json'))
 assert rec['status']=='PASS' and rec['returncode']==0 and rec['completed_updates']==end
 assert rec['argv']==['bash',row['wrapper'],*row['launch_argv']]
 assert status['status']==row['expected_terminal_status'] and status['completed_updates']==end
 assert run['device']=='cuda:1' and run['num_envs']==8192 and run['seed']==20261015
 assert run['allocation_updates']==hist['allocated_updates'] and run['task']==hist['task'] and run['dependencies']==hist['dependencies']
 assert run['source_sha256']==hist['source_files'] and sha(p/'run.json')==rec['run_sha256']
 assert canonical({k:run.get(k) for k in effective})==canonical(effective)
 assert cfg['status']=='PASS' and cfg['differences']==[] and len(cfg['allowed_collision_differences'])==96
 assert cfg['triton_enable_fp_fusion']==(i<3)
 for key in ['WORLD_SIZE','RANK','LOCAL_RANK','CUDA_VISIBLE_DEVICES','MICRODUCK_WARM_START']:assert cfg['compiler_environment'][key] is None
 expected={**mat['static_file_hashes'],**hist['source_files'],**built['added_meshes']};expected[built['only_modified_static_file']]=built['corrected_xml_sha256']
 for name,h in expected.items():assert sha(source/name)==h,(p.name,name)
 numeric={str(f.relative_to(source)) for f in source.glob('src/**/*') if f.suffix in ('.py','.cpp')};assert numeric=={name for name in hist['source_files'] if name.startswith('src/')}
 if previous_end is not None:assert rec['started_unix']>=previous_end
 previous_end=rec['ended_unix']
 if i==0:assert run['resume'] is None and rec['parent_sha256'] is None
 else:
  prev=plan['phases'][i-1];filename=f'checkpoint_{prev["expected_completed_updates"]:06d}.pt' if i<=2 else f'resume_{prev["expected_completed_updates"]:06d}.pt';parent=Path(prev['output'])/filename;assert run['resume']==str(parent) and rec['parent_sha256']==sha(parent)
  restored=read(p/'resume-verification.json');assert restored['status']=='PASS' and restored['actor_critic_normalizers_optimizer_exact'] and restored['checkpoint_sha256']==sha(parent)
 filename=f'checkpoint_{end:06d}.pt' if i<2 else f'resume_{end:06d}.pt';selected=p/filename;d=torch.load(selected,map_location='cpu',weights_only=False);v=d['infos']['v2'];assert v['completed_updates']==end and d['infos']['env_state']['common_step_counter']==24*end;assert len(v['cuda_rng'])==2 and v['source_sha256']==hist['source_files'];assert v['recovery']['stage']==oldstate['recovery']['stage'] and v['recovery']['polish']==oldstate['recovery']['polish']
 for name,h in rec['checkpoint_hashes'].items():assert sha(p/name)==h,name
 files={f.name for f in p.glob('evaluation_*.json')};assert files=={x['file'] for x in hist['evaluations']}
 for name in sorted(files):
  e=read(p/name);assert e['config_sha256']==sha(p/'run.json') and e['seeds']==[2026101501,2026101502] and e['samples_per_group']==128 and e['duration_s']==8. and e['deterministic']
  assert e['iteration']==int(Path(name).stem.rsplit('_',1)[1])
  if i>=2:assert sha(p/f'checkpoint_{e["iteration"]:06d}.pt')==e['checkpoint_sha256']
  native_evals.append(dict(phase=p.name,iteration=e['iteration'],wins=sum(g['wins'] for c in e['cases'] for g in c['groups'].values()),trials=sum(g['trials'] for c in e['cases'] for g in c['groups'].values()),polish=e['decision']['polish'],right_side_rate=e['decision']['rates']['right_side'],report_sha256=sha(p/name)))
 if i==0:assert read(p/'historical-stop-event.json')['before_completed_update']==12 and not files
 rows.append(dict(index=i,name=p.name,completed_updates=end,common_step_counter=24*end,task=run['task'],num_envs=8192,device='cuda:1',fp_fusion=cfg['triton_enable_fp_fusion'],source_files_exact=len(hist['source_files']),static_and_collision_files_exact=len(expected),selected_checkpoint=filename,selected_checkpoint_sha256=sha(selected),native_exit_code=rec['returncode'],native_recovery_stage=v['recovery']['stage'],native_polish=v['recovery']['polish']))
assert len(native_evals)==19 and [e['iteration'] for e in native_evals]==[250,500,750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3328,3456,3506,3584,3712,3762]
# Recheck actual compiled physical identities without modifying the frozen preflight evidence.
a=mujoco.MjModel.from_xml_path(str(BASE/'sources/.static-cache/src/mjlab_microduck/custom_v2/assets/robot.xml'));z=mujoco.MjModel.from_xml_path(str(BASE/'sources/paired-feedback-allocation-001/src/mjlab_microduck/custom_v2/assets/robot.xml'))
fields=['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','body_parentid','jnt_type','jnt_bodyid','jnt_pos','jnt_axis','jnt_range','jnt_limited','qpos0','actuator_trnid','actuator_ctrlrange','actuator_gear']
for field in fields:assert np.array_equal(getattr(a,field),getattr(z,field)),field
assert a.ngeom==75 and z.ngeom==107 and a.nq==z.nq and a.nv==z.nv
for i in range(a.ngeom):
 name=mujoco.mj_id2name(a,mujoco.mjtObj.mjOBJ_GEOM,i);j=mujoco.mj_name2id(z,mujoco.mjtObj.mjOBJ_GEOM,name);assert j>=0
 for field in ['geom_bodyid','geom_type','geom_pos','geom_quat','geom_size','geom_friction','geom_solref','geom_solimp','geom_contype','geom_conaffinity','geom_condim','geom_margin','geom_gap']:
  assert np.array_equal(getattr(a,field)[i],getattr(z,field)[j]),(name,field)
legacy=read(ROOT/'publication-receipt.json');assert legacy['manifest_files_verified_as_git_blobs']==186
subprocess.run(['git','merge-base','--is-ancestor',legacy['commit'],'HEAD'],cwd=PUB,check=True)
subprocess.run(['git','diff','--exit-code',legacy['commit'],'HEAD','--',legacy['archived_result']],cwd=PUB,check=True)
assert read(BASE/'final-evaluation-status.json')['status']=='FINAL_COMPARISON_RUNS_FINISHED_PENDING_REVIEW'
assert read(BASE/'regression-replay/status.json')['status']=='COMPLETED_PENDING_TRACE_REVIEW' and read(BASE/'normalizer-ablation/status.json')['status']=='COMPLETED_PENDING_REVIEW'
summary=read(BASE/'final-evaluation/comparison-summary.json');assert summary['model_checks']['completed_updates']==3762
report=dict(status='PASS_REQUESTED_TRAINING_AND_EVIDENCE_AUDIT',verified_unix=time.time(),legacy_result_publication=legacy,legacy_published_tree_unchanged=True,frozen_document_sha256=sha(ROOT/'STRICT_3762_SINGLE_GPU.md'),review_sha256=sha(ROOT/'document-review.json'),phases=rows,native_evaluations=native_evals,single_gpu_sequential_training=True,formal_retries=0,completed_updates=3762,control_steps_per_environment=90288,aggregate_environment_control_steps=3762*8192*24,collision_delta=dict(original_geoms=75,new_geoms=107,added_shells=32,mass_kg=float(z.body_mass.sum()),existing_model_fields_exact=fields,existing_geom_fields_exact=True,robot_xml_sha256='8ae0d89041950d565bbcd6272ab87317f46132ff1eef99b922782ecbe2b95308'),final_artifact=summary['model_checks'],behavior_qualified_as_replacement=False,hardware_qualified=False,limitations=['New +63deg neck command causes marked body instability despite restored angular accuracy','Transient physical joint-limit penetration occurs in the head battery','2500 right-side regression is preserved and reproducible; normalizer swaps localize it to the network component','Frozen original protocol clarified by the one-way-polish review addendum; no numerical recipe change made'],pending_delivery_publication=True)
(ROOT/'training-completion-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],phases=len(rows),evaluations=len(native_evals),updates=3762,mass_kg=report['collision_delta']['mass_kg'],behavior_qualified=False)))
