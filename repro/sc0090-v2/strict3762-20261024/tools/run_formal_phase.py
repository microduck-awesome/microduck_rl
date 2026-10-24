"""Launch one unchanged historical allocation after its reviewed qualification gates."""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, re, subprocess, time
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def canonical(x):
 if isinstance(x,dict):return {k:canonical(v) for k,v in x.items()}
 if isinstance(x,list):return [canonical(v) for v in x]
 if isinstance(x,str):return re.sub(r' at 0x[0-9a-fA-F]+(?=>)','',x)
 return x

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--phase',type=int,required=True,choices=range(10));a=parser.parse_args()
 review=read(ROOT/'document-review.json');assert review['status']=='PASS_FOR_QUALIFICATION_ONLY'
 for name,h in review['locked_files'].items():assert sha(ROOT/name)==h,('Changed reviewed file',name)
 assert str(Path(__file__).relative_to(ROOT)) in review['locked_files']
 plan=read(ROOT/'execution-plan.json');row=plan['phases'][a.phase];hist=read(ROOT/'history-inventory.json')['phases'][a.phase]
 q=Path(plan['qualification_root']);formal=Path(plan['formal_root']);out=Path(row['output'])
 assert not out.exists(),('Never overwrite/restart partial allocation',out)
 formal.mkdir(parents=True,exist_ok=True)
 lock=open(q/'process.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 boundary=read(q/'update12-boundary-process.json')
 assert boundary['status']=='PASS' and boundary['completed_updates']==12 and boundary['common_step_counter']==288
 assert boundary['no_evaluation'] and boundary['no_update13']
 assert boundary['historical_entry_sha256']==sha(ROOT/'tools/historical_entry.py')
 assert boundary['document_sha256']==sha(ROOT/'STRICT_3762_SINGLE_GPU.md')
 # Revalidate static files, chronological numeric code, and collision assets.
 materialized=read(ROOT/'materialized-sources.json');built=materialized['phases'][a.phase]
 expected={**materialized['static_file_hashes'],**hist['source_files'],**built['added_meshes']}
 expected[built['only_modified_static_file']]=built['corrected_xml_sha256']
 for name,h in expected.items():assert sha(Path(row['source'])/name)==h,name
 proof={}
 for kind,directory,envs,iterations in [('smoke','smoke64',64,5),('batch','batch',8192,row['qualification_iterations'])]:
  record=read(q/row['name']/f'{kind}-process.json');report_path=q/row['name']/directory/'report.json';report=read(report_path)
  assert record['status']=='PASS' and record['report_sha256']==sha(report_path)
  assert report['status']=='PASS' and report['task']==hist['task'] and report['device']=='cuda:1'
  assert report['envs']==envs and report['iterations']==iterations and report['robot_xml_sha256']==built['corrected_xml_sha256']
  assert report['physics_cuda_graph'] and min(report['motor_graph_replays'])>0 and report['onnx_exported']
  proof[kind]=sha(report_path)
  if kind=='batch' and a.phase>0:
   assert report['throughput']['measured_updates']>=10 and report['throughput']['median_end_to_end_steps_per_s']>100000
 import torch
 parent=None
 if a.phase:
  previous=plan['phases'][a.phase-1];parent=Path(row['replay_trainer_argv'][row['replay_trainer_argv'].index('--resume')+1])
  assert parent.parent==Path(previous['output'])
  prior=read(formal/f'{previous["name"]}-process.json');assert prior['status']=='PASS'
  assert prior['checkpoint_hashes'][parent.name]==sha(parent)
  d=torch.load(parent,map_location='cpu',weights_only=False);v=d['infos']['v2']
  assert v['completed_updates']==previous['expected_completed_updates'] and len(v['cuda_rng'])==2
  assert d['infos']['env_state']['common_step_counter']==24*previous['expected_completed_updates']
  if row['head_stage'] is not None:assert v['recovery']['polish'],'Native head recipe requires recovery polish'
 env=os.environ.copy();contract=plan['environment_contract']
 for key in contract['unset']:env.pop(key,None)
 for key in ('SC0090_CUDA_COMPAT_DIR','SC0090_V2_PYTHON'):env[key]=contract[key]
 check=formal/f'{row["name"]}-config.json'
 with (formal/f'{row["name"]}-config.log').open('w') as log:
  subprocess.run(['bash',row['wrapper'],str(ROOT/'tools/check_phase_config.py'),'--phase',row['name'],'--output',str(check)],cwd=row['source'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert read(check)['status']=='PASS'
 argv=['bash',row['wrapper'],*row['launch_argv']]
 receipt=dict(status='RUNNING',pid=os.getpid(),argv=argv,cwd=row['source'],started_unix=time.time(),qualification_hashes=proof,parent_sha256=sha(parent) if parent else None,document_review_sha256=sha(ROOT/'document-review.json'))
 record=formal/f'{row["name"]}-process.json';record.write_text(json.dumps(receipt,indent=2)+'\n')
 with (formal/f'{row["name"]}.log').open('w') as log:
  result=subprocess.run(argv,cwd=row['source'],env=env,stdout=log,stderr=subprocess.STDOUT)
 receipt.update(status='EXITED',returncode=result.returncode,ended_unix=time.time());record.write_text(json.dumps(receipt,indent=2)+'\n')
 assert result.returncode==0,('Original trainer failed; preserve attempt',out)
 status=read(out/'status.json');assert status['status']==row['expected_terminal_status'] and status['completed_updates']==row['expected_completed_updates']
 run=read(out/'run.json');assert run['source_sha256']==hist['source_files'] and run['dependencies']==hist['dependencies']
 cfg=read(check.with_suffix('.config.json'))
 assert canonical({k:run.get(k) for k in cfg})==canonical(cfg),'Effective runtime config differs from checked config'
 assert run['num_envs']==8192 and run['device']=='cuda:1' and run['seed']==20261015
 n=row['expected_completed_updates'];filename=f'checkpoint_{n:06d}.pt' if a.phase<2 else f'resume_{n:06d}.pt'
 d=torch.load(out/filename,map_location='cpu',weights_only=False);assert d['infos']['v2']['completed_updates']==n
 assert d['infos']['env_state']['common_step_counter']==24*n and len(d['infos']['v2']['cuda_rng'])==2
 if a.phase==0:
  assert not list(out.glob('evaluation*')) and read(out/'historical-stop-event.json')['before_completed_update']==12
 expected_evaluations={item['file'] for item in hist['evaluations']}
 assert {p.name for p in out.glob('evaluation_*.json')}==expected_evaluations
 for name in expected_evaluations:
  evaluation=read(out/name)
  assert evaluation['seeds']==[2026101501,2026101502] and evaluation['samples_per_group']==128 and evaluation['duration_s']==8.
  assert evaluation['iteration']==int(Path(name).stem.rsplit('_',1)[1])
 receipt.update(status='PASS',completed_updates=n,checkpoint_hashes={p.name:sha(p) for p in out.glob('*.pt')},run_sha256=sha(out/'run.json'),native_recovery=d['infos']['v2']['recovery'])
 record.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
