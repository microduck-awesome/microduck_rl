"""Run reviewed artifact checks and sequential final comparisons after the training chain."""
from pathlib import Path
import fcntl,hashlib,json,os,subprocess,time
BASE=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261024-strict3762')
ROOT=BASE/'publication/repro/sc0090-v2/strict3762-20261024'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert read(BASE/'chain-status.json')['status']=='TRAINING_CHAIN_FINISHED_PENDING_FINAL_AUDIT'
 lock=open(BASE/'qualification/process.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 status_path=BASE/'final-evaluation-status.json';assert not status_path.exists(),'Preserve existing evaluation attempt'
 plan=read(ROOT/'final-evaluation-plan.json');source=Path(plan['source']);contract=read(ROOT/'execution-plan.json')['environment_contract']
 mat=read(ROOT/'materialized-sources.json');built=mat['phases'][-1];expected={**mat['static_file_hashes'],**plan['historical_numerical_sources'],**built['added_meshes']};expected[built['only_modified_static_file']]=built['corrected_xml_sha256']
 for name,h in expected.items():assert sha(source/name)==h,name
 for name,h in plan['added_evaluation_helpers'].items():assert sha(source/name)==h,name
 assert sha(source/plan['artifact_verifier'])==plan['artifact_verifier_sha256']
 assert sha(source/plan['joint_limit_analyzer']['script'])==plan['joint_limit_analyzer']['sha256']
 env=os.environ.copy()
 for key in contract['unset']:env.pop(key,None)
 for key in ('SC0090_CUDA_COMPAT_DIR','SC0090_V2_PYTHON'):env[key]=contract[key]
 env['PYTHONFAULTHANDLER']='1'
 state=dict(status='ARTIFACT_VERIFICATION_RUNNING',pid=os.getpid(),started_unix=time.time(),completed_jobs=[],driver_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'final-evaluation-plan.json'))
 def save():
  state['updated_unix']=time.time();tmp=status_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(status_path)
 def run(name,argv,log):
  state.update(status='RUNNING',current_job=name);save()
  with log.open('w') as out:
   code=subprocess.run(['bash',str(source/'scripts/v2_python.sh'),*argv],cwd=source,env=env,stdout=out,stderr=subprocess.STDOUT).returncode
  assert code==0,(name,'Abnormal exit; preserve attempt',code)
  state['completed_jobs'].append(name);save();print('COMPLETED',name,flush=True)
 try:
  save();run('artifact-verification',[plan['artifact_verifier']],BASE/'artifact-verification.log')
  output=Path(plan['output']);assert read(output/'artifact-verification.json')['status']=='PASS_ARTIFACTS_ONLY'
  for job in plan['jobs']:
   argv=job['argv'];assert argv[argv.index('--device')+1]=='cuda:1'
   run(job['name'],argv,output/(job['name']+'.log'))
  for name in ('new','old'):
   run('joint-limits-'+name,[plan['joint_limit_analyzer']['script'],'--head-dir',str(output/('head-'+name)),'--output',str(output/('joint-limits-'+name+'.json'))],output/('joint-limits-'+name+'.log'))
  state.update(status='FINAL_COMPARISON_RUNS_FINISHED_PENDING_REVIEW');save()
 except BaseException as error:
  state.update(status='FAILED',error=repr(error));save();raise
if __name__=='__main__':main()
