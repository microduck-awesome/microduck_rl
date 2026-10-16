import hashlib,json,os,pathlib,subprocess,time
r=pathlib.Path(__file__).resolve().parent;src=r/'walking-repaired-prep-worktree';out=r/'walking-discovery-allocation-001'
q=r/'walking-final-qualification.json';d=json.loads(q.read_text());assert d['status']=='PASS'
for name,digest in d['source_sha256'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==digest,name
for name,digest in d['assets_sha256'].items():assert hashlib.sha256((src/name).read_bytes()).hexdigest()==digest,name
assert hashlib.sha256((src/'scripts/evaluate_v2_walking.py').read_bytes()).hexdigest()==d['evaluator_sha256']
cmd=[str(src/'scripts/v2_python.sh'),'scripts/train_v2_walking.py','--device','cuda:1','--num-envs','32768','--iterations','256','--qualification',str(q),'--output',str(out)]
env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')
start=time.time();record=dict(status='STARTING',start_unix=start,arguments=cmd,qualification_sha256=hashlib.sha256(q.read_bytes()).hexdigest(),planned_updates=256,planned_controls=201326592)
p=r/'walking-discovery-formal-process.json'
def save():
 t=p.with_suffix('.tmp');t.write_text(json.dumps(record,indent=2)+'\n');t.replace(p)
save()
with (r/'walking-discovery-allocation-001.log').open('w') as log:
 proc=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT);record.update(status='RUNNING',pid=proc.pid);save();record['returncode']=proc.wait()
record.update(status='PROCESS_EXITED',finish_unix=time.time(),wall_seconds=time.time()-start);save()
if (out/'status.json').exists():
 status=json.loads((out/'status.json').read_text());record.update(training_status=status['status'],actual_controls=status['allocation_controls'],full_process_control_per_second=status['allocation_controls']/record['wall_seconds'])
 (out/'actual-wall-time.json').write_text(json.dumps(record,indent=2)+'\n');save()
if record['returncode']!=0:raise SystemExit(record['returncode'])
