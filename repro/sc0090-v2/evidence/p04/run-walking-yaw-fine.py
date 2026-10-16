import hashlib,json,os,pathlib,signal,statistics,subprocess,time
r=pathlib.Path(__file__).resolve().parent;src=r/'walking-yaw-fine-worktree';p=r/'walking-yaw-fine-plan.json'
plan=json.loads(p.read_text());plan['phases']=[]
patterns=('ccd overflow','broadphase overflow','narrowphase overflow','nefc overflow','njmax_nnz overflow','height field collision overflow','contact overflow','contact buffer overflow','constraint buffer full')
env=dict(os.environ,SC0090_CUDA_COMPAT_DIR='/home/l40-vm01/Documents/ai-research/microduck/microduck_rl/artifacts/cuda-compat-12.9/usr/local/cuda-12.9/compat')
common=[str(src/'scripts/v2_python.sh'),'scripts/train_v2_walking.py','--task',plan['task'],'--device','cuda:1','--resume',plan['parent'],'--source-transition',plan['source_transition']]
def save():
 t=p.with_suffix('.tmp');t.write_text(json.dumps(plan,indent=2)+'\n');t.replace(p)
def execute(name,cmd):
 phase=dict(name=name,arguments=cmd,start_unix=time.time());plan['phases'].append(phase);plan['status']=name.upper()+'_RUNNING';save();logpath=r/(name+'.log');rejected=False
 with logpath.open('w') as log:
  proc=subprocess.Popen(cmd,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT);phase['pid']=proc.pid;save()
  while proc.poll() is None:
   text=logpath.read_text(errors='replace').lower();found=[x for x in patterns if x in text]
   if found and not rejected:
    rejected=True;phase['capacity_rejection']=found;proc.send_signal(signal.SIGTERM);save()
   time.sleep(2)
 phase.update(returncode=proc.returncode,wall_seconds=time.time()-phase['start_unix']);save()
 assert proc.returncode==0 and not rejected,phase
 return phase
try:
 smoke=r/'walking-yaw-fine-smoke-64/report.json'
 while not smoke.exists():
  status=r/'walking-yaw-fine-smoke-64/status.json'
  if status.exists():
   s=json.loads(status.read_text());proc=pathlib.Path(f'/proc/{s["pid"]}/stat')
   if not proc.exists() or proc.read_text().split()[2]=='Z':raise RuntimeError('Small smoke exited without completed report')
  time.sleep(2)
 s=json.loads(smoke.read_text());assert s['status']=='ALLOCATION_COMPLETE' and s['completed_updates']==261
 assert s['terminal_bootstrap_samples']>=32 and s['training_metrics']['oracle_calls']==120
 assert max(s['training_metrics']['oracle_max_abs'])<1e-6
 while pathlib.Path(f'/proc/{s["pid"]}/stat').exists() and pathlib.Path(f'/proc/{s["pid"]}/stat').read_text().split()[2]!='Z':time.sleep(2)
 assert '12 passed' in (r/'walking-yaw-fine-cpu.log').read_text()
 out=r/'walking-yaw-fine-throughput-32768'
 actual=execute('walking-yaw-fine-actual-n',common+['--num-envs','32768','--iterations','16','--validation-only','--output',str(out)])
 a=json.loads((out/'report.json').read_text());assert a['status']=='ALLOCATION_COMPLETE' and a['completed_updates']==272
 assert a['source_sha256']==s['source_sha256'];assert a['training_metrics']['oracle_calls']==384 and max(a['training_metrics']['oracle_max_abs'])<1e-6
 assert a['contact_peak']<=a['contact_capacity'];assert a['throughput']['median_control_per_second']>=100000,a['throughput']
 metrics=[json.loads(line)['metrics'] for line in (out/'ppo-diagnostics.jsonl').read_text().splitlines()]
 quality=dict(max_mean_kl=max(x['PPO/kl_mean'] for x in metrics),max_clip=max(x['PPO/clip_fraction'] for x in metrics),last5_median_kl=statistics.median(x['PPO/kl_mean'] for x in metrics[-5:]),last5_median_clip=statistics.median(x['PPO/clip_fraction'] for x in metrics[-5:]))
 assert quality['max_mean_kl']<.5 and quality['max_clip']<.8 and quality['last5_median_kl']<.1 and quality['last5_median_clip']<.5,quality
 q=r/'walking-yaw-fine-qualification.json';qualification=dict(status='PASS',created_unix=time.time(),num_envs=32768,source_sha256=a['source_sha256'],actual_n_quality=quality,throughput=a['throughput'],validation_reports={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [smoke,out/'report.json',r/'walking-yaw-fine-cpu.log']},source_transition_sha256=hashlib.sha256(pathlib.Path(plan['source_transition']).read_bytes()).hexdigest(),scope='One additive yaw reward; exact256 parent restore for formal, never validation272 weights')
 q.write_text(json.dumps(qualification,indent=2)+'\n');plan['qualification']=str(q);save()
 out=r/'walking-yaw-fine-allocation-001'
 formal=execute('walking-yaw-fine-formal',common+['--num-envs','32768','--iterations','128','--qualification',str(q),'--output',str(out)])
 state=json.loads((out/'status.json').read_text());assert state['status']=='ALLOCATION_COMPLETE' and state['completed_updates']==384
 wall={**formal,'actual_controls':state['allocation_controls'],'full_process_control_per_second':state['allocation_controls']/formal['wall_seconds'],'scope':'Entire formal process including imports/init/save, may include concurrent independent evaluations'}
 (out/'actual-wall-time.json').write_text(json.dumps(wall,indent=2)+'\n')
 plan['status']='FORMAL_COMPLETE_EVALUATIONS_SEPARATE';plan['actual_formal_controls']=state['allocation_controls'];save()
except BaseException as error:
 plan['status']='FAILED';plan['error']=repr(error);save();raise
