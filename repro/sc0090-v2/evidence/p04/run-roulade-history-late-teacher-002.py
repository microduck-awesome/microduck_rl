"""One bounded64k late-landing diagnostic after SitStand releases GPU1."""
import os,sys,time,json,hashlib,traceback,importlib.util,gc
from pathlib import Path
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'roulade-history-worktree';os.chdir(W);sys.path.insert(0,str(W/'scripts'));plan=json.loads((R/'roulade-history-late-teacher-plan-001.json').read_text());state=R/'roulade-history-late-teacher-status-002.json';assert not state.exists();completed=0;started=time.time();reports=[]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write(**kw):
 p=state.with_suffix('.tmp');p.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),completed_diagnostic_controls=completed,diagnostic_controls_cap=64000,new_formal_controls=0,**kw),indent=2)+'\n');p.replace(state)
write(status='PRELOADING_CPU_WAITING_GPU1')
try:
 assert sha(Path(plan['student']))==plan['student_sha256'] and sha(Path(plan['teacher']))==plan['teacher_sha256'];probe=Path(plan['probe']);assert sha(probe)==plan['probe_sha256']
 spec=importlib.util.spec_from_file_location('roulade_late_teacher_probe',probe);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 import torch
 torch.set_num_threads(2);write(status='WAITING_SIT_COMPLETE_AND_PID_EXIT');deadline=time.time()+7200
 while True:
  d=json.loads((R/'sitstand-near-teacher-lr3e6-pipeline-status.json').read_text())
  if d['status']=='SITSTAND_NEAR_TEACHER_COMPLETE_NO_AUTOMATIC_CONTINUATION' and not Path('/proc/'+str(d['pid'])).exists():break
  if d['status']=='FAILED_REQUIRES_DIAGNOSIS':raise RuntimeError('Sit predecessor failed; do not automatically takeGPU')
  assert time.time()<deadline,'Bounded queue expired'
  time.sleep(2)
 torch.cuda.set_device('cuda:1')
 for arm in plan['arms']:
  for stage in plan['conditions']:
   out=R/f'roulade-history-late-teacher-{arm}-{stage}-001';write(status='RUNNING_DIAGNOSTIC',arm=arm,stage=stage,output=str(out))
   sys.argv=[str(probe),'--checkpoint',plan['student'],'--teacher',plan['teacher'],'--arm',arm,'--stage',stage,'--num-envs','64','--seed','2612','--device','cuda:1','--output',str(out)];module.main()
   count=json.loads((out/'diagnostic-count.json').read_text());assert count['completed_controls']==16000;completed+=16000;reports.append(str(out/'report.json'));gc.collect();torch.cuda.empty_cache()
 assert completed==64000;write(status='FOUR_LATE_LANDING_DIAGNOSTICS_COMPLETE',reports=reports,wall_s=time.time()-started)
except BaseException:
 write(status='STOPPED_REQUIRES_REVIEW',reports=reports,exception=traceback.format_exc());raise
