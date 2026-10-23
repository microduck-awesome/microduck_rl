"""Qualify and run one bounded historical head phase after an optional service."""
import argparse,json,subprocess,sys,time,hashlib
from pathlib import Path
from finalize3762 import equal
ROOT=Path(__file__).resolve().parents[1]
TASK='Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline'

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',type=int,required=True,choices=(1,2,4));p.add_argument('--parent',type=Path,required=True);p.add_argument('--update',type=int,required=True);p.add_argument('--iterations',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--wait-unit');a=p.parse_args()
 out=a.output.resolve();out.mkdir(exist_ok=False,parents=True)
 def status(state,**extra):
  tmp=out/'status.tmp';tmp.write_text(json.dumps(dict(status=state,stage=a.stage,parent=str(a.parent),parent_update=a.update,updated_unix=time.time(),**extra),indent=2)+'\n');tmp.replace(out/'status.json')
 def run(name,args):
  status('RUNNING',step=name)
  with (out/f'{name}.log').open('w') as log:subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 try:
  status('WAITING')
  if a.wait_unit:
   while subprocess.check_output(['systemctl','--user','show',a.wait_unit,'--property=ActiveState','--value'],text=True).strip() in ('active','activating','deactivating'):time.sleep(30)
   final_status=json.loads((ROOT.parent/'final-evaluation/status.json').read_text())
   assert final_status['status']=='EVALUATION_COMPLETE_REQUIRES_REVIEW',final_status
  import torch
  parent=a.parent.resolve()
  for rank in (0,1):
   s=json.loads((parent/f'rank{rank}/status.json').read_text());assert s['status']=='ALLOCATION_COMPLETE' and s['completed_updates']==a.update,s
   cp=torch.load(parent/f'rank{rank}/resume_{a.update:06d}.pt',weights_only=False,map_location='cpu')
   assert cp['infos']['v2']['recovery']['polish'],'Recovery not mastered'
  if a.stage==1:
   run('head-old',['scripts/evaluate_v2_head.py','--task',TASK,'--checkpoint','/home/l40-vm01/data2/experiments/microduck-v2-20261015/baseline/paired-feedback-allocation-001/checkpoint_003762.pt','--output',out/'head-old','--device','cuda:0','--repeats',8])
  head=['--head-precision','--head-precision-stage',a.stage]
  for n,it,label in [(64,5,'smoke64'),(4096,12,'batch4096')]:
   run(label,['scripts/smoke_v2_ppo.py','--task',TASK,'--device','cuda:0','--num-envs',n,'--iterations',it,'--output-dir',out/label,'--report',out/f'{label}.json',*head])
   assert json.loads((out/f'{label}.json').read_text())['status']=='PASS'
  common=['-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train3762_ddp.py','--task',TASK,'--device','cuda:0','--num-envs',4096,'--preflight',out/'batch4096.json','--resume-root',parent,'--resume-update',a.update,*head]
  run('ddp-boundary',[*common,'--iterations',6,'--eval-interval',2,'--eval-samples',2,'--validation-only','--output',out/'ddp-boundary'])
  states=[]
  for rank in (0,1):
   r=out/f'ddp-boundary/rank{rank}';s=json.loads((r/'status.json').read_text());assert s['status']=='ALLOCATION_COMPLETE' and s['completed_updates']==a.update+6,s
   restore=json.loads((r/'resume-verification.json').read_text());assert restore['actor_critic_normalizers_optimizer_exact']
   states.append(torch.load(r/f'resume_{a.update+6:06d}.pt',weights_only=False,map_location='cpu'))
  for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict'):equal(states[0][key],states[1][key])
  (out/'qualification.json').write_text(json.dumps(dict(status='PASS',stage=a.stage,resume_update=a.update,global_envs=8192,actor_critic_normalizers_optimizer_equal=True,qualification_weights_discarded=True),indent=2)+'\n')
  del states,cp
  run('formal',[*common,'--iterations',a.iterations,'--eval-interval',250 if a.stage==1 else 128,'--eval-samples',128,'--output',out/'formal'])
  end=a.update+a.iterations
  for rank in (0,1):
   s=json.loads((out/f'formal/rank{rank}/status.json').read_text());assert s['status']=='ALLOCATION_COMPLETE' and s['completed_updates']==end,s
  new=out/f'formal/rank0/resume_{end:06d}.pt'
  run('recovery',['scripts/evaluate3762_recovery.py','--checkpoint',new,'--output',out/'recovery','--device','cuda:0','--stage',5,'--samples',128])
  run('head',['scripts/evaluate_v2_head.py','--task',TASK,'--checkpoint',new,'--output',out/'head','--device','cuda:0','--repeats',8])
  report=json.loads((out/'recovery/report.json').read_text())
  native_gate=all(g['wins']/g['trials']>=.7 for c in report['cases'] for g in c['groups'].values())
  status('PHASE_COMPLETE_REQUIRES_REVIEW',completed_updates=end,recovery_original_gate_pass=native_gate,checkpoint=str(new),checkpoint_sha256=hashlib.sha256(new.read_bytes()).hexdigest())
 except BaseException as exc:
  status('FAILED',error=repr(exc));raise
if __name__=='__main__':main()
