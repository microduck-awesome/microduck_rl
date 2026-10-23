"""Bounded continuation of the already-qualified, unchanged stage-4 recipe."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TASK='Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline'

def main():
 p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--update',type=int,required=True);p.add_argument('--iterations',type=int,default=512);p.add_argument('--qualification-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);parent=a.parent.resolve();q=a.qualification_root.resolve()
 def status(state,**extra):
  tmp=out/'status.tmp';tmp.write_text(json.dumps(dict(status=state,stage=4,parent=str(parent),parent_update=a.update,updated_unix=time.time(),**extra),indent=2)+'\n');tmp.replace(out/'status.json')
 def run(name,args):
  status('RUNNING',step=name)
  with (out/f'{name}.log').open('w') as log:subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
 try:
  qual=json.loads((q/'qualification.json').read_text());assert qual['status']=='PASS' and qual['stage']==4 and qual['global_envs']==8192
  for rank in (0,1):
   d=json.loads((parent/f'rank{rank}/status.json').read_text());assert d['status']=='ALLOCATION_COMPLETE' and d['completed_updates']==a.update
   manifest=json.loads((parent/f'rank{rank}/run.json').read_text());assert manifest['head_precision_recipe']['stage']==4
   for name,h in manifest['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
  (out/'qualification.json').write_text(json.dumps({**qual,'reused_from':str(q),'all_parent_training_source_hashes_unchanged':True},indent=2)+'\n')
  run('formal',['-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train3762_ddp.py','--task',TASK,'--device','cuda:0','--num-envs',4096,'--preflight',q/'batch4096.json','--resume-root',parent,'--resume-update',a.update,'--head-precision','--head-precision-stage',4,'--iterations',a.iterations,'--eval-interval',128,'--eval-samples',128,'--output',out/'formal'])
  end=a.update+a.iterations
  for rank in (0,1):
   d=json.loads((out/f'formal/rank{rank}/status.json').read_text());assert d['status']=='ALLOCATION_COMPLETE' and d['completed_updates']==end
  cp=out/f'formal/rank0/resume_{end:06d}.pt'
  run('recovery',['scripts/evaluate3762_recovery.py','--checkpoint',cp,'--output',out/'recovery','--device','cuda:0','--stage',5,'--samples',128])
  run('head',['scripts/evaluate_v2_head.py','--task',TASK,'--checkpoint',cp,'--output',out/'head','--device','cuda:0','--repeats',8])
  status('PHASE_COMPLETE_REQUIRES_REVIEW',completed_updates=end,checkpoint=str(cp))
 except BaseException as e:status('FAILED',error=repr(e));raise
if __name__=='__main__':main()
