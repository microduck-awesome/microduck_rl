"""Fixed camera pair: one64 allocation and original two-stage evaluation per GPU."""
from pathlib import Path
import argparse,gc,hashlib,json,os,runpy,sys,time

p=argparse.ArgumentParser();p.add_argument('arm',choices=('mild','control'));args=p.parse_args()
r=Path(__file__).resolve().parent;w=r/'velstand-camera-worktree'
arm=args.arm;device='cuda:0' if arm=='mild' else 'cuda:1';weight=-.01 if arm=='mild' else 0.
status=r/f'velstand-camera-{arm}-pipeline-status.json';assert not status.exists()
out=r/f'velstand-camera-{arm}-allocation-001';start=time.monotonic()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(**kw):
 tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(dict(updated_unix=time.time(),pid=os.getpid(),arm=arm,device=device,stage_maximum_controls=75497472,automatic_further_allocation=False,**kw),indent=2)+'\n');tmp.replace(status)
try:
 if arm=='control':assert read(r/'ballkick-finish-pipeline-status.json')['status']=='BALLKICK_STAGE_COMPLETE_NO_FURTHER_TRAINING'
 state(status='IMPORTING',formal_controls=0)
 sys.path.insert(0,str(w/'scripts'))
 ns=runpy.run_path(str(w/'scripts/train_v2_velstand.py'),run_name='camera_trainer')
 import torch
 torch.cuda.set_device(device)
 q=read(r/'velstand-camera-qualification-001.json');assert ns['source_hashes'](w)==q['source_sha256']
 for name,digest in q['assets_sha256'].items():assert sha(w/name)==digest,name
 protocol=read(r/'velstand-camera-eval-protocol.json')
 assert sha(r/'evaluate_v2_velstand_commands.py')==protocol['commands_sha256']
 assert sha(r/'evaluate_v2_velstand_survey_device.py')==protocol['survey_device_sha256']
 state(status='FORMAL_TRAINING',formal_controls=0,end_update=320)
 sys.argv=[str(w/'scripts/train_v2_velstand.py'),'--output',str(out),'--device',device,'--num-envs','24576','--iterations','64','--resume',str(r/'velstand-v8-allocation-002/checkpoint_000256.pt'),'--source-transition',str(r/'velstand-camera-source-transition-002.json'),'--native-range-budget','16','--camera-weight',str(weight),'--standing-teacher',str(r/'head2-micro-exposure-allocation-001/checkpoint_001856.pt'),'--walking-teacher',str(r/'walking-yaw-fine-allocation-002/checkpoint_000512.pt'),'--qualification',str(r/'velstand-camera-qualification-001.json')]
 ns['main']()
 report=read(out/'report.json');assert report['status']=='ALLOCATION_COMPLETE' and report['completed_updates']==320 and report['allocation_controls']==37748736
 wall=time.monotonic()-start
 (out/'outer-process-wall.json').write_text(json.dumps(dict(wall_seconds=wall,controls=37748736,controls_per_second=37748736/wall,scope='Full worker import/qualification/train/save/export through completed main, before evaluation'),indent=2)+'\n')
 gc.collect();torch.cuda.empty_cache()
 checkpoint=out/'checkpoint_000320.pt';evaluations=[]
 for stage in ('nominal','measured'):
  for kind,script in (('commands','evaluate_v2_velstand_commands.py'),('survey','evaluate_v2_velstand_survey_device.py')):
   dest=r/f'velstand-camera-{arm}-0320-{kind}-{stage}'
   state(status='EVALUATING',phase=dest.name,formal_controls=37748736)
   sys.argv=[str(r/script),'--checkpoint',str(checkpoint),'--output',str(dest),'--device',device,'--stage',stage]
   phase_start=time.monotonic();runpy.run_path(str(r/script),run_name='__main__')
   result=read(dest/'report.json');assert result['status'] in ('EVALUATION_COMPLETE','STUDENT_SURVEY_COMPLETE')
   (dest/'outer-phase-wall.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-phase_start,scope='Evaluation in worker with reused imports; no training'),indent=2)+'\n')
   evaluations.append(str(dest/'report.json'));gc.collect();torch.cuda.empty_cache()
 state(status='ARM_BUDGET_COMPLETE_EVALUATIONS_REVIEW_READY',formal_controls=37748736,evaluations=evaluations)
except BaseException as exc:
 state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc),formal_controls=read(out/'status.json')['allocation_controls'] if (out/'status.json').exists() else 0)
 raise
