"""Four fixed-clock frozen-policy batteries; zero formal controls."""
import os,sys,json,time,hashlib,gc,importlib.util,traceback
from pathlib import Path
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'sitstand-transition-release-worktree';os.chdir(W);sys.path.insert(0,str(W/'scripts'));state=R/'sitstand-fresh-m4-m16-eval-status.json';started=time.monotonic()
def write(v):
 v.update(pid=os.getpid(),updated_unix=time.time(),formal_controls_added=0);p=state.with_suffix('.tmp');p.write_text(json.dumps(v,indent=2)+'\n');p.replace(state)
write(dict(status='IMPORTING',diagnostic_cap=614400));print('WORKER_PID',os.getpid(),flush=True)
import torch
import train_v2_sitstand as train
spec=importlib.util.spec_from_file_location('paired_sitstand_eval',W/'scripts/evaluate_v2_sitstand_clock_fixed.py');e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
models={'M4':R/'sitstand-fresh-release-actual16384-001/checkpoint_000016.pt','M16':R/'sitstand-fresh-m16-actual16384-002/checkpoint_000016.pt'};results={};hashes=train.source_hashes(W)
try:
 for label,cp in models.items():
  meta=torch.load(cp,map_location='cpu',weights_only=False)['infos']['v2'];assert meta['source_sha256']==hashes and meta['completed_updates']==16 and meta['lineage']is None
  for stage in('nominal','measured'):
   out=R/('sitstand-fresh-'+label.lower()+'-0016-'+stage);write(dict(status='RUNNING',arm=label,stage=stage,output=str(out),diagnostic_cap=614400));print('EVAL',label,stage,flush=True)
   sys.argv=['eval','--checkpoint',str(cp),'--output',str(out),'--device','cuda:1','--stage',stage,'--num-envs','256','--seed','2603'];e.main();results[label+'_'+stage]=str(out/'report.json');gc.collect();torch.cuda.empty_cache()
 assert train.source_hashes(W)==hashes
 pairing={};summary={}
 for stage in('nominal','measured'):
  aa=torch.load(R/('sitstand-fresh-m4-0016-'+stage)/'traces.pt',map_location='cpu',weights_only=False);bb=torch.load(R/('sitstand-fresh-m16-0016-'+stage)/'traces.pt',map_location='cpu',weights_only=False)
  pairing[stage]={k:dict(exact=torch.equal(aa['initial'][k],bb['initial'][k]),max_abs=float((aa['initial'][k]-bb['initial'][k]).abs().max()))for k in aa['initial']}
  assert all(v['exact']for v in pairing[stage].values()),pairing
  for label in('M4','M16'):
   report=json.loads(Path(results[label+'_'+stage]).read_text());assert report['numerical_source_verified']and report['actor_normalizer_unchanged'];summary[label+'_'+stage]={k:[{a:v[a]for a in('tail_hold','faults','head_step_reached','head_step_tail_band_hold','tilt_over45','head_tail_mae_deg')}for v in vals]for k,vals in report['groups'].items()}
 review=dict(status='FOUR_FROZEN_EVALUATIONS_COMPLETE',diagnostic_controls=614400,formal_controls_added=0,total_sitstand_formal_controls_unchanged=226492416,reports=results,initial_pairing=pairing,summary=summary,source_unchanged=True,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Frozen16updated actors, sameactualP04/clock-corrected original256x12s battery. M differs only training optimization; inferenceactorconfig identical. No furtherPPO authorization.')
 train.write_json(R/'sitstand-fresh-m4-m16-eval-review-001.json',review);write(dict(status=review['status'],diagnostic_controls=614400,worker_wall_s=time.monotonic()-started,report=str(R/'sitstand-fresh-m4-m16-eval-review-001.json')))
except BaseException as ex:write(dict(status='FAILED_REQUIRES_DIAGNOSIS',exception=repr(ex)));traceback.print_exc();raise
