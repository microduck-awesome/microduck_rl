"""Partition the complete checkpoint set into bounded-memory evaluation batches."""
import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parents[1];base=root/'docs/evidence/official-walk/corrected/best-search/all-20s-voltage';base.mkdir(parents=True,exist_ok=True)
run=next((root/'logs/rsl_rl/microduck_v2_official_baseline').glob('*dr_fresh'));files=sorted(run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]));iterations=[int(p.stem.split('_')[-1]) for p in files]
results=[];reports=[]
for offset in range(0,len(iterations),35):
 subset=iterations[offset:offset+35];out=base/f'batch-{offset//35:02d}'
 if not (out/'results.json').exists():
  out.mkdir(exist_ok=True)
  command=['bash','scripts/v2_python.sh','scripts/search_best_walk_checkpoint.py','--run',str(run),'--output',str(out),'--iterations',*map(str,subset),'--seconds','20','--replicas','12','--voltages','10.2','11.1','12.4','--seed','2026102603']
  with (out/'process.log').open('w') as log:subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
 d=json.loads((out/'results.json').read_text());assert d['checkpoints_evaluated']==len(subset);results.extend(d['results']);reports.append(d)
 print(json.dumps({'completed_checkpoints':len(results),'total_checkpoints':len(iterations),'batch':offset//35,'batch_best':d['results'][0]['iteration']}),flush=True)
assert len(set(d['paired_nuisance_sha256'] for d in reports))==1,'Paired nuisance draws differ across batches'
assert sorted(r['iteration'] for r in results)==iterations
results.sort(key=lambda x:(x['total_trials']-x['stable_trials'],x['near_limit_trials'],x['selection_score_lower_is_better']))
report={k:v for k,v in reports[0].items() if k not in ('results','checkpoints_evaluated','elapsed_s')};report.update(checkpoints_evaluated=len(results),all_existing_checkpoints_evaluated=True,results=results,elapsed_s=sum(d['elapsed_s'] for d in reports),batches=len(reports),batch_nuisance_match_verified=True)
(base/'results.json').write_text(json.dumps(report,indent=2)+'\n');(base/'ranking.csv').write_text('iteration,score,coverage,stable,total,near_limit\n'+''.join(f"{x['iteration']},{x['selection_score_lower_is_better']},{x['responsive_motion_fraction']},{x['stable_trials']},{x['total_trials']},{x['near_limit_trials']}\n" for x in results));print(json.dumps({'top':[{k:r[k] for k in ('iteration','selection_score_lower_is_better','responsive_motion_fraction','stable_trials','total_trials')} for r in results[:12]]},indent=2))
