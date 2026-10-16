import gc,json,runpy,sys,torch
from pathlib import Path
root=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
sys.path.insert(0,str(Path('scripts').resolve()))
sys.argv=['scripts/train_v2_ground_pick.py','--task','Mjlab-GroundPick-Flat-MicroDuck-V2-Repaired-S0-O0-R1','--output',str(root/'groundpick-smoke-002'),'--device','cuda:0','--num-envs','64','--iterations','5','--validation-only','--force-timeouts']
runpy.run_path(sys.argv[0],run_name='__main__')
gc.collect();torch.cuda.empty_cache()
for stage in ('nominal','measured'):
 sys.argv=['scripts/evaluate_v2_ground_pick.py','--checkpoint',str(root/'groundpick-smoke-002/checkpoint_000005.pt'),'--output',str(root/('groundpick-evaluator-smoke-002-'+stage)),'--device','cuda:0','--stage',stage,'--num-envs','32']
 runpy.run_path(sys.argv[0],run_name='__main__')
 gc.collect();torch.cuda.empty_cache()
 reports=[root/('groundpick-evaluator-smoke-002-'+stage)/'report.json' for stage in ('nominal','measured')]
for p in reports:
 r=json.loads(p.read_text());assert r['actor_normalizer_unchanged'] and max(r['oracle_max_abs'])<2e-6 and r['camera_protocol']['samples']==2401
(root/'groundpick-evaluator-smoke-qualification-002.json').write_text(json.dumps({'status':'PASS','reports':[str(p) for p in reports],'scope':'Software/oracle/source/frozen actor/camera validation, random5-update behavior not skill acceptance'},indent=2)+'\n')
