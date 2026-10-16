import json,os,runpy,sys,time
from pathlib import Path
root=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
runpy.run_path('tests/test_v2_sitstand_lesson_cfg.py',run_name='__main__')
while not (root/'sitstand-clear-0256-measured/report.json').exists():
 if not Path('/proc/3826567').exists():raise RuntimeError('Prior measured evaluation ended without report')
 time.sleep(2)
old=json.loads((root/'sitstand-clear-0256-measured/report.json').read_text())
assert old['numerical_source_verified'] and old['actor_normalizer_unchanged']
sys.argv=['scripts/train_v2_sitstand.py','--task','Mjlab-SitStand-Flat-MicroDuck-V2-ClearSeat-Lesson4s-Repaired-S0-O0-R1','--output',str(root/'sitstand-lesson4-smoke-001'),'--device','cuda:0','--num-envs','64','--iterations','5','--validation-only','--force-timeouts','--resume',str(root/'sitstand-clear-allocation-001/checkpoint_000256.pt'),'--source-transition',str(root/'sitstand-short-lesson-worktree/source-transition.json')]
runpy.run_path('scripts/train_v2_sitstand.py',run_name='__main__')
