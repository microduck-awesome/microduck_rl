"""One interpreter for CPU admission then mature64x5, avoiding duplicate imports."""
import os,sys,runpy
from pathlib import Path
r=Path(__file__).resolve().parent;w=r/'ballkick-finish-worktree'
os.chdir(w);runpy.run_path(str(w/'tests/validate_v2_ballkick_finish_cpu.py'),run_name='__main__')
sys.argv=['scripts/train_v2_ballkick.py','--output',str(r/'ballkick-finish-smoke-001'),'--device','cuda:1','--num-envs','64','--iterations','5','--resume',str(r/'ballkick-right-allocation-001/checkpoint_000085.pt'),'--source-transition',str(r/'ballkick-finish-source-transition.json'),'--validation-only','--force-timeouts']
runpy.run_path(str(w/'scripts/train_v2_ballkick.py'),run_name='__main__')
