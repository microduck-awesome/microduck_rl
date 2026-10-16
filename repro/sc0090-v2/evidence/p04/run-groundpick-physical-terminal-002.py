import gc,runpy,sys,torch
sys.argv=['scripts/validate_v2_ground_pick_reach.py','--output','runs-v2/groundpick-reach-hold-002']
runpy.run_path(sys.argv[0],run_name='__main__')
gc.collect();torch.cuda.empty_cache()
sys.argv=['tests/validate_v2_ground_pick_terminal.py','--output','runs-v2/groundpick-terminal-validation-001.json']
runpy.run_path(sys.argv[0],run_name='__main__')
