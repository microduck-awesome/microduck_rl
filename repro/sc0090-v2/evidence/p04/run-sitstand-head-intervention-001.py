import os,json,time,sys,hashlib,importlib.util,traceback,gc
from pathlib import Path
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04')
W=R/'sitstand-history-hold320-worktree'
MARK=R/'sitstand-head-intervention-status-001.json'
plan=json.loads((R/'sitstand-head-intervention-plan-001.json').read_text())
probe=R/'probe-sitstand-head-intervention-001.py'
assert hashlib.sha256(probe.read_bytes()).hexdigest()==plan['probe_sha256']
start=time.time();completed=0;reports=[]
def write(**kw):
    out=dict(pid=os.getpid(),updated_unix=time.time(),diagnostic_cap=153600,completed_diagnostic_controls=completed,new_formal_controls=0,**kw)
    tmp=MARK.with_suffix('.tmp');tmp.write_text(json.dumps(out,indent=2)+'\n');tmp.replace(MARK)
write(status='IMPORTING')
try:
    spec=importlib.util.spec_from_file_location('frozen_sitstand_head_probe',probe)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    import torch
    for arm in plan['arms']:
        for stage in plan['conditions']:
            out=R/f'sitstand-head-intervention-{arm}-{stage}-001'
            assert not out.exists()
            shared_before=json.loads((R/'groundpick-health-history-allocation-001/status.json').read_text())
            write(status='RUNNING_DIAGNOSTIC',arm=arm,stage=stage,output=str(out),ground_update_before=shared_before['completed_updates'])
            sys.argv=[str(probe),'--checkpoint',plan['checkpoint'],'--output',str(out),'--head-control',arm,'--stage',stage,'--num-envs','64','--seed','2603','--device','cuda:0']
            mod.main()
            count=json.loads((out/'diagnostic-count.json').read_text());assert count['completed_controls']==38400
            completed+=count['completed_controls'];reports.append(str(out/'report.json'))
            gc.collect();torch.cuda.empty_cache()
    assert completed==153600
    write(status='FOUR_FIXED_DIAGNOSTICS_COMPLETE_REVIEW_REQUIRED',reports=reports,wall_seconds=time.time()-start)
except BaseException:
    write(status='STOPPED_REQUIRES_REVIEW',reports=reports,exception=traceback.format_exc(),wall_seconds=time.time()-start)
    raise
