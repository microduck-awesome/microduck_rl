import os,json,sys,time,hashlib,importlib.util,traceback,gc
from pathlib import Path
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'groundpick-health-history-worktree'
sys.path.insert(0,str(W/'scripts'));plan=json.loads((R/'groundpick-local-endpoint-plan-001.json').read_text())
state=R/'groundpick-local-endpoint-status-001.json';assert not state.exists();started=time.time();completed=0;reports=[]
def write(**kw):
    p=state.with_suffix('.tmp');p.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),completed_diagnostic_controls=completed,diagnostic_cap=153600,formal_controls=0,**kw),indent=2)+'\n');p.replace(state)
write(status='IMPORTING')
try:
    probe=Path(plan['probe']);assert hashlib.sha256(probe.read_bytes()).hexdigest()==plan['probe_sha256']
    assert hashlib.sha256(Path(plan['parent']).read_bytes()).hexdigest()==plan['parent_sha256']
    spec=importlib.util.spec_from_file_location('groundpick_local_endpoint_probe',probe);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    import torch
    assert not Path('/proc/444216').exists()
    assert json.loads((R/'groundpick-health-history-pipeline-status.json').read_text())['status']=='GROUNDPICK_HEALTH_HISTORY_COMPLETE_NO_AUTOMATIC_CONTINUATION'
    for arm in plan['arms']:
        for stage in plan['conditions']:
            out=R/f'groundpick-local-endpoint-{arm}-{stage}-001';assert not out.exists()
            write(status='RUNNING_DIAGNOSTIC',arm=arm,stage=stage,output=str(out))
            sys.argv=[str(probe),'--checkpoint',plan['parent'],'--output',str(out),'--arm',arm,'--stage',stage,'--seed','2607','--num-envs','64','--device','cuda:0']
            mod.main();count=json.loads((out/'diagnostic-count.json').read_text());assert count['completed_controls']==38400
            completed+=38400;reports.append(str(out/'report.json'));gc.collect();torch.cuda.empty_cache()
    assert completed==153600
    write(status='FOUR_LOCAL_ENDPOINT_DIAGNOSTICS_COMPLETE',reports=reports,wall_seconds=time.time()-started)
except BaseException:
    write(status='STOPPED_REQUIRES_REVIEW',reports=reports,exception=traceback.format_exc(),wall_seconds=time.time()-started);raise
