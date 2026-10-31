"""Print one reproducible phase; execute only when --execute is supplied."""
import argparse,json,os,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TASK='Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline'

def commands(phase,base,phases):
    name=phase['name'];out=base/name;pre=out/'batch4096.json'
    py=['bash','repro/python.sh'];head=[]
    if phase['head_stage'] is not None:head=['--head-precision','--head-precision-stage',str(phase['head_stage'])]
    parent=[]
    if phase['start']:
        previous=phases[phases.index(phase)-1]
        parent=['--resume-root',str(base/previous['name']/'formal'),'--resume-update',str(phase['start'])]
    result=[]
    result.append(('environment',[*py,'repro/check.py','--environment','--gpu','--output',str(out/'environment.json')]))
    for count,steps,label in [(64,5,'smoke64'),(4096,12,'batch4096')]:
        result.append((label,[*py,'scripts/smoke_v2_ppo.py','--task',TASK,'--device','cuda:0','--num-envs',str(count),'--iterations',str(steps),'--output-dir',str(out/label),'--report',str(out/(label+'.json')),*head]))
    common=[*py,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','scripts/train3762_ddp.py','--task',TASK,'--device','cuda:0','--num-envs','4096','--preflight',str(pre),*parent,*head]
    result.append(('boundary',[*common,'--iterations','6','--eval-interval','2','--eval-samples','2','--validation-only','--output',str(out/'boundary')]))
    result.append(('verify-boundary',[*py,'repro/verify_ranks.py','--run',str(out/'boundary'),'--update',str(phase['start']+6),'--output',str(out/'boundary-verification.json'),*(['--resume-check'] if parent else [])]))
    result.append(('formal',[*common,'--iterations',str(phase['updates']),'--eval-interval',str(phase['eval_interval']),'--eval-samples','128','--output',str(out/'formal')]))
    result.append(('verify-formal',[*py,'repro/verify_ranks.py','--run',str(out/'formal'),'--update',str(phase['end']),'--output',str(out/'rank-verification.json'),*(['--resume-check'] if parent else [])]))
    if phase['head_stage'] is not None or phase['end']==3762:
        cp=out/f"formal/rank0/resume_{phase['end']:06d}.pt"
        result.append(('recovery',[*py,'scripts/evaluate3762_recovery.py','--checkpoint',str(cp),'--output',str(out/'recovery'),'--device','cuda:0','--stage','5','--samples','128']))
        result.append(('head',[*py,'scripts/evaluate_v2_head.py','--task',TASK,'--checkpoint',str(cp),'--output',str(out/'head'),'--device','cuda:0','--repeats','8']))
    if phase['end']==6804:
        result.append(('onnx',[*py,'repro/verify_ranks.py','--run',str(out/'formal'),'--update','6804','--onnx','--resume-check','--output',str(out/'artifact-verification.json')]))
    return result

def main():
    phases=json.loads((ROOT/'repro/phases.json').read_text())
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=[s['name'] for s in phases],required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--execute',action='store_true',help='Actually run smoke/qualification/training/evaluation for this phase')
    a=p.parse_args();base=a.output_root.resolve();phase=next(s for s in phases if s['name']==a.phase)
    cmds=commands(phase,base,phases)
    print(f"# {phase['name']}: {phase['start']} -> {phase['end']}; global environments=8192",flush=True)
    for label,cmd in cmds:print(f'# {label}\n'+shlex.join(cmd),flush=True)
    if not a.execute:return
    out=base/phase['name']
    if out.exists():raise SystemExit('Output exists; choose a fresh root or preserve/rename the interrupted phase before retrying')
    if phase['start']:
        previous=phases[phases.index(phase)-1];parent=base/previous['name']/'formal'
        for rank in (0,1):
            status=json.loads((parent/f'rank{rank}/status.json').read_text())
            assert status['status']=='ALLOCATION_COMPLETE' and status['completed_updates']==phase['start']
            assert (parent/f"rank{rank}/resume_{phase['start']:06d}.pt").is_file()
    out.mkdir(parents=True)
    (out/'commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
    env=dict(os.environ);env.setdefault('CUDA_VISIBLE_DEVICES','0,1')
    for label,cmd in cmds:
        with (out/(label+'.log')).open('w') as log:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Phase complete. Review recovery/head reports before executing the next phase.',flush=True)
if __name__=='__main__': main()
