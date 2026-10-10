"""Serial matched training jobs with durable progress; smoke is not formal."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['benchmark','core','r1'],required=True)
    parser.add_argument('--label',required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,7,1])
    parser.add_argument('--device',choices=['cpu','cuda:0'],default='cuda:0')
    args=parser.parse_args()
    ws=Path(__file__).resolve().parents[2]
    study=ws/'sc0090-datasets/v5-testbench-sim2real-20261011'
    root=study/'runs'/f'{args.stage}-{args.label}'
    root.mkdir(exist_ok=False)
    if args.stage=='benchmark':jobs=[{'policy':'P03','seed':42,'num_envs':n,'iterations':5} for n in [256,512,1024]]
    else:
        budget=json.loads((study/'reports/frozen-training-budget.json').read_text())
        groups=['P00','P01','P02','P03'] if args.stage=='core' else ['P04','P05','P06','P07']
        jobs=[{'policy':p,'seed':s,'num_envs':budget['num_envs'],'iterations':budget['iterations']}
              for s in args.seeds for p in groups]
    state={'stage':args.stage,'device':args.device,'status':'running','jobs':[{**j,'status':'pending'} for j in jobs]}
    def write():
        state['updated_at']=datetime.now(timezone.utc).isoformat()
        (root/'queue.json').write_text(json.dumps(state,indent=2)+'\n')
    write()
    for i,job in enumerate(jobs):
        name=f"{job['policy']}-seed{job['seed']}-n{job['num_envs']}"
        output=root/name
        stage='benchmark' if args.stage=='benchmark' else 'train'
        cmd=[sys.executable,str(Path(__file__).with_name('sc0090_train.py')),'--stage',stage,
             '--policy',job['policy'],'--seed',str(job['seed']),'--num-envs',str(job['num_envs']),
             '--iterations',str(job['iterations']),'--device',args.device,'--output',str(output)]
        state['jobs'][i]['status']='running';state['jobs'][i]['output']=str(output.relative_to(study));write()
        print(json.dumps({'started':name,'stage':stage,'output':str(output)}),flush=True)
        with (root/(name+'.log')).open('w') as log:
            result=subprocess.run(cmd,cwd=ws/'microduck_rl',stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            state['jobs'][i]['status']='failed';state['status']='failed';state['exit_code']=result.returncode;write()
            raise SystemExit(f'Job failed: {name}; see retained log')
        report=json.loads((output/'report.json').read_text())
        if not report['passed']:raise SystemExit(f'Unqualified run report: {name}')
        state['jobs'][i]['status']='passed';write()
        if args.stage in ['core','r1']:
            index=study/'policy-index.csv'
            with index.open() as f:
                reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
            manifest=json.loads((output/'manifest.json').read_text())
            for row in rows:
                if row['group']==job['policy'] and int(row['seed'])==job['seed']:
                    row.update(status='trained_pending_runtime_qualification',
                        checkpoint=str((output/report['checkpoint']).relative_to(study)),
                        onnx=str((output/'policy.onnx').relative_to(study)),
                        onnx_sha256=manifest['onnx_sha256'],interaction_steps=manifest['train_interactions'])
            with index.open('w') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        print(json.dumps({'completed':name,'report':report}),flush=True)
    state['status']='passed';write()


if __name__=='__main__':main()
