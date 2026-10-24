"""Run only reviewed, isolated qualifications. This tool cannot launch formal training."""
from pathlib import Path
import argparse, fcntl, hashlib, json, os, subprocess, time
ROOT = Path(__file__).resolve().parents[1]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, required=True, choices=range(10))
    parser.add_argument('--kind', required=True, choices=('smoke','batch'))
    args = parser.parse_args()
    review = json.loads((ROOT/'document-review.json').read_text())
    assert review['status'] == 'PASS_FOR_QUALIFICATION_ONLY'
    for name, sha in review['locked_files'].items():
        assert digest(ROOT/name) == sha, ('Changed reviewed file', name)
    plan = json.loads((ROOT/'execution-plan.json').read_text())
    phase = plan['phases'][args.phase]
    dest = Path(plan['qualification_root'])/phase['name']/('smoke64' if args.kind=='smoke' else 'batch')
    assert not dest.exists(), ('Preserve existing attempt; do not overwrite', dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock = open(Path(plan['qualification_root'])/'process.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    contract = plan['environment_contract']
    for name in contract['unset']: env.pop(name,None)
    for name in ('SC0090_CUDA_COMPAT_DIR','SC0090_V2_PYTHON'): env[name]=contract[name]
    # Recheck current sources, resources, dependencies, and effective numerical config.
    check = dest.parent/(args.kind+'-config.json')
    with (dest.parent/(args.kind+'-config.log')).open('w') as log:
        subprocess.run(['bash',phase['wrapper'],str(ROOT/'tools/check_phase_config.py'),
                        '--phase',phase['name'],'--output',str(check)],
                       cwd=phase['source'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert json.loads(check.read_text())['status']=='PASS'
    if args.kind=='batch':
        smoke = json.loads((dest.parent/'smoke64/report.json').read_text())
        assert smoke['status']=='PASS' and smoke['envs']==64 and smoke['iterations']==5
    argv = ['bash',phase['wrapper'],*phase[args.kind+'_argv']]
    receipt = dict(status='RUNNING',pid=os.getpid(),argv=argv,cwd=phase['source'],started_unix=time.time(),
                   document_review_sha256=digest(ROOT/'document-review.json'),formal_weights_used=False)
    record = dest.parent/(args.kind+'-process.json')
    record.write_text(json.dumps(receipt,indent=2)+'\n')
    with (dest.parent/(args.kind+'.log')).open('w') as log:
        result = subprocess.run(argv,cwd=phase['source'],env=env,stdout=log,stderr=subprocess.STDOUT)
    receipt.update(status='EXITED',returncode=result.returncode,ended_unix=time.time())
    record.write_text(json.dumps(receipt,indent=2)+'\n')
    assert result.returncode==0, f"Qualification failed; inspect {dest.parent/(args.kind+'.log')}"
    report=json.loads((dest/'report.json').read_text())
    assert report['status']=='PASS' and report['device']=='cuda:1'
    assert report['robot_xml_sha256']=='8ae0d89041950d565bbcd6272ab87317f46132ff1eef99b922782ecbe2b95308'
    assert report['envs']==(64 if args.kind=='smoke' else 8192)
    assert report['iterations']==(5 if args.kind=='smoke' else phase['qualification_iterations'])
    receipt.update(status='PASS',report_sha256=digest(dest/'report.json'))
    if args.kind=='batch' and args.phase>0:
        throughput=report['throughput']
        receipt['formal_throughput_gate_passed']=(throughput['measured_updates']>=10 and throughput['median_end_to_end_steps_per_s']>100000)
        if not receipt['formal_throughput_gate_passed']:receipt['status']='FORMAL_GATE_FAILED'
    record.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    assert receipt['status']=='PASS'
if __name__=='__main__':main()
