"""Compare walking reports with model10250 under the historical fixed protocol."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=json.loads(a.report.read_text())
    ref=json.loads((ROOT/'docs/evidence/official-walk/corrected/best-search/selected/selection.json').read_text())['selection_result']
    assert report['seed']==2026102603 and report['seconds_per_trial']==20 and report['replicas']==12
    assert report['voltages']==[10.2,11.1,12.4] and report['protocol_version']=='private-rng-v2'
    assert report['ppo_updates']==0
    if 'robot' in report:
        assert report['robot']=='V2 SC0090'
    else:
        # The exact archived 139-checkpoint report predates the robot label.
        archived=ROOT/'docs/evidence/official-walk/corrected/best-search/all-20s-voltage/results.json'
        assert hashlib.sha256(a.report.read_bytes()).digest()==hashlib.sha256(archived.read_bytes()).digest()
    baseline={c['case']:c for c in ref['cases']};results=[]
    for r in report['results']:
        assert r['total_trials']==156 and {c['case'] for c in r['cases']}==set(baseline)
        entry={k:r[k] for k in ('iteration','sha256','stable_trials','total_trials','near_limit_trials','selection_score_lower_is_better','balanced_tracking_error','responsive_motion_fraction')}
        entry['reference']={k:ref[k] for k in ('stable_trials','total_trials','near_limit_trials','selection_score_lower_is_better','balanced_tracking_error','responsive_motion_fraction')}
        entry['score_delta']=r['selection_score_lower_is_better']-ref['selection_score_lower_is_better']
        entry['cases']=[]
        for c in r['cases']:
            b=baseline[c['case']];assert c['command']==b['command']
            entry['cases'].append({'case':c['case'],'command':c['command'],'velocity':[c[k] for k in ('mean_vx','mean_vy','mean_yaw')],'reference_velocity':[b[k] for k in ('mean_vx','mean_vy','mean_yaw')],'failed_trials':c['failed_trials'],'near_limit_occupancy_max':c['near_limit_occupancy_max']})
        results.append(entry)
    result={'scope':'Historical paired development protocol. Report behavior deltas, not weight equality; no new equivalence tolerance is invented.','results':results}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__': main()
