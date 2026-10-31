"""Compare identical-protocol recovery and head metrics with the stored reference."""
import argparse,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--recovery',type=Path,required=True);p.add_argument('--head',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    recovery=json.loads(a.recovery.read_text());head=json.loads(a.head.read_text())
    ref_head=json.loads((ROOT/'docs/evidence/rise3762-collision/final/head-new.json').read_text())
    ref=json.loads((ROOT/'docs/evidence/rise3762-collision/final/comparison-summary.json').read_text())['new']
    assert recovery['evaluation_stage']==5
    assert len(recovery['cases'])==2 and {c['seed'] for c in recovery['cases']}=={2026101501,2026101502}
    assert recovery['checkpoint_sha256']==head['checkpoint_sha256'],'Reports must use the same policy'
    assert head['action_mode']=='checkpoint_policy' and head['current_ranges']==ref_head['current_ranges'] and head['final_ranges']==ref_head['final_ranges']
    key=lambda r:(r['seed'],r['name'],r['range'],tuple(r['target']),r['trials'])
    assert sorted(map(key,head['records']))==sorted(map(key,ref_head['records'])),'Head protocol/case mismatch'
    groups={}
    for case in recovery['cases']:
        assert set(case['groups'])==set(ref['recovery'])
        for name,g in case['groups'].items():
            assert g['trials']==128
            total=groups.setdefault(name,dict(wins=0,trials=0))
            total['wins']+=g['wins'];total['trials']+=g['trials']
    result={'checkpoint_sha256':head['checkpoint_sha256'],'recovery':{},'head':{},'scope':'Same stored development protocols; deltas are measured minus reference. No new numerical equivalence tolerance is invented.'}
    for name,g in groups.items():result['recovery'][name]={**g,'reference':ref['recovery'][name],'wins_delta':g['wins']-ref['recovery'][name]['wins']}
    for name in ('zero','small','current','final'):
        rows=[r for r in head['records'] if r['range']==name];total=sum(r['trials'] for r in rows)
        means=[sum(r['step']['steady_mae_deg_mean'][i]*r['trials'] for r in rows)/total for i in range(4)]
        result['head'][name]={'mean_axis_mae_deg':means,'reference':ref['head'][name]['mean_axis_mae_deg'],'delta_deg':[x-y for x,y in zip(means,ref['head'][name]['mean_axis_mae_deg'])],'invalid_trials':sum(r['invalid_trials'] for r in rows)}
    result['recovery_reference_count_matched']=all(g['wins_delta']==0 for g in result['recovery'].values())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
