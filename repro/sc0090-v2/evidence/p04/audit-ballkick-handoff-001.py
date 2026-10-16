from pathlib import Path
import torch,json,math,hashlib
R=Path(__file__).resolve().parent;torch.set_num_threads(2);torch.cuda.set_device('cuda:0')
results={};witnesses=[]
for stage in ('nominal','measured'):
 old=torch.load(R/f'ballkick-finish-0170-{stage}/traces.pt',weights_only=False,map_location='cuda:0')
 new=torch.load(R/f'ballkick-standing-handoff-{stage}-001/traces.pt',weights_only=False,map_location='cuda:0')
 initial={k:dict(exact=torch.equal(v,new['initial'][k]),max_abs=float((v-new['initial'][k]).abs().max())) for k,v in old['initial'].items()}
 prefix={}
 for k,v in old['measurements'].items():
  other=new['measurements'][k]
  prefix[k]=dict(exact=torch.equal(v[:100],other[:100]),max_abs=float((v[:100].float()-other[:100].float()).abs().max()))
 arms={}
 for label,t in [('original',old),('handoff',new)]:
  x=t['measurements'];flags=dict(head=x['head_contact'],self_contact=x['self_contact'],low=x['height']<.08,tilt=x['tilt']>math.pi/4,bad_ball=x['bad_contact'],feedback=~x['feedback_ready'],nonfinite=~x['finite'])
  guards={}
  for name,mask in flags.items():
   first=torch.where(mask,torch.arange(250,device=mask.device)[:,None],250).amin(0)
   guards[name]=dict(any_trials=int(mask.any(0).sum()),before2s_trials=int(mask[:100].any(0).sum()),after2s_trials=int(mask[100:].any(0).sum()),sample_fraction=float(mask.float().mean()),first_seconds_min=float(first.min()*.02),first_seconds_median=float(first.float().median()*.02))
   if label=='handoff' and name in ('head','self_contact','low','tilt') and mask.any():
    ids=mask.any(0).nonzero().flatten()[:8]
    for env in ids.tolist():
     step=int(first[env]);witnesses.append(dict(stage=stage,cause=name,world=env,step=step,qpos=x['raw_qpos'][step,env].cpu(),qvel=x['raw_qvel'][step,env].cpu(),height=float(x['height'][step,env]),tilt=float(x['tilt'][step,env]),origin=new['initial']['origins'][env].cpu()))
  arms[label]=dict(guards=guards,body_tail_hold=int(x['standing'][-25:].all(0).sum()),rolling_tail_hold=int(x['rolling_target'][-50:].all(0).sum()))
 results[stage]=dict(initial=initial,pre_switch_measurements=prefix,arms=arms)
torch.save(witnesses,R/'ballkick-handoff-violation-witnesses-001.pt')
report=dict(status='HANDOFF_GUARDS_AUDITED',results=results,diagnostic_controls=128000,formal_controls=0,scope='Offline GPU reductions of original170 and teacher diagnostic; prefixes explicitly compared. Whole-episode clean guard remains unchanged; witnesses are actual sampled qpos, not fabricated poses.')
(R/'ballkick-standing-handoff-review-001.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:dict(initial_exact=all(x['exact']for x in v['initial'].values()),prefix_exact=all(x['exact']for x in v['pre_switch_measurements'].values()),arms=v['arms'])for k,v in results.items()},indent=2))
