from pathlib import Path
import json,torch,math
r=Path(__file__).resolve().parent;d=torch.load(r/'roulade-endpoint-0256-nominal/traces.pt',weights_only=False,map_location='cpu')['measurements'];idx=torch.arange(d['height'].shape[1]);rows=[]
for group in range(5):
 sel=idx%8<4 if group==0 else idx%8==group+3;tail=slice(-25,None)
 fields={'height':((d['height']-.112).abs()<.012),'tilt':(d['tilt']<math.radians(15)),'linear':(d['linear'].norm(dim=-1)<.05),'omega':(d['omega'].norm(dim=-1)<.5),'bothfeet':d['feet'].all(-1),'no_head':~d['head_contact'],'no_self':~d['self_contact'],'legs_home':(d['physical_q'][:,:,[0,1,2,3,4,9,10,11,12,13]].abs().mean(-1)<.2),'head_home':(d['physical_q'][:,:,5:9].abs().amax(-1)<.1)}
 near=(d['height']>.09)&(d['tilt']<math.radians(30))&d['feet'].all(-1)&(~d['head_contact'])&(~d['self_contact'])
 rows.append(dict(group='standing' if group==0 else (90,180,270,330)[group-1],n=int(sel.sum()),tail_gate_pass={k:int(v[tail,sel].all(0).sum()) for k,v in fields.items()},tail_height_mean=float(d['height'][tail,sel].mean()),tail_tilt_mean_deg=float(torch.rad2deg(d['tilt'][tail,sel]).mean()),tail_omega_mean=float(d['omega'][tail,sel].norm(dim=-1).mean()),tail_q_mean=d['physical_q'][tail,sel].mean((0,1)).tolist(),near_upright_after02_any=int(near[10:,sel].any(0).sum()),head_latch_any=int(d['head_latch'][:,sel].any(0).sum()),frontier330_any=int((d['frontier'][:,sel]>math.radians(330)).any(0).sum())))
(r/'roulade-endpoint-0256-nominal-behavior-audit.json').write_text(json.dumps(dict(status='SAVED_TRACE_AUDIT',rows=rows,scope='Post02 near-upright is a loose diagnostic, not success; final hold gates remain original'),indent=2)+'\n');print(json.dumps(rows,indent=2))
