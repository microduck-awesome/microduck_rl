from pathlib import Path
import torch,json,hashlib,re,math
R=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261015/p04');W=R/'roulade-history-worktree';S=R/'roulade-history-formal-001/checkpoint_001536.pt';T=R/'head2-micro-exposure-allocation-001/checkpoint_001856.pt';torch.set_num_threads(2)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def norm(x):
 if isinstance(x,dict):return {k:norm(v)for k,v in x.items()}
 if isinstance(x,list):return [norm(v)for v in x]
 return re.sub(r' at 0x[0-9a-fA-F]+','',x)if isinstance(x,str)else x
sa,ta=[json.loads((p.parent/'run.json').read_text())for p in [S,T]];at,bt=[v['environment']['observations']['actor']['terms']for v in [sa,ta]];assert list(at)==list(bt)
assert all(norm(at[k])==norm(bt[k])for k in at if k not in ['head_command','body_command'])
assert at['head_command']['params']=={'dim':4} and at['body_command']['params']=={'dim':6}
actions=[norm(x['environment']['actions'])for x in [sa,ta]];archives=[Path(x['joint_pos']['archive_path'])for x in actions];man=[{str(p.relative_to(root)):sha(p)for p in root.rglob('*')if p.is_file()}for root in archives];assert man[0]and man[0]==man[1]
for x in actions:x['joint_pos']['archive_path']='<verified-identical-archive>'
assert actions[0]==actions[1]
core=[k for k in ta['source_sha256']if k.startswith(('src/mjlab_microduck/sc0090_v2/','src/mjlab_microduck/actuator/'))or k in ('src/mjlab_microduck/custom_v2/fast_native.py','src/mjlab_microduck/custom_v2/head_exploration.py')];assert all(sa['source_sha256'][k]==ta['source_sha256'][k]==sha(W/k)for k in core)
rows=[]
for stage in ['nominal','measured']:
 p=R/f'roulade-history-1536-{stage}-001/traces.pt';tr=torch.load(p,map_location='cpu',weights_only=False);x=tr['measurements'];k=124
 gate=x['head_latch'][k]&(x['frontier'][k]>=math.radians(330))&(x['tilt'][k]<math.pi/4)&(x['height'][k]>.04)&(x['omega'][k].norm(dim=-1)<2)&x['feedback_ready'][k]&x['finite'][k]
 groups=[]
 for name,mask in [('standing',torch.arange(256)%8<4),*[(f'mid{deg}',torch.arange(256)%8==i+4)for i,deg in enumerate([90,180,270,330])]]:
  selected=gate&mask;groups.append(dict(group=name,trials=int(mask.sum()),eligible=int(selected.sum()),height_m=float(x['height'][k,mask].mean()),tilt_deg=float(torch.rad2deg(x['tilt'][k,mask]).mean()),omega_rad_s=float(x['omega'][k,mask].norm(dim=-1).mean()),frontier_deg=float(torch.rad2deg(x['frontier'][k,mask]).mean()),head_contact_trials=int(x['head_contact'][k,mask].sum()),self_contact_trials=int(x['self_contact'][k,mask].sum())))
 rows.append(dict(stage=stage,groups=groups,trace_sha256=sha(p)))
result=dict(status='CPU_CONTRACT_PASS_LOCAL_HANDOFF_NOT_YET_TESTED',student=str(S),student_sha256=sha(S),teacher=str(T),teacher_sha256=sha(T),matched_core_files=len(core),matched_archive_files=len(man[0]),actor_base48_and_action_equal=True,head_and_body_commands='Literal zero-padding equals teacherHOME targets; all13commands assertedzero inprobe. Teacher own frozen normalizer, actual current rawobs and previous executedaction.',cached_evidence=rows,limitations=['CPU gate evidence is at old reward boundary after control124; liveprobe uses actual preaction at125, so eligibilitymaydiffer.','Midroll reset progress/latch is precredited; gate does not prove standing-originfullroll.','Head/self contacts initially allowed because folded landing is the failure being tested, tailcriteria remain unchanged.'])
(R/'roulade-history-landing-domain-001.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'rows':rows},indent=2))
