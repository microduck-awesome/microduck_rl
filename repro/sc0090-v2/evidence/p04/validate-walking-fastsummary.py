import hashlib,importlib.util,json,pathlib,time
import torch
r=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fastsummary',r/'evaluate_v2_walking_fastsummary.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(2);device='cuda:1'
# Known irregular touchdown history: each foot's first event is censored.
pos=torch.zeros(6,1,2,2,device=device);pos[:,0,0,0]=torch.arange(6,device=device)*.1;pos[:,0,1,1]=torch.arange(6,device=device)*.2
touch=torch.zeros(6,1,2,dtype=torch.bool,device=device);touch[[1,3,5],0,0]=True;touch[[0,4],0,1]=True
actual=m.stride_lengths(pos,touch);torch.testing.assert_close(actual,torch.tensor([.2,.8,.2],device=device),atol=1e-7,rtol=1e-6)
assert m.stride_lengths(pos,torch.zeros_like(touch)).numel()==0
assert m.stride_lengths(pos,touch[:1].expand_as(touch)&False).numel()==0
cases=0;largest=0.;reports=[];timer=time.perf_counter()
for checkpoint in (65,129):
 for stage in ('nominal','measured'):
  path=r/f'walking-discovery-{checkpoint:06d}-{stage}/report.json';report=json.loads(path.read_text());reports.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  for seed in (2026101612,2026101613):
   trace=torch.load(path.parent/f'trace-{seed}.pt',map_location=device,weights_only=False)['measurements']
   records=[x for x in report['records'] if x['seed']==seed];cursor=0
   for row in records:
    sl=slice(cursor,cursor+row['trials']);cursor+=row['trials']
    for name,a,b in m.PHASES:
     contact=trace['contact'][a:b,sl]
     prev=trace['contact'][a-1:b-1,sl] if a else torch.cat((contact[:1],contact[:-1]))
     touches=contact&~prev
     lengths=m.stride_lengths(trace['feet_position'][a:b,sl,:,:2],touches)
     old=row['phases'][name]
     assert len(lengths)==old['same_foot_stride_count'],(checkpoint,stage,seed,row['name'],name)
     if len(lengths):
      delta=abs(float(lengths.mean())-old['same_foot_stride_mean_m']);largest=max(largest,delta);assert delta<2e-7,delta
     else:assert old['same_foot_stride_mean_m'] is None
     cases+=1
   assert cursor==trace['contact'].shape[1]
# Read-only timing comparison of the old and new exact output order on real data.
pos=trace['feet_position'][:100,:,:,:2];contact=trace['contact'][:100];touches=contact&~torch.cat((contact[:1],contact[:-1]))
def old_loop(positions,touch):
 last_pos=torch.zeros_like(positions[0]);seen=torch.zeros_like(touch[0]);items=[]
 for i in range(len(touch)):
  valid=touch[i]&seen;items.append((positions[i]-last_pos).norm(dim=-1)[valid]);last_pos=torch.where(touch[i][...,None],positions[i],last_pos);seen|=touch[i]
 return torch.cat(items)
timings={};values={}
for name,function in [('old_loop',old_loop),('vectorized',m.stride_lengths)]:
 torch.cuda.synchronize(device);start=time.perf_counter();values[name]=function(pos,touches);torch.cuda.synchronize(device);timings[name]=time.perf_counter()-start
assert torch.equal(values['old_loop'],values['vectorized'])
report=dict(status='PASS',real_case_phases_checked=cases,largest_saved_mean_abs_delta=largest,irregular_event_and_empty_guards=True,real_stride_vector_bitwise_equal=True,timing_seconds=timings,validation_wall_seconds=time.perf_counter()-timer,source_reports=reports,source_sha256=hashlib.sha256((r/'evaluate_v2_walking_fastsummary.py').read_bytes()).hexdigest(),scope='Existing GPU traces only, no new simulation or policy training. Timing is a single diagnostic call while other workloads run, not whole-evaluation speed.')
(r/'walking-fastsummary-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
