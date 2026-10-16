"""One bounded head-loss experiment; qualification then64 formal then eval."""
from pathlib import Path
import gc,hashlib,json,os,runpy,sys,time
R=Path(__file__).resolve().parent;W=R/'velstand-head-bias-worktree'
STATUS=R/'velstand-head-bias-pipeline-status-001.json'
assert not STATUS.exists()
started=time.monotonic()
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def state(**kw):
    tmp=STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),
        stage_maximum_formal_controls=37748736,automatic_second_allocation=False,**kw),indent=2)+'\n');tmp.replace(STATUS)
state(status='IMPORTING',formal_controls=0)
sys.path.insert(0,str(W/'scripts'))
ns=runpy.run_path(str(W/'scripts/train_v2_velstand.py'),run_name='head_bias_trainer')
import torch
from mjlab_microduck.custom_v2.velstand_ppo import teacher_axis_loss,BC_AXIS_SCOPE
torch.cuda.set_device('cuda:0');torch.set_num_threads(2)
plan=read(R/'velstand-head-bias-plan-001.json')
transition=R/'velstand-head-bias-source-transition-001.json'
parent=R/'velstand-head-free-bc-allocation-001/checkpoint_000320.pt'
stand=R/'head2-micro-exposure-allocation-001/checkpoint_001856.pt'
walk=R/'walking-yaw-fine-allocation-002/checkpoint_000512.pt'
reference=read(R/'velstand-head-free-bc-qualification-001.json')
hashes=ns['source_hashes'](W)
assert hashes==read(transition)['target_source_sha256']
def verify_sources():
    assert ns['source_hashes'](W)==hashes
    for name,digest in reference['assets_sha256'].items():assert sha(W/name)==digest,name
def train(out,n,it,*,validation,warm=False,qualification=None):
    verify_sources()
    argv=[str(W/'scripts/train_v2_velstand.py'),'--output',str(out),'--device','cuda:0',
        '--num-envs',str(n),'--iterations',str(it),'--camera-weight','-.01',
        '--standing-teacher',str(stand),'--walking-teacher',str(walk),
        '--native-range-budget','8' if validation else '16']
    if warm:argv+=['--warm-start',str(walk)]
    else:argv+=['--resume',str(parent),'--source-transition',str(transition)]
    if validation:argv+=['--validation-only','--force-timeouts']
    if qualification:argv+=['--qualification',str(qualification)]
    sys.argv=argv;ns['main']()
    report=read(out/'report.json')
    assert report['status']=='ALLOCATION_COMPLETE'
    assert report['allocation_controls']==n*24*it
    assert report['lineage']['bc_axis_scope']==BC_AXIS_SCOPE
    assert report['feedback_audit']['unexplained_invalid_lanes']==0
    assert report['camera_audit']['raw_gated_fault_nonfinite_totals'][3]==0
    gc.collect();torch.cuda.empty_cache()
    return report
try:
    state(status='CPU_WEIGHTED_HISTORY_CHECKS',formal_controls=0)
    from mjlab_microduck.custom_v2.head_bias import update_history
    # Independent explicit weighted history, including reset lanes and commands.
    torch.manual_seed(73403)
    moment=torch.zeros(3,4,dtype=torch.float64);mass=torch.zeros(3,dtype=torch.float64)
    records=[[],[],[]];max_error=0.
    for step in range(120):
        angle=torch.randn(3,4,dtype=torch.float64)*.2
        command=torch.tensor([[0.,.2,-.1,.05],[.1,0.,.2,-.1],[-.1,.1,0.,.2]],dtype=torch.float64)*(1 if step<50 else -1)
        gate=torch.tensor([1.,0. if step<60 else 1.,.3 if step%3 else .8],dtype=torch.float64)
        fresh=torch.tensor([step==0,step in (0,90),step in (0,65)])
        moment,mass,cost=update_history(moment,mass,angle,command,gate,fresh,.02)
        expected=[]
        for lane in range(3):
            if fresh[lane]:records[lane]=[]
            records[lane].append((angle[lane].clone(),float(gate[lane])))
            error=torch.zeros(4,dtype=torch.float64)
            for lag,(q,g) in enumerate(reversed(records[lane])):
                error+=.02*.98**lag*g*(q-command[lane])
            expected.append(-error.abs().mean()*gate[lane])
        oracle=torch.stack(expected)
        torch.testing.assert_close(cost,oracle,rtol=1e-12,atol=1e-14)
        max_error=max(max_error,float((cost-oracle).abs().max()))
        assert bool((cost<=0).all()) and bool((mass>=0).all()) and bool((mass<=1).all())
        if step<60:assert cost[1]==0 and mass[1]==0
    # Constant bias converges; alternating gait oscillation cancels in the EMA.
    m=torch.zeros(2,4,dtype=torch.float64);h=torch.zeros(2,dtype=torch.float64)
    for step in range(800):
        angle=torch.tensor([[.2]*4,[.2*(-1)**step]*4],dtype=torch.float64)
        m,h,c=update_history(m,h,angle,torch.zeros_like(m),torch.ones(2),torch.zeros(2,dtype=torch.bool),.02)
    assert abs(float(c[0])+.2)<1e-6 and abs(float(c[1]))<.003
    cpu=dict(status='PASS',weighted_history_max_abs_error=max_error,constant_bias_cost=float(c[0]),alternating_cost=float(c[1]),
        command_rebase_reset_fallen_partial_gate=True,source_sha256=hashes)
    (R/'velstand-head-bias-cpu-001.json').write_text(json.dumps(cpu,indent=2)+'\n')
    state(status='SMALL64_SOFTWARE_VALIDATION',formal_controls=0)
    small=train(R/'velstand-head-bias-smoke64-001',64,5,validation=True,warm=True)
    assert small['onnx_max_abs_error']<2e-5 and small['terminal_bootstrap_samples']>=32
    state(status='ACTUAL24576_MATURE_VALIDATION',formal_controls=0)
    actual=train(R/'velstand-head-bias-actual24576-001',24576,16,validation=True)
    assert actual['completed_updates']==336 and actual['terminal_bootstrap_samples']>=12288
    verify_sources()
    qualification=R/'velstand-head-bias-qualification-001.json'
    q=dict(schema=1,status='PASS',created_unix=time.time(),source_sha256=hashes,
        assets_sha256=reference['assets_sha256'],num_envs=24576,steps_per_env=24,
        allowed_camera_weights=[-.01],head_bias_weight=1.,head_bias_tau_s=1.,stage_maximum_controls=37748736,maximum_updates=64,
        bc_axis_scope=BC_AXIS_SCOPE,parent_checkpoint_sha256=sha(parent),
        smoke_onnx_max_abs=small['onnx_max_abs_error'],actual_throughput=actual['throughput'],
        actual_feedback=actual['feedback_audit'],actual_terminal_bootstraps=actual['terminal_bootstrap_samples'],
        validation_inherited=False,physics_camera_sampling_unchanged_from_qualified_pair=True,
        evidence_sha256={str(p):sha(p) for p in [R/'velstand-head-bias-cpu-001.json',R/'velstand-head-bias-smoke64-001/report.json',R/'velstand-head-bias-actual24576-001/report.json',transition,R/'velstand-head-bias-plan-001.json']})
    qualification.write_text(json.dumps(q,indent=2)+'\n')
    state(status='FORMAL_TRAINING',formal_controls=0)
    formal_started=time.monotonic()
    out=R/'velstand-head-bias-allocation-001'
    report=train(out,24576,64,validation=False,qualification=qualification)
    assert report['completed_updates']==384
    (out/'outer-phase-wall.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-formal_started,
        controls=37748736,scope='Formal phase only, imports and qualification separately excluded'),indent=2)+'\n')
    reports=[]
    protocol=read(R/'velstand-camera-eval-protocol.json')
    assert sha(R/'evaluate_v2_velstand_commands.py')==protocol['commands_sha256']
    assert sha(R/'evaluate_v2_velstand_survey_device.py')==protocol['survey_device_sha256']
    for stage in ('nominal','measured'):
        for kind,script in [('commands','evaluate_v2_velstand_commands.py'),('survey','evaluate_v2_velstand_survey_device.py')]:
            dest=R/f'velstand-head-bias-0384-{kind}-{stage}'
            state(status='EVALUATING',phase=dest.name,formal_controls=37748736)
            sys.argv=[str(R/script),'--checkpoint',str(out/'checkpoint_000384.pt'),
                '--output',str(dest),'--device','cuda:0','--stage',stage]
            runpy.run_path(str(R/script),run_name='__main__')
            result=read(dest/'report.json');assert result['status'] in ('EVALUATION_COMPLETE','STUDENT_SURVEY_COMPLETE')
            reports.append(str(dest/'report.json'));gc.collect();torch.cuda.empty_cache()
    state(status='BOUNDED_HEAD_BIAS_STAGE_COMPLETE_REVIEW_REQUIRED',formal_controls=37748736,
        reports=reports,whole_pipeline_wall_s=time.monotonic()-started)
except BaseException as exc:
    out=R/'velstand-head-bias-allocation-001/status.json'
    state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc),
        formal_controls=read(out)['allocation_controls'] if out.exists() else 0)
    raise
