"""One bounded head-loss experiment; qualification then64 formal then eval."""
from pathlib import Path
import gc,hashlib,json,os,runpy,sys,time
R=Path(__file__).resolve().parent;W=R/'velstand-head-free-bc-worktree'
STATUS=R/'velstand-head-free-bc-pipeline-status-001.json'
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
ns=runpy.run_path(str(W/'scripts/train_v2_velstand.py'),run_name='head_free_bc_trainer')
import torch
from mjlab_microduck.custom_v2.velstand_ppo import teacher_axis_loss,BC_AXIS_SCOPE
torch.cuda.set_device('cuda:0');torch.set_num_threads(2)
plan=read(R/'velstand-head-free-bc-plan-001.json')
transition=R/'velstand-head-free-bc-source-transition-001.json'
parent=R/'velstand-v8-allocation-002/checkpoint_000256.pt'
stand=R/'head2-micro-exposure-allocation-001/checkpoint_001856.pt'
walk=R/'walking-yaw-fine-allocation-002/checkpoint_000512.pt'
reference=read(R/'velstand-camera-qualification-001.json')
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
    state(status='CPU_LOSS_AND_GRADIENT_CHECKS',formal_controls=0)
    p=torch.linspace(-.9,.8,8*14,dtype=torch.float64).reshape(8,14).requires_grad_()
    target=torch.sin(torch.arange(8*14,dtype=torch.float64)).reshape(8,14)
    weights=torch.linspace(.4,1.7,8,dtype=torch.float64)
    stand_mask=torch.tensor([True,False,False,True,False,True,True,False])
    loss=teacher_axis_loss(p,target,stand_mask,weights)
    gradient=torch.autograd.grad(loss,p)[0]
    expected=2*weights[:,None]*(p.detach()-target)/(8*14)
    expected[~stand_mask,5:9]=0
    torch.testing.assert_close(gradient,expected,rtol=1e-14,atol=1e-15)
    old_loss=(weights*(p-target).square().mean(-1)).mean()
    all_stand=teacher_axis_loss(p,target,torch.ones_like(stand_mask),weights)
    torch.testing.assert_close(all_stand,old_loss,rtol=0,atol=0)
    old_gradient=torch.autograd.grad(old_loss,p)[0]
    legs=[0,1,2,3,4,9,10,11,12,13]
    torch.testing.assert_close(gradient[:,legs],old_gradient[:,legs],rtol=0,atol=0)
    torch.testing.assert_close(gradient[stand_mask],old_gradient[stand_mask],rtol=0,atol=0)
    assert torch.count_nonzero(gradient[~stand_mask,5:9])==0
    cpu=dict(status='PASS',scope=BC_AXIS_SCOPE,head_walk_gradient_zero=True,
        leg_gradients_exact_old=True,standing_full_loss_and_gradients_exact_old=True,
        mixed_gradient_max_error=float((gradient-expected).abs().max()),source_sha256=hashes)
    (R/'velstand-head-free-bc-cpu-001.json').write_text(json.dumps(cpu,indent=2)+'\n')
    state(status='SMALL64_SOFTWARE_VALIDATION',formal_controls=0)
    small=train(R/'velstand-head-free-bc-smoke64-001',64,5,validation=True,warm=True)
    assert small['onnx_max_abs_error']<2e-5 and small['terminal_bootstrap_samples']>=32
    state(status='ACTUAL24576_MATURE_VALIDATION',formal_controls=0)
    actual=train(R/'velstand-head-free-bc-actual24576-001',24576,16,validation=True)
    assert actual['completed_updates']==272 and actual['terminal_bootstrap_samples']>=12288
    verify_sources()
    qualification=R/'velstand-head-free-bc-qualification-001.json'
    q=dict(schema=1,status='PASS',created_unix=time.time(),source_sha256=hashes,
        assets_sha256=reference['assets_sha256'],num_envs=24576,steps_per_env=24,
        allowed_camera_weights=[-.01],stage_maximum_controls=37748736,maximum_updates=64,
        bc_axis_scope=BC_AXIS_SCOPE,parent_checkpoint_sha256=sha(parent),
        smoke_onnx_max_abs=small['onnx_max_abs_error'],actual_throughput=actual['throughput'],
        actual_feedback=actual['feedback_audit'],actual_terminal_bootstraps=actual['terminal_bootstrap_samples'],
        validation_inherited=False,physics_camera_sampling_unchanged_from_qualified_pair=True,
        evidence_sha256={str(p):sha(p) for p in [R/'velstand-head-free-bc-cpu-001.json',R/'velstand-head-free-bc-smoke64-001/report.json',R/'velstand-head-free-bc-actual24576-001/report.json',transition,R/'velstand-head-free-bc-plan-001.json']})
    qualification.write_text(json.dumps(q,indent=2)+'\n')
    state(status='FORMAL_TRAINING',formal_controls=0)
    formal_started=time.monotonic()
    out=R/'velstand-head-free-bc-allocation-001'
    report=train(out,24576,64,validation=False,qualification=qualification)
    assert report['completed_updates']==320
    (out/'outer-phase-wall.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-formal_started,
        controls=37748736,scope='Formal phase only, imports and qualification separately excluded'),indent=2)+'\n')
    reports=[]
    protocol=read(R/'velstand-camera-eval-protocol.json')
    assert sha(R/'evaluate_v2_velstand_commands.py')==protocol['commands_sha256']
    assert sha(R/'evaluate_v2_velstand_survey_device.py')==protocol['survey_device_sha256']
    for stage in ('nominal','measured'):
        for kind,script in [('commands','evaluate_v2_velstand_commands.py'),('survey','evaluate_v2_velstand_survey_device.py')]:
            dest=R/f'velstand-head-free-bc-0320-{kind}-{stage}'
            state(status='EVALUATING',phase=dest.name,formal_controls=37748736)
            sys.argv=[str(R/script),'--checkpoint',str(out/'checkpoint_000320.pt'),
                '--output',str(dest),'--device','cuda:0','--stage',stage]
            runpy.run_path(str(R/script),run_name='__main__')
            result=read(dest/'report.json');assert result['status'] in ('EVALUATION_COMPLETE','STUDENT_SURVEY_COMPLETE')
            reports.append(str(dest/'report.json'));gc.collect();torch.cuda.empty_cache()
    state(status='BOUNDED_HEAD_BC_STAGE_COMPLETE_REVIEW_REQUIRED',formal_controls=37748736,
        reports=reports,whole_pipeline_wall_s=time.monotonic()-started)
except BaseException as exc:
    out=R/'velstand-head-free-bc-allocation-001/status.json'
    state(status='STOPPED_REQUIRES_REVIEW',error=repr(exc),
        formal_controls=read(out)['allocation_controls'] if out.exists() else 0)
    raise
