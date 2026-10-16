"""Fixed first-trial visual review of two frozen models, no policy changes."""
from pathlib import Path
import gc,hashlib,importlib.util,json,os,runpy,sys,time
os.environ['MUJOCO_EGL_DEVICE_ID']='0'
R=Path(__file__).resolve().parent;W=R/'velstand-head-free-bc-worktree'
STATUS=R/'velstand-head-bc-visual-status-001.json'
assert not STATUS.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def state(**kw):
    t=STATUS.with_suffix('.tmp');t.write_text(json.dumps(dict(pid=os.getpid(),updated_unix=time.time(),formal_controls=0,diagnostic_control_cap=16800,**kw),indent=2)+'\n');t.replace(STATUS)
state(status='IMPORTING')
sys.path.insert(0,str(W/'scripts'))
ns=runpy.run_path(str(W/'scripts/train_v2_velstand.py'),run_name='visual_imports')
import torch,mujoco
torch.cuda.set_device('cuda:0')
assert ns['source_hashes'](W)==read(R/'velstand-head-free-bc-qualification-001.json')['source_sha256']
source=R/'evaluate_v2_velstand_commands.py'
spec=importlib.util.spec_from_file_location('visual_same_command_evaluator',source)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
original=m.WalkingMeasurements;output=[None]
class Capture(original):
    def __init__(self,env,**kw):
        super().__init__(env,**kw)
        mujoco.mj_saveModel(env.sim.mj_model,str(output[0]/f'visual-model-{env.cfg.seed}.mjb'))
    def read(self):
        values=super().read();values['raw_qpos']=self.env.sim.data.qpos;values['raw_qvel']=self.env.sim.data.qvel
        return values
m.WalkingMeasurements=Capture
try:
    products=[]
    for arm,allocation in [('full-bc','velstand-camera-mild-allocation-001'),('leg-bc','velstand-head-free-bc-allocation-001')]:
        out=R/f'velstand-head-bc-visual-{arm}-capture-001';output[0]=out
        checkpoint=R/allocation/'checkpoint_000320.pt'
        state(status='CAPTURING_GPU_PHYSICS',arm=arm,output=str(out))
        sys.argv=[str(source),'--checkpoint',str(checkpoint),'--output',str(out),
            '--device','cuda:0','--stage','nominal','--repeats','1','--seeds','2026101612']
        m.main()
        (out/'visual-capture-provenance.json').write_text(json.dumps(dict(status='CAPTURE_COMPLETE',
            wrapper_sha256=sha(Path(__file__)),evaluator_sha256=sha(source),report_sha256=sha(out/'report.json'),
            capture='Read-only actual GPU post-control qpos/qvel. Separate21-world diagnostic; original8s command timing and metrics unchanged.',
            visual_timing='CPU kinematic forward on a separate saved model only; never an extra forward on live simulation.',
            scope='First fixed trial in every case, not selected successes. Not the same trajectory as the336-world battery and not CPU dynamics/sim2real proof.'),indent=2)+'\n')
        gc.collect();torch.cuda.empty_cache()
        dest=R/f'velstand-head-bc-visual-{arm}-001';renderer=R/'render-velstand-head-bc-review-001.py'
        state(status='RENDERING_SAVED_QPOS',arm=arm,output=str(dest))
        sys.argv=[str(renderer),'--capture',str(out),'--seed','2026101612','--output',str(dest),'--label',arm]
        runpy.run_path(str(renderer),run_name='__main__')
        assert read(dest/'manifest.json')['status']=='RENDER_COMPLETE'
        products.append(str(dest/'walking-review.mp4'))
    state(status='VISUAL_ARTIFACTS_COMPLETE_INSPECTION_REQUIRED',diagnostic_controls=16800,videos=products)
except BaseException as exc:
    state(status='FAILED_REQUIRES_REVIEW',error=repr(exc));raise
