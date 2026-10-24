"""CPU-only all-14 physical limit diagnostics from a saved head battery, without policy changes."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mujoco
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.scene import Scene
from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS
HEAD=('neck_pitch','head_pitch','head_yaw','head_roll')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def longest(mask):
 run=np.zeros(mask.shape[1:],dtype=np.int32);peak=run.copy()
 for row in mask:
  run=np.where(row,run+1,0);peak=np.maximum(peak,run)
 return peak.max(axis=0)*.02

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--head-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
 report=json.loads((a.head_dir/'report.json').read_text());assert a.output.resolve()!=a.head_dir/'report.json'
 assert not a.output.exists(),'Preserve existing diagnosis'
 cfg=load_env_cfg(report['task'],play=True)
 # World count changes storage capacity, not joint topology; compile the actual scene on CPU.
 cfg.scene.num_envs=1;model=Scene(cfg.scene,device='cpu').compile()
 mapping=[]
 for name in POLICY_JOINTS:
  matches=[i for i in range(model.njnt) if mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i).split('/')[-1]==name]
  assert len(matches)==1,(name,matches)
  i=matches[0];assert model.jnt_limited[i] and model.jnt_type[i]==mujoco.mjtJoint.mjJNT_HINGE
  mapping.append(dict(name=name,model_name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,i),joint_id=i,qpos_address=int(model.jnt_qposadr[i]),lower_rad=float(model.jnt_range[i,0]),upper_rad=float(model.jnt_range[i,1])))
 assert len(mapping)==14 and len({r['qpos_address'] for r in mapping})==14
 adr=[r['qpos_address'] for r in mapping];lo=np.array([r['lower_rad'] for r in mapping]);hi=np.array([r['upper_rad'] for r in mapping]);rows=[]
 for seed in sorted({r['seed'] for r in report['records']}):
  path=a.head_dir/f'traces-{seed}.npz'
  with np.load(path) as trace:
   qpos=trace['qpos'];assert qpos.shape[0]==300 and qpos.shape[-1]==model.nq
   assert np.isfinite(qpos).all(),'Nonfinite physical state in head trace'
   q=qpos[...,adr].astype(np.float64)
   head_offsets={}
   for axis,name in enumerate(HEAD):
    delta=q[...,POLICY_JOINTS.index(name)]-trace['physical_head_rad'][...,axis]
    assert float(delta.max()-delta.min())<2e-5,(name,'Physical head channel does not align with model qpos mapping')
    head_offsets[name]=float(np.median(delta))
   clearance=np.minimum(q-lo,hi-q)*180/np.pi
   lower_over=np.maximum(lo-q,0)*180/np.pi;upper_over=np.maximum(q-hi,0)*180/np.pi
   longest_half=longest(clearance<=.5);longest_one=longest(clearance<=1.)
   joints=[]
   for j,item in enumerate(mapping):
    worst=np.unravel_index(np.argmin(clearance[...,j]),clearance[...,j].shape)
    joints.append(dict(**item,min_clearance_deg=float(clearance[...,j].min()),max_lower_overrun_deg=float(lower_over[...,j].max()),max_upper_overrun_deg=float(upper_over[...,j].max()),fraction_within_half_degree=float((clearance[...,j]<=.5).mean()),fraction_within_one_degree=float((clearance[...,j]<=1.).mean()),longest_within_half_degree_s=float(longest_half[j]),longest_within_one_degree_s=float(longest_one[j]),worst_step=int(worst[0]),worst_env=int(worst[1]),minimum_position_deg=float(q[...,j].min()*180/np.pi),maximum_position_deg=float(q[...,j].max()*180/np.pi)))
   rows.append(dict(seed=seed,trace_sha256=sha(path),shape=list(qpos.shape),physical_head_mapping_offsets_rad=head_offsets,joints=joints))
 result=dict(status='DIAGNOSTICS_COMPLETE_NOT_BEHAVIOR_ACCEPTANCE',report_sha256=sha(a.head_dir/'report.json'),checkpoint_sha256=report['checkpoint_sha256'],joint_order=list(POLICY_JOINTS),compiled_scene_nq=int(model.nq),seeds=rows,scope='All 14 policy joints across the full 6-second standing/head battery, at 50 Hz post-step qpos. Mouth excluded. Near-limit durations are diagnostics, not new training gates.',limitations=['Does not cover 200 Hz intermediate substep extremes','Does not cover the independent recovery battery or walking','Small constraint penetration is reported, not silently rounded to zero'],script_sha256=sha(Path(__file__)))
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(status=result['status'],seeds=len(rows),joints=14,minimum_clearance_deg=min(j['min_clearance_deg'] for r in rows for j in r['joints']))),flush=True)
if __name__=='__main__':main()
