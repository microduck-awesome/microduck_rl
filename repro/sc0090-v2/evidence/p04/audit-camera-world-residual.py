"""CPU-only, independent world-frame SO(3) audit of saved command traces."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import torch

def multiply(a,b):
    return np.concatenate((a[...,:1]*b[...,:1]-(a[...,1:]*b[...,1:]).sum(-1,keepdims=True),
        a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+np.cross(a[...,1:],b[...,1:])),axis=-1)

def world_velocity(quat,dt=.005):
    q=np.asarray(quat,dtype=np.float64)
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    inv=q[:-1].copy();inv[...,1:]*=-1
    relative=multiply(q[1:],inv)  # World rotation: q_next * inverse(q_previous).
    relative=np.where(relative[...,:1]<0,-relative,relative)
    size=np.linalg.norm(relative[...,1:],axis=-1,keepdims=True)
    scale=2*np.arctan2(size,relative[...,:1])/np.maximum(size,1e-30)/dt
    return relative[...,1:]*scale

def fixtures():
    t=np.arange(51)*.005
    initial=np.array([np.cos(.4),0,np.sin(.4),0])
    for axis in range(3):
        for rate in (-1.7,0.,2.3):
            q=np.zeros((len(t),4));q[:,0]=np.cos(rate*t/2);q[:,axis+1]=np.sin(rate*t/2)
            q=multiply(q,initial);q[::3]*=-1
            actual=world_velocity(q);expected=np.zeros_like(actual);expected[:,axis]=rate
            assert np.max(np.abs(actual-expected))<1e-10
    return dict(constant_world_axis_with_tilt_and_sign_flips=9,max_abs_bound_rad_s=1e-10)

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();torch.set_num_threads(2);tests=fixtures()
    root=a.directory;report=json.loads((root/'report.json').read_text());rows=[];checks=[]
    assert report['status']=='EVALUATION_COMPLETE'
    for check in report['checks']:
        seed=check['seed'];path=root/f'trace-{seed}.pt';digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==check['trace_sha256']
        trace=torch.load(path,map_location='cpu',weights_only=False);m=trace['measurements'];q=trace['camera'].numpy();omega=world_velocity(q)
        assert omega.shape[:2]==(4*len(m['command']),m['command'].shape[1])
        command=np.repeat(m['command'].numpy(),4,axis=0);residual=omega.copy();residual[...,2]-=command[...,2]
        failed=m['failed'].numpy().astype(bool).cumsum(axis=0)>0
        records=[x for x in report['records'] if x['seed']==seed];index=0;max_scalar_error=0.
        for case in records:
            sl=slice(index,index+case['trials']);index+=case['trials']
            for phase,start,end in report['command_protocol']:
                v=omega[start*4:end*4,sl];rv=residual[start*4:end*4,sl]
                raw=np.sqrt(np.square(v).sum(-1).mean(0));excess=np.sqrt(np.square(rv).sum(-1).mean(0))
                survived=~failed[end-1,sl];linear=m['linear'][end-25:end,sl].numpy().mean((0,1))
                discrepancy=abs(float(raw.mean())-case['phases'][phase]['camera']['angular_speed_rms_rad_s']);max_scalar_error=max(max_scalar_error,discrepancy)
                rows.append(dict(seed=seed,case=case['name'],phase=phase,trials=case['trials'],survived_trials=int(survived.sum()),
                    total_camera_rms_per_trial_mean_rad_s=float(raw.mean()),requested_yaw_residual_rms_per_trial_mean_rad_s=float(excess.mean()),
                    survived_total_rms_per_trial_mean_rad_s=float(raw[survived].mean()) if survived.any() else None,
                    survived_requested_yaw_residual_rms_per_trial_mean_rad_s=float(excess[survived].mean()) if survived.any() else None,
                    world_angular_axes_rms_rad_s=np.sqrt(np.square(v).mean((0,1))).tolist(),world_angular_axes_mean_rad_s=v.mean((0,1)).tolist(),
                    requested_yaw_rad_s=float(command[start*4,sl,2].mean()),linear_tail_mean_m_s=linear.tolist(),
                    intended_head_motion=bool(np.any(np.abs(m['head_command'][start:end,sl].numpy())>1e-9))))
        assert index==m['command'].shape[1]
        # Independent float64 quaternion log vs existing float32 scalar distance.
        assert max_scalar_error<.002,max_scalar_error
        checks.append(dict(seed=seed,trace_sha256=digest,max_total_rms_difference_from_original_rad_s=max_scalar_error))
    result=dict(status='PASS_SAVED_TRACE_AUDIT',input_report_sha256=hashlib.sha256((root/'report.json').read_bytes()).hexdigest(),checkpoint_sha256=report['checkpoint_sha256'],stage=report['stage'],fixtures=tests,checks=checks,rows=rows,
        scope='Independent finite-interval world SO(3) rotation at200Hz, not instantaneous Jacobian angular velocity used by training reward. Subtract only original requested body yaw about world vertical. Head commands remain explicitly labelled; residual is not automatically unwanted jitter. Healthy columns use original latched walking health, not stationary hold. No new simulation or actor mutation.')
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],rows=len(rows),checks=checks)))

if __name__=='__main__':main()
