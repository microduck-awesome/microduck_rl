"""Verify synchronized states and, optionally, normalized ONNX parity on CPU."""
import argparse, hashlib, json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--update',type=int,required=True)
    p.add_argument('--resume-check',action='store_true');p.add_argument('--onnx',action='store_true')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    import torch
    torch.set_num_threads(2)
    def equal(x,y):
        if isinstance(x,torch.Tensor):
            if not torch.equal(x,y): raise ValueError('Rank tensor mismatch')
        elif isinstance(x,dict):
            if x.keys()!=y.keys(): raise ValueError('Rank keys mismatch')
            for k in x: equal(x[k],y[k])
        elif isinstance(x,(tuple,list)):
            if len(x)!=len(y): raise ValueError('Rank sequence mismatch')
            for u,v in zip(x,y): equal(u,v)
        elif x!=y: raise ValueError('Rank scalar mismatch')
    paths=[a.run/f'rank{rank}/resume_{a.update:06d}.pt' for rank in (0,1)]
    states=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    for rank,state in enumerate(states):
        folder=a.run/f'rank{rank}'
        status=json.loads((folder/'status.json').read_text())
        assert status['status']=='ALLOCATION_COMPLETE' and status['completed_updates']==a.update,status
        assert state['infos']['v2']['completed_updates']==a.update
        if a.resume_check:
            restore=json.loads((folder/'resume-verification.json').read_text())
            assert restore['actor_critic_normalizers_optimizer_exact']
    for key in ('actor_state_dict','critic_state_dict','optimizer_state_dict'): equal(states[0][key],states[1][key])
    for field in ('stage','polish'): equal(states[0]['infos']['v2']['recovery'][field],states[1]['infos']['v2']['recovery'][field])
    report={'status':'PASS','update':a.update,'rank_states_equal':True,'resume_restore_checked':a.resume_check,'checkpoint_sha256':hashlib.sha256(paths[0].read_bytes()).hexdigest()}
    if a.onnx:
        import numpy as np,onnx,onnxruntime as ort
        from dataclasses import asdict
        from tensordict import TensorDict
        import mjlab.tasks
        from mjlab.tasks.registry import load_rl_cfg
        from rsl_rl.utils import resolve_callable,resolve_obs_groups
        from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS
        onnx_path=a.run/'rank0/rank0.onnx'
        model=onnx.load(str(onnx_path));onnx.checker.check_model(model)
        meta={item.key:item.value for item in model.metadata_props}
        assert tuple(meta['joint_names'].split(','))==POLICY_JOINTS
        config=asdict(load_rl_cfg('Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline'))
        obs=TensorDict({'actor':torch.zeros(1,61),'critic':torch.zeros(1,74)},batch_size=[1])
        groups=resolve_obs_groups(obs,config['obs_groups'],['actor','critic'])
        for key in ('cnn_cfg','distribution_cfg'):
            if config['actor'].get(key) is None: config['actor'].pop(key,None)
        if config['actor'].get('rnn_type') is None:
            for key in ('rnn_type','rnn_hidden_dim','rnn_num_layers'): config['actor'].pop(key,None)
        policy=resolve_callable(config['actor'].pop('class_name'))(obs,groups,'actor',14,**config['actor'])
        policy.load_state_dict(states[0]['actor_state_dict']);policy.eval()
        session=ort.InferenceSession(str(onnx_path),providers=['CPUExecutionProvider'])
        mean=policy.obs_normalizer.mean.detach().numpy();std=policy.obs_normalizer.std.detach().numpy()
        rng=np.random.default_rng(20261023)
        samples=[np.zeros((1,61),np.float32),np.linspace(-.1,.1,61,dtype=np.float32)[None]]
        samples.extend((mean+rng.normal(size=(1,61))*std*.5).astype(np.float32) for _ in range(64))
        error=0.
        for x in samples:
            result=session.run(None,{session.get_inputs()[0].name:x})[0]
            with torch.inference_mode(): expected=policy({'actor':torch.as_tensor(x)}).numpy()
            assert result.shape==(1,14) and np.isfinite(result).all()
            np.testing.assert_allclose(result,expected,atol=2e-5,rtol=2e-5)
            error=max(error,float(np.abs(result-expected).max()))
        report.update(onnx_sha256=hashlib.sha256(onnx_path.read_bytes()).hexdigest(),parity_samples=len(samples),onnx_native_max_abs_error=error)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
