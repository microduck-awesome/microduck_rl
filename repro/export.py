"""Export a walking checkpoint with the original normalization-aware exporter."""
import argparse
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.parent.mkdir(parents=True,exist_ok=True)
from mjlab_microduck.export import run_export,ExportConfig
from mjlab_microduck.custom_v2.microduck_v2 import TASK_ID
run_export(TASK_ID,ExportConfig(onnx_file=str(a.output),checkpoint_file=str(a.checkpoint),num_envs=1,device='cpu'))
import hashlib,inspect,json
from dataclasses import asdict
import numpy as np,onnx,onnxruntime as ort,torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from mjlab.tasks.registry import load_rl_cfg
from mjlab_microduck.custom_v2.microduck_v2 import POLICY_JOINTS
model=onnx.load(str(a.output));onnx.checker.check_model(model)
meta={p.key:p.value for p in model.metadata_props};assert tuple(meta['joint_names'].split(','))==POLICY_JOINTS
obs=TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]);cfg=asdict(load_rl_cfg(TASK_ID).actor)
allowed=inspect.signature(MLPModel.__init__).parameters
kwargs={k:v for k,v in cfg.items() if k in allowed and k not in {'self','obs','obs_groups','obs_set','output_dim'}}
actor=MLPModel(obs,{'actor':['actor']},'actor',14,**kwargs)
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);actor.load_state_dict(saved['actor_state_dict'],strict=True);actor.eval()
session=ort.InferenceSession(str(a.output),providers=['CPUExecutionProvider'])
assert session.get_inputs()[0].shape[-1]==61 and session.get_outputs()[0].shape[-1]==14
rng=np.random.default_rng(20261026);sample=rng.normal(0,.15,(64,61)).astype('float32');sample[:,5]=-1.
with torch.inference_mode():expected=actor(TensorDict({'actor':torch.from_numpy(sample)},batch_size=[64])).numpy()
actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:x[None]})[0] for x in sample],axis=0)
assert np.isfinite(actual).all();np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-4)
report={'status':'PASS','checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),'onnx_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'parity_samples':64,'onnx_native_max_abs_error':float(np.abs(actual-expected).max()),'normalization_included_verified_by_parity':True}
a.output.with_suffix('.verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
