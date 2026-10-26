"""Package the selected existing checkpoint via the official normalized ONNX exporter."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
import mjlab.tasks
from mjlab.tasks.registry import load_rl_cfg
from mjlab_microduck.custom_v2.microduck_v2 import TASK_ID
from mjlab_microduck.export import run_export,ExportConfig

p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();selection=json.loads(a.selection.read_text());assert selection['all_existing_checkpoints_evaluated'];winner=selection['results'][0];source=Path(winner['checkpoint']);assert hashlib.sha256(source.read_bytes()).hexdigest()==winner['sha256'];a.output.mkdir(parents=True,exist_ok=True)
shutil.copy2(source,a.output/source.name)
for name in ('agent.yaml','env.yaml'):shutil.copy2(source.parent/'params'/name,a.output/name)
result=run_export(TASK_ID,ExportConfig(onnx_file=str(a.output/'policy.onnx'),checkpoint_file=str(source),num_envs=1,device='cpu'))
# Compare the official exported model to the unmodified native actor, including normalization.
obs=TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]);cfg=asdict(load_rl_cfg(TASK_ID).actor);allowed=inspect.signature(MLPModel.__init__).parameters;kwargs={k:v for k,v in cfg.items() if k in allowed and k not in {"self","obs","obs_groups","obs_set","output_dim"}};actor=MLPModel(obs,{'actor':['actor']},'actor',14,**kwargs);saved=torch.load(source,map_location='cpu',weights_only=False);actor.load_state_dict(saved['actor_state_dict'],strict=True);actor.eval()
import onnx
import onnxruntime as ort
onnx.checker.check_model(onnx.load(result.onnx_path));session=ort.InferenceSession(str(result.onnx_path),providers=['CPUExecutionProvider']);assert session.get_inputs()[0].shape[-1]==61;assert session.get_outputs()[0].shape[-1]==14
rng=np.random.default_rng(20261026);sample=rng.normal(0,.15,(64,61)).astype('float32');sample[:,5]=-1.
with torch.inference_mode():expected=actor(TensorDict({'actor':torch.from_numpy(sample)},batch_size=[64])).numpy()
actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:x[None]})[0] for x in sample],axis=0);assert np.isfinite(actual).all();np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-4)
report={'selected_iteration':winner['iteration'],'source_checkpoint':str(source),'checkpoint_sha256':winner['sha256'],'onnx_sha256':hashlib.sha256(result.onnx_path.read_bytes()).hexdigest(),'official_exporter':'mjlab_microduck.export.run_export','normalization_included_verified_by_parity':True,'onnx_native_max_abs_error':float(np.max(np.abs(actual-expected))),'actor_input':61,'policy_actions':14,'selection_report':str(a.selection.resolve()),'selection_result':winner,'training_status':'PAUSED_BY_USER','ppo_updates_during_search':0,'hardware_validated':False}
(a.output/'selection.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='selection_result'},indent=2))
