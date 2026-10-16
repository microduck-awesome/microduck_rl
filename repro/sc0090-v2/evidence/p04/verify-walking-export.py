import copy,hashlib,json,pathlib
import numpy as np
import onnxruntime as ort
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
r=pathlib.Path(__file__).resolve().parent
out=r/'walking-controller-smoke-64'
checkpoint=out/'checkpoint_000005.pt'
state=torch.load(checkpoint,map_location='cpu',weights_only=False)
run=json.loads((out/'run.json').read_text())
observations=torch.cat([torch.load(r/f'walking-evaluator-qualification-{stage}/trace-2026101614.pt',map_location='cpu',weights_only=False)['initial']['actor'] for stage in ('nominal','measured')])
cfg=copy.deepcopy(run['agent']['actor'])
assert cfg.get('cnn_cfg') is None and cfg.get('rnn_type') is None
allowed=('hidden_dims','activation','obs_normalization','distribution_cfg')
cfg={k:cfg[k] for k in allowed}
model=MLPModel(TensorDict({'actor':observations},batch_size=[len(observations)]),{'actor':['actor']},'actor',14,**cfg).eval()
model.load_state_dict(state['actor_state_dict'],strict=True)
onnx=out/(out.name+'.onnx')
options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
session=ort.InferenceSession(str(onnx),sess_options=options,providers=['CPUExecutionProvider'])
assert session.get_inputs()[0].shape==[1,61] and session.get_outputs()[0].shape==[1,14]
actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:x[None].numpy()})[0] for x in observations])
with torch.inference_mode():expected=model(TensorDict({'actor':observations},batch_size=[len(observations)])).numpy()
error=float(np.abs(actual-expected).max());assert error<1e-5,error
report=dict(status='PASS',samples=len(observations),max_abs_error=error,actor_shape=[1,61],action_shape=[1,14],normalizer_baked=True,source='actual nominal+measured qualification initial actor observations',checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),onnx_sha256=hashlib.sha256(onnx.read_bytes()).hexdigest(),scope='offline inference check only; no hardware/deployment')
(r/'walking-controller-export-check.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
