"""CPU export through the installed native mjlab runner; no simulation or training."""
from pathlib import Path
from types import SimpleNamespace
import json,torch,onnx,hashlib
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from mjlab.rl.runner import MjlabOnPolicyRunner
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261023/correct-routing-retrain-008/rise/formal-001')
OUT=Path('/home/l40-vm01/data2/experiments/microduck-v2-20261023/rise1400-hardware-delivery')
torch.set_num_threads(2)
saved=torch.load(SOURCE/'assessed-1400.pt',map_location='cpu',weights_only=False)
r=saved['infos']['sim2real_baseline_20261021'];cfg=json.loads((SOURCE/'allocation.json').read_text())['agent']
assert r['completed_updates']==1400 and r['training_phase']=='robust'
a=cfg['actor'];model=MLPModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),cfg['obs_groups'],'actor',14,hidden_dims=a['hidden_dims'],activation=a['activation'],obs_normalization=a['obs_normalization'],distribution_cfg=a['distribution_cfg'])
model.load_state_dict(saved['actor_state_dict'],strict=True);model.eval()
MjlabOnPolicyRunner.export_policy_to_onnx(SimpleNamespace(alg=SimpleNamespace(get_policy=lambda:model)),str(OUT),'policy.onnx')
# Existing V2 metadata describes the unchanged runtime interface. Replace all
# variable training identity fields with this immutable checkpoint's contract.
meta=json.loads((ROOT.parent/'20261022-evening/rise-8040/onnx-metadata.json').read_text())
meta['run_path']=str(SOURCE/'assessed-1400.pt')
meta['microduck_training_phase']=r['training_phase']
meta['microduck_randomization']=json.dumps(r['randomization_contract'],sort_keys=True)
meta['microduck_geometry_sha256']=r['randomization_contract']['geometry_sha256']
meta['microduck_nominal_mass_kg']=str(r['randomization_contract']['mass_kg'])
meta['microduck_checkpoint_sha256']=hashlib.sha256((SOURCE/'assessed-1400.pt').read_bytes()).hexdigest()
meta['microduck_recipe_sha256']=r['recipe_sha256']
meta['microduck_actuator_routing']='corrected-name-matched-15-axis;mouth-independent-policy14'
graph=onnx.load(OUT/'policy.onnx');onnx.helper.set_model_props(graph,meta);onnx.checker.check_model(graph);onnx.save(graph,OUT/'policy.onnx')
print('EXPORTED',OUT/'policy.onnx')
