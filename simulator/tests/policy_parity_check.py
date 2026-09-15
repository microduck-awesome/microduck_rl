"""Compare shipped/cached ONNX against checkpoint actors, including normalization.

Run from the repository root; requires the local source training checkpoints.
This checks inference only and does not re-export or update any model.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
import mjlab.tasks
from mjlab.tasks.registry import load_rl_cfg


def check(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    results = []
    for slot, policy in manifest['policies'].items():
        checkpoint = torch.load(policy['checkpoint'], map_location='cpu', weights_only=False)
        cfg = asdict(load_rl_cfg(policy['task']).actor)
        assert cfg['class_name'] == 'MLPModel' and cfg['obs_normalization']
        actor = MLPModel(TensorDict({'actor': torch.zeros(1, 61)}, batch_size=[1]),
                         {'actor': ['actor']}, 'actor', 14,
                         **{k: cfg[k] for k in ('hidden_dims', 'activation', 'obs_normalization', 'distribution_cfg')})
        actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
        actor.eval()
        generator = torch.Generator().manual_seed(90210)
        mean, std = actor.obs_normalizer._mean, actor.obs_normalizer._std
        observations = torch.cat([torch.zeros(1, 61), *[
            mean + scale * std * torch.randn(32, 61, generator=generator) for scale in (.1, 1., 3.)]])
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(manifest_path.parent / policy['file']),
                                       sess_options=options, providers=['CPUExecutionProvider'])
        with torch.inference_mode():
            expected = actor(TensorDict({'actor': observations}, batch_size=[len(observations)])).numpy()
        actual = np.concatenate([session.run(None, {session.get_inputs()[0].name: row[None].numpy()})[0]
                                 for row in observations])
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        results.append({'slot': slot, 'checkpoint': policy['checkpoint'], 'samples': len(observations),
                        'max_absolute_error': float(np.max(np.abs(actual - expected)))})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-cache', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = [Path('simulator/models/manifest.json')]
    if args.include_cache:
        paths.extend(p for p in Path('simulator/models/cache').glob('*.json') if p.name != 'manifest.json')
    report = [result for path in paths for result in check(path)]
    output = Path('simulator/logs/policy_parity_check.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
