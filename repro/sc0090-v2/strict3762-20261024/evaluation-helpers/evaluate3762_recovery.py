"""Nominal recovery with the checkpoint's actual position-delay recipe."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tarfile

import torch
from tensordict import TensorDict
import mjlab.tasks
from mjlab.tasks.registry import load_rl_cfg
from rsl_rl.utils import resolve_callable,resolve_obs_groups
from train_v2_standup import evaluate


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:1')
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--stage', type=int, default=5, choices=range(6), help='Fixed final recovery frontier for comparable evaluation')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    task='Mjlab-StandUp-Flat-MicroDuck-V2-PairedError-Baseline'
    parent=json.loads((args.checkpoint.parent/'run.json').read_text())
    assert parent['task']==task
    state=torch.load(args.checkpoint,map_location=args.device,weights_only=False)
    position=state['infos']['v2'].get('position_robustness_recipe')
    config=asdict(load_rl_cfg(task))
    obs=TensorDict({'actor':torch.zeros(1,61,device=args.device),
        'critic':torch.zeros(1,74,device=args.device)},batch_size=[1])
    groups=resolve_obs_groups(obs,config['obs_groups'],['actor','critic'])
    for option in ('cnn_cfg','distribution_cfg'):
        if config['actor'].get(option) is None:config['actor'].pop(option,None)
    if config['actor'].get('rnn_type') is None:
        for option in ('rnn_type','rnn_hidden_dim','rnn_num_layers'):
            config['actor'].pop(option,None)
    constructor=resolve_callable(config['actor'].pop('class_name'))
    policy=constructor(obs,groups,'actor',14,**config['actor']).to(args.device)
    policy.load_state_dict(state['actor_state_dict'],strict=True)
    policy.eval()
    sources=[*Path('src').glob('**/*.py'),Path(__file__),Path('scripts/train_v2_standup.py')]
    with tarfile.open(args.output/'source.tar.gz','w:gz') as archive:
        for source in sources:
            archive.add(source,arcname=str(source.relative_to(Path.cwd()) if source.is_absolute() else source))
    with torch.inference_mode():
        cases=evaluate(task,policy,args.device,args.stage,args.samples,
            position_recipe=position,evidence_dir=args.output)
    for name,value in state['actor_state_dict'].items():
        torch.testing.assert_close(policy.state_dict()[name],value,rtol=0,atol=0)
    report=dict(schema=1,task=task,evaluation_stage=args.stage,
        robot_xml_sha256=hashlib.sha256(Path('src/mjlab_microduck/custom_v2/assets/robot.xml').read_bytes()).hexdigest(),
        checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        completed_updates=state['infos']['v2']['completed_updates'],
        position_observation_recipe=position,actor_and_normalizer_unchanged=True,
        protocol='nominal physical conditions; additive corruption disabled; actual checkpoint delay retained',
        cases=cases,source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(args.output)


if __name__=='__main__':main()
