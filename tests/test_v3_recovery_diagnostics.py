"""CPU oracle cases for V3 recovery limit and contact evidence."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_v3_recovery_clearance import reduce_contacts, update_joint_metrics, update_home_metrics


def test_joint_limit_evidence_counts_sustained_and_hard_limit_violations():
    limits = torch.tensor([[[-1.,1.]]*14]*2)
    q = torch.zeros((2,14))
    smallest = torch.full_like(q,float('inf'))
    streak = torch.zeros_like(q,dtype=torch.int32)
    longest = torch.zeros_like(streak)
    violation = torch.zeros_like(streak)
    for step in range(205):
        q[0,3] = 1.02 if step == 100 else .96
        q[1,9] = .97 if step < 20 or step >= 100 else 0.
        smallest,streak,longest,violation = update_joint_metrics(
            q,limits,smallest,streak,longest,violation,.05)
    assert torch.isclose(smallest[0,3],torch.tensor(-.02),atol=1e-6)
    assert longest[0,3] == 205 and violation[0,3] == 1
    assert longest[1,9] == 105 and violation[1,9] == 0
    assert torch.isclose(smallest[0,0],torch.tensor(1.))
    assert (longest[0,:3] == 0).all()


def test_contact_classification_and_sparse_reduction():
    # terrain=0, head=2/3, feet=4/5. Include duplicate contacts, an
    # inactive positive distance, bad world ids and a slot past nacon.
    world=torch.tensor([0,0,0,1,1,1,-1,3,1,0])
    pair=torch.tensor([[0,2],[2,0],[0,4],[7,8],[8,0],[0,3],
                       [0,2],[0,2],[0,3],[0,5]])
    dist=torch.tensor([-.001,-.003,-.002,-.004,-.005,.001,-.2,-.2,-.006,-.2])
    out=reduce_contacts(world,pair,dist,torch.tensor([9]),2,0,[2,3],[4,5])
    assert out['head_ground'][0].tolist()==[1,1]
    torch.testing.assert_close(out['head_ground'][1],torch.tensor([.003,.006]))
    assert out['feet_ground'][0].tolist()==[1,0]
    torch.testing.assert_close(out['feet_ground'][1],torch.tensor([.002,0.]))
    assert out['other_body_ground'][0].tolist()==[0,1]
    torch.testing.assert_close(out['other_body_ground'][1],torch.tensor([0.,.005]))
    assert out['self'][0].tolist()==[0,1]
    torch.testing.assert_close(out['self'][1],torch.tensor([0.,.004]))


def test_home_error_uses_each_joints_nonzero_absolute_home():
    home=torch.linspace(-.7,.7,15).repeat(2,1)
    absolute=torch.zeros_like(home);signed=torch.zeros_like(home);peak=torch.zeros_like(home)
    q=home.clone();q[0,5]+=.1;q[1,14]-=.2
    absolute,signed,peak=update_home_metrics(q,home,absolute,signed,peak)
    q=home.clone();q[0,5]-=.1;q[1,14]+=.05
    absolute,signed,peak=update_home_metrics(q,home,absolute,signed,peak)
    torch.testing.assert_close(absolute[0,5],torch.tensor(.2))
    torch.testing.assert_close(signed[0,5],torch.tensor(0.),atol=1e-7,rtol=0)
    torch.testing.assert_close(peak[0,5],torch.tensor(.1))
    torch.testing.assert_close(absolute[1,14],torch.tensor(.25))
    torch.testing.assert_close(signed[1,14],torch.tensor(-.15))
    torch.testing.assert_close(peak[1,14],torch.tensor(.2))
    assert torch.count_nonzero(peak)==2
