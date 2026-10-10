"""Frozen voltage-specific neural torque with continuous private history.

The original serialized model owns the network/parameters. MuJoCo alone owns
body integration; never restart the trajectory simulator per command segment.
"""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from .physics import make_spec


class CpuNeural:
    def __init__(self, study, settings, mass):
        study=Path(study);self.settings=np.array(settings,dtype=float)
        artifacts=json.loads((study/'protocol.json').read_text())['dynamics']['cross_evaluation']['artifacts']
        matches=[r for r in artifacts if r['vin']==float(self.settings[3])]
        if len(matches)!=1:raise ValueError('Neural dynamics supports only frozen10.2/11.1/12.4Vbranches')
        row=matches[0];path=study.parent/row['path_from_dataset_root']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Frozen neural dynamics identity mismatch')
        self.source_path=path;self.source_sha256=row['sha256']
        config=json.loads(path.with_name('config.json').read_text())
        from servo_bam import neural_dynamics
        if hashlib.sha256(Path(neural_dynamics.__file__).read_bytes()).hexdigest()!=config['source_sha256']['neural_dynamics.py']:
            raise ValueError('Neural feature/torque source differs from frozen branch')
        self.features=neural_dynamics.features
        self.motor_and_control=neural_dynamics.motor_and_control
        self.frozen=torch.jit.load(str(path),map_location='cpu').eval()
        if self.frozen.memory or self.frozen.network.kind!='actuator_mlp':
            raise ValueError('Expected the frozen tail_plain directMLP branch')
        for parameter in self.frozen.parameters():parameter.requires_grad_(False)
        self.p=self.frozen.parameters_vector
        self.tensor_settings=torch.tensor(self.settings,dtype=torch.float64)[None]
        geometry={'armature':float(self.p[2]),'q_offset':float(self.p[5])}
        self.model=make_spec(geometry,self.settings,mass).compile();self.data=mujoco.MjData(self.model)
        self.reset()

    def reset(self,q=0.,v=0.):
        mujoco.mj_resetData(self.model,self.data)
        self.data.qpos[0],self.data.qvel[0]=q,v;mujoco.mj_forward(self.model,self.data)
        self.history=None;self.state=torch.zeros((2,1,24),dtype=torch.float64)
        self.steps=0

    def step(self,goal,enabled=True):
        q,v=float(self.data.qpos[0]),float(self.data.qvel[0])
        tq,tv,tg,te=[torch.tensor([value],dtype=torch.float64) for value in [q,v,goal,float(enabled)]]
        with torch.inference_mode():
            frame=self.features(tq,tv,tg,te,self.tensor_settings,self.p)
            if self.history is None:self.history=frame[:,None,:].repeat(1,10,1)
            self.history=torch.cat([self.history[:,1:],frame[:,None,:]],1)
            raw,self.state=self.frozen.network(self.history,self.state)
            _,external,duty=self.motor_and_control(tq,tv,tg,te,self.tensor_settings,self.p)
            motor=.5*torch.tanh(raw[:,0])*te
            drive=self.p[0]*self.tensor_settings[:,3]*self.tensor_settings[:,8]/self.p[1]
            emf=self.p[0]*self.p[0]*tv/self.p[1]
            motor=torch.clamp(motor,(-drive-emf)*te,(drive-emf)*te)
            budget=.1*torch.nn.functional.softplus(raw[:,1])+.001
            inertia=self.tensor_settings[:,4]+self.p[2]
            stop=inertia/self.tensor_settings[:,0]*tv+motor+external
            friction=torch.clamp(stop,-budget,budget)
            applied=motor-friction
            expected_v=tv+(applied+external)/inertia*self.tensor_settings[:,0]
            if not torch.isfinite(applied).all() or torch.any(expected_v.abs()>100):
                raise FloatingPointError('Neural model left its qualified numerical velocity domain')
            self.data.ctrl[0]=float(applied[0]);control=float(duty[0]*self.settings[3])
        mujoco.mj_step(self.model,self.data);self.steps+=1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError('Nonfinite neural MuJoCo state')
        return q,v,control
