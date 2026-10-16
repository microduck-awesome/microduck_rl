"""Measured S0 registers consumed by the deployed native LS3 algorithm.

The virtual calibration reproduces the research affine in the firmware's
per-joint zero/scale form. It is not physical robot calibration. Full healthy
gates include the newest packet, driver velocity, native/CAD bounds and delayed
history query. No neural features, neural warmup or neural fault gate is used.
"""
import torch
from mjlab_microduck.custom_v2.firmware_ls3 import FirmwareLS3
from mjlab_microduck.sc0090.observer import FrozenObserver
from .stream import CausalStream


class FirmwareNativeStream(FirmwareLS3):
    _tensor=CausalStream._tensor
    _append=CausalStream._append
    _last=CausalStream._last
    _select=CausalStream._select
    command=CausalStream.command
    pid=CausalStream.pid

    def __init__(self,n,weight_path,mode,*,device='cpu'):
        if mode!='O0':raise ValueError('Firmware native LS3 is non-neural')
        super().__init__(n,device)
        self.model=FrozenObserver(weight_path,'O0').to(device)
        self.zero=512.
        self.scale=.005624596222247351
        self.count_limits=(20,1001)
        self.q_limits=(-3.,3.)
        z=lambda *shape,dtype=torch.float32:torch.zeros(n,*shape,device=device,dtype=dtype)
        self.command_t=z(8,dtype=torch.int64);self.command_values=z(8,3)
        self.pid_t=z(2,dtype=torch.int64);self.pid_values=z(2,4)
        self.source_fault=z(dtype=torch.int64)
        self.front_status=z(dtype=torch.int64)
        self.packet_time=z(dtype=torch.int64)
        self.packet_q=z();self.packet_count=z();self.packet_good=z(dtype=torch.bool)
        self.position_min=torch.full((n,),-3.,device=device)
        self.position_max=torch.full((n,),3.,device=device)

    def reset(self,ids=None):
        if ids is None:ids=slice(None)
        super().reset(ids)
        for name in ('command_t','command_values','pid_t','pid_values','source_fault',
                     'front_status','packet_time','packet_q','packet_count','packet_good'):
            getattr(self,name)[ids]=0

    @torch.compiler.disable(recursive=False)
    def read(self,start_us,end_us,available_us,count,speed,load,voltage,valid=True,*,mask=None):
        del speed,load,voltage
        start,end,available=[self._tensor(t) for t in (start_us,end_us,available_us)]
        count=self._tensor(count,torch.float32)
        good=self._tensor(valid,torch.bool)&(start>0)&(start<=end)&(end<=available)
        good&=torch.isfinite(count)&(count>=0)&(count<=1023)
        if mask is None:mask=torch.ones(self.n,device=self.device,dtype=torch.bool)
        scale=self.model.mapping[1]*self.scale
        zero=self.zero-self.model.mapping[0]/scale
        q=(count-zero)*scale
        self.push(end,q,good,mask)
        for name,value in (('packet_time',end),('packet_q',q),('packet_count',count),('packet_good',good)):
            storage=getattr(self,name);storage.copy_(torch.where(mask,value,storage))
        self.front_status.copy_(torch.where(mask,torch.where(good,0,2),self.front_status))
        return good,self.front_status

    @torch.compiler.disable(recursive=False)
    def tick(self,*args,**kwargs):
        # The firmware's non-neural consumer queries native history directly.
        return None

    def query(self,query_us,wall_us):
        value,aligned,_=super().query(query_us,wall_us)
        wall=self._tensor(wall_us)
        healthy=self.packet_good&self.velocity_ready&(self.packet_time>0)
        healthy&=(self.packet_time<=wall)&(wall-self.packet_time<=50000)
        healthy&=(self.packet_count>=20)&(self.packet_count<=1001)
        healthy&=(self.packet_q>=self.position_min-.02)&(self.packet_q<=self.position_max+.02)
        healthy&=(value[:,0]>=self.position_min-.02)&(value[:,0]<=self.position_max+.02)
        valid=aligned&healthy
        return torch.where(valid[:,None],value,float('nan')),valid
