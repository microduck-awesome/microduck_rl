"""GPU version of the deployed ee77131 non-neural LS3 history functions.

Input positions are already calibrated robot radians, as in the UART owner.
This is NOT the bench P04 CausalStream: it fits native points directly, has
no neural history, and permits no forecast at the delayed policy query.
Current packet health, calibrated joint bounds and the controller's fail-stop
remain separate gates; a successful old-history query does not override them.
"""
import torch


class FirmwareLS3:
    @torch.inference_mode(False)
    def __init__(self,n,device):
        self.n,self.device=n,device
        self.times=torch.zeros(n,8,dtype=torch.int64,device=device)
        self.positions=torch.zeros(n,8,device=device)
        self.valid=torch.zeros(n,8,dtype=torch.bool,device=device)
        self.driver_count=torch.zeros(n,dtype=torch.int64,device=device)
        self.driver_time=torch.zeros(n,3,dtype=torch.int64,device=device)
        self.driver_position=torch.zeros(n,3,device=device)
        self.velocity_ready=torch.zeros(n,dtype=torch.bool,device=device)
        self.current_velocity=torch.zeros(n,device=device)
        self._rows=torch.arange(n,device=device)

    def reset(self,ids):
        for name in ('times','positions','valid','driver_count','driver_time',
                     'driver_position','velocity_ready','current_velocity'):
            getattr(self,name)[ids]=0

    @staticmethod
    def fit(times,q,target):
        # Match the C loop's FP64 operation order on three physical timestamps.
        relative=(times-times[:,0,None]).double()*1e-6
        values=q.double()
        mt=((relative[:,0]+relative[:,1])+relative[:,2])/3.
        mq=((values[:,0]+values[:,1])+values[:,2])/3.
        x=relative-mt[:,None];y=values-mq[:,None]
        numerator=(x[:,0]*y[:,0]+x[:,1]*y[:,1])+x[:,2]*y[:,2]
        denominator=(x[:,0]*x[:,0]+x[:,1]*x[:,1])+x[:,2]*x[:,2]
        slope=numerator/denominator.clamp_min(1e-30)
        position=mq+slope*((target-times[:,0]).double()*1e-6-mt)
        return position.float(),slope.float(),denominator>1e-12

    @staticmethod
    def append(storage,value,mask):
        shifted=torch.cat((storage[:,1:],value[:,None]),dim=1)
        storage.copy_(torch.where(mask[:,None],shifted,storage))

    def push(self,time,q,read_ok,mask=None):
        time=torch.as_tensor(time,device=self.device,dtype=torch.int64).expand(self.n)
        if mask is None:mask=torch.ones(self.n,device=self.device,dtype=torch.bool)
        finite=torch.isfinite(q)
        bad=mask&(~finite|(time<=0))
        self.driver_count.masked_fill_(bad,0)
        self.driver_time.masked_fill_(bad[:,None],0)
        self.driver_position.masked_fill_(bad[:,None],0.)
        new=mask&finite&(time>0)&((self.driver_count==0)|(time>self.driver_time[:,-1]))
        gap=new&(self.driver_count>0)&(time-self.driver_time[:,-1]>25000)
        self.driver_count.masked_fill_(gap,0)
        self.append(self.driver_time,time,new)
        self.append(self.driver_position,q,new)
        self.driver_count.copy_(torch.where(new,(self.driver_count+1).clamp_max(3),self.driver_count))
        _,velocity,den_ok=self.fit(self.driver_time,self.driver_position,time)
        ready=new&(self.driver_count>=3)&den_ok&torch.isfinite(velocity)&(velocity.abs()<=40)
        self.velocity_ready.copy_(torch.where(mask,ready,self.velocity_ready))
        self.current_velocity.copy_(torch.where(ready,velocity,self.current_velocity))
        # The UART stores explicit invalid received points too, but ignores
        # non-increasing history timestamps exactly as md_history_push does.
        append=mask&(time>0)&((self.times[:,-1]==0)|(time>self.times[:,-1]))
        self.append(self.times,time,append)
        self.append(self.positions,q,append)
        self.append(self.valid,ready&read_ok&finite,append)

    def query(self,target,wall):
        target=torch.as_tensor(target,device=self.device,dtype=torch.int64).expand(self.n)
        wall=torch.as_tensor(wall,device=self.device,dtype=torch.int64).expand(self.n)
        indices=torch.arange(8,device=self.device)[None]
        available=(self.times>0)&(self.times<=wall[:,None])
        first=torch.where(available,indices,8).amin(-1)
        newest=torch.where(available,indices,-1).amax(-1)
        last=torch.where(available&(self.times>=target[:,None]),indices,8).amin(-1)
        last=torch.where(last==8,newest,last)
        last=torch.maximum(last,first+2).clamp(2,7)
        selection=last[:,None]+torch.arange(-2,1,device=self.device)[None]
        times=self.times.gather(1,selection)
        q=self.positions.gather(1,selection)
        position,velocity,den_ok=self.fit(times,q,target)
        intervals=times[:,1:]-times[:,:-1]
        valid=(available.sum(-1)>=3)&(target>0)&(target<=wall)
        valid&=target>=self.times[self._rows,first.clamp_max(7)]
        valid&=target<=self.times[self._rows,newest.clamp_min(0)]
        valid&=self.valid.gather(1,selection).all(-1)&torch.isfinite(q).all(-1)
        valid&=((intervals>0)&(intervals<=25000)).all(-1)&den_ok
        valid&=torch.isfinite(position)&torch.isfinite(velocity)&(velocity.abs()<=40)
        return torch.stack((position,velocity),-1),valid,times[:,-1]-times[:,0]
