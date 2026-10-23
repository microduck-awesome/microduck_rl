"""Physical head step-response and 200 Hz camera metrics, computed on device."""
import math
import torch


def first_time(mask, dt):
    time = torch.arange(1,mask.shape[0]+1,device=mask.device)*dt
    return torch.where(mask,time[:,None,None],float('inf')).amin(0)


def response_metrics(position, initial, target, commanded_step, *, dt=.02, hold_s=.1):
    """All tensors use physical HOME-relative radians, independent of the observer.

    position contains the POST-control physical samples. Delay10 measures 10%
    directed motion from the measured initial pose. Time90 is the beginning of
    a sustained target band (10% of command step, minimum .002 rad), not a single
    transient crossing. A nonresponse remains infinity, never a zero latency.
    """
    active = commanded_step.abs() > 1e-9
    amplitude = commanded_step.abs().clamp_min(1e-9)
    sign = commanded_step.sign()
    progress = sign[None]*(position-initial[None])
    delay = first_time(progress >= .1*amplitude[None],dt)
    error = position-target[None]
    band = error.abs() <= torch.maximum(.1*amplitude,amplitude.new_full((),.002))[None]
    hold = max(1,math.ceil(hold_s/dt))
    attained = band.unfold(0,hold,1).all(-1)
    time90 = first_time(attained,dt)
    tail = error[-max(1,round(.5/dt)):]
    return dict(
        active=active, delay10_s=torch.where(active,delay,float('nan')),
        time90_s=torch.where(active,time90,float('nan')),
        overshoot_percent=torch.where(active,(sign[None]*error).clamp_min(0).amax(0)/amplitude*100,float('nan')),
        steady_signed_error_rad=tail.mean(0),steady_mae_rad=tail.abs().mean(0),
        steady_rmse_rad=tail.square().mean(0).sqrt(),
        initial_error_rad=initial-(target-commanded_step),
        maximum_excursion_rad=(position-initial[None]).abs().amax(0))


def quaternion_distance(a,b):
    """Small-angle stable SO(3) distance, invariant to quaternion sign."""
    a = torch.nn.functional.normalize(a,dim=-1)
    b = torch.nn.functional.normalize(b,dim=-1)
    b = torch.where((a*b).sum(-1,keepdim=True)<0,-b,b)
    return 4*torch.atan2((a-b).norm(dim=-1),(a+b).norm(dim=-1))


def camera_metrics(quaternions, *, dt=.005, exposure_s=.02):
    omega = quaternion_distance(quaternions[1:],quaternions[:-1])/dt
    width = round(exposure_s/dt)+1
    windows = quaternions.unfold(0,width,1).movedim(-1,1)
    span = torch.zeros(windows.shape[0],windows.shape[2],device=quaternions.device)
    # Maximum angular distance inside each exposure; a return to the starting
    # orientation cannot hide motion that happened in the middle of exposure.
    for left in range(width):
        for right in range(left+1,width):
            span = torch.maximum(span,quaternion_distance(windows[:,left],windows[:,right]))
    return dict(angular_speed_rms_rad_s=omega.square().mean(0).sqrt(),
                angular_speed_p95_rad_s=torch.quantile(omega,.95,dim=0),
                exposure_span_p95_rad=torch.quantile(span,.95,dim=0),
                exposure_span_max_rad=span.amax(0))


def summarize_response(metrics):
    def clean(value):
        return [float(x) if math.isfinite(float(x)) else None for x in value.detach().cpu()]
    out = {}
    for name,value in metrics.items():
        if name == 'active':
            continue
        if name in ('delay10_s','time90_s'):
            active = metrics['active']
            out[name.replace('_s','_reached_fraction')] = clean(
                torch.where(active.any(0),torch.isfinite(value).sum(0)/active.sum(0).clamp_min(1),float('nan')))
            # Include nonresponders as infinity in quantiles, reported as null.
            # This avoids reporting a fast median from only the successful tail.
            out[name+'_median_all_trials'] = clean(value.median(0).values)
            ordered = value.sort(dim=0).values
            out[name+'_p95_all_trials'] = clean(ordered[max(0,math.ceil(.95*len(value))-1)])
        else:
            factor = 180/math.pi if name.endswith('_rad') else 1.
            unit_name = name[:-4]+'_deg' if name.endswith('_rad') else name
            out[unit_name+'_mean'] = clean(value.mean(0)*factor)
            out[unit_name+'_p95'] = clean(value.sort(dim=0).values[max(0,math.ceil(.95*len(value))-1)]*factor)
    return out
