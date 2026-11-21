"""Unscaled VJPs from one shared forward; only the auxiliary is projected."""
import torch
from revisions.gim_pizscore_parameter_merge_v1.objective import configure_physics, optimizer_for
from revisions.gim_pizscore_fixed_recipe_v1.objective import training_loss, clip_gradients

@torch.no_grad()
def merge(params, gd, gx, rho):
    assert len(params) == len(gd) == len(gx) and rho >= 0
    # Direct original supervised assignment preserves unused-parameter semantics.
    if rho == 0:
        for p, d in zip(params, gd):
            p.grad = None if d is None else d.detach().float()
        return dict(rho=0., auxiliary_disabled=True)
    dd = torch.zeros((), dtype=torch.float64, device=params[0].device)
    xx, dx = dd.clone(), dd.clone()
    for d, x in zip(gd, gx):
        if d is not None: dd += d.detach().double().square().sum()
        if x is not None: xx += x.detach().double().square().sum()
        if d is not None and x is not None: dx += (d.detach().double()*x.detach().double()).sum()
    if not bool(torch.isfinite(torch.stack((dd,xx,dx))).all()):
        raise FloatingPointError('nonfinite gradient reduction')
    active = bool(dd > 0) and bool(xx > 0)
    a = rho*torch.sqrt(dd/xx) if active else dd.new_zeros(())
    c = torch.minimum(a*dx, dd.new_zeros(()))/dd if active else dd.new_zeros(())
    hh, dh, actual = dd.new_zeros(()), dd.new_zeros(()), dd.new_zeros(())
    for p, d, x in zip(params, gd, gx):
        assert p.dtype == torch.float32
        if d is None and x is None:
            p.grad = None
            continue
        d64 = d.detach().double() if d is not None else torch.zeros_like(p, dtype=torch.float64)
        x64 = x.detach().double() if x is not None else torch.zeros_like(p, dtype=torch.float64)
        h = a*x64-c*d64 if active else torch.zeros_like(d64)
        p.grad = (d64+h).float()
        hh += h.square().sum(); dh += (d64*h).sum()
        actual += (p.grad.double()-d64).square().sum()
    nd, nx = float(dd.sqrt()), float(xx.sqrt())
    return dict(rho=rho, data_gradient_norm=nd, auxiliary_gradient_norm=nx, a=float(a), c=float(c),
                cosine=float(dx/(dd*xx).sqrt()) if active else None,
                projection_triggered=bool(c < 0), removed_component_norm=float(-c*dd.sqrt()),
                auxiliary_ratio_before=float(a*xx.sqrt()/dd.sqrt()) if nd else 0.,
                auxiliary_ratio_after=float(hh.sqrt()/dd.sqrt()) if nd else 0.,
                data_dot_auxiliary_after=float(dh), auxiliary_norm_after=float(hh.sqrt()),
                fp32_writeback_difference_norm=float(actual.sqrt()))

def assign_gradients(data, physical, params, rho):
    enabled = rho != 0 and physical is not None
    gd = torch.autograd.grad(data, params, retain_graph=enabled, allow_unused=True)
    gx = torch.autograd.grad(physical, params, allow_unused=True) if enabled else (None,)*len(params)
    return merge(params, gd, gx, rho)

def backward(pred, target, valid, model, physics, drivers, config):
    data = training_loss(pred, target, valid)
    terms = {}; physical = None
    if config['physics_mode'] == 'fixed':
        with torch.autocast(device_type=pred.device.type, enabled=False):
            _, terms = physics(pred.float(), None, drivers=drivers)
            physical = .5*sum(terms.values())
    if not bool(torch.isfinite(data)) or (physical is not None and not bool(torch.isfinite(physical))):
        raise FloatingPointError('nonfinite objective')
    stats = assign_gradients(data, physical, list(model.parameters()), config['rho'])
    pn, _ = clip_gradients(model, None)
    stats.update(predictor_norm_before_clip=pn, predictor_clip_coefficient=min(1.,5./(pn+1e-6)),
                 prediction_dtype=str(pred.dtype), gradient_dtype='torch.float32',
                 data=float(data.detach()), auxiliary=float(physical.detach()) if physical is not None else 0.,
                 terms={k:float(v.detach()) for k,v in terms.items()})
    return stats
