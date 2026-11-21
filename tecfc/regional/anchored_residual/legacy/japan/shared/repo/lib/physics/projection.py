"""One-sided auxiliary gradient projection used by the matched GIM runs."""
import torch

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

