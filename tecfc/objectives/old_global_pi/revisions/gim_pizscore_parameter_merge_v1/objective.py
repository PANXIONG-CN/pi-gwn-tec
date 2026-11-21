"""One forward; unscaled branch VJPs; merge at FP32 leaf parameters."""
import torch
from revisions.gim_pizscore_fixed_recipe_v1.objective import training_loss, clip_gradients

def configure_physics(physics, mode):
    if physics is not None:
        for p in physics.parameters():
            p.requires_grad_(mode == 'learned')
            if mode == 'fixed':
                with torch.no_grad(): p.zero_()

def optimizer_for(model, physics, config):
    groups = [{'params': list(model.parameters()), 'name': 'predictor', 'lr': config['predictor_lr']}]
    if config['physics_mode'] == 'learned':
        groups.append({'params': list(physics.parameters()), 'name': 'physics', 'lr': config['physics_lr']})
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids))
    return torch.optim.Adam(groups, lr=config['predictor_lr'], betas=(.9,.999), eps=1e-8,
                            weight_decay=0., amsgrad=False, fused=False)

def assign_gradients(data, physical, model, physics, *, mode, lam, diagnostics=True):
    params = list(model.parameters())
    gd = torch.autograd.grad(data, params, retain_graph=physical is not None, allow_unused=True)
    gp = (None,) * len(params)
    if physical is not None:
        sp = list(physics.parameters()) if mode == 'learned' else []
        combined = torch.autograd.grad(physical, params + sp, allow_unused=True)
        gp = combined[:len(params)]
        for p, g in zip(sp, combined[len(params):]):
            if g is None: raise ValueError('missing learned log-variance gradient')
            p.grad = g.detach().float().mul(lam)
    # Scalar reductions use FP64; tensors assigned to the original FP32 parameters.
    dd = torch.zeros((), device=params[0].device, dtype=torch.float64)
    pp = dd.clone(); dot = dd.clone(); delta2 = dd.clone()
    changed = torch.zeros((), device=params[0].device, dtype=torch.int64)
    elements = 0
    for p, d, g in zip(params, gd, gp):
        if d is None and g is None:
            p.grad = None
            continue
        assert p.dtype == torch.float32
        if g is None:
            total = d.detach().float()
        elif d is None:
            total = g.detach().float().mul(lam)
        else:
            total = d.detach().float() + lam * g.detach().float()
        p.grad = total
        if diagnostics:
            d64 = d.double() if d is not None else torch.zeros_like(total, dtype=torch.float64)
            g64 = g.double() if g is not None else torch.zeros_like(total, dtype=torch.float64)
            dd += d64.square().sum(); pp += g64.square().sum(); dot += (d64*g64).sum()
            delta = total.double() - d64
            delta2 += delta.square().sum(); changed += torch.count_nonzero(delta); elements += total.numel()
    if not diagnostics: return {}
    nd, np, dp, actual = float(dd.sqrt()), float(pp.sqrt()), float(dot), float(delta2.sqrt())
    return dict(data_gradient_norm=nd, unscaled_physics_gradient_norm=np,
                weighted_physics_gradient_norm=lam*np, physics_data_ratio=lam*np/nd if nd else None,
                parameter_gradient_cosine=dp/(nd*np) if nd and np else None,
                merged_gradient_difference_norm=actual, changed_gradient_elements=int(changed),
                gradient_elements=elements, changed_gradient_fraction=int(changed)/elements if elements else None)

def backward(pred, target, valid, model, physics, drivers, config):
    data = training_loss(pred, target, valid)
    terms = {}; physical = None
    if config['physics_mode'] != 'none':
        with torch.autocast(device_type=pred.device.type, enabled=False):
            aggregate, terms = physics(pred.float(), None, drivers=drivers)
            physical = aggregate if config['physics_mode'] == 'learned' else .5 * sum(terms.values())
    total = data if physical is None else data + config['lambda_phys'] * physical
    if not bool(torch.isfinite(total)): raise FloatingPointError('nonfinite objective')
    stats = assign_gradients(data, physical, model, physics, mode=config['physics_mode'], lam=config['lambda_phys'])
    pn, sn = clip_gradients(model, physics if config['physics_mode'] == 'learned' else None)
    stats.update(predictor_norm_before_clip=pn, predictor_clip_coefficient=min(1.,5./(pn+1e-6)),
                 physics_norm_before_clip=sn, prediction_dtype=str(pred.dtype), gradient_dtype='torch.float32')
    return float(total.detach()), {**{k:float(v.detach()) for k,v in terms.items()},
                                  'data':float(data.detach()), 'physics':float(physical.detach()) if physical is not None else 0.}, stats
