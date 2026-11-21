"""Common supervised path; optional training-only empirical regularization."""
import torch
from lib.masked_regression import (build_supervised_loss_policy, build_strict_supervised_loss,
                                   masked_loss_from_callable)
from lib.normalization import MinMax01Scaler

_NATIVE = build_strict_supervised_loss(build_supervised_loss_policy(dataset='gim'), MinMax01Scaler(0,147.1))


def training_loss(pred, target, valid):
    return masked_loss_from_callable(_NATIVE, pred, target, valid,
                                     strict_explicit_mask_only=True)[0]


def objective(pred, target, valid, physics=None, target_drivers=None):
    supervised = training_loss(pred, target, valid)
    if physics is None:
        # Deliberately no target-driver read, physics computation, or RNG call.
        return supervised, {'data': supervised.detach()}
    with torch.autocast(device_type=pred.device.type, enabled=False):
        phys, terms = physics(pred.float(), None, drivers=target_drivers())
        total = supervised + .1 * phys
    if not torch.isfinite(total).all():
        raise FloatingPointError('nonfinite total loss')
    return total, dict(data=supervised.detach(), physics=phys.detach(), **terms)


def optimizer_for(model, physics=None):
    groups=[{'params':list(model.parameters()), 'name':'predictor'}]
    if physics is not None:
        groups.append({'params':list(physics.parameters()), 'name':'physics'})
    ids=[id(p) for g in groups for p in g['params']]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate optimizer ownership')
    return torch.optim.Adam(groups,lr=.003,weight_decay=0.,betas=(.9,.999),eps=1e-8,amsgrad=False,fused=False)


def check_finite(model, physics=None, *, gradients=False):
    for module in [model, physics]:
        if module is None: continue
        for name,p in module.named_parameters():
            value=p.grad if gradients else p
            if gradients and value is None and module is model and name in inactive_predictor_names(model):continue
            if value is None or not torch.isfinite(value).all():
                raise FloatingPointError(('gradient' if gradients else 'parameter')+':'+name)


def clip_gradients(model, physics=None):
    check_finite(model,physics,gradients=True)
    predictor_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
    # Separate norm prevents six learned loss weights shrinking predictor gradients.
    physics_norm=0. if physics is None else torch.nn.utils.clip_grad_norm_(physics.parameters(),5.,error_if_nonfinite=True)
    return float(predictor_norm),float(physics_norm)


def inactive_predictor_names(model):
    # Frozen GWN graph path bypasses residual convolutions; final layer returns
    # its skip before the final graph convolution/BN output reaches the loss.
    if not getattr(model,'gcn_bool',False):return set()
    last=len(model.bn)-1
    return {n for n,_ in model.named_parameters() if n.startswith('residual_convs.')
            or n.startswith('bn.'+str(last)+'.') or n.startswith('gconv.'+str(last)+'.')}
