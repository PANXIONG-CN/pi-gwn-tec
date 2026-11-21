"""Causal inference input construction; no fitting and no target errors."""
import importlib.util
import sys
from pathlib import Path
import numpy as np

from tecfc.features import combined_inputs as core


def global_input(day, family, day_reader, climate, drivers=None):
    day = np.datetime64(day, 'D'); previous = day-np.timedelta64(1, 'D')
    origin = int(day.astype('datetime64[s]').astype(np.int64))
    history = day_reader(str(previous))
    assert history.shape == (12, 71, 72) and np.isfinite(history).all()
    x = np.array(history/np.float32(147.1), dtype=np.float32)
    if family in ['G','PI','EMP']:
        return x
    if family in ['C','C-in','C-plus']:
        if family == 'C':
            background, geom = np.zeros_like(x), np.zeros_like(x)
        else:
            cl = climate[int(str(previous)[5:7])-1]; valid = np.isfinite(history)
            numerator = history[valid].sum(dtype=np.float64)
            denominator = cl[valid].sum(dtype=np.float64)
            assert np.isfinite(denominator) and denominator>0
            alpha = np.float32(np.clip(numerator/denominator,.3,3.))
            background = np.array(cl*alpha/np.float32(147.1), np.float32)
            geom = core.c16.solar(previous)
        x = np.stack((x,background,geom), axis=1)
        if family == 'C-plus':
            extra = [day_reader(str(day-np.timedelta64(lag,'D')))/np.float32(147.1) for lag in [2,3,27]]
            x = np.concatenate((x,np.stack(extra,axis=1)),axis=1)
    elif family == 'G-geo':
        x = np.concatenate((x[...,None],core.f15.geometry(core.f15.history_times(str(day)))),axis=-1)
    elif family == 'G-DE':
        assert drivers is not None
        x = np.concatenate((x[...,None],drivers.standardized(core.f15.history_times(str(day)))),axis=-1)
    elif family in ['G-long','G-plus-M','G-plus-S']:
        parts = [x]+[day_reader(str(day-np.timedelta64(lag,'D')))/np.float32(147.1) for lag in [2,3,27]]
        x = np.stack(parts,axis=-1)
        if family != 'G-long':
            x = np.concatenate((x,core.f15.geometry(core.f15.history_times(str(day)))),axis=-1)
    else:
        raise ValueError(family)
    assert np.isfinite(x).all()
    assert np.max(core.f15.history_times(str(day))) <= origin-7200
    return x


def regional_history(raw, valid, fallback):
    assert raw.shape == valid.shape and raw.shape[0] == 89 and valid.dtype == bool
    result = raw.copy(); last = np.full(raw.shape[1],-10000); value = np.zeros(raw.shape[1],np.float32)
    for k in range(89):
        last[valid[k]] = k; value[valid[k]] = raw[k,valid[k]]
        recent = ~valid[k] & ((k-last)*900 <= 21600)
        result[k,recent] = value[recent]
        other = ~valid[k] & ~recent
        result[k,other] = fallback[k//8,other]
    assert np.isfinite(result).all()
    return result


def module(path, name):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
