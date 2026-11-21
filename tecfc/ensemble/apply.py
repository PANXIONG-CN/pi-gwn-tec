"""Check frozen fusion, mapping and gates after all member parity checks."""
import json
from pathlib import Path
import numpy as np
from tecfc.scoring.confirm_records import SEEDS, sha, read, write, now


def mapping(lat,lon):
    la,lo=np.meshgrid(lat,lon,indexing='ij');yy=(87.5-la.ravel().astype(float))/2.5;xx=(lo.ravel().astype(float)+180)/5
    iy=yy.astype(int);ix=xx.astype(int);dy=yy-iy;dx=xx-ix
    return np.stack([iy*73+ix,iy*73+ix+1,(iy+1)*73+ix,(iy+1)*73+ix+1]),np.stack([(1-dy)*(1-dx),(1-dy)*dx,dy*(1-dx),dy*dx])


def remap(x,mp):return sum(np.asarray(x[...,j],np.float64)*w for j,w in zip(*mp))


def fuse(means,frozen):
    # Preserve the original 13-family axis including zero coefficients.
    p=np.stack([means[f] if f in means else np.zeros_like(next(iter(means.values()))) for f in frozen['family_order']])
    w=np.asarray(frozen['full_validation_weights'],float)
    return np.einsum('lf,fln->ln',w,p.reshape(len(p),12,-1)).reshape(12,71,73).astype(np.float32)
