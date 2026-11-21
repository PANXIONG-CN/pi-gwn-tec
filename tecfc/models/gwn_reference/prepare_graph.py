"""Derive the periodic unique-longitude graph and rank-10 SVD initialization."""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

HERE=Path(__file__).resolve().parent
SOURCE=Path(sys.argv[1])  # eight-neighbour 71x73 GIM adjacency (scripts/prepare_assets.py builds it)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(SOURCE)=='c5e278353e30339f66d78ca08e4c29305a435d1f40db2f9ee3d26ecd6883cd00'
a=np.load(SOURCE,mmap_mode='r',allow_pickle=False);assert a.shape==(5183,5183) and a.dtype==np.float32
r,c=np.nonzero(a);assert len(r)==45787 and (a[r,c]==1).all()
r=(r//73)*72+(r%73)%72;c=(c//73)*72+(c%73)%72
q=sp.coo_matrix((np.ones(len(r),np.float32),(r,c)),shape=(5112,5112)).tocsr();q.data[:]=1;q.eliminate_zeros()
assert q.nnz==45576 and (q!=q.T).nnz==0 and (q.diagonal()==1).all()
# Old GWNET asym_adj computes D^-1 A in float32.  Iterative SVD finds its
# leading singular vectors without materializing a 5112 x 5112 dense SVD.
t=sp.diags(1/np.asarray(q.sum(axis=1)).ravel()).dot(q).astype(np.float32).tocsr()
start=time.time()
u,s,vt=svds(t.astype(np.float64),k=10,which='LM',tol=1e-9,maxiter=100000,v0=np.random.default_rng(1947).standard_normal(5112))
order=np.argsort(s)[::-1];u,s,vt=u[:,order],s[order],vt[order]
res=max(np.max(np.linalg.norm(t@vt.T-u*s,axis=0)),np.max(np.linalg.norm(t.T@u-vt.T*s,axis=0)))
assert res<1e-6
out=HERE/'prepared'/'graph_svd_v1.npz';assert not out.exists()
np.savez_compressed(out,indptr=q.indptr,indices=q.indices,data=q.data,shape=np.array(q.shape),u=u.astype(np.float32),s=s.astype(np.float32),v=vt.T.astype(np.float32))
info={'source':str(SOURCE),'source_sha256':sha(SOURCE),'graph_shape':list(q.shape),'graph_nnz':int(q.nnz),'construction':'boolean union of original 73-column graph after identifying columns 0 and 72; periodic 72-longitude graph','svd':'scipy sparse ARPACK rank 10 largest singular triplets of float32 D^-1 A, evaluated in float64; stored float32; deterministic v0 seed 1947','singular_values':s.tolist(),'max_residual_norm':float(res),'elapsed_seconds':time.time()-start,'output_sha256':sha(out)}
(HERE/'prepared'/'GRAPH_IDENTITY.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info),flush=True)
