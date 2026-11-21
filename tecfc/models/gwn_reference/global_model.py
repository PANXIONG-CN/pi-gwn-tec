"""I/O and longitude-quotient adapter around the unchanged global GWNET source."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from model.GWN.GWN import GWNET

HERE=Path(__file__).resolve().parent

class GlobalGWN(nn.Module):
 def __init__(self,device):
  super().__init__()
  with np.load(HERE/'prepared/graph_svd_v1.npz',allow_pickle=False) as z:
   q=sp.csr_matrix((z['data'],z['indices'],z['indptr']),shape=tuple(z['shape'])); assert q.shape==(5112,5112) and q.nnz==45576
   u=torch.from_numpy(z['u'].copy()).to(device);s=torch.from_numpy(z['s'].copy()).to(device);v=torch.from_numpy(z['v'].copy()).to(device)
  args=SimpleNamespace(adj_mx=q.toarray(),num_nodes=5112,dropout=.3,blocks=4,layers=2,gcn_bool=True,addaptadj=True,
   adjtype='doubletransition',randomadj=False,aptonly=False,kernel_size=2,nhid=32,residual_channels=32,
   dilation_channels=32,input_window=12,output_window=12,driver_fusion='none')
  # The source calls a full torch.svd() to obtain its first ten singular
  # triplets. These precomputed ARPACK triplets are the same mathematical
  # rank-10 SVD initialization, with residual and provenance recorded.
  original_svd=torch.svd
  def precomputed_svd(a,*args,**kwargs):
   assert tuple(a.shape)==(5112,5112) and a.device.type==device.type
   return u,s,v
  try:
   torch.svd=precomputed_svd
   self.gwn=GWNET(args,device,dim_in=1,dim_out=1)
  finally:torch.svd=original_svd
 def forward(self,history,restore=True):
  if history.ndim!=4 or tuple(history.shape[1:])!=(12,71,72):raise ValueError('history must be [B,12,71,72]')
  p=self.gwn(history.reshape(history.shape[0],12,5112,1)).reshape(history.shape[0],12,71,72)
  return torch.cat((p,p[...,:1]),dim=-1) if restore else p
