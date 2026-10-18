import os
import torch
import torch.nn as nn


def _to_edge_index_from_adj(A: torch.Tensor) -> torch.Tensor:
    """Convert dense adjacency [N,N] -> edge_index [2,E] (undirected)."""
    N = A.size(0)
    idx = (A > 0).nonzero(as_tuple=False)
    if idx.numel() == 0:
        # Self-loops fallback
        idx = torch.arange(N, device=A.device)
        return torch.stack([idx, idx], dim=0)
    return idx.t().contiguous()


class TEC_MoLLM(nn.Module):
    """
    适配 Vendor TEC_MoLLM 的轻量封装，使之符合 统一预测器接口：
    - __init__(args_predictor, device, dim_in, dim_out)
    - forward(x) 接收 [B,L,N,dim_in]，输出 [B,H,N,dim_out]
    其中 dim_in=1, dim_out=1 对 GIMtec 有效。
    """

    def __init__(self, args_predictor, device, dim_in, dim_out):
        super().__init__()
        self.device = device
        self.dim_in = dim_in
        self.dim_out = dim_out

        # 只用本地 vendor 适配（去除任何绝对路径依赖）
        from model.TEC_MoLLM.vendor.tec_mollm import TEC_MoLLM as _VendorTEC
        # 邻接矩阵（71x73 网格 8 邻接），统一由 lib/datasets/gimtec_adj 提供
        try:
            from lib.datasets.gimtec_adj import load_or_build_adj
            import numpy as np
            A_np, _ = load_or_build_adj(dataset='GIMtec', num_nodes=args_predictor.num_nodes,
                                        graph_tag=getattr(args_predictor, 'graph_tag', 'grid8'))
            A = torch.from_numpy(A_np.astype('float32'))
        except Exception:
            # 兜底：稀疏自环
            A = torch.eye(args_predictor.num_nodes, dtype=torch.float32)
        self.register_buffer('edge_index', _to_edge_index_from_adj(A))

        # 将字符串形的元组转为真正的 tuple
        def _as_tuple(s, default):
            try:
                return tuple(eval(s)) if isinstance(s, str) else tuple(s)
            except Exception:
                return default

        temporal_channels = _as_tuple(getattr(args_predictor, 'temporal_channels', (64, 128)), (64, 128))
        temporal_strides = _as_tuple(getattr(args_predictor, 'temporal_strides', (2, 2)), (2, 2))
        use_node_tod_emb = bool(getattr(args_predictor, 'use_node_tod_emb', True))
        d_e = int(getattr(args_predictor, 'd_e', 16))
        tod_bins = int(getattr(args_predictor, 'tod_bins', 12))
        d_emb_total = (2 * d_e) if use_node_tod_emb else 0

        self.model = _VendorTEC(
            num_nodes=args_predictor.num_nodes,
            d_emb=d_emb_total,
            spatial_in_base=self.dim_in,
            spatial_out=int(getattr(args_predictor, 'spatial_out', 16)),
            heads=int(getattr(args_predictor, 'heads', 2)),
            temporal_channels=temporal_channels,
            temporal_strides=temporal_strides,
            patch_len=int(getattr(args_predictor, 'patch_len', 4)),
            d_llm=int(getattr(args_predictor, 'd_llm', 768)),
            llm_layers=int(getattr(args_predictor, 'llm_layers', 3)),
            pred_horizon=int(getattr(args_predictor, 'output_window', 12)),
            temporal_seq_len=int(getattr(args_predictor, 'input_window', 12)),
            node_chunk=int(getattr(args_predictor, 'node_chunk', 512)),
            use_ln=bool(getattr(args_predictor, 'use_ln', False)),
            hf_model_name=str(getattr(args_predictor, 'hf_model_name', 'gpt2-large')),
            hf_cache_dir=str(getattr(args_predictor, 'hf_cache_dir', os.path.expanduser('~/.cache/huggingface'))),
            use_node_tod_emb=use_node_tod_emb,
            d_e=d_e,
            tod_bins=tod_bins,
            use_lora=bool(getattr(args_predictor, 'use_lora', False)),
            lora_r=int(getattr(args_predictor, 'lora_r', 32)),
            lora_alpha=int(getattr(args_predictor, 'lora_alpha', 64)),
            lora_targets=str(getattr(args_predictor, 'lora_targets', 'c_attn')),
            dropout_after_llm=float(getattr(args_predictor, 'dropout_after_llm', 0.1)),
        )

        self.to(self.device)
        self.model.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, N, dim_in] -> 仅取基础通道
        if x.size(-1) > 1:
            x = x[..., :1]
        tod = None  # 最小实现：不使用时间编码
        y = self.model(x, tod, self.edge_index)
        # 输出: [B, H, N, 1] -> 若 dim_out>1，仅重复到 dim_out（一般为 1）
        if self.dim_out > 1:
            y = y.repeat(1, 1, 1, self.dim_out)
        return y
