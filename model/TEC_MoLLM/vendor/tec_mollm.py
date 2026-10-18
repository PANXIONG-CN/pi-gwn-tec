import torch
import torch.nn as nn
from lib.runtime_options import runtime_option
from .modules import SpatialEncoder, MultiScaleConvEmbedder, LatentPatching, LLMBackbone, PredictionHead


class TEC_MoLLM(nn.Module):
    def __init__(
        self,
        num_nodes=5183,
        d_emb=0,
        spatial_in_base=1,
        spatial_out=11,
        heads=2,
        temporal_channels=(64, 128),
        temporal_strides=(2, 2),
        patch_len=4,
        d_llm=768,
        llm_layers=3,
        pred_horizon=12,
        temporal_seq_len=12,
        node_chunk=512,
        use_ln: bool = False,
        hf_model_name: str = None,
        hf_cache_dir: str = None,
        use_node_tod_emb: bool = False,
        d_e: int = 16,
        tod_bins: int = 12,
        use_lora: bool = False,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_targets: str = 'c_attn',
        dropout_after_llm: float = 0.1,
    ):
        super().__init__()
        self.spatial = SpatialEncoder(in_ch=spatial_in_base + d_emb, out_ch=spatial_out, heads=heads)
        t_in = self.spatial.out_dim
        self.temporal = MultiScaleConvEmbedder(in_ch=t_in, ch_list=temporal_channels, strides=temporal_strides)
        conv_out_len = temporal_seq_len
        for s in temporal_strides:
            conv_out_len //= s
        if conv_out_len <= 0:
            conv_out_len = 1
        if patch_len <= 0 or (conv_out_len % patch_len) != 0:
            patch_len = 1
        self.patch = LatentPatching(latent_dim=self.temporal.out_ch, patch_len=patch_len, d_llm=d_llm)
        self.llm = LLMBackbone(layers=llm_layers, d_llm=d_llm,
                               hf_model_name=hf_model_name, cache_dir=hf_cache_dir, local_only=True,
                               freeze=not use_lora,
                               use_lora=use_lora, lora_r=lora_r, lora_alpha=lora_alpha,
                               lora_targets=lora_targets, lora_dropout=0.1)
        num_patches = max(1, conv_out_len // patch_len)
        self.head = PredictionHead(in_dim=d_llm * num_patches, out_dim=pred_horizon)
        self.num_nodes = num_nodes
        self.pred_horizon = pred_horizon
        self.node_chunk = max(1, int(node_chunk))
        self.use_ln = bool(use_ln)
        # light embeddings
        self.use_node_tod_emb = bool(use_node_tod_emb)
        self.tod_bins = int(tod_bins)
        self.d_e = int(d_e)
        if self.use_node_tod_emb:
            self.node_emb = nn.Embedding(num_nodes, self.d_e)
            self.tod_emb = nn.Embedding(self.tod_bins, self.d_e)
        self.dropout_after_llm = float(dropout_after_llm)
        if self.use_ln:
            self.ln = nn.LayerNorm(self.spatial.out_dim)

    def forward(self, x, tod, edge_index):
        # x: [B, L, N, C=1]; tod: [B, L, N, 1] (unused in minimal variant)
        B, L, N, C = x.shape
        # Spatial per-timestep: [B,L,N,C] -> [L,B,N,C] -> [L*B,N,C]
        x_4g = x.permute(1, 0, 2, 3).reshape(-1, N, C)
        x_in = x_4g
        # light Node+TOD embedding concat
        if self.use_node_tod_emb:
            nodes = torch.arange(self.num_nodes, device=x.device)
            e_node = self.node_emb(nodes).view(1, N, -1).repeat(L*B, 1, 1)
            steps = torch.arange(L, device=x.device)
            tod_idx = (steps % self.tod_bins).repeat_interleave(B)
            e_pos = self.tod_emb(tod_idx).view(L*B, 1, -1).expand(L*B, N, -1)
            x_in = torch.cat([x_in, e_node, e_pos], dim=-1)
        s = self.spatial(x_in, edge_index)  # [L*B, N, Cs]
        # Equal-dim residual only
        if s.size(-1) == x_in.size(-1):
            s = s + x_in
        # else: residual disabled (dims mismatch)
        # Back to [B,N,L,Cs]
        Cs = s.size(-1)
        s = s.view(L, B, N, Cs).permute(1, 2, 0, 3)
        # Temporal + LLM + head with node chunking to limit peak memory
        L_out = self.pred_horizon
        y_all = x.new_zeros((B, L_out, N, 1))
        # Optional safety cap on B*Nc via PIGWN_NODE_SAFE_CAP.
        # If not set, use user-configured node_chunk directly for max throughput.
        safe_cap_env = runtime_option('NODE_SAFE_CAP').strip()
        chunk = int(self.node_chunk)
        if safe_cap_env:
            try:
                safe_cap = int(safe_cap_env)
                if safe_cap > 0:
                    max_nc = max(1, safe_cap // max(1, B))
                    chunk = min(chunk, max_nc)
            except Exception:
                pass
        for start in range(0, N, chunk):
            end = min(N, start + chunk)
            s_ch = s[:, start:end]  # [B, Nc, L, Cs]
            if hasattr(self, 'ln'):
                s_ch = self.ln(s_ch)
            Bc, Nc = B, (end - start)
            t = s_ch.reshape(-1, L, Cs).transpose(1, 2)  # [B*Nc, Cs, L]
            t = self.temporal(t).transpose(1, 2)        # [B*Nc, L', Ct]
            p = self.patch(t)                           # [B*Nc, P, d_llm]
            attn_mask = torch.ones(p.shape[:-1], device=p.device, dtype=torch.long)
            h = self.llm(inputs_embeds=p, attention_mask=attn_mask)
            import torch.nn.functional as F
            if self.dropout_after_llm and self.dropout_after_llm > 0:
                h = F.dropout(h, p=self.dropout_after_llm, training=self.training)
            y = self.head(h)                            # [B*Nc, L_out]
            y = y.view(Bc, -1, Nc, 1)                  # [B, L_out, Nc, 1]
            y_all[:, :, start:end, :] = y
        return y_all
