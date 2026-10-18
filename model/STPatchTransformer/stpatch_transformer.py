from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class STPatchTransformer(nn.Module):
    """Space-time patch Transformer baseline (non-HF, lightweight encoder).

    Input : [B, T_in, N, C] (C can be 1 or 1+5 drivers if enabled upstream)
    Output: [B, T_out, N, D]
    """

    def __init__(self, args, device: str, dim_in: int, dim_out: int):
        super().__init__()
        self.device = device

        self.grid_h = int(getattr(args, "grid_h", 0))
        self.grid_w = int(getattr(args, "grid_w", 0))
        self.num_nodes = int(getattr(args, "num_nodes", 0))
        if self.grid_h <= 0 or self.grid_w <= 0:
            raise ValueError("STPatchTransformer requires grid_h/grid_w in config (see conf/STPatchTransformer/*.conf)")
        if (self.grid_h * self.grid_w) != self.num_nodes:
            raise ValueError(f"grid_h*grid_w != num_nodes: {self.grid_h}*{self.grid_w} != {self.num_nodes}")

        self.input_window = int(getattr(args, "input_window", 12))
        self.output_window = int(getattr(args, "output_window", 12))

        self.space_patch = max(1, int(getattr(args, "space_patch", 8)))
        self.time_patch = max(1, int(getattr(args, "time_patch", 2)))

        self.d_model = int(getattr(args, "d_model", 192))
        self.n_layers = int(getattr(args, "n_layers", 6))
        self.n_heads = int(getattr(args, "n_heads", 6))
        self.dropout = float(getattr(args, "dropout", 0.1))
        self.mlp_ratio = float(getattr(args, "mlp_ratio", 4.0))
        self.attention_mode = str(getattr(args, "attention_mode", "global"))
        self.local_radius = int(getattr(args, "local_radius", 1))
        if self.attention_mode not in {"global", "local_mask"}:
            raise ValueError(f"Unsupported attention_mode: {self.attention_mode}")
        if self.local_radius < 0:
            raise ValueError(f"local_radius must be >= 0, got {self.local_radius}")

        self.dim_in = int(dim_in)
        self.dim_out = int(dim_out)

        # Padded grid dims for patching
        self.h_pad = int(math.ceil(self.grid_h / self.space_patch) * self.space_patch)
        self.w_pad = int(math.ceil(self.grid_w / self.space_patch) * self.space_patch)
        self.hp = int(self.h_pad // self.space_patch)
        self.wp = int(self.w_pad // self.space_patch)
        self.n_spatial = int(self.hp * self.wp)

        self.t_pad = int(math.ceil(self.input_window / self.time_patch) * self.time_patch)
        self.n_time = int(self.t_pad // self.time_patch)
        self.seq_len = int(self.n_time * self.n_spatial)
        if self.seq_len > 1024:
            raise ValueError(f"STPatchTransformer seq_len={self.seq_len} exceeds 1024; adjust patch sizes.")

        patch_vec_dim = int(self.time_patch * self.space_patch * self.space_patch * self.dim_in)
        self.patch_proj = nn.Linear(patch_vec_dim, self.d_model, bias=True)

        self.time_pos = nn.Embedding(self.n_time, self.d_model)
        self.space_pos = nn.Embedding(self.n_spatial, self.d_model)
        self.emb_drop = nn.Dropout(self.dropout)

        ffn_dim = int(self.d_model * self.mlp_ratio)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=ffn_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=self.n_layers, norm=nn.LayerNorm(self.d_model))

        # Predict all horizon steps from the last time-patch tokens (one token per spatial patch)
        patch_area = int(self.space_patch * self.space_patch)
        self.head = nn.Linear(self.d_model, self.output_window * patch_area * self.dim_out, bias=True)

        # Precompute positional indices for [time-major] token layout:
        # [t0: all patches], [t1: all patches], ...
        time_ids = torch.arange(self.n_time, dtype=torch.long).unsqueeze(1).repeat(1, self.n_spatial).reshape(-1)
        space_ids = torch.arange(self.n_spatial, dtype=torch.long).repeat(self.n_time)
        self.register_buffer("_token_time_ids", time_ids, persistent=False)
        self.register_buffer("_token_space_ids", space_ids, persistent=False)
        if self.attention_mode == "local_mask":
            self.register_buffer("_attn_mask", self._build_local_attention_mask(), persistent=False)
        else:
            self._attn_mask = None

    def _build_local_attention_mask(self) -> torch.Tensor:
        """Restrict attention to a spatially local patch neighborhood.

        Tokens remain fully connected across time, but each token only attends to
        patches within a Chebyshev-radius neighborhood on the patch grid. Radius=1
        corresponds to an 8-neighbor patch window, mirroring the repo's grid-local
        GWN prior while keeping the Transformer backbone intact.
        """
        patch_rows = torch.arange(self.hp, dtype=torch.long).unsqueeze(1).repeat(1, self.wp).reshape(-1)
        patch_cols = torch.arange(self.wp, dtype=torch.long).repeat(self.hp)
        row_delta = (patch_rows[:, None] - patch_rows[None, :]).abs()
        col_delta = (patch_cols[:, None] - patch_cols[None, :]).abs()
        allow_spatial = torch.maximum(row_delta, col_delta) <= int(self.local_radius)
        token_allow = allow_spatial.index_select(0, self._token_space_ids).index_select(1, self._token_space_ids)
        return ~token_allow

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        # source: [B, T, N, C]
        if not isinstance(source, torch.Tensor) or source.dim() != 4:
            raise ValueError(f"STPatchTransformer expects 4D tensor [B,T,N,C], got: {type(source)} {getattr(source,'shape',None)}")
        source = source.contiguous()
        B, T, N, C = source.shape
        if int(N) != int(self.num_nodes):
            raise ValueError(f"num_nodes mismatch: N={int(N)} vs cfg={self.num_nodes}")
        if int(T) != int(self.input_window):
            # Keep strict to avoid silent train/test protocol drift.
            raise ValueError(f"input_window mismatch: T={int(T)} vs cfg={self.input_window}")
        if int(C) != int(self.dim_in):
            raise ValueError(f"dim_in mismatch: C={int(C)} vs ctor={self.dim_in}")

        # [B,T,C,H,W]
        x = source.view(B, T, self.grid_h, self.grid_w, C).permute(0, 1, 4, 2, 3).contiguous()

        # Spatial pad on the right/bottom (padded nodes will be cropped out from outputs)
        if self.h_pad != self.grid_h or self.w_pad != self.grid_w:
            x = F.pad(x, (0, self.w_pad - self.grid_w, 0, self.h_pad - self.grid_h))

        # Time pad on the left (keep the newest frames aligned)
        if self.t_pad != T:
            pad_t = int(self.t_pad - T)
            if pad_t > 0:
                zeros = torch.zeros(B, pad_t, C, self.h_pad, self.w_pad, device=x.device, dtype=x.dtype)
                x = torch.cat([zeros, x], dim=1)

        # Patchify: [B, n_time, n_spatial, patch_vec_dim]
        x = x.view(B, self.n_time, self.time_patch, C, self.h_pad, self.w_pad)
        x = x.view(B, self.n_time, self.time_patch, C, self.hp, self.space_patch, self.wp, self.space_patch)
        x = x.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()  # [B,nt,hp,wp,tp,ps,ps,C]
        x = x.view(B, self.n_time, self.n_spatial, -1)

        tok = self.patch_proj(x)  # [B,nt,ns,d_model]
        tok = tok.view(B, self.seq_len, self.d_model)
        tok = tok + self.time_pos(self._token_time_ids) + self.space_pos(self._token_space_ids)
        tok = self.emb_drop(tok)

        attn_mask = self._attn_mask if self.attention_mode == "local_mask" else None
        enc = self.encoder(tok, mask=attn_mask)  # [B,seq,d_model]

        # Use only the last time-patch tokens for prediction (most recent context).
        start = (self.n_time - 1) * self.n_spatial
        last = enc[:, start : start + self.n_spatial, :]  # [B,ns,d_model]

        y_patch = self.head(last)  # [B,ns,T_out*patch_area*D]

        # Unpatchify: [B,T_out,H_pad,W_pad,D] -> crop -> [B,T_out,N,D]
        ps = self.space_patch
        patch_area = ps * ps
        y_patch = y_patch.view(B, self.n_spatial, self.output_window, patch_area, self.dim_out)
        y_patch = y_patch.view(B, self.hp, self.wp, self.output_window, ps, ps, self.dim_out)
        y_grid = y_patch.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, self.output_window, self.h_pad, self.w_pad, self.dim_out)
        y_grid = y_grid[:, :, : self.grid_h, : self.grid_w, :].contiguous()
        return y_grid.reshape(B, self.output_window, self.num_nodes, self.dim_out)
