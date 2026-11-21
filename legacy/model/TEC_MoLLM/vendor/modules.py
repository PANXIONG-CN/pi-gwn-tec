import os
import math
import torch
import torch.nn as nn
from lib.runtime_options import runtime_option

try:
    from transformers import AutoModel  # type: ignore
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, ks=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, padding=(k - 1) // 2),
                nn.GroupNorm(1, out_ch),
                nn.GELU(),
            )
            for k in ks
        ])
        self.final = nn.Conv1d(out_ch * len(ks), out_ch, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat([m(x) for m in self.convs], dim=1)
        return self.final(y)


class MultiScaleConvEmbedder(nn.Module):
    def __init__(self, in_ch: int, ch_list=(64, 128), strides=(2, 2)):
        super().__init__()
        assert len(ch_list) == len(strides)
        layers = []
        cur = in_ch
        for oc, s in zip(ch_list, strides):
            layers.append(MultiScaleConvBlock(cur, oc, s))
            cur = oc
        self.net = nn.Sequential(*layers)
        self.out_ch = cur

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentPatching(nn.Module):
    def __init__(self, latent_dim: int, patch_len: int, d_llm: int):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(latent_dim * patch_len, d_llm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        B, L, D = x.shape
        l = self.patch_len
        if l <= 0:
            l = 1
        P = max(1, L // l)
        x = x[:, : P * l, :].contiguous().view(B, P, l, D).reshape(B, P, l * D)
        return self.proj(x)  # [B, P, d_llm]


class SpatialEncoder(nn.Module):
    """Spatial encoder with optional GATv2 (if torch_geometric is installed).
    Fallback: simple residual + normalized adjacency aggregation.
    """

    def __init__(self, in_ch: int, out_ch: int, heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.use_tg = False
        self.register_buffer('_A_hat', None, persistent=False)
        try:
            from torch_geometric.nn import GATv2Conv  # type: ignore
            self.gat = GATv2Conv(in_ch, out_ch, heads=heads, dropout=dropout, concat=True, add_self_loops=True)
            self.out_dim = out_ch * heads
            self.use_tg = True
        except Exception:
            # Fallback: linear projections + adjacency aggregation
            self.lin_self = nn.Linear(in_ch, out_ch * heads)
            self.lin_neigh = nn.Linear(in_ch, out_ch * heads)
            self.out_dim = out_ch * heads

    def _agg_with_adj(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]; edge_index: [2, E]
        B, N, C = x.shape
        device = x.device
        # Cache A_hat on first use or if device changes
        A_hat = self._A_hat
        if (A_hat is None) or (A_hat.device != device):
            src = edge_index[0]
            dst = edge_index[1]
            idx_i = torch.cat([src, torch.arange(N, device=device)])
            idx_j = torch.cat([dst, torch.arange(N, device=device)])
            val = torch.ones(idx_i.size(0), device=device)
            A = torch.sparse_coo_tensor(torch.stack([idx_i, idx_j]), val, (N, N))
            deg = torch.sparse.sum(A, dim=1).to_dense().clamp(min=1.0)
            inv_deg = 1.0 / deg
            A_dense = A.to_dense()
            A_hat = inv_deg.unsqueeze(1) * A_dense
            self._A_hat = A_hat  # cache
        y = torch.einsum('ij,bjc->bic', A_hat, x)
        return y

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        if self.use_tg:
            B, N, C = x.shape
            y = self.gat(x.reshape(-1, C), edge_index)
            return y.view(B, N, self.out_dim)
        # fallback
        h_self = self.lin_self(x)
        h_nei = self._agg_with_adj(x, edge_index)
        h_nei = self.lin_neigh(h_nei)
        import torch.nn.functional as F
        return F.gelu(h_self + h_nei)


class TinyTransformerEncoder(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj_in = nn.Linear(d_model, d_model)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask=None):
        # inputs_embeds: [B, P, d_model]
        x = self.proj_in(inputs_embeds)
        # Convert mask to src_key_padding_mask if provided (True means to mask)
        pad_mask = None
        if attention_mask is not None:
            pad_mask = (attention_mask == 0)
        out = self.enc(x, src_key_padding_mask=pad_mask)
        return out


class LLMBackbone(nn.Module):
    def __init__(self, layers=3, d_llm=768,
                 lora_r=32, lora_alpha=64, hf_model_name: str = None, cache_dir: str = None, local_only: bool = True, freeze: bool = True,
                 use_lora: bool = False, lora_targets: str = 'c_attn', lora_dropout: float = 0.1):
        super().__init__()
        self.d_llm = d_llm
        self.use_hf = _HAS_TRANSFORMERS
        if use_lora and not self.use_hf:
            raise RuntimeError(
                "[LLMBackbone] use_lora=True 但 transformers 不可用。"
                "请先在环境中安装 transformers 与 peft，并确保 HF 模型已可离线加载。"
            )
        if self.use_hf:
            # prefer gpt2-large by default, load from local cache only (unless caller sets local_only=False)
            name = hf_model_name or os.environ.get('HF_MODEL_NAME', 'gpt2-large')
            cdir = cache_dir or os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
            try:
                # 兼容两种离线布局：
                # 1) 标准 HF cache_dir（内部含 models--*/snapshots/*）
                # 2) 直接把模型文件平铺到某个目录（含 config.json / pytorch_model.bin 等）
                use_cache_dir = True
                model_src = name
                try:
                    if isinstance(name, str) and os.path.isdir(name):
                        model_src = name
                        use_cache_dir = False
                    elif local_only and isinstance(cdir, str) and os.path.isdir(cdir):
                        if os.path.isfile(os.path.join(cdir, 'config.json')):
                            model_src = cdir
                            use_cache_dir = False
                except Exception:
                    # 若检查失败，保守走原始 cache_dir 路径
                    use_cache_dir = True
                    model_src = name

                if use_cache_dir:
                    base = AutoModel.from_pretrained(name, cache_dir=cdir, local_files_only=local_only)
                else:
                    base = AutoModel.from_pretrained(model_src, local_files_only=local_only)
            except Exception as e:
                if use_lora:
                    raise RuntimeError(
                        f"[LLMBackbone] use_lora=True 但 HF 模型离线不可加载：name={name} cache_dir={cdir} local_only={local_only}。"
                        "请先完成 HF 预下载并验证 local_files_only=True 可加载。"
                    ) from e
                try:
                    # fallback to gpt2 base to keep HF path and avoid tiny fallback
                    base = AutoModel.from_pretrained('gpt2', cache_dir=cdir, local_files_only=local_only)
                except Exception:
                    self.use_hf = False
                    base = None
            if self.use_hf and base is not None:
                # Keep only first `layers`
                try:
                    base.h = nn.ModuleList(list(base.h)[:layers])
                except Exception:
                    # some architectures may not expose .h; ignore layer slicing
                    pass
                # Reduce memory: disable KV cache; gradient checkpointing opt-in via env
                try:
                    base.config.use_cache = False
                except Exception:
                    pass
                # PIGWN_ENABLE_GC takes precedence over the historical environment key.
                from lib.runtime_options import gradient_checkpointing_enabled
                enable_gc = gradient_checkpointing_enabled()
                try:
                    if enable_gc and hasattr(base, 'gradient_checkpointing_enable'):
                        # Prefer non-reentrant checkpointing for更少的调度开销（新版本Transformers支持）
                        try:
                            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                        except Exception:
                            base.gradient_checkpointing_enable()
                        try:
                            setattr(base.config, 'gradient_checkpointing', True)
                        except Exception:
                            pass
                except Exception:
                    pass
                # Force safe attention implementation to avoid CUDA SDPA flash issues
                # Prefer eager (math) attention in Transformers when available
                try:
                    # Newer Transformers
                    setattr(base.config, 'attn_implementation', 'eager')
                except Exception:
                    pass
                for attr in ['_attn_implementation', '_attn_implementation_internal']:
                    try:
                        setattr(base.config, attr, 'eager')
                    except Exception:
                        pass
                # Optional LoRA injection
                if use_lora:
                    try:
                        from peft import get_peft_model, LoraConfig
                    except Exception as e:
                        raise RuntimeError(
                            "[LLMBackbone] use_lora=True 但 peft 不可用。请先安装 peft 并确保版本兼容。"
                        ) from e
                    targets = [t.strip() for t in str(lora_targets).split(',') if t.strip()]
                    try:
                        lcfg = LoraConfig(
                            r=int(lora_r),
                            lora_alpha=int(lora_alpha),
                            target_modules=targets,
                            lora_dropout=float(lora_dropout),
                            bias='none',
                        )
                        base = get_peft_model(base, lcfg)
                    except Exception as e:
                        raise RuntimeError(
                            f"[LLMBackbone] LoRA 注入失败：targets={targets} r={lora_r} alpha={lora_alpha} dropout={lora_dropout}"
                        ) from e
                    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
                    if int(trainable) <= 0:
                        raise RuntimeError(
                            "[LLMBackbone] LoRA 注入后未发现可训练参数（trainable==0）。"
                            "请检查 lora_targets 是否匹配模型层名，或 peft/transformers 版本是否兼容。"
                        )
                elif freeze:
                    for p in base.parameters():
                        p.requires_grad_(False)
                self.model = base
        if not self.use_hf:
            # Fallback tiny transformer
            self.model = TinyTransformerEncoder(d_model=min(d_llm, 256), nhead=4, num_layers=2, dim_feedforward=512)
            self.in_proj = nn.Linear(d_llm, min(d_llm, 256))
            self.out_proj = nn.Linear(min(d_llm, 256), d_llm)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask=None):
        if self.use_hf:
            # Select SDPA via PIGWN_SDPA: 'math' (default), 'mem', or 'flash'.
            mode = runtime_option('SDPA', 'math').strip().lower()
            # Prefer new torch.nn.attention.sdpa_kernel; fallback to torch.backends.cuda.sdp_kernel
            ctx = None
            try:
                import torch.nn as _nn
                if hasattr(_nn, 'attention') and hasattr(_nn.attention, 'sdpa_kernel'):
                    if mode == 'flash':
                        ctx = _nn.attention.sdpa_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
                    elif mode in ('mem','memory','efficient'):
                        ctx = _nn.attention.sdpa_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)
                    else:
                        ctx = _nn.attention.sdpa_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
            except Exception:
                ctx = None
            if ctx is None:
                try:
                    if mode == 'flash':
                        ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
                    elif mode in ('mem','memory','efficient'):
                        ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)
                    else:
                        ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
                except Exception:
                    class _Noop:
                        def __enter__(self, *a, **kw):
                            return self
                        def __exit__(self, *a, **kw):
                            return False
                    ctx = _Noop()
            with ctx:
                out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            return out.last_hidden_state
        # Fallback path
        z = self.in_proj(inputs_embeds)
        h = self.model(z, attention_mask)
        return self.out_proj(h)


class PredictionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, ratio=4, p=0.1):
        super().__init__()
        hid = max(8, in_dim // ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid), nn.GELU(), nn.Dropout(p), nn.Linear(hid, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x.view(x.size(0), -1))
