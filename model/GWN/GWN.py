import torch
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

from lib.physics.driver_scaler import (
    DRIVER_CHANNEL_ORDER,
    RAW_SCALING,
    TRAIN_ZSCORE,
    DriverHistoryBatch,
)

try:
    from model.GWN.driver_fusion import (
        HistoryFiLM, HistoryResidualGate,
        align_film_driver_history,
        required_film_config,
    )
except ImportError:  # pragma: no cover - compatibility with execution from model/
    from GWN.driver_fusion import HistoryFiLM, HistoryResidualGate, align_film_driver_history, required_film_config


def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    lap = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(lap, 1, which='LM')
        lambda_max = lambda_max[0]
    lap = sp.csr_matrix(lap)
    m, _ = lap.shape
    identity = sp.identity(m, format='csr', dtype=lap.dtype)
    lap = (2 / lambda_max * lap) - identity
    return lap.astype(np.float32).todense()


class NConv(nn.Module):
    def __init__(self):
        super(NConv, self).__init__()

    def forward(self, x, adj):
        x = torch.einsum('ncvl,vw->ncwl', (x, adj))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(Linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = NConv()
        c_in = (order*support_len+1)*c_in
        self.mlp = Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GWNET(nn.Module):
    def __init__(self, args, device, dim_in, dim_out):
        super(GWNET, self).__init__()
        self.adj_mx = args.adj_mx
        self.num_nodes = args.num_nodes
        self.feature_dim = dim_in

        self.dropout = args.dropout
        self.blocks = args.blocks
        self.layers =  args.layers
        self.gcn_bool = args.gcn_bool
        self.addaptadj = args.addaptadj
        self.adjtype = args.adjtype
        self.randomadj = args.randomadj
        self.aptonly = args.aptonly
        self.kernel_size = args.kernel_size
        self.nhid = args.nhid
        self.residual_channels = args.residual_channels
        self.dilation_channels = args.dilation_channels
        self.skip_channels = self.nhid * 8
        self.end_channels = self.nhid * 16
        self.input_window = args.input_window
        self.output_window = args.output_window
        self.output_dim = dim_out
        self.device = device

        requested_driver_fusion = getattr(args, 'driver_fusion', None)
        self.driver_fusion_mode = str(requested_driver_fusion or 'none').strip().lower()
        valid_driver_fusions = {'none', 'raw_concat', 'zscore_concat', 'film', 'residual_gate', 'causal_crossattn'}
        if self.driver_fusion_mode not in valid_driver_fusions:
            raise ValueError(
                f"unsupported driver_fusion={self.driver_fusion_mode!r}; "
                f"expected one of {sorted(valid_driver_fusions)}"
            )
        self.expected_driver_source_sha256 = None
        self.expected_driver_scaler_sha256 = None
        self.expected_driver_scaling = None
        self.source_feature_dim = self.feature_dim
        self.input_ablation = getattr(args, 'input_mask', None)
        if self.driver_fusion_mode != 'none':
            required_driver_fields = (
                'driver_scaling',
                'predictor_driver_channels',
                'driver_source_sha256',
                'driver_scaler_sha256',
            )
            missing = [field for field in required_driver_fields if not hasattr(args, field)]
            if missing:
                raise ValueError(
                    "missing required driver config: " + ', '.join(sorted(missing))
                )
            channels = getattr(args, 'predictor_driver_channels')
            if isinstance(channels, (str, bytes)) or tuple(channels) != DRIVER_CHANNEL_ORDER:
                raise ValueError(
                    f"predictor driver channel order mismatch: expected {DRIVER_CHANNEL_ORDER}"
                )
            expected_scaling = (
                RAW_SCALING if self.driver_fusion_mode == 'raw_concat' else TRAIN_ZSCORE
            )
            if str(getattr(args, 'driver_scaling')) != expected_scaling:
                raise ValueError(
                    f"driver scaling mismatch: expected {expected_scaling}, "
                    f"got {getattr(args, 'driver_scaling')!r}"
                )
            self.expected_driver_scaling = expected_scaling
            self.expected_driver_source_sha256 = self._validated_sha256_arg(
                getattr(args, 'driver_source_sha256'), 'driver_source_sha256'
            )
            self.expected_driver_scaler_sha256 = self._validated_sha256_arg(
                getattr(args, 'driver_scaler_sha256'), 'driver_scaler_sha256'
            )
            if int(self.input_window) != 12:
                raise ValueError(
                    f"CAC-R1 driver history requires input_window=12, got {self.input_window}"
                )
            if self.driver_fusion_mode in {'raw_concat', 'zscore_concat'}:
                if self.feature_dim != 6:
                    raise ValueError(
                        f"{self.driver_fusion_mode} requires GWNET dim_in=6, got {self.feature_dim}"
                    )
                self.source_feature_dim = 1
            elif self.feature_dim != 1:
                raise ValueError(f"film requires GWNET dim_in=1, got {self.feature_dim}")

        self._logger = getLogger()

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=self.feature_dim,
                                    out_channels=self.residual_channels,
                                    kernel_size=(1, 1))

        from model.GWN.attention import NodewiseCausalTemporalCrossAttention
        self.history_crossattn = NodewiseCausalTemporalCrossAttention() if self.driver_fusion_mode == 'causal_crossattn' else None
        self.history_gate = HistoryResidualGate(self.residual_channels) if self.driver_fusion_mode == 'residual_gate' else None
        self.history_film = None
        if self.driver_fusion_mode == 'film':
            frozen_config = required_film_config()
            missing = [field for field in frozen_config if not hasattr(args, field)]
            if missing:
                raise ValueError(
                    "missing required FiLM config: " + ', '.join(sorted(missing))
                )
            mismatched = {
                field: (frozen_value, getattr(args, field))
                for field, frozen_value in frozen_config.items()
                if getattr(args, field) != frozen_value
            }
            if mismatched:
                raise ValueError(f"FiLM frozen config mismatch: {mismatched}")
            self.history_film = HistoryFiLM(self.residual_channels)


        if self.adj_mx is None:
            self.supports = None
        else:
            self.cal_adj(self.adjtype)
            self.supports = [torch.tensor(i).to(self.device) for i in self.adj_mx]
        if self.randomadj:
            self.aptinit = None
        else:
            self.aptinit = self.supports[0]
        if self.aptonly:
            self.supports = None

        receptive_field = self.output_dim

        self.supports_len = 0
        if self.supports is not None:
            self.supports_len += len(self.supports)

        if self.gcn_bool and self.addaptadj:
            if self.aptinit is None:
                if self.supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10).to(self.device),
                                             requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes).to(self.device),
                                             requires_grad=True).to(self.device)
                self.supports_len += 1
            else:
                if self.supports is None:
                    self.supports = []
                m, p, n = torch.svd(self.aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(self.device)
                self.supports_len += 1

        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            for i in range(self.layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                   out_channels=self.dilation_channels,
                                                   kernel_size=(1, self.kernel_size), dilation=new_dilation))
                # print(self.filter_convs[-1])
                # Use Conv2d to match residual tensor shape [B, C, N, T]
                self.gate_convs.append(nn.Conv2d(in_channels=self.residual_channels,
                                                 out_channels=self.dilation_channels,
                                                 kernel_size=(1, self.kernel_size), dilation=new_dilation))
                # print(self.gate_convs[-1])
                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=(1, 1)))
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(self.residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(GCN(self.dilation_channels, self.residual_channels,
                                          self.dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=self.skip_channels,
                                    out_channels=self.end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=self.end_channels,
                                    out_channels=self.output_window,
                                    kernel_size=(1, 1),
                                    bias=True)
        self.receptive_field = receptive_field
        self._logger.info('receptive_field: '+str(self.receptive_field))

    @staticmethod
    def _validated_sha256_arg(value, field_name):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"{field_name} must be a 64-character SHA-256 digest")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a SHA-256 hex digest") from exc
        return value.lower()

    def _prepare_driver_inputs(self, source, driver_history):
        if self.driver_fusion_mode == 'none':
            if driver_history is not None:
                raise ValueError("driver history was supplied to a no-driver GWNET")
            return source, None
        if not isinstance(source, torch.Tensor) or source.ndim != 4:
            raise ValueError("GWNET source must have shape [B,L,N,C]")
        if not torch.is_floating_point(source) or not bool(torch.isfinite(source).all()):
            raise ValueError("GWNET source contains non-finite or non-floating values")
        if int(source.size(1)) != int(self.input_window):
            raise ValueError(
                f"GWNET source history mismatch: expected {self.input_window}, got {int(source.size(1))}"
            )
        if int(source.size(2)) != int(self.num_nodes):
            raise ValueError(
                f"GWNET source node count mismatch: expected {self.num_nodes}, got {int(source.size(2))}"
            )
        if int(source.size(3)) != int(self.source_feature_dim):
            raise ValueError(
                f"GWNET source channel mismatch: expected {self.source_feature_dim}, got {int(source.size(3))}"
            )
        if driver_history is None:
            raise ValueError(f"driver history is required for {self.driver_fusion_mode}")
        if not isinstance(driver_history, DriverHistoryBatch):
            raise ValueError("driver history must be a strict DriverHistoryBatch")
        if int(driver_history.cadence_seconds) != 7200:
            raise ValueError(
                f"driver cadence mismatch: expected 7200 seconds, got {driver_history.cadence_seconds}"
            )
        driver_history.validate(
            expected_history_steps=int(self.input_window),
            expected_num_nodes=int(self.num_nodes),
            expected_source_driver_sha256=self.expected_driver_source_sha256,
            expected_driver_scaler_sha256=self.expected_driver_scaler_sha256,
            expected_scaling=self.expected_driver_scaling,
            expected_batch_size=int(source.size(0)),
        )
        values = driver_history.values
        if values.device != source.device:
            raise ValueError("GWNET source/driver device mismatch")
        if values.dtype != source.dtype:
            raise ValueError("GWNET source/driver dtype mismatch")
        if self.input_ablation is not None:
            removed = torch.tensor(self.input_ablation, device=values.device,
                                   dtype=torch.bool).logical_not()
            values = values.masked_fill(removed, 0.0)
        if self.driver_fusion_mode in {'raw_concat', 'zscore_concat'}:
            return torch.cat((source, values), dim=-1), values
        return source, values

    def assert_film_zero_initialization(self):
        if self.history_film is None:
            raise ValueError("GWNET is not configured for FiLM")
        self.history_film.assert_exact_zero_initialization()

    def forward(self, source, driver_history=None):
        inputs, strict_driver_values = self._prepare_driver_inputs(source, driver_history)
        inputs = inputs.transpose(1, 3)  # (batch_size, feature_dim, num_nodes, input_window)
        inputs = nn.functional.pad(inputs, (1, 0, 0, 0))  # (batch_size, feature_dim, num_nodes, input_window+1)

        # zero_pad = torch.zeros_like(inputs[..., 0:1]).to(self.device)
        # inputs = torch.cat([inputs, zero_pad], dim=-1)

        in_len = inputs.size(3)
        if in_len < self.receptive_field:
            # print(xxx)
            x = nn.functional.pad(inputs, (self.receptive_field-in_len, 0, 0, 0))
        else:
            # For long input sequences, computing all sliding outputs (T - receptive_field + 1) is memory-heavy
            # and unnecessary when we only care about the most recent prediction. For a causal WaveNet, the last
            # output depends only on the last receptive_field steps, so we keep a tail window here.
            x = inputs[..., -self.receptive_field:].contiguous()
        x = self.start_conv(x)  # (batch_size, residual_channels, num_nodes, self.receptive_field)
        if self.history_film is not None:
            aligned_drivers = align_film_driver_history(
                strict_driver_values,
                receptive_field=self.receptive_field,
            )
            x = self.history_film(x, aligned_drivers)
        if self.history_gate is not None:
            x = self.history_gate(x, align_film_driver_history(strict_driver_values, receptive_field=self.receptive_field))
        if self.history_crossattn is not None:
            x = self.history_crossattn(x, strict_driver_values)
        # x = torch.einsum('rd,bdnt->brnt', self.U2, x)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = self.supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*
            # (dilation, init_dilation) = self.dilations[i]
            # residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # (batch_size, residual_channels, num_nodes, self.receptive_field)
            # dilated convolution
            filter = self.filter_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            gate = torch.sigmoid(gate)
            x = filter * gate
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            # parametrized skip connection
            s = x
            # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
            s = self.skip_convs[i](s)
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except(Exception):
                skip = 0
            skip = s + skip
            # (batch_size, skip_channels, num_nodes, receptive_field-kernel_size+1)
            if self.gcn_bool and self.supports is not None:
                # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
                # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            else:
                # (batch_size, dilation_channels, num_nodes, receptive_field-kernel_size+1)
                x = self.residual_convs[i](x)
                # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            # residual: (batch_size, residual_channels, num_nodes, self.receptive_field)
            x = x + residual[:, :, :, -x.size(3):]
            # (batch_size, residual_channels, num_nodes, receptive_field-kernel_size+1)
            x = self.bn[i](x)
        x = F.relu(skip)
        # (batch_size, skip_channels, num_nodes, self.output_dim)
        x = F.relu(self.end_conv_1(x))
        # (batch_size, end_channels, num_nodes, self.output_dim)
        x = self.end_conv_2(x)
        # (batch_size, output_window, num_nodes, self.output_dim)
        # When input sequence is longer than receptive_field, the temporal dimension here can be > 1.
        # Downstream trainer expects a single-step output on the last dimension (equivalent to the most recent time).
        try:
            if isinstance(x, torch.Tensor) and x.dim() == 4 and x.size(-1) > 1:
                x = x[..., -1:].contiguous()
        except Exception:
            pass
        return x

    def cal_adj(self, adjtype):
        if adjtype == "scalap":
            self.adj_mx = [calculate_scaled_laplacian(self.adj_mx)]
        elif adjtype == "normlap":
            self.adj_mx = [calculate_normalized_laplacian(self.adj_mx).astype(np.float32).todense()]
        elif adjtype == "symnadj":
            self.adj_mx = [sym_adj(self.adj_mx)]
        elif adjtype == "transition":
            self.adj_mx = [asym_adj(self.adj_mx)]
        elif adjtype == "doubletransition":
            self.adj_mx = [asym_adj(self.adj_mx), asym_adj(np.transpose(self.adj_mx))]
        elif adjtype == "identity":
            self.adj_mx = [np.diag(np.ones(self.adj_mx.shape[0])).astype(np.float32)]
        else:
            assert 0, "adj type not defined"

