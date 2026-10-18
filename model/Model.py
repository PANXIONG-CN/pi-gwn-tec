import torch
import torch.nn as nn
import importlib
from lib.model_names import canonical_model_name
try:
    # support running as a module (python -m model.Run)
    from model.Pretrain_model.GPTST import GPTST_Model
except Exception:
    # fallback for legacy direct execution paths
    from Pretrain_model.GPTST import GPTST_Model

class Fusion(nn.Module):
    def __init__(self, dim):
        super(Fusion, self).__init__()
        self.HS_fc = nn.Linear(dim, dim, bias=True)
        self.HT_fc = nn.Linear(dim, dim, bias=True)
        self.output_fc = nn.Linear(dim, dim, bias=True)

    def forward(self, flow_eb, time_eb):
        XS = self.HS_fc(flow_eb)
        XT = self.HT_fc(time_eb)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.multiply(z, flow_eb), torch.multiply(1 - z, time_eb))
        H = self.output_fc(H)
        return H

class Enhance_model(nn.Module):
    def __init__(self, args, args_predictor):
        super(Enhance_model, self).__init__()
        self.num_node = args.num_nodes
        self.input_base_dim = args.input_base_dim
        self.input_extra_dim = args.input_extra_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.embed_dim = args.embed_dim
        self.embed_dim_spa = args.embed_dim_spa
        self.HS = args.HS
        self.HT = args.HT
        self.HT_Tem = args.HT_Tem
        self.num_route = args.num_route
        self.mode = args.mode
        self.model = canonical_model_name(args.model)
        self.load_pretrain_path = args.load_pretrain_path
        self.log_dir = args.log_dir

        if self.mode == 'eval':
            self.pretrain_model = GPTST_Model(args)
            self.load_pretrained_model()
            self.fusion = Fusion(self.hidden_dim)
            self.lin_test = nn.Linear(self.input_base_dim, self.hidden_dim)

        # For baseline predictors without GPT-ST enhancement, both 'ori' and 'test'
        # should use raw input base dim. Only 'eval' (enhanced) uses hidden_dim.
        #
        # Some predictors optionally accept 5 external drivers channels (AE/Dst/Kp/F107/cosSZA).
        # Keep this behavior explicitly opt-in via use_drivers_input.
        use_drivers_input = getattr(args, 'use_drivers_input', None)
        if use_drivers_input is None:
            use_drivers_input = False
        model_key = str(self.model or '').strip().lower()
        self._expect_drivers_input = (model_key in {'gwn', 'stpatchtransformer'} and bool(use_drivers_input))
        if self.mode in ['ori', 'test']:
            dim_in = self.input_base_dim + (5 if self._expect_drivers_input else 0)
        else:
            dim_in = self.hidden_dim
        dim_out = self.output_dim

        if self.model == 'MTGNN':
            from MTGNN.MTGNN import MTGNN
            self.predictor = MTGNN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'STGCN':
            from STGCN.stgcn import STGCN
            self.predictor = STGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'STSGCN':
            from STSGCN.STSGCN import STSGCN
            self.predictor = STSGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'ASTGCN':
            from ASTGCN.ASTGCN import ASTGCN
            self.predictor = ASTGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'GWN':
            try:
                from model.GWN.GWN import GWNET
            except Exception:
                from GWN.GWN import GWNET
            self.predictor = GWNET(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'STPatchTransformer':
            from model.STPatchTransformer.stpatch_transformer import STPatchTransformer
            self.predictor = STPatchTransformer(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'TGCN':
            from TGCN.TGCN import TGCN
            self.predictor = TGCN(args_predictor, args.device, dim_in)
        elif self.model == 'STFGNN':
            from STFGNN.STFGNN import STFGNN
            self.predictor = STFGNN(args_predictor, dim_in)
        elif self.model == 'STGODE':
            from STGODE.STGODE import ODEGCN
            self.predictor = ODEGCN(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'STWA':
            from ST_WA.ST_WA import STWA
            self.predictor = STWA(args_predictor, args.device, dim_in)
        elif self.model == 'MSDR':
            from MSDR.gmsdr_model import GMSDRModel
            args_predictor.input_dim = dim_in
            self.predictor = GMSDRModel(args_predictor, args.device)
        elif self.model == 'DMVSTNET':
            from DMVSTNET_demand.DMVSTNET import DMVSTNet
            self.predictor = DMVSTNet(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'CCRNN':
            from CCRNN_demand.CCRNN import EvoNN2
            self.predictor = EvoNN2(args_predictor, args.device, dim_in, dim_out)
        elif self.model == 'STMGCN':
            from STMGCN_demand.STMGCN import ST_MGCN
            self.predictor = ST_MGCN(args_predictor, args.device, dim_in, dim_out)
        else:
            # Dynamic plugin fallback: model/<Name>/<lowercase>.py with class <Name>
            try:
                module = importlib.import_module(f"model.{self.model}.{self.model.lower()}")
                Predictor = getattr(module, self.model)
                try:
                    # Try common ctor signature
                    self.predictor = Predictor(args_predictor, args.device, dim_in, dim_out)
                except TypeError:
                    # Fallback to minimal signature
                    self.predictor = Predictor(args_predictor, args.device, dim_in)
            except Exception as e:
                raise ValueError(f"Unknown model {self.model} or plugin load failed: {e}")

    def load_pretrained_model(self):
        self.pretrain_model.load_state_dict(torch.load(self.log_dir + self.load_pretrain_path))
        for param in self.pretrain_model.parameters():
            param.requires_grad = False

    def forward(self, source, label, batch_seen=None, drivers_input=None):
        if self.mode in ['ori', 'test']:
            return self.forward_ori(source, label, batch_seen, drivers_input=drivers_input)
        else:
            return self.forward_pretrain(source, label, batch_seen)

    def forward_pretrain(self, source, label, batch_seen=None):
        x_pretrain_flow, _, _, _, _ = self.pretrain_model(source, label)
        x_t1 = self.lin_test(source[..., :self.input_base_dim])
        pretrain_eb = self.fusion(x_pretrain_flow, x_t1)
        if self.model == 'CCRNN':
            if label is None:
                x_predic = self.predictor(pretrain_eb, None, None)
            else:
                x_predic = self.predictor(pretrain_eb, label[:, :, :, 0:self.input_base_dim], None)
        else:
            x_predic = self.predictor(pretrain_eb)
        return x_predic, x_predic, x_predic, x_predic, x_predic

    def forward_ori(self, source, label=None, batch_seen=None, drivers_input=None):
        if self.model == 'CCRNN':
            if label is None:
                x_predic = self.predictor(source[:, :, :, 0:self.input_base_dim], None, None)
            else:
                x_predic = self.predictor(source[:, :, :, 0:self.input_base_dim], label[:, :, :, 0:self.input_base_dim], None)
        else:
            if self._expect_drivers_input:
                import torch as _torch
                base = source[:, :, :, 0:self.input_base_dim]
                if drivers_input is None:
                    # zero-fill 5 driver channels if upstream drivers are unavailable
                    zeros = _torch.zeros(base.size(0), base.size(1), base.size(2), 5, device=base.device, dtype=base.dtype)
                    x_in = _torch.cat([base, zeros], dim=-1)
                else:
                    x_in = _torch.cat([base, drivers_input.to(base.device)], dim=-1)
                x_predic = self.predictor(x_in)
            else:
                x_predic = self.predictor(source[:, :, :, 0:self.input_base_dim])
        return x_predic, x_predic, x_predic, x_predic, x_predic


def build_paper_predictor(config, adjacency, device, drivers=None):
    from lib.paper_recipe import build_predictor
    return build_predictor(config, adjacency, device, drivers)
