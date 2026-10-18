import configparser
import math
import os


def parse_args(DATASET, parser):
    # Robust path resolution: base on repo root
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(os.path.dirname(_here))
    config_file = os.path.join(_repo, "conf", "STPatchTransformer", f"{DATASET}.conf")
    config_file = os.path.normpath(config_file)

    config = configparser.ConfigParser()
    read_ok = config.read(config_file)
    if not read_ok or "data" not in config:
        raise FileNotFoundError(f"STPatchTransformer config not found or invalid: {config_file}")

    # data
    parser.add_argument("--num_nodes", type=int, default=int(config["data"]["num_nodes"]))
    parser.add_argument("--grid_h", type=int, default=int(config["data"].get("grid_h", fallback="0")))
    parser.add_argument("--grid_w", type=int, default=int(config["data"].get("grid_w", fallback="0")))
    parser.add_argument("--input_window", type=int, default=int(config["data"]["input_window"]))
    parser.add_argument("--output_window", type=int, default=int(config["data"]["output_window"]))
    parser.add_argument("--output_dim", type=int, default=int(config["data"].get("output_dim", fallback="1")))

    # model
    parser.add_argument("--space_patch", type=int, default=int(config.get("model", "space_patch", fallback="8")))
    parser.add_argument("--time_patch", type=int, default=int(config.get("model", "time_patch", fallback="2")))
    parser.add_argument("--d_model", type=int, default=int(config.get("model", "d_model", fallback="192")))
    parser.add_argument("--n_layers", type=int, default=int(config.get("model", "n_layers", fallback="6")))
    parser.add_argument("--n_heads", type=int, default=int(config.get("model", "n_heads", fallback="6")))
    parser.add_argument("--dropout", type=float, default=float(config.get("model", "dropout", fallback="0.1")))
    parser.add_argument("--mlp_ratio", type=float, default=float(config.get("model", "mlp_ratio", fallback="4.0")))
    parser.add_argument(
        "--attention_mode",
        type=str,
        default=str(config.get("model", "attention_mode", fallback="global")),
        choices=["global", "local_mask"],
    )
    parser.add_argument("--local_radius", type=int, default=int(config.get("model", "local_radius", fallback="1")))

    args, _ = parser.parse_known_args()

    # Derived, paper-facing stats (saved into outputs/.../config.yaml via predictor_args)
    try:
        H = int(getattr(args, "grid_h", 0))
        W = int(getattr(args, "grid_w", 0))
        ps = max(1, int(getattr(args, "space_patch", 8)))
        pt = max(1, int(getattr(args, "time_patch", 2)))
        T = int(getattr(args, "input_window", 0))
        if H > 0 and W > 0 and T > 0:
            H_pad = int(math.ceil(H / ps) * ps)
            W_pad = int(math.ceil(W / ps) * ps)
            n_spatial = int((H_pad // ps) * (W_pad // ps))
            n_time = int(math.ceil(T / pt))
            setattr(args, "n_spatial_patches", n_spatial)
            setattr(args, "n_time_patches", n_time)
            setattr(args, "seq_len", int(n_spatial * n_time))
            setattr(args, "attention_mode", str(getattr(args, "attention_mode", "global")))
            setattr(args, "local_radius", int(getattr(args, "local_radius", 1)))
    except Exception:
        pass

    return args
