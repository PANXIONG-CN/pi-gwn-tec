import argparse
import configparser
import importlib
import os

def get_predictor_params(args):
    from lib.model_names import canonical_model_name
    args.model = canonical_model_name(args.model)
    # get the based paras of predictors
    _here = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(_here, '..', 'conf', 'GPTST_pretrain', 'params_predictors.conf')
    config_file = os.path.normpath(config_file)
    config = configparser.ConfigParser()
    read_ok = config.read(config_file)
    if not read_ok or 'train' not in config:
        raise FileNotFoundError(f'配置文件不存在或损坏: {config_file}')

    # NOTE: 主训练入口使用“单短横线 + 长参数名”（例如 -horizon/-holdout_*）。
    # 这里必须禁用默认 "-h" help，否则 parse_known_args 会把 "-horizon" 误判为 help 并提前退出。
    parser_pred = argparse.ArgumentParser(prefix_chars='--', description='predictor_based_arguments', allow_abbrev=False, add_help=False)
    parser_pred.add_argument('--help', action='help', help='show this help message and exit')
    # train
    parser_pred.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
    parser_pred.add_argument('--epochs', default=config['train']['epochs'], type=int)
    parser_pred.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
    parser_pred.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
    parser_pred.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
    parser_pred.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
    parser_pred.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
    parser_pred.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
    parser_pred.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
    parser_pred.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
    parser_pred.add_argument('--debug', default=config['train']['debug'], type=eval)
    parser_pred.add_argument('--real_value', default=config['train']['real_value'], type=eval, help='use real value for loss calculation')

    if args.model == 'MTGNN':
        from model.MTGNN.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred)
    elif args.model == 'STGCN':
        from model.STGCN.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred)
    elif args.model == 'STSGCN':
        from model.STSGCN.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred)
    elif args.model == 'ASTGCN':
        from model.ASTGCN.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred)
    elif args.model == 'GWN':
        from model.GWN.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred)
    elif args.model == 'TGCN':
        from model.TGCN.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred)
    elif args.model == 'STFGNN':
        from model.STFGNN.args import parse_args
        args_predictor = parse_args(args.dataset, args, parser_pred)
    elif args.model == 'STGODE':
        from model.STGODE.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred)
    elif args.model == 'STWA':
        from model.ST_WA.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred)
    elif args.model == 'MSDR':
        from model.MSDR.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred)
    elif args.model == 'DMVSTNET':
        from model.DMVSTNET_demand.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred)
    elif args.model == 'CCRNN':
        from model.CCRNN_demand.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred)
    elif args.model == 'STMGCN':
        from model.STMGCN_demand.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred)
    elif args.model == 'OptimizedCSA_WTConvLSTM':
        from model.OptimizedCSA_WTConvLSTM.args import parse_args
        args_predictor = parse_args(args.dataset, parser_pred)
    else:
        # Dynamic plugin fallback for custom predictors
        try:
            mod = importlib.import_module(f"model.{args.model}.args")
            parse_args = getattr(mod, 'parse_args')
            args_predictor = parse_args(args.dataset, parser_pred)
        except Exception as e:
            raise ValueError(f"Unknown predictor {args.model} or plugin load failed: {e}")

    return args_predictor
