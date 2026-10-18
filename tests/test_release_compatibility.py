"""Regression checks for the public names and runtime switches."""
from types import SimpleNamespace
from contextlib import nullcontext
from pathlib import Path
import sys
import pytest
import torch
from lib.runtime_options import gradient_checkpointing_enabled


@pytest.mark.parametrize("name", ["STPatchGPT", "STPatchTransformer"])
@pytest.mark.parametrize("drivers", [False, True])
def test_patch_predictor_alias_parser_checkpoint_and_forward(monkeypatch, name, drivers):
    from lib.Params_pretrain import parse_args
    from lib.Params_predictor import get_predictor_params
    from model.Model import Enhance_model
    monkeypatch.setattr(sys, "argv", ["Run.py", "-dataset", "GIMtec", "-mode", "ori", "-model", name])
    args = parse_args("cpu")
    pred = get_predictor_params(args)
    assert args.model == "STPatchTransformer"
    assert pred.grid_h * pred.grid_w == pred.num_nodes == 5183
    # Exercise the real wrapper with a small grid; no data files or training.
    for key, value in dict(num_nodes=16, horizon=2, input_base_dim=1, output_dim=1,
                           use_drivers_input=drivers, model=name, device="cpu").items():
        setattr(args, key, value)
    small = SimpleNamespace(grid_h=4, grid_w=4, num_nodes=16, input_window=4,
                            output_window=2, space_patch=2, time_patch=2,
                            d_model=16, n_layers=1, n_heads=2, dropout=0.,
                            mlp_ratio=2., attention_mode="global", local_radius=1)
    old_alias = Enhance_model(args, small).eval()
    args.model = "STPatchTransformer"
    canonical = Enhance_model(args, small).eval()
    canonical.load_state_dict(old_alias.state_dict(), strict=True)
    assert old_alias._expect_drivers_input is drivers
    x = torch.randn(2, 4, 16, 1)
    driver_input = torch.randn(2, 4, 16, 5) if drivers else None
    with torch.no_grad():
        first = old_alias(x, None, drivers_input=driver_input)[0]
        second = canonical(x, None, drivers_input=driver_input)[0]
    assert first.shape == (2, 2, 16, 1)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    if drivers:
        with torch.no_grad():
            zero_drivers = canonical(x, None, drivers_input=torch.zeros_like(driver_input))[0]
        assert not torch.allclose(second, zero_drivers)


@pytest.mark.parametrize("env,expected", [
    ({}, False),
    ({"PIGWN_ENABLE_GC": "yes"}, True),
    ({"PIGWN_ENABLE_GC": "TRUE"}, True),
    ({"PIGWN_ENABLE_GC": "0"}, False),
    ({"PIGWN_ENABLE_GC": ""}, False),
    ({"PIGWN_ENABLE_GC": "invalid"}, False),
])
def test_gradient_checkpoint_switch_precedence(env, expected):
    assert gradient_checkpointing_enabled(env) is expected


@pytest.mark.parametrize("env,backend", [
    ({}, "math"),
    ({"PIGWN_SDPA": "mem"}, "mem"),
    ({"PIGWN_SDPA": "flash"}, "flash"),
    ({"PIGWN_SDPA": "math"}, "math"),
    ({"PIGWN_SDPA": ""}, "math"),
])
def test_attention_environment_reaches_forward_backend(monkeypatch, env, backend):
    from model.TEC_MoLLM.vendor import modules
    monkeypatch.delenv("PIGWN_SDPA", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    # Test the forward integration without downloading an HF model or using a GPU.
    class IdentityHF(torch.nn.Module):
        def forward(self, inputs_embeds, attention_mask=None):
            return SimpleNamespace(last_hidden_state=inputs_embeds)
    backbone = modules.LLMBackbone.__new__(modules.LLMBackbone)
    torch.nn.Module.__init__(backbone)
    backbone.use_hf = True
    backbone.model = IdentityHF()
    selected = []
    def capture_backend(**options):
        selected.append(options)
        return nullcontext()
    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", capture_backend)
    x = torch.randn(2, 3, 16)
    torch.testing.assert_close(backbone(x), x)
    assert selected == [{"enable_flash": backend == "flash", "enable_math": backend == "math",
                         "enable_mem_efficient": backend == "mem"}]


@pytest.mark.parametrize("env,expected_batches", [
    ({}, [10]),
    ({"PIGWN_NODE_SAFE_CAP": "4"}, [4, 4, 2]),
    ({"PIGWN_NODE_SAFE_CAP": "2"}, [2, 2, 2, 2, 2]),
    ({"PIGWN_NODE_SAFE_CAP": ""}, [10]),
])
def test_node_cap_environment_controls_forward_chunks(monkeypatch, env, expected_batches):
    from model.TEC_MoLLM.vendor import modules
    from model.TEC_MoLLM.vendor.tec_mollm import TEC_MoLLM
    monkeypatch.delenv("PIGWN_NODE_SAFE_CAP", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(modules, "_HAS_TRANSFORMERS", False)
    predictor = TEC_MoLLM(num_nodes=5, spatial_out=4, heads=2,
                         temporal_channels=(16,), temporal_strides=(1,),
                         patch_len=1, d_llm=16, pred_horizon=2,
                         temporal_seq_len=4, node_chunk=5).eval()
    batches = []
    hook = predictor.temporal.register_forward_pre_hook(
        lambda module, inputs: batches.append(inputs[0].shape[0]))
    with torch.no_grad():
        y = predictor(torch.randn(2, 4, 5, 1), None,
                      torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]))
    hook.remove()
    assert y.shape == (2, 2, 5, 1) and torch.isfinite(y).all()
    assert batches == expected_batches


@pytest.mark.parametrize("cwd", [".", "model"])
def test_tecmollm_config_does_not_depend_on_working_directory(monkeypatch, cwd):
    import argparse
    from model.TEC_MoLLM.args import parse_args
    monkeypatch.chdir(Path(__file__).resolve().parents[1] / cwd)
    monkeypatch.setattr(sys, "argv", ["Run.py"])
    args = parse_args("GIMtec", argparse.ArgumentParser())
    # Values from conf/TEC_MoLLM/GIMtec.conf, not the built-in fallbacks.
    assert (args.heads, args.spatial_out, args.d_llm, args.use_lora, args.scheduler) == (3, 11, 1280, True, "cosine")


def test_tecmollm_missing_config_is_an_error(monkeypatch):
    import argparse
    from model.TEC_MoLLM.args import parse_args
    monkeypatch.setattr(sys, "argv", ["Run.py"])
    with pytest.raises(FileNotFoundError):
        parse_args("NoSuchDataset", argparse.ArgumentParser())
