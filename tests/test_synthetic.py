"""Synthetic checks of the scoring, interval, ensemble, objective and input code (no data, no weights)."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def test_confirmation_scorer_self_test():
    from tecfc.scoring.confirm_score import synthetic
    assert synthetic()['status'] == 'passed'


def test_pooled_summary_and_identical_pair():
    from tecfc.scoring import pooled
    rng = np.random.default_rng(0)
    days = np.arange(np.datetime64('2023-01-01'), np.datetime64('2023-03-01')).astype('datetime64[s]').astype(np.int64)
    stats = np.zeros((5, len(days), 12, 4))
    stats[..., 0] = 10
    stats[..., 1] = rng.uniform(10, 20, stats.shape[:-1])
    stats[..., 2] = np.sqrt(stats[..., 1] * 10)
    row = pooled.summary(stats)
    expected = np.mean([np.sqrt(stats[s, ..., 1].sum() / stats[s, ..., 0].sum()) for s in range(5)])
    assert row['RMSE'] == pytest.approx(expected)
    pair = pooled.paired(stats, stats.copy(), days)
    assert pair['mean_delta'] == 0 and pair['ci95'] == [0.0, 0.0]


def test_interval_level_rule_matches_record():
    from tecfc.intervals import spec
    record = json.loads((REPO / 'records/scoring/interval_calibration_records.json').read_text())
    n = record['systems']['global_S2']['static']['n_calibration_days']
    assert spec.level_numerator(n, spec.ALPHAS['0.10']) == -(-((n + 1) * 9) // 10)
    assert spec.level(n, spec.ALPHAS['0.10']) == min(1.0, spec.level_numerator(n, spec.ALPHAS['0.10']) / n)
    assert spec.rank(n, 1000, spec.ALPHAS['0.10']) == -(-(spec.level_numerator(n, spec.ALPHAS['0.10']) * 1000) // n)


def test_static_calibration_quantile_is_an_order_statistic():
    from tecfc.intervals import calibration
    rng = np.random.default_rng(1)
    scores = rng.exponential(size=(12, 40, 30))
    q, pool, n_days = calibration.static_quantiles(scores, np.arange(20))
    for l in range(12):
        values = np.sort(scores[l][:20].ravel())
        assert q['0.10'][l] in values and pool[l] == values.size and n_days == 20


def test_fusion_and_bilinear_mapping():
    from tecfc.ensemble.apply import fuse, mapping, remap
    rng = np.random.default_rng(2)
    order = ['A', 'B', 'C']
    means = {f: rng.normal(size=(12, 71, 73)) for f in ('A', 'C')}
    weights = rng.uniform(size=(12, 3)); weights[:, 1] = 0
    fused = fuse(means, dict(family_order=order, full_validation_weights=weights.tolist()))
    manual = sum(weights[:, j, None, None] * means[f] for j, f in enumerate(order) if f in means)
    np.testing.assert_allclose(fused, manual.astype(np.float32), rtol=1e-6, atol=1e-5)
    lat, lon = np.array([30.3, 41.0]), np.array([100.2, 120.7])
    mp = mapping(lat, lon)
    np.testing.assert_allclose(mp[1].sum(0), 1.0)
    np.testing.assert_allclose(remap(np.ones((12, 5183)), mp), 1.0)


def test_gradient_projection_ratio_equals_rho():
    import torch
    from tecfc.objectives.penalty import assign_gradients
    torch.manual_seed(0)
    layer = torch.nn.Linear(4, 3)
    x = torch.randn(8, 4)
    for rho in (0.01, 0.1, 1.0):
        data = layer(x).square().mean()
        physical = (layer(x) - 1).abs().mean() * 1e4  # penalty gradient far larger than the data gradient
        stats = assign_gradients(data, physical, list(layer.parameters()), rho)
        assert stats['auxiliary_ratio_before'] == pytest.approx(rho, rel=1e-9)
        assert stats['auxiliary_ratio_after'] <= rho * (1 + 1e-9)


def test_signed_solar_geometry():
    from tecfc.features.geometry import signed_cos_sza
    times = np.array([np.datetime64('2023-03-20T12:00', 's').astype(np.int64)])
    lon = np.array([0.0, 180.0]); lat = np.array([0.0, 0.0])
    c = signed_cos_sza(times, lon, lat)[0]
    assert c[0] > 0.99 and c[1] < -0.99


def test_configurations_resolve():
    from tecfc.training import families
    mapping = families.locations('/data/gimtec', '/data/assets')
    for path in sorted((REPO / 'configs').glob('*.json')):
        paper = families.load(path)
        assert paper['seeds'] == [42, 1024, 2025, 3407, 5000]
        if 'runtime_config' in paper:
            config = families.runtime_config(paper, 42, mapping)
            text = json.dumps(config)
            assert '{GIMTEC}' not in text and '{ASSETS}' not in text and '{TECFC}' not in text
            assert config['seed'] == 42
            for name, change in paper.get('release_hash_changes', {}).items():
                key, file = name.rsplit('/', 1)
                folder = {'cg_reference_sha256': 'tecfc/models/simvp_reference', 'reference_sha256': 'tecfc/models/gwn_reference',
                          'physics/empirical_sha256': 'tecfc/objectives/empirical',
                          'simvp/sha256': 'tecfc/models/simvp_reference'}[key]
                assert families.sha(REPO / folder / file) == change['release_sha256']
        else:
            assert (REPO / paper['recipe_record']).is_file()
