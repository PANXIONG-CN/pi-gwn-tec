"""exp0028 fixed specification: constants and exact integer rules only. No file access.

Everything here is fixed by PREREGISTRATION_CN.txt before any coverage is computed.
"""

SEED = 20260924
REPLICATES = 5000
BLOCK_DAYS = 7
HORIZONS = 12
LEADS = tuple(2 * (i + 1) for i in range(HORIZONS))
ALPHAS = {'0.10': (1, 10), '0.20': (1, 5)}      # alpha = a / b, kept as exact integers
NOMINAL = {'0.10': 0.90, '0.20': 0.80}
BAND = {'0.10': (0.88, 0.92), '0.20': (0.78, 0.82)}
HIGH_TEC = 60.0
DAY = 86400
ROLL_FIRST, ROLL_LAST, ROLL_MIN = 366, 2, 365   # window D-366 .. D-2 (365 calendar days), 365 forecast days required
LAST_LEAD_OFFSET = 22 * 3600                    # latest target time of a forecast day = origin + 22 h
CI_QUANTILES = (0.025, 0.975)
MAIN_ALPHA = '0.10'

# system -> (region, prediction key, calibration rule for method S, evaluation years, period split)
SYSTEMS = {
    'global_S2':    dict(region='global', method='S2',     static='old_evaluation', eval_years=(2023, 2024, 2025, 2026),
                         P1=(2023, 2024, 2025), P2=(2026,)),
    'china_final':  dict(region='china', method='final',   static='year_2023', eval_years=(2024, 2025, 2026),
                         P1=(2024, 2025), P2=(2026,)),
    'china_anchor': dict(region='china', method='anchor',  static='year_2023', eval_years=(2024, 2025, 2026),
                         P1=(2024, 2025), P2=(2026,)),
    'japan_final':  dict(region='japan', method='final',   static='year_2023', eval_years=(2024, 2025, 2026),
                         P1=(2024, 2025), P2=(2026,)),
    'japan_anchor': dict(region='japan', method='anchor',  static='year_2023', eval_years=(2024, 2025, 2026),
                         P1=(2024, 2025), P2=(2026,)),
}

# Check A: expected RMSE (None = no preregistered expected value, observed only)
EXPECTED_RMSE = {
    'global_S2':    {'ext': 4.139160, 'y2026': 3.093689},
    'china_final':  {'ext': 6.021815, 'y2026': 4.863077},
    'japan_final':  {'ext': 3.940818, 'y2026': 3.119169},
    'china_anchor': {'ext': 6.418629, 'y2026': None},
    'japan_anchor': {'ext': 4.259275, 'y2026': None},
}
CHECK_A_TOLERANCE = 1e-6
CHECK_C_TOLERANCE = 0.01


def level_numerator(n_days, alpha):
    """m = ceil((n+1)(1-alpha)) in exact integer arithmetic (alpha = a/b)."""
    a, b = alpha
    return -(-((n_days + 1) * (b - a)) // b)


def rank(n_days, n_pool, alpha):
    """1-based rank k = ceil(p * N) of the pooled order statistic, p = min(1, m / n_days)."""
    m = level_numerator(n_days, alpha)
    if m >= n_days:
        return n_pool
    return -(-(m * n_pool) // n_days)


def level(n_days, alpha):
    m = level_numerator(n_days, alpha)
    return min(1.0, m / n_days)


def wording(coverage, akey):
    lo, hi = BAND[akey]
    if coverage < lo:
        return '覆盖不足'
    if coverage > hi:
        return '偏保守'
    return '接近名义水平'
