

import numpy as np
from scipy import stats


def diebold_mariano_test(errors_a, errors_b, power=2, h=1):

    errors_a = np.asarray(errors_a)
    errors_b = np.asarray(errors_b)
    n = len(errors_a)
    loss_a = np.abs(errors_a) ** power
    loss_b = np.abs(errors_b) ** power
    d = loss_a - loss_b

    d_mean = np.mean(d)
    # Newey-West style variance with (h-1) lag autocovariance correction
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        gamma = np.cov(d[:-lag], d[lag:])[0, 1] if n > lag else 0.0
        var_d += 2 * gamma
    var_d = max(var_d, 1e-12) / n

    dm_stat = d_mean / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def wilcoxon_signed_rank_test(errors_a, errors_b):
    
    a = np.abs(np.asarray(errors_a))
    b = np.abs(np.asarray(errors_b))
    try:
        stat, p_value = stats.wilcoxon(a, b)
    except ValueError:
        stat, p_value = float("nan"), float("nan")
    return float(stat), float(p_value)


def run_significance_tests(proposed_errors, baseline_errors_dict):
  
    results = []
    for name, errs in baseline_errors_dict.items():
        n = min(len(proposed_errors), len(errs))
        dm_stat, dm_p = diebold_mariano_test(proposed_errors[:n], errs[:n])
        w_stat, w_p = wilcoxon_signed_rank_test(proposed_errors[:n], errs[:n])
        results.append({
            "baseline": name,
            "DM_statistic": dm_stat,
            "DM_p_value": dm_p,
            "Wilcoxon_statistic": w_stat,
            "Wilcoxon_p_value": w_p,
            "significant_5pct": bool(dm_p < 0.05 and w_p < 0.05),
        })
    return results
