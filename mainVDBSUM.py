import time

from utils.runner import run_sweep

import numpy as np

from core.cal import cal_W_SIM, cal_sum_rate_given_p
from core.vdBsum import solve_vd_bsum


def solve(task):
    W_T_1, W_T, G, phase_init, Pt, sigma2, K, N, L, qos_threshold = task
    p_init = np.full(K, Pt / K, dtype=np.float64)
    t0 = time.perf_counter()
    _, phase, p, _, _ = solve_vd_bsum(W_T_1, W_T, G, Pt, sigma2, phase_init, p_init, qos_threshold,
                                      max_outer=100, max_pdmm=50, max_ipdd=50)
    runtime = time.perf_counter() - t0
    W_SIM = cal_W_SIM(phase, W_T_1, W_T, L)
    return cal_sum_rate_given_p(G, W_SIM, K, np.maximum(p, 0.0), sigma2)[0], runtime


if __name__ == "__main__":
    run_sweep("VD-BSUM", solve, tolerance=1.0e-6, qos_aware=True)
