import time

from utils.runner import run_sweep

import numpy as np

from core.algoAO import algo_aoga
from core.cal import cal_W_SIM, cal_sum_rate_given_p


def solve(task):
    W_T_1, W_T, G, phase_init, Pt, sigma2, K, N, L, qos_threshold = task
    p_init = np.full(K, Pt / K, dtype=np.float64)
    t0 = time.perf_counter()
    _, phase, p, _ = algo_aoga(phase_init.copy(), Pt, p_init, W_T_1, W_T, L, N, K, G, sigma2,
                               max_outer=100, max_backtrack=100, max_iwf=100)
    runtime = time.perf_counter() - t0
    W_SIM = cal_W_SIM(phase, W_T_1, W_T, L)
    return cal_sum_rate_given_p(G, W_SIM, K, p, sigma2)[0], runtime


if __name__ == "__main__":
    run_sweep("AO", solve, tolerance=1.0e-7, qos_aware=False)
