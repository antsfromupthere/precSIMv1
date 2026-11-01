import time

from utils.runner import run_sweep

import numpy as np
import torch

from core.cal import cal_W_SIM, cal_sum_rate_given_p
from core.projAdam import proj_adam_qos
torch.set_default_dtype(torch.float64)


def solve(task):
    W_T_1, W_T, G, phase_init, Pt, sigma2, K, N, L, qos_threshold = task
    torch.set_num_threads(1)
    p_init = np.full(K, Pt / K, dtype=float)
    t0 = time.perf_counter()
    _, theta, p, _, _ = proj_adam_qos(
        W_T_1, W_T, G[np.newaxis, :, :], Pt, sigma2, 100, 0.45, 0.0045, torch.device("cpu"),
        phase_init, p_init, qos_threshold, lambda_max=100.0, dual_lr=1.5,
    )
    runtime = time.perf_counter() - t0
    phase = np.exp(1j * np.asarray(theta, dtype=np.float64)[0].T)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    W_SIM = cal_W_SIM(phase, W_T_1, W_T, L)
    return cal_sum_rate_given_p(G, W_SIM, K, p, sigma2)[0], runtime


if __name__ == "__main__":
    run_sweep("Projected-Adam-QoS", solve, tolerance=1.0e-6, qos_aware=True)
