import time

from utils.runner import run_sweep

import numpy as np
import torch

from core.cal import cal_W_SIM, cal_sum_rate_given_p
from core.propAlgo import phy_awa_pri_dual


def solve(task):
    W_T_1, W_T, G, phase_init, Pt, sigma2, K, N, L, qos_threshold = task
    torch.set_num_threads(1)
    logits_init = np.ones((1, K), dtype=np.float64)
    t0 = time.perf_counter()
    _, theta, p, _, _ = phy_awa_pri_dual(
        W_T_1, W_T, G[np.newaxis, :, :], K, K, L, N, Pt, sigma2,
        iterations=100, lr=0.45, power_lr=0.0045, device=torch.device("cpu"),
        phase_init=phase_init, logits_init=logits_init, qos_threshold=qos_threshold,
        lambda_max=100.0, dual_lr=1.5, phase_ema_decay=0.9,
    )
    runtime = time.perf_counter() - t0
    phase = np.exp(1j * np.asarray(theta, dtype=np.float64)[0].T)
    p = np.maximum(np.asarray(p, dtype=np.float64).reshape(-1), 0.0)
    W_SIM = cal_W_SIM(phase, W_T_1, W_T, L)
    return cal_sum_rate_given_p(G, W_SIM, K, p, sigma2)[0], runtime


if __name__ == "__main__":
    run_sweep("NR-QoS", solve, tolerance=0.0, qos_aware=True)
