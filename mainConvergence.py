import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import torch

from core.propAlgo import phy_awa_pri_dual
from utils.sim_model import make_realizations, make_system

K = 4
L = 6
N = 49
P_max_dBm = 15.0
qos_threshold = 1.0
MonteCarloStep = 300
selected_realizations = (220, 119, 111, 29)

if __name__ == "__main__":
    torch.set_num_threads(1)
    W_T, W_T_1, Corr_T_sqrt, pathloss, Pt, sigma2 = make_system(L, N, K, P_max_dBm)
    channels, phases = make_realizations(Corr_T_sqrt, pathloss, K, N, L, MonteCarloStep)

    sum_rate = []
    min_rate = []
    dual = []
    for mc in selected_realizations:
        output = phy_awa_pri_dual(
            W_T_1, W_T, channels[mc][np.newaxis, :, :], K, K, L, N, Pt, sigma2,
            iterations=100, lr=0.45, power_lr=0.0045, device=torch.device("cpu"),
            phase_init=phases[mc], logits_init=np.ones((1, K), dtype=np.float64),
            qos_threshold=qos_threshold, keep_history=True,
        )
        history = output[-1]
        sum_rate.append(history["sum_rate"])
        min_rate.append(history["min_rate"])
        dual.append(np.stack(history["dual"]))
        first_feasible = int(np.argmax(np.asarray(history["min_rate"]) >= qos_threshold))
        print(f"realization {mc}: first feasible iteration = {first_feasible}, "
              f"final sum rate = {history['sum_rate'][-1]:.3f} bps/Hz, "
              f"peak duals = {np.round(np.max(dual[-1], axis=0), 2)}")

    Path("results").mkdir(exist_ok=True)
    np.savez("results/convergence.npz", realizations=np.asarray(selected_realizations),
             sum_rate=np.asarray(sum_rate), min_rate=np.asarray(min_rate), dual=np.asarray(dual))
