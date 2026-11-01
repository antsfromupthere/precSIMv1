import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

import numpy as np
import torch

from core.ddpg import DDPGEnvironment, evaluate_actor, make_actor, make_config
from utils.sim_model import make_realizations, make_system

CONFIGURATIONS = ((4, 49), (6, 49), (8, 49), (6, 64))
K = 4
P_max_dBm = 15.0
qos_threshold = 1.0
seed = 42


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--mc", type=int, default=300)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    configurations = CONFIGURATIONS if args.L is None else ((args.L, args.N),)

    for L, N in configurations:
        configuration_seed = seed + L * 10_000 + N * 10
        W_T, W_T_1, Corr_T_sqrt, pathloss, Pt, sigma2 = make_system(L, N, K, P_max_dBm)
        channels_np, _ = make_realizations(Corr_T_sqrt, pathloss, K, N, L, args.mc)
        channels = torch.as_tensor(channels_np, device=device).cfloat()

        environment = DDPGEnvironment(W_T_1, W_T, pathloss, L, N, K, Pt, sigma2, qos_threshold, device)
        checkpoint = torch.load(args.weights_dir / f"L{L}_N{N}.pt", map_location=device, weights_only=True)
        actor = make_actor(environment, make_config(L, N, K))
        actor.load_state_dict(checkpoint["actor_state_dict"])

        training_actions = checkpoint["training_best_actions"].to(device)
        with torch.no_grad():
            metrics = environment.physical_metrics(channels[:training_actions.shape[0]], training_actions)
        training_rate = float(np.mean(metrics["sum_rate"].cpu().numpy()))

        sum_rates, minimum_rates, feasible = evaluate_actor(
            environment, actor, channels, args.rollout_steps, args.restarts,
            configuration_seed + 9_000_000,
        )
        print(f"L = {L}, N = {N}, "
              f"DDPG (training) = {training_rate:.3f} bps/Hz over {training_actions.shape[0]} channels, "
              f"DDPG (inference) = {float(np.mean(sum_rates)):.3f} bps/Hz over {args.mc} channels, "
              f"Avg. Min-Rate = {float(np.mean(minimum_rates)):.3f} bps/Hz, "
              f"Succ. Prob. = {float(np.mean(feasible)):.4f}", flush=True)


if __name__ == "__main__":
    main()
