import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

import numpy as np
import torch

from core.ddpg import DDPGEnvironment, DDPGTrainer, evaluate_actor, make_config
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
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps-per-episode", type=int, default=26_000)
    parser.add_argument("--train-channels", type=int, default=100)
    parser.add_argument("--progress-interval", type=int, default=5_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "ddpg")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    configurations = CONFIGURATIONS if args.L is None else ((args.L, args.N),)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for L, N in configurations:
        configuration_seed = seed + L * 10_000 + N * 10
        W_T, W_T_1, Corr_T_sqrt, pathloss, Pt, sigma2 = make_system(L, N, K, P_max_dBm)
        channels_np, _ = make_realizations(Corr_T_sqrt, pathloss, K, N, L, args.train_channels)
        channels = torch.as_tensor(channels_np, device=device).cfloat()

        environment = DDPGEnvironment(W_T_1, W_T, pathloss, L, N, K, Pt, sigma2, qos_threshold, device)
        trainer = DDPGTrainer(environment, make_config(L, N, K), seed)

        training_best_sum = None
        training_best_action = None
        for episode in range(1, args.episodes + 1):
            episode_best_sum, episode_best_action = trainer.train_episode(
                channels,
                args.steps_per_episode,
                configuration_seed + episode * 100_003,
                args.progress_interval,
            )
            if training_best_sum is None:
                training_best_sum = episode_best_sum.clone()
                training_best_action = episode_best_action.clone()
            else:
                better = episode_best_sum > training_best_sum
                training_best_sum = torch.where(better, episode_best_sum, training_best_sum)
                training_best_action = torch.where(better.unsqueeze(1), episode_best_action,
                                                   training_best_action)
            print(f"L = {L}, N = {N}, episode {episode}/{args.episodes}, "
                  f"DDPG (training) = {float(training_best_sum.mean().cpu()):.3f} bps/Hz", flush=True)
            torch.save(
                {
                    "actor_state_dict": trainer.actor.state_dict(),
                    "training_best_actions": training_best_action.cpu(),
                },
                args.output_dir / f"L{L}_N{N}.pt",
            )

        sum_rates, _, _ = evaluate_actor(environment, trainer.actor, channels, 64, 1,
                                         configuration_seed + 9_000_000)
        print(f"L = {L}, N = {N}, frozen actor on the training channels = "
              f"{float(np.mean(sum_rates)):.3f} bps/Hz", flush=True)


if __name__ == "__main__":
    main()
