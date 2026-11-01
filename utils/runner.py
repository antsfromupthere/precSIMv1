import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import csv
import itertools
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from utils.sim_model import make_realizations, make_system

K = 4


def run_sweep(method, solve, tolerance, qos_aware):
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", nargs="+", type=int, default=[6])
    parser.add_argument("--N", nargs="+", type=int, default=[49])
    parser.add_argument("--Pmax", nargs="+", type=float, default=[15.0])
    parser.add_argument("--Rmin", nargs="+", type=float, default=[1.0])
    parser.add_argument("--mc", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)
    path = Path("results") / f"{method}.csv"
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        if write_header:
            writer.writerow(["L", "N", "Pmax_dBm", "Rmin", "realizations", "avg_sum_rate",
                             "feasible_count", "succ_prob", "avg_runtime_s"])
        for L, N, P_max_dBm, qos_threshold in itertools.product(args.L, args.N, args.Pmax, args.Rmin):
            W_T, W_T_1, Corr_T_sqrt, pathloss, Pt, sigma2 = make_system(L, N, K, P_max_dBm)
            channels, phases = make_realizations(Corr_T_sqrt, pathloss, K, N, L, args.mc)
            tasks = [
                (W_T_1, W_T, channels[mc], phases[mc], Pt, sigma2, K, N, L, qos_threshold)
                for mc in range(args.mc)
            ]
            if args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    outputs = list(pool.map(solve, tasks, chunksize=1))
            else:
                outputs = [solve(task) for task in tasks]

            rates = np.stack([output[0] for output in outputs])
            runtimes = np.asarray([output[1] for output in outputs])
            sum_rates = rates.sum(axis=1)
            feasible = np.max(np.maximum(qos_threshold - rates, 0.0), axis=1) <= tolerance
            flag = int(feasible.sum())
            if qos_aware:
                avg_sumrate = float(sum_rates[feasible].mean()) if flag else float("nan")
            else:
                avg_sumrate = float(sum_rates.mean())

            print(f"{method}: L = {L}, N = {N}, Pmax = {P_max_dBm} dBm, Rmin = {qos_threshold} bps/Hz, "
                  f"Avg. Sum-Rate = {avg_sumrate:.3f} bps/Hz, "
                  f"Succ. Prob. = {flag / args.mc:.4f} ({flag}/{args.mc}), "
                  f"Avg. Runtime = {runtimes.mean():.3f} sec", flush=True)
            writer.writerow([L, N, P_max_dBm, qos_threshold, args.mc, avg_sumrate, flag,
                             flag / args.mc, float(runtimes.mean())])
            stream.flush()
