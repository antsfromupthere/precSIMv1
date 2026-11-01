import numpy as np
import torch
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)


class PhyAwaNet(torch.nn.Module):
    def __init__(self, L, N, M, K, W_T_1, W_T, total_power_watt, device):
        super().__init__()
        self.L = L
        self.N = N
        self.M = M
        self.K = K
        self.total_power_watt = total_power_watt

        W_T_1 = np.asarray(W_T_1)
        self.register_buffer("W_T_1", torch.from_numpy(W_T_1).to(device).cfloat())

        W_T = np.asarray(W_T)
        self.register_buffer("W_T", torch.from_numpy(W_T).to(device).cfloat())

        self.theta = torch.nn.Parameter(
            torch.rand(1, L, N, device=device) * 2 * np.pi
        )
        self.power_logits = torch.nn.Parameter(
            torch.randn(1, K, device=device)
        )

    def forward(self, H_H, sigma2):

        power_ratios = F.softmax(self.power_logits, dim=1)
        P_vector = power_ratios * self.total_power_watt

        phase_shifts = torch.exp(1j * self.theta)

        W_T_1 = self.W_T_1.unsqueeze(0)
        W_SIM = phase_shifts[:, 0].unsqueeze(-1) * W_T_1

        W_T = self.W_T.unsqueeze(0)
        for l in range(1, self.L):
            temp = torch.matmul(W_T, W_SIM)
            W_SIM = phase_shifts[:, l].unsqueeze(-1) * temp

        H_eff = torch.matmul(H_H, W_SIM)

        signal_gains = torch.abs(torch.diagonal(H_eff, dim1=1, dim2=2)) ** 2
        signal_power = P_vector * signal_gains

        total_matrix_power = torch.abs(H_eff) ** 2
        interference_matrix = total_matrix_power * P_vector.unsqueeze(1)
        total_power_received = torch.sum(interference_matrix, dim=2)
        interference_power = total_power_received - signal_power

        sinr = signal_power / (interference_power + sigma2)
        rates = torch.log2(1 + sinr)
        sum_rate = torch.sum(rates, dim=1)

        return sum_rate, rates, self.theta, P_vector


def phy_awa_pri_dual(W_T_1, W_T, H_H_batch, M, K, L, N, P_total_watt, sigma2,
                     iterations, lr, power_lr, device, phase_init, logits_init,
                     qos_threshold, lambda_min=1e-6, lambda_max=1.0 * 1e2, dual_lr=1.5,
                     phase_ema_decay=0.9, keep_history=False):

    H_H_tensor = torch.from_numpy(H_H_batch).cfloat().to(device)

    model = PhyAwaNet(L, N, M, K, W_T_1, W_T, P_total_watt, device).to(device)

    theta_init = np.angle(np.asarray(phase_init))[None, :, :].transpose(0, 2, 1)
    model.theta = torch.nn.Parameter(
        torch.from_numpy(theta_init).float().to(device)
    )

    model.power_logits = torch.nn.Parameter(
        torch.from_numpy(np.asarray(logits_init)).float().to(device)
    )

    optimizer = torch.optim.Adam([
        {'params': model.theta, 'lr': lr},
        {'params': model.power_logits, 'lr': power_lr},
    ])

    lambdas = torch.full((K,), 0.0, dtype=torch.get_default_dtype(), device=device)

    best_feasible = None
    best_unconstrained = None
    history = {"sum_rate": [], "min_rate": [], "dual": []}

    phase_ema = torch.exp(1j * model.theta.detach()).clone()

    def record_candidate(sum_rate, rates, theta, powers):
        nonlocal best_feasible, best_unconstrained
        candidate_sum = float(sum_rate.mean().item())
        candidate = (
            candidate_sum,
            theta.detach().clone(),
            powers.detach().clone(),
            rates.detach().clone(),
        )
        if best_unconstrained is None or candidate_sum > best_unconstrained[0]:
            best_unconstrained = candidate

        feasible_mask = (rates >= qos_threshold).all(dim=1)
        if feasible_mask.any():
            feasible_sum = float(sum_rate[feasible_mask].mean().item())
            if best_feasible is None or feasible_sum > best_feasible[0]:
                best_feasible = (feasible_sum, candidate[1], candidate[2], candidate[3])

    def record_history(sum_rate, rates):
        history["sum_rate"].append(float(sum_rate.mean().item()))
        history["min_rate"].append(float(rates.min().item()))
        history["dual"].append(lambdas.detach().cpu().numpy().copy())

    model.train()

    for i in range(iterations):
        optimizer.zero_grad()

        sum_rate, rates, current_thetas, current_powers = model(H_H_tensor, sigma2)
        record_candidate(sum_rate, rates, current_thetas, current_powers)
        if keep_history:
            record_history(sum_rate, rates)

        constraint_violation = qos_threshold - rates
        g_mean = constraint_violation.mean(dim=0)

        lagrangian = -sum_rate.mean() + (lambdas.detach() * g_mean).sum()

        lagrangian.backward()
        optimizer.step()

        with torch.no_grad():
            norm_grad = g_mean / (torch.norm(g_mean) + 1e-6)
            lambdas += dual_lr * norm_grad
            lambdas.clamp_(min=lambda_min, max=lambda_max)
            phase_ema.mul_(phase_ema_decay).add_(
                torch.exp(1j * model.theta.detach()),
                alpha=1.0 - phase_ema_decay,
            )

    with torch.no_grad():
        sum_rate, rates, current_thetas, current_powers = model(H_H_tensor, sigma2)
        record_candidate(sum_rate, rates, current_thetas, current_powers)
        if keep_history:
            record_history(sum_rate, rates)

        model.theta.copy_(torch.angle(phase_ema))
        record_candidate(*model(H_H_tensor, sigma2))

    is_feasible = best_feasible is not None
    selected = best_feasible if is_feasible else best_unconstrained
    final_sum_rate, final_thetas, final_powers, final_rates = selected

    output = (
        final_sum_rate,
        final_thetas.cpu().numpy(),
        final_powers.cpu().numpy(),
        final_rates.cpu().numpy(),
        is_feasible,
    )
    if keep_history:
        return output + (history,)
    return output
