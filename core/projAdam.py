import math

import numpy as np
import torch


def project_power_feasible_set(power, total_power):
    original_shape = power.shape
    vector = power.reshape(-1)
    nonnegative = torch.clamp(vector, min=0.0)
    if bool((nonnegative.sum() <= total_power).item()):
        return nonnegative.reshape(original_shape)

    sorted_values, _ = torch.sort(vector, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0)
    indices = torch.arange(
        1,
        vector.numel() + 1,
        device=vector.device,
        dtype=vector.dtype,
    )
    support = sorted_values - (cumulative - total_power) / indices > 0
    rho = torch.nonzero(support, as_tuple=False)[-1, 0]
    threshold = (cumulative[rho] - total_power) / (rho + 1)
    projected = torch.clamp(vector - threshold, min=0.0)
    return projected.reshape(original_shape)


def physical_rates(theta, power, channel, W_T_1, W_T, sigma2):
    phase_coefficients = torch.exp(1j * theta)
    W_sim = phase_coefficients[:, 0].unsqueeze(-1) * W_T_1.unsqueeze(0)
    W_T_batch = W_T.unsqueeze(0)
    for layer in range(1, theta.shape[1]):
        W_sim = phase_coefficients[:, layer].unsqueeze(-1) * torch.matmul(
            W_T_batch,
            W_sim,
        )

    effective_channel = torch.matmul(channel, W_sim)
    gains = torch.abs(effective_channel) ** 2
    desired_gains = torch.diagonal(gains, dim1=1, dim2=2)
    signal = power * desired_gains
    total_received = torch.sum(gains * power.unsqueeze(1), dim=2)
    interference = total_received - signal
    sinr = signal / (interference + sigma2)
    rates = torch.log2(1.0 + sinr)
    return rates.sum(dim=1), rates


def proj_adam_qos(W_T_1, W_T, channel_batch, total_power, sigma2, iterations, phase_lr,
                  power_lr, device, phase_init, power_init, qos_threshold,
                  lambda_max=1.0e2, dual_lr=1.5):

    channel = torch.as_tensor(channel_batch, device=device).cfloat()
    W_T_1_tensor = torch.as_tensor(W_T_1, device=device).cfloat()
    W_T_tensor = torch.as_tensor(W_T, device=device).cfloat()

    theta_initial = np.angle(np.asarray(phase_init)).T[np.newaxis, :, :]
    theta_initial = np.where(theta_initial < 0.0, theta_initial + 2.0 * np.pi, theta_initial)
    theta = torch.nn.Parameter(
        torch.as_tensor(theta_initial, device=device, dtype=torch.float32)
    )
    power = torch.nn.Parameter(
        torch.as_tensor(
            np.asarray(power_init)[np.newaxis, :],
            device=device,
            dtype=torch.float32,
        )
    )

    with torch.no_grad():
        theta.clamp_(0.0, 2.0 * math.pi)
        power.copy_(project_power_feasible_set(power, total_power))

    optimizer = torch.optim.Adam(
        [
            {"params": [theta], "lr": phase_lr},
            {"params": [power], "lr": power_lr},
        ]
    )
    dual = torch.zeros(power.shape[1], device=device)
    best_feasible = None
    best_unconstrained = None

    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        sum_rate, rates = physical_rates(theta, power, channel, W_T_1_tensor, W_T_tensor, sigma2)

        with torch.no_grad():
            candidate = {
                "sum_rate": float(sum_rate.mean().item()),
                "theta": theta.detach().clone(),
                "power": power.detach().clone(),
                "rates": rates.detach().clone(),
            }
            if best_unconstrained is None or candidate["sum_rate"] > best_unconstrained["sum_rate"]:
                best_unconstrained = candidate

            feasible = bool((rates >= qos_threshold).all().item())
            if feasible and (best_feasible is None or candidate["sum_rate"] > best_feasible["sum_rate"]):
                best_feasible = candidate

        violation = qos_threshold - rates
        mean_violation = violation.mean(dim=0)
        lagrangian = -sum_rate.mean() + (dual.detach() * mean_violation).sum()
        lagrangian.backward()
        optimizer.step()

        with torch.no_grad():
            theta.clamp_(0.0, 2.0 * math.pi)
            power.copy_(project_power_feasible_set(power, total_power))

            normalized_violation = mean_violation / (torch.norm(mean_violation) + 1.0e-6)
            dual.add_(dual_lr * normalized_violation)
            dual.clamp_(min=1.0e-6, max=lambda_max)

    result = best_feasible if best_feasible is not None else best_unconstrained
    return (
        float(result["sum_rate"]),
        result["theta"].cpu().numpy(),
        result["power"].cpu().numpy(),
        result["rates"].cpu().numpy(),
        best_feasible is not None,
    )
