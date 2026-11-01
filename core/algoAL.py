import math
import numpy as np
from core.cal import cal_W_SIM, cal_gradient, cal_power_gradient, cal_sum_rate_given_p


def project_simplex(vector, total):
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered)
    support = ordered - (cumulative - total) / np.arange(1, values.size + 1) > 0
    if not np.any(support):
        return np.full_like(values, total / values.size)
    rho = np.nonzero(support)[0][-1]
    threshold = (cumulative[rho] - total) / float(rho + 1)
    projected = np.maximum(values - threshold, 0.0)
    positive = projected > 0.0
    projected[positive] += (total - projected.sum()) / positive.sum()
    return projected


def eval_augmented_lagrangian(phase, p, G, W_T_1, W_T, sigma2, qos_threshold, dual, penalty):
    W_SIM = cal_W_SIM(phase, W_T_1, W_T, phase.shape[1])
    rates, gama, H_fit = cal_sum_rate_given_p(G, W_SIM, p.size, p, sigma2)
    sum_rate = float(np.sum(rates))
    violation = qos_threshold - rates
    shifted_dual = np.maximum(0.0, dual + penalty * violation)
    augmentation = float(np.sum((shifted_dual**2 - dual**2) / (2.0 * penalty)))
    score = sum_rate - augmentation
    rate_weights = 1.0 + shifted_dual
    return score, sum_rate, rates, gama, H_fit, rate_weights


def algo_al_qos(W_T_1, W_T, G, Pt, sigma2, phase_init, power_init, qos_threshold,
                max_outer=100, max_line_search=50, power_steps=5, phase_steps=1,
                init_penalty=1.0, penalty_growth=2.0, max_penalty=1.0e4,
                tol=1.0e-6, armijo=1.0e-4, backtrack=0.5):

    G = np.asarray(G, dtype=np.complex128)
    phase = np.asarray(phase_init, dtype=np.complex128).copy()
    phase /= np.abs(phase)

    K = G.shape[0]
    p = project_simplex(np.asarray(power_init, dtype=float).reshape(K), Pt)
    N, L = phase.shape
    dual = np.zeros(K, dtype=float)
    penalty = float(init_penalty)
    previous_violation = math.inf
    best_feasible = None
    best_restoration = None

    def evaluate():
        return eval_augmented_lagrangian(phase, p, G, W_T_1, W_T, sigma2,
                                         qos_threshold, dual, penalty)

    def record_candidate():
        nonlocal best_feasible, best_restoration
        _, sum_rate, rates, _, _, _ = evaluate()
        maximum_violation = float(np.max(np.maximum(qos_threshold - rates, 0.0)))
        if maximum_violation <= tol and (
            best_feasible is None or sum_rate > best_feasible[0]
        ):
            best_feasible = (sum_rate, phase.copy(), p.copy(), rates.copy())
        restoration_key = (maximum_violation, -sum_rate)
        if best_restoration is None or restoration_key < (
            best_restoration[0], best_restoration[1]
        ):
            best_restoration = (maximum_violation, -sum_rate,
                                phase.copy(), p.copy(), rates.copy())
        return sum_rate, rates, maximum_violation

    record_candidate()
    for outer in range(max_outer):

        for _ in range(power_steps):
            current_score, _, _, _, H_fit, rate_weights = evaluate()
            gradient = cal_power_gradient(H_fit, p, sigma2, rate_weights)
            maximum_gradient = float(np.max(np.abs(gradient)))
            if maximum_gradient <= 1.0e-14:
                break
            proposal_scale = Pt / maximum_gradient
            projected = project_simplex(p + proposal_scale * gradient, Pt)
            direction = projected - p
            directional_derivative = float(gradient @ direction)
            if directional_derivative <= 1.0e-14:
                break
            step_length = 1.0
            accepted = False
            original_power = p.copy()
            for _ in range(max_line_search):
                p = original_power + step_length * direction
                trial_score = evaluate()[0]
                if trial_score >= current_score + armijo * step_length * directional_derivative:
                    accepted = True
                    record_candidate()
                    break
                step_length *= backtrack
            if not accepted:
                p = original_power
                break

        for _ in range(phase_steps):
            current_score, _, _, gama, H_fit, rate_weights = evaluate()
            gradient = cal_gradient(phase, H_fit, gama, p, W_T_1, W_T, L, N, K, G,
                                    sigma2, rate_weights=rate_weights)
            maximum_gradient = float(np.max(np.abs(gradient)))
            if maximum_gradient <= 1.0e-14:
                break
            proposal_scale = math.pi / maximum_gradient
            direction = proposal_scale * gradient
            directional_derivative = float(np.sum(gradient * direction))
            step_length = 1.0
            accepted = False
            original_angles = np.angle(phase)
            for _ in range(max_line_search):
                phase = np.exp(1j * (original_angles + step_length * direction))
                trial_score = evaluate()[0]
                if trial_score >= current_score + armijo * step_length * directional_derivative:
                    accepted = True
                    record_candidate()
                    break
                step_length *= backtrack
            if not accepted:
                phase = np.exp(1j * original_angles)
                break

        sum_rate, rates, maximum_violation = record_candidate()
        violation = qos_threshold - rates
        dual = np.maximum(0.0, dual + penalty * violation)
        if maximum_violation > 0.75 * previous_violation:
            penalty = min(penalty * penalty_growth, max_penalty)
        previous_violation = maximum_violation

    if best_feasible is not None:
        result_sum, result_phase, result_power, result_rates = best_feasible
        feasible = True
    else:
        _, negative_sum, result_phase, result_power, result_rates = best_restoration
        result_sum = -negative_sum
        feasible = False
    return float(result_sum), result_phase, result_power, result_rates, feasible
