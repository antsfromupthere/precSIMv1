import numpy as np
from core.cal import cal_W_SIM

TINY = np.finfo(float).tiny


def unit_modulus_projection(vector):
    vector = np.asarray(vector, dtype=np.complex128)
    magnitude = np.abs(vector)
    result = np.ones_like(vector)
    nonzero = magnitude > TINY
    result[nonzero] = vector[nonzero] / magnitude[nonzero]
    return result


def nonnegative_ball_projection(vector, radius):
    result = np.maximum(np.asarray(vector, dtype=float), 0.0)
    norm = float(np.linalg.norm(result))
    if norm > radius and norm > 0.0:
        result *= radius / norm
    return result


def nonnegative_budget_projection(vector, Pt):
    result = np.maximum(np.asarray(vector, dtype=float), 0.0)
    if float(np.sum(result)) <= Pt:
        return result
    sorted_values = np.sort(result)[::-1]
    cumulative = np.cumsum(sorted_values)
    indices = np.arange(1, result.size + 1)
    active = np.flatnonzero(sorted_values * indices > cumulative - Pt)
    threshold = (cumulative[active[-1]] - Pt) / (active[-1] + 1)
    return np.maximum(result - threshold, 0.0)


def real_halfspace_projection(point, normal, offset):
    point = np.asarray(point, dtype=float)
    normal = np.asarray(normal, dtype=float)
    violation = float(normal @ point + offset)
    if violation <= 0.0:
        return point.copy()
    norm_squared = float(normal @ normal)
    if norm_squared <= TINY:
        return point.copy()
    return point - (violation / norm_squared) * normal


def complex_halfspace_projection(point, normal, offset):
    point = np.asarray(point, dtype=np.complex128)
    normal = np.asarray(normal, dtype=np.complex128)
    violation = float(np.real(np.vdot(point, normal)) + offset)
    if violation <= 0.0:
        return point.copy()
    norm_squared = float(np.real(np.vdot(normal, normal)))
    if norm_squared <= TINY:
        return point.copy()
    return point - (violation / norm_squared) * normal


def recover_feasible_power(H_fit, physical_power, Pt, gamma, sigma2,
                           maximum_cycles=1000, tolerance=1.0e-10):
    physical_power = np.asarray(physical_power, dtype=float)
    if gamma <= 0.0:
        return nonnegative_budget_projection(physical_power, Pt)

    gains = np.abs(H_fit) ** 2
    K = gains.shape[0]
    normals = []
    offsets = []
    coupling = np.zeros((K, K), dtype=float)
    noise_vector = np.zeros(K, dtype=float)
    for user in range(K):
        desired_gain = float(gains[user, user])
        if desired_gain <= TINY:
            return None
        normal = gamma * gains[user] / desired_gain
        normal[user] = -1.0
        normals.append(normal)
        offsets.append(gamma * sigma2 / desired_gain)
        coupling[user] = gamma * gains[user] / desired_gain
        coupling[user, user] = 0.0
        noise_vector[user] = gamma * sigma2 / desired_gain

    if float(np.max(np.abs(np.linalg.eigvals(coupling)))) >= 1.0 - 1.0e-12:
        return None
    try:
        minimum_power = np.linalg.solve(np.eye(K) - coupling, noise_vector)
    except np.linalg.LinAlgError:
        return None
    if np.any(minimum_power < -1.0e-9) or float(np.sum(minimum_power)) > Pt + 1.0e-9:
        return None

    point = physical_power.copy()
    corrections = [np.zeros_like(point) for _ in range(K + 1)]
    for cycle in range(1, maximum_cycles + 1):
        previous = point.copy()

        shifted = point + corrections[0]
        projected = nonnegative_budget_projection(shifted, Pt)
        corrections[0] = shifted - projected
        point = projected

        for index, (normal, offset) in enumerate(zip(normals, offsets), start=1):
            shifted = point + corrections[index]
            projected = real_halfspace_projection(shifted, normal, offset)
            corrections[index] = shifted - projected
            point = projected

        maximum_violation = max(
            0.0,
            max(float(normal @ point + offset) for normal, offset in zip(normals, offsets)),
            float(-np.min(point)),
            float(np.sum(point) - Pt),
        )
        if np.linalg.norm(point - previous) <= tolerance and maximum_violation <= tolerance:
            break

    maximum_violation = max(
        0.0,
        max(float(normal @ point + offset) for normal, offset in zip(normals, offsets)),
        float(-np.min(point)),
        float(np.sum(point) - Pt),
    )
    if maximum_violation > 1.0e-7:
        return None
    return nonnegative_budget_projection(point, Pt)


def evaluate_rates(W_T_1, W_T, G, phase, amplitude, sigma2):
    W_SIM = cal_W_SIM(phase, W_T_1, W_T)
    H_fit = G @ W_SIM
    power = np.square(np.maximum(np.asarray(amplitude, dtype=float), 0.0))
    received = np.abs(H_fit) ** 2 * power[None, :]
    signal = np.diag(received)
    interference = np.sum(received, axis=1) - signal
    gama = signal / np.maximum(interference + sigma2, TINY)
    rates = np.log2(1.0 + gama)
    return float(np.sum(rates)), rates, gama, H_fit


def wmmse_auxiliaries(H_fit, amplitude, sigma2):
    total_received = np.sum(np.abs(H_fit) ** 2 * amplitude[None, :] ** 2, axis=1) + sigma2
    desired = np.diag(H_fit) * amplitude
    receiver = desired / np.maximum(total_received, TINY)
    mse = np.real(1.0 - np.conj(receiver) * desired)
    weight = 1.0 / np.maximum(mse, 1.0e-12)
    return receiver, weight


def power_qos_ipdd_projection(target, gains, user, gamma, sigma2, eta_initial, eta_decay,
                              tolerance_squared, maximum_iterations):
    if gamma <= 0.0:
        return np.asarray(target, dtype=float).copy()

    target = np.asarray(target, dtype=float)
    gains = np.asarray(gains, dtype=float)
    z = target.copy()
    dual = np.zeros_like(target)
    eta = float(eta_initial)
    p = target.copy()

    for iteration in range(1, maximum_iterations + 1):
        desired_z = np.zeros_like(target)
        desired_z[user] = gains[user] * z[user]
        interference_z = gains * z
        interference_z[user] = 0.0
        normal_p = gamma * interference_z - desired_z

        z_tilde = z + eta * dual
        denominator = 2.0 + 1.0 / eta
        unconstrained_p = (2.0 * target + z_tilde / eta) / denominator
        p = real_halfspace_projection(unconstrained_p, normal_p, gamma * sigma2)

        desired_p = np.zeros_like(target)
        desired_p[user] = gains[user] * p[user]
        interference_p = gains * p
        interference_p[user] = 0.0
        normal_z = gamma * interference_p - desired_p

        unconstrained_z = p - eta * dual
        z_new = real_halfspace_projection(unconstrained_z, normal_z, gamma * sigma2)
        residual = float(np.sum(np.square(p - z_new)))
        dual += (z_new - p) / eta
        z = z_new
        if residual <= tolerance_squared:
            return p
        eta = max(eta * eta_decay, 1.0e-12)

    return p


def optimize_power_pdmm(H_fit, amplitude, receiver, weight, Pt, gamma, sigma2, mu_initial,
                        mu_growth, mu_growth_every, eta_initial, eta_decay,
                        tolerance_squared, maximum_pdmm_iterations, maximum_ipdd_iterations):
    gains = np.abs(H_fit) ** 2
    quadratic = np.sum((weight * np.abs(receiver) ** 2)[:, None] * gains, axis=0)
    linear = np.real(weight * np.conj(receiver) * np.diag(H_fit))

    point = np.asarray(amplitude, dtype=float).copy()
    radius = float(np.sqrt(Pt))
    mu = float(mu_initial)

    for iteration in range(1, maximum_pdmm_iterations + 1):
        projections = [nonnegative_ball_projection(point, radius)]
        for user in range(point.size):
            projections.append(power_qos_ipdd_projection(
                point, gains[user], user, gamma, sigma2, eta_initial, eta_decay,
                tolerance_squared, maximum_ipdd_iterations,
            ))

        projection_sum = np.sum(projections, axis=0)
        average_projection = projection_sum / len(projections)
        point_new = (linear + mu * projection_sum) / (quadratic + mu * len(projections))
        residual = float(np.sum(np.square(point_new - average_projection)))
        point = point_new
        if residual <= tolerance_squared:
            break
        if iteration % mu_growth_every == 0:
            mu *= mu_growth

    return nonnegative_ball_projection(point, radius)


def layer_vectors(G, prefix, suffix, amplitude):
    suffix_channel = (suffix.conj().T @ G.conj().T).T
    return (
        suffix_channel[:, None, :]
        * np.conj(prefix.T)[None, :, :]
        * amplitude[None, :, None]
    )


def apply_phase_qos_matrix(vectors_for_user, user, gamma, vector):
    scales = np.ones(vectors_for_user.shape[0], dtype=float)
    scales[user] = -1.0 / gamma
    inner_products = vectors_for_user.conj() @ vector
    return (scales * inner_products) @ vectors_for_user


def phase_qos_ipdd_projection(target, vectors_for_user, user, gamma, sigma2, eta_initial,
                              eta_decay, tolerance_squared, maximum_iterations):
    if gamma <= 0.0:
        return np.asarray(target, dtype=np.complex128).copy()

    target = np.asarray(target, dtype=np.complex128)
    kappa = target.copy()
    dual = np.zeros_like(target)
    eta = float(eta_initial)
    phi = target.copy()

    for iteration in range(1, maximum_iterations + 1):
        normal_phi = apply_phase_qos_matrix(vectors_for_user, user, gamma, kappa)
        kappa_tilde = kappa + eta * dual
        denominator = 1.0 + 1.0 / (2.0 * eta)
        unconstrained_phi = (target + kappa_tilde / (2.0 * eta)) / denominator
        phi = complex_halfspace_projection(unconstrained_phi, normal_phi, sigma2)

        normal_kappa = apply_phase_qos_matrix(vectors_for_user, user, gamma, phi)
        unconstrained_kappa = phi - eta * dual
        kappa_new = complex_halfspace_projection(unconstrained_kappa, normal_kappa, sigma2)

        residual = float(np.sum(np.abs(phi - kappa_new) ** 2))
        dual += (kappa_new - phi) / eta
        kappa = kappa_new
        if residual <= tolerance_squared:
            return phi
        eta = max(eta * eta_decay, 1.0e-12)

    return phi


def optimize_phase_layer_pdmm(G, phase, amplitude, layer, receiver, weight, gamma, sigma2,
                              mu_initial, mu_growth, mu_growth_every, eta_initial, eta_decay,
                              tolerance_squared, maximum_pdmm_iterations,
                              maximum_ipdd_iterations, prefix, suffix):
    vectors = layer_vectors(G, prefix, suffix, amplitude)
    K, _, N = vectors.shape
    wmmse_scale = weight * np.abs(receiver) ** 2
    quadratic = np.einsum("k,kin,kim->nm", wmmse_scale, vectors, vectors.conj(), optimize=True)
    quadratic = 0.5 * (quadratic + quadratic.conj().T)
    diagonal_vectors = vectors[np.arange(K), np.arange(K)]
    linear = np.einsum("k,kn->n", weight * receiver, diagonal_vectors, optimize=True)

    eigenvalues, eigenvectors = np.linalg.eigh(quadratic)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    point = np.asarray(phase[:, layer], dtype=np.complex128).copy()
    mu = float(mu_initial)

    for iteration in range(1, maximum_pdmm_iterations + 1):
        projections = [unit_modulus_projection(point)]
        for user in range(K):
            projections.append(phase_qos_ipdd_projection(
                point, vectors[user], user, gamma, sigma2, eta_initial, eta_decay,
                tolerance_squared, maximum_ipdd_iterations,
            ))

        projection_sum = np.sum(projections, axis=0)
        average_projection = projection_sum / len(projections)
        right_hand_side = linear + mu * projection_sum
        transformed = eigenvectors.conj().T @ right_hand_side
        point_new = eigenvectors @ (transformed / (eigenvalues + mu * len(projections)))
        residual = float(np.sum(np.abs(point_new - average_projection) ** 2))
        point = point_new
        if residual <= tolerance_squared:
            break
        if iteration % mu_growth_every == 0:
            mu *= mu_growth

    return unit_modulus_projection(point)


def solve_vd_bsum(W_T_1, W_T, G, Pt, sigma2, phase_initial, power_initial, qos_threshold,
                  max_outer=100, max_pdmm=50, max_ipdd=50, outer_tolerance=1.0e-4,
                  projection_tolerance_squared=1.0e-4, eta_initial=100.0, eta_decay=0.8,
                  mu_initial=0.1, mu_growth=2.0, mu_growth_every=4, tol=1.0e-6):

    W_T_1 = np.asarray(W_T_1, dtype=np.complex128)
    W_T = np.asarray(W_T, dtype=np.complex128)
    G = np.asarray(G, dtype=np.complex128)
    phase = np.asarray(phase_initial, dtype=np.complex128).copy()
    initial_power = np.asarray(power_initial, dtype=float).reshape(-1)

    phase = np.column_stack(
        [unit_modulus_projection(phase[:, layer]) for layer in range(phase.shape[1])]
    )
    amplitude = nonnegative_ball_projection(
        np.sqrt(np.maximum(initial_power, 0.0)),
        np.sqrt(Pt),
    )
    gamma = float(2.0**qos_threshold - 1.0)

    best_feasible = None
    best_any = None

    def record_candidate():
        nonlocal best_feasible, best_any
        sum_rate, rates, _, H_fit = evaluate_rates(W_T_1, W_T, G, phase, amplitude, sigma2)
        candidate = (sum_rate, phase.copy(), amplitude.copy(), rates.copy())
        if best_any is None or sum_rate > best_any[0]:
            best_any = candidate
        if np.all(rates >= qos_threshold - tol):
            if best_feasible is None or sum_rate > best_feasible[0]:
                best_feasible = candidate

        recovered_power = recover_feasible_power(H_fit, np.square(amplitude), Pt, gamma, sigma2)
        if recovered_power is not None:
            recovered_amplitude = np.sqrt(np.maximum(recovered_power, 0.0))
            recovered_sum, recovered_rates, _, _ = evaluate_rates(
                W_T_1, W_T, G, phase, recovered_amplitude, sigma2,
            )
            recovered_feasible = bool(np.all(recovered_rates >= qos_threshold - tol))
            if recovered_feasible and (best_feasible is None or recovered_sum > best_feasible[0]):
                best_feasible = (recovered_sum, phase.copy(), recovered_amplitude, recovered_rates)
        return sum_rate, H_fit

    previous_sum_rate, H_fit = record_candidate()

    for outer_iteration in range(1, max_outer + 1):
        receiver, weight = wmmse_auxiliaries(H_fit, amplitude, sigma2)
        previous_amplitude = amplitude.copy()
        proposed_amplitude = optimize_power_pdmm(
            H_fit, amplitude, receiver, weight, Pt, gamma, sigma2, mu_initial, mu_growth,
            mu_growth_every, eta_initial, eta_decay, projection_tolerance_squared,
            max_pdmm, max_ipdd,
        )
        recovered_block_power = recover_feasible_power(
            H_fit, np.square(proposed_amplitude), Pt, gamma, sigma2,
        )
        if recovered_block_power is not None:
            amplitude = np.sqrt(np.maximum(recovered_block_power, 0.0))
        else:
            amplitude = previous_amplitude

        layers = phase.shape[1]
        suffixes = [np.eye(phase.shape[0], dtype=np.complex128) for _ in range(layers)]
        for layer_index in range(layers - 2, -1, -1):
            suffixes[layer_index] = (
                suffixes[layer_index + 1] @ (phase[:, layer_index + 1, None] * W_T)
            )
        prefix = W_T_1.copy()
        for layer in range(layers):
            phase[:, layer] = optimize_phase_layer_pdmm(
                G, phase, amplitude, layer, receiver, weight, gamma, sigma2, mu_initial,
                mu_growth, mu_growth_every, eta_initial, eta_decay,
                projection_tolerance_squared, max_pdmm, max_ipdd, prefix, suffixes[layer],
            )
            prefix = W_T @ (phase[:, layer, None] * prefix)

        sum_rate, H_fit = record_candidate()
        if abs(sum_rate - previous_sum_rate) <= outer_tolerance:
            break
        previous_sum_rate = sum_rate

    feasible = best_feasible is not None
    selected = best_feasible if feasible else best_any
    selected_sum, selected_phase, selected_amplitude, selected_rates = selected
    return float(selected_sum), selected_phase, np.square(selected_amplitude), selected_rates, feasible
