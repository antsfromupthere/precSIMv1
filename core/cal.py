import numpy as np
from core.iwf import iwf


def cal_W_SIM(phase_transmit, W_T_1, W_T, L=None):

    phase_transmit = np.asarray(phase_transmit, dtype=np.complex128)
    W_T_1 = np.asarray(W_T_1, dtype=np.complex128)
    W_T = np.asarray(W_T, dtype=np.complex128)

    if L is None:
        L = phase_transmit.shape[1]

    W_SIM = phase_transmit[:, 0][:, None] * W_T_1

    for l in range(L - 1):
        temp = W_T @ W_SIM
        W_SIM = phase_transmit[:, l + 1][:, None] * temp

    return W_SIM


def cal_sum_rate_given_p(G, W_SIM, K, p, sigma2):

    H_fit = G @ W_SIM

    gama = np.zeros(K, dtype=float)
    R = np.zeros(K, dtype=float)

    for ii in range(K):
        signal = np.abs(H_fit[ii, ii])**2 * p[ii]

        interference = (
            np.sum(np.abs(H_fit[ii, :])**2 * p)
            - np.abs(H_fit[ii, ii])**2 * p[ii]
        )

        gama[ii] = signal / (interference + sigma2)

        R[ii] = np.log2(1.0 + gama[ii])

    return R, gama, H_fit


def cal_sum_rate(G, W_SIM, K, Pt, sigma2, max_iwf=100, tol_iwf=1e-6):

    H_fit = G @ W_SIM

    p = iwf(Pt, sigma2, H_fit, K, tol=tol_iwf, max_iter=max_iwf)

    gains = np.abs(H_fit) ** 2
    received = gains @ p + sigma2
    desired = np.diag(gains) * p
    gama = desired / (received - desired)
    R = np.log2(1.0 + gama)

    return R, gama, H_fit, p


def cal_gradient(
    phase_transmit,
    H_fit,
    gama,
    p,
    W_T_1,
    W_T,
    L,
    N,
    K,
    G,
    sigma2,
    rate_weights=None,
):
    phase_transmit = np.asarray(phase_transmit, dtype=np.complex128)
    H_fit = np.asarray(H_fit, dtype=np.complex128)
    gama = np.asarray(gama, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    W_T_1 = np.asarray(W_T_1, dtype=np.complex128)
    W_T = np.asarray(W_T, dtype=np.complex128)
    G = np.asarray(G, dtype=np.complex128)
    if rate_weights is None:
        rate_weights = np.ones(K, dtype=float)
    rate_weights = np.asarray(rate_weights, dtype=float).reshape(-1)

    PD_transmit_phase = np.zeros((N, L), dtype=float)

    left_products = [W_T_1.copy()]
    for layer in range(1, L):
        left_products.append(
            W_T @ (phase_transmit[:, layer - 1, None] * left_products[-1])
        )
    right_products = [np.empty((K, N))] * L
    right_products[L - 1] = G.copy()
    for layer in range(L - 2, -1, -1):
        right_products[layer] = right_products[layer + 1] @ (
            phase_transmit[:, layer + 1, None] * W_T
        )

    received_power = np.sum(np.abs(H_fit) ** 2 * p[None, :], axis=1) + sigma2
    delta = 1.0 / received_power
    coefficients = -delta[:, None] * gama[:, None] * p[None, :]
    diagonal = np.arange(K)
    coefficients[diagonal, diagonal] = delta * p
    coefficients *= rate_weights[:, None]

    for layer in range(L):
        derivative_factors = (
            right_products[layer][:, :, None]
            * left_products[layer][None, :, :]
            * phase_transmit[None, :, layer, None]
        )
        imaginary_products = np.imag(
            np.conj(derivative_factors) * H_fit[:, None, :]
        )
        PD_transmit_phase[:, layer] = (
            2.0
            / np.log(2.0)
            * np.sum(coefficients[:, None, :] * imaginary_products, axis=(0, 2))
        )

    return PD_transmit_phase


def cal_power_gradient(H_fit, p, sigma2, rate_weights):

    H_fit = np.asarray(H_fit, dtype=np.complex128)
    p = np.asarray(p, dtype=float).reshape(-1)
    K = p.size
    rate_weights = np.asarray(rate_weights, dtype=float).reshape(-1)

    gains = np.abs(H_fit) ** 2
    desired = np.diag(gains)
    interference = gains @ p - desired * p + float(sigma2)
    total_received = interference + desired * p

    jacobian = gains * (
        1.0 / total_received[:, None] - 1.0 / interference[:, None]
    ) / np.log(2.0)
    diagonal = np.arange(K)
    jacobian[diagonal, diagonal] = desired / (
        np.log(2.0) * total_received
    )
    return rate_weights @ jacobian
