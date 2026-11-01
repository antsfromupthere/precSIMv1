import numpy as np
from core.cal import cal_W_SIM, cal_gradient, cal_sum_rate, cal_sum_rate_given_p


def algo_aoga(phase_transmit, Pt, p, W_T_1, W_T, L, N, K, G, sigma2,
              max_outer=100, max_backtrack=100, max_iwf=100, tol=1e-6):

    W_SIM = cal_W_SIM(phase_transmit, W_T_1, W_T, L)
    R, gama, H_fit = cal_sum_rate_given_p(G, W_SIM, K, p, sigma2)
    C_old = float(np.sum(R))
    C_new = C_old ** 2

    count = 1
    phase_phase_transmit = np.angle(phase_transmit)

    while abs(C_new - C_old) >= C_old * tol and count <= max_outer:
        C_old = C_new

        PD_transmit_phase = cal_gradient(phase_transmit, H_fit, gama,
                                         p, W_T_1, W_T, L, N, K, G, sigma2)

        max_pd = float(np.max(PD_transmit_phase))
        if not np.isfinite(max_pd) or max_pd == 0.0:
            C_new = C_old
            break
        yy = np.pi / max_pd

        C_old_1 = C_old
        C_new = 0.0
        count_2 = 1

        phase_transmit_temp = phase_transmit.copy()

        while C_new < C_old_1 and count_2 <= max_backtrack:
            phase_phase_transmit_temp = phase_phase_transmit + yy * PD_transmit_phase
            phase_transmit_temp = np.exp(1j * phase_phase_transmit_temp)
            W_SIM = cal_W_SIM(phase_transmit_temp, W_T_1, W_T, L)
            R, gama, H_fit, p = cal_sum_rate(G, W_SIM, K, Pt, sigma2, max_iwf)
            C_new = float(np.sum(R))

            yy *= 0.5
            count_2 += 1

        phase_transmit = phase_transmit_temp
        phase_phase_transmit = np.angle(phase_transmit)
        count += 1

    return C_new, phase_transmit, p, R
