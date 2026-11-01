import numpy as np


def iwf(P_total, sigma2, H, K, tol=1e-6, max_iter=100):
    H = np.asarray(H, dtype=np.complex128)

    p = np.full(K, P_total / K, dtype=float)
    p_new = np.zeros_like(p)
    gama = np.zeros(K, dtype=float)
    R = np.zeros(K, dtype=float)
    noise = np.zeros(K, dtype=float)
    water_floor = np.zeros(K, dtype=float)

    for ii in range(K):
        row = H[ii, :]
        noise[ii] = np.sum(np.abs(row) ** 2 * p) - (np.abs(H[ii, ii]) ** 2) * p[ii] + sigma2
        gama[ii] = (np.abs(H[ii, ii]) ** 2 * p[ii]) / noise[ii]
        R[ii] = np.log2(1.0 + gama[ii])

    sum_R_old = np.sum(R)
    sum_R_new = 2.0 * sum_R_old
    it = 1

    while abs(sum_R_new - sum_R_old) >= sum_R_old * tol and it <= max_iter:
        sum_R_old = sum_R_new

        for ii in range(K):
            water_floor[ii] = noise[ii] / (np.abs(H[ii, ii]) ** 2)

        order = np.argsort(water_floor)
        wf_sorted = water_floor[order]

        P_sum = 0.0
        idx_break = K - 1

        for ii in range(K - 1):
            P_sum = P_sum + (ii + 1) * (wf_sorted[ii + 1] - wf_sorted[ii])
            if P_sum >= P_total:
                idx_break = ii
                break

        if P_sum >= P_total:
            i_act = idx_break
            denom = i_act + 1.0
            PA_WF_part = wf_sorted[i_act + 1] - wf_sorted[:i_act + 1] - (P_sum - P_total) / denom
            PA_WF = np.concatenate([PA_WF_part, np.zeros(K - PA_WF_part.size)])
        else:
            i_act = K - 1
            denom = i_act + 1.0
            PA_WF = wf_sorted[i_act] - wf_sorted[:i_act + 1] + (P_total - P_sum) / denom

        p_new[order] = PA_WF

        p = (1.0 / K) * p_new + ((K - 1.0) / K) * p

        for ii in range(K):
            row = H[ii, :]
            noise[ii] = np.sum(np.abs(row) ** 2 * p) - (np.abs(H[ii, ii]) ** 2) * p[ii] + sigma2
            gama[ii] = (np.abs(H[ii, ii]) ** 2 * p[ii]) / noise[ii]
            R[ii] = np.log2(1.0 + gama[ii])

        sum_R_new = np.sum(R)
        it += 1

    return p
