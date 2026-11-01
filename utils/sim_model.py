import numpy as np
from utils.pl_model import make_pathloss
from utils.setSeed import set_seed


def make_W_TnCorr_T(n_max, N, wavelength, d_layer, d_element_SIM):

    W_T = np.zeros((N, N), dtype=np.complex128)
    Corr_T = np.zeros((N, N), dtype=np.float64)

    for mm1 in range(N):
        m_z = int(np.ceil((mm1 + 1) / n_max))
        m_x = (mm1 % n_max) + 1

        for mm2 in range(N):

            n_z = int(np.ceil((mm2 + 1) / n_max))
            n_x = (mm2 % n_max) + 1

            d_temp = np.sqrt((m_x - n_x)**2 + (m_z - n_z)**2) * d_element_SIM
            d_temp2 = np.sqrt(d_layer**2 + d_temp**2)

            W_T[mm2, mm1] = (
                wavelength**2 / 4
                * (
                    d_layer / (d_temp2 * d_temp2)
                    * (1 / (2 * np.pi * d_temp2) - 1j / wavelength)
                )
                * np.exp(1j * 2 * np.pi * d_temp2 / wavelength)
            )

            Corr_T[mm2, mm1] = np.sinc(2 * d_temp / wavelength)

    return W_T, Corr_T


def make_W_T_1(n_max, N, K, wavelength, d_layer, d_element_SIM):

    W_T_1 = np.zeros((N, K), dtype=np.complex128)

    for mm in range(N):
        m_y = int(np.ceil((mm + 1) / n_max))
        m_x = (mm % n_max) + 1

        for nn in range(K):
            nn1 = nn + 1

            d_transmit = np.sqrt(
                d_layer**2
                + (
                    (m_x - (1 + n_max) / 2.0) * d_element_SIM
                    - (nn1 - (1 + K) / 2.0) * wavelength / 2.0
                )**2
                + (
                    (m_y - (1 + n_max) / 2.0) * d_element_SIM
                )**2
            )

            W_T_1[mm, nn] = (
                wavelength**2 / 4.0
                * (
                    d_layer / (d_transmit * d_transmit)
                    * (1.0 / (2.0 * np.pi * d_transmit) - 1j / wavelength)
                )
                * np.exp(1j * 2.0 * np.pi * d_transmit / wavelength)
            )

    return W_T_1


def make_system(L, N, K, P_max_dBm, n_max=10, ant_gain=5.0, noise_dBm=-104.0):
    wavelength = 3.0e8 / 28.0e9
    d_element_SIM = wavelength / 2.0
    d_layer = 5.0 * wavelength / L
    W_T, Corr_T = make_W_TnCorr_T(n_max, N, wavelength, d_layer, d_element_SIM)
    W_T_1 = make_W_T_1(n_max, N, K, wavelength, d_layer, d_element_SIM)
    _, pathloss = make_pathloss(K, wavelength)
    eigvals, eigvecs = np.linalg.eigh(Corr_T)
    Corr_T_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0.0))) @ eigvecs.conj().T
    Pt = 10.0 ** (P_max_dBm / 10.0) * 10.0 ** (ant_gain / 10.0)
    sigma2 = 10.0 ** (noise_dBm / 10.0)
    return W_T, W_T_1, Corr_T_sqrt, pathloss, Pt, sigma2


def make_realizations(Corr_T_sqrt, pathloss, K, N, L, MonteCarloStep, seed=42):
    set_seed(seed)
    D = np.diag(np.sqrt(pathloss))
    channels = []
    phases = []
    for mc in range(MonteCarloStep):
        G_in = np.sqrt(0.5) * (
            np.random.randn(K, N) + 1j * np.random.randn(K, N)
        )
        G = D @ G_in @ Corr_T_sqrt
        phase_transmit_init = np.random.randn(N, L) + 1j * np.random.randn(N, L)
        phase_transmit_init = phase_transmit_init / np.abs(phase_transmit_init)
        channels.append(G)
        phases.append(phase_transmit_init)
    return np.stack(channels), np.stack(phases)
