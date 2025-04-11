# シュレディンガー・リウヴィル時間発展
# propagator.py
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.sparse import kron, identity, csc_matrix
from scipy.interpolate import interp1d


def schrodinger_propagation(H0, E, psi0, tlist, mu_x, mu_y, mu_z=None):
    dim = H0.shape[0]
    E_vec = E.get_vector_field()
    E_interp = [
        interp1d(tlist, E_vec[0], kind='cubic', fill_value="extrapolate"),
        interp1d(tlist, E_vec[1], kind='cubic', fill_value="extrapolate")
    ]
    def rhs(t, y):
        Ex_t = E_interp[0](t)
        Ey_t = E_interp[1](t)
        H = H0 + mu_x * Ex_t + mu_y * Ey_t
        return -1j * H @ y

    y0 = psi0.astype(np.complex128)
    sol = solve_ivp(rhs, [tlist[0], tlist[-1]], y0, t_eval=tlist, method='RK45')
    return sol.y.T


def liouville_propagation(H0, E, rho0, tlist, mu_x, mu_y, mu_z=None):
    dim = H0.shape[0]
    rho_flat0 = rho0.flatten()
    E_vec = E.get_vector_field()
    E_interp = [
        interp1d(tlist, E_vec[0], kind='cubic', fill_value="extrapolate"),
        interp1d(tlist, E_vec[1], kind='cubic', fill_value="extrapolate")
    ]
    def rhs(t, rho_flat):
        rho = rho_flat.reshape(dim, dim)
        Ex_t = E_interp[0](t)
        Ey_t = E_interp[1](t)
        H = H0 + mu_x * Ex_t + mu_y * Ey_t
        return (-1j * (H @ rho - rho @ H)).flatten()

    sol = solve_ivp(rhs, [tlist[0], tlist[-1]], rho_flat0, t_eval=tlist, method='RK45')
    return sol.y.T.reshape(-1, dim, dim)