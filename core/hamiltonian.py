# hamiltonian.py
import numpy as np
from core.basis import VJMBasis
from core.states import StateVector, DensityMatrix
from core.electric_field import ElectricField
from typing import Callable


def generate_free_hamiltonian(basis: VJMBasis, h=1.0, omega=1.0, delta_omega=0.0, B=1.0, alpha=0.0):
    """
    分子の自由ハミルトニアン H0 を生成（単位は任意、一貫性があればOK）
    E(V, J) = h*ω*(V+1/2) - h*Δω*(V+1/2)**2 + (B - α*(V+1/2))*J*(J+1)

    Parameters
    ----------
    h : float
        ディラック定数（任意単位）
    omega : float
        振動固有周波数
    delta_omega : float
        振動の非調和性補正項
    B : float
        回転定数
    alpha : float
        振動-回転相互作用定数
    """
    size = basis.size()
    H0 = np.zeros((size, size), dtype=np.float64)
    for i, (V, J, M) in enumerate(basis.basis):
        vterm = V + 0.5
        energy = h * omega * vterm - h * delta_omega * vterm**2
        energy += (B - alpha * vterm) * J * (J + 1)
        H0[i, i] = energy
    return H0


def transition_dipole_moment_linear_molecule(quanta1, quanta2, axis='z'):
    """
    直線分子における2つの量子状態間の遷移双極子モーメントの回転準位依存部分を計算する

    Parameters
    ----------
    quanta1 : list or array-like
        初期状態の量子数 [V, J, M] もしくは [V, J] もしくは [V]
    quanta2 : list or array-like
        最終状態の量子数 [V, J, M] もしくは [V, J] もしくは [V]

    Returns
    -------
    float
        遷移双極子モーメントの絶対値（数値）

    Notes
    -----
    - 選択則に従わない遷移には 0 を返します
    """
    quanta1 = np.array(quanta1)
    quanta2 = np.array(quanta2)
    delta = quanta2 - quanta1

    # 振動部分（ΔV = ±1）
    if delta[0] == 1:
        tdm = np.sqrt(quanta1[0] + 1)
    elif delta[0] == -1:
        tdm = np.sqrt(quanta2[0] + 1)
    else:
        return 0.0  # ΔV != ±1 → 禁制遷移

    # 回転部分（ΔJ = ±1）
    if len(delta) >= 2:
        if delta[1] == 1:
            tdm *= np.sqrt(quanta1[1] / (2 * quanta1[1] + 1)) / 2
        elif delta[1] == -1:
            tdm *= np.sqrt((quanta1[1] + 1) / (2 * quanta1[1] + 1)) / 2
        else:
            return 0.0  # ΔJ ≠ ±1 → 禁制

    # 磁気部分（ΔM = ±1, 0）
    if len(delta) == 3:
        J1 = quanta1[1]
        M1 = quanta1[2]
        J2 = quanta2[1]
        M2 = quanta2[2]
        if axis == 'x':
            if delta[1] == 1:
                if delta[2] in [1, -1]:
                    tdm *= -delta[2] * np.sqrt((J2 + delta[2] * M2 + 1) * (J2 + delta[2] * M2 + 2) /
                                ((2 * J2 + 1) * (2 * J2 + 3))) / 2
                else:
                    return 0.0
            elif delta[1] == -1:
                if delta[2] in [1, -1]:
                    tdm *= delta[2] * np.sqrt((J1 - delta[2] * M1 + 1) * (J1 - delta[2] * M1 + 2) /
                                ((2 * J1 + 1) * (2 * J1 + 3))) / 2
                else:
                    return 0.0
            else:
                return 0.0  # ΔJ ≠ ±1 → 禁制
        elif axis == 'y':
            if delta[1] == 1:
                if delta[2] in [1, -1]:
                    tdm *= -1j * np.sqrt((J2 - delta[2] * M2 + 1) * (J2 - delta[2] * M2 + 2) /
                                ((2 * J2 + 1) * (2 * J2 + 3))) / 2
                else:
                    return 0.0
            elif delta[1] == -1:
                if delta[2] in [1, -1]:
                    tdm *= 1j * np.sqrt((J1 + delta[2] * M1 + 1) * (J1 + delta[2] * M1 + 2) /
                                ((2 * J1 + 1) * (2 * J1 + 3))) / 2
                else:
                    return 0.0
            else:
                return 0.0
        # z軸の場合
        # ΔM = 0 の場合のみ値が非ゼロ
        elif axis == 'z':
            if delta[1] == 1:
                if delta[2] == 0:
                    tdm *= np.sqrt((J2 + 1 - M2) * (J2 + 1 + M2) /
                                ((2 * J2 + 1) * (2 * J2 + 3)))
                else:
                    return 0.0
            elif delta[1] == -1:
                if delta[2] == 0:
                    tdm *= np.sqrt((J1 + 1 - M1) * (J1 + 1 + M1) /
                                ((2 * J1 + 1) * (2 * J1 + 3)))
                else:
                    return 0.0
            else:
                return 0.0
            

    return tdm


def generate_dipole_matrix(basis: VJMBasis, mu0:float = 1.0, tdm_func: Callable = transition_dipole_moment_linear_molecule, axis='z'):
    """
    遷移双極子行列を生成（数値）
    tdm_func: quanta1, quanta2 -> 値 を返す関数（numpyベース）
    """
    size = basis.size()
    mu = np.zeros((size, size), dtype=np.complex128)
    for i, q1 in enumerate(basis.basis):
        for j, q2 in enumerate(basis.basis):
            val = tdm_func(q1, q2, axis=axis)
            if val != 0:
                mu[i, j] = val
    return mu * mu0