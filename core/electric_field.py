# 電場波形生成
# electric_field.py
import numpy as np
from numpy import pi
from scipy.fft import fft, ifft, fftfreq

class ElectricField:
    """
    電場波形を表現するクラス（偏光、包絡線、GDD/TOD付き）
    """
    def __init__(self, tlist, envelope_func, carrier_freq, amplitude=1.0,
                 polarization=np.array([1.0]), gdd=0.0, tod=0.0):
        """
        Parameters
        ----------
        tlist : np.ndarray
            時間軸（fs）
        envelope_func : Callable
            包絡線関数（例: lambda t: np.exp(-(t/50)**2)）
        carrier_freq : float
            キャリア周波数（rad/fs）
        amplitude : float
            電場の振幅
        polarization : np.ndarray
            ジョーンズベクトル（例: [1, 1j]）
        gdd : float
            群遅延分散（fs^2）
        tod : float
            三次分散（fs^3）
        """
        self.tlist = tlist
        self.envelope = envelope_func(tlist)
        self.omega = carrier_freq
        self.amplitude = amplitude
        self.polarization = polarization / np.linalg.norm(polarization)
        self.gdd = gdd
        self.tod = tod
        self._generate_E_field()

    def _generate_E_field(self):
        envelope = self.envelope * self.amplitude
        carrier = np.exp(1j * self.omega * self.tlist)
        field = envelope * carrier

        # スペクトル → 分散位相の付加
        freq = fftfreq(len(self.tlist), d=(self.tlist[1] - self.tlist[0]))
        E_freq = fft(field)
        phase = np.where(
            freq >= 0,
            np.exp(-1j * np.pi * self.gdd * (2*pi*freq - self.omega)**2 + self.tod * (2*np.pi*freq - self.omega)**3),
            np.exp(+1j * np.pi * self.gdd * (2*pi*freq + self.omega)**2 + self.tod * (2*np.pi*freq + self.omega)**3)  
        )
        E_freq_disp = E_freq * phase
        self.E_complex = ifft(E_freq_disp)
        self.E_real = np.real(self.E_complex)  # 実部も保存

    def __call__(self, t):
        """
        任意の時刻 t における電場値（補間）を返す
        """
        return np.interp(t, self.tlist, self.E_real)

    def get_vector_field(self):
        """
        ベクトル電場（偏光含む）を返す（実部）：shape = (len(tlist), polarization_dim)
        """
        return np.real(np.outer(self.E_complex, self.polarization))

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.tlist, np.real(self.E_complex), label='Re E(t)')
        plt.plot(self.tlist, np.imag(self.E_complex), label='Im E(t)')
        plt.xlabel("Time (fs)")
        plt.ylabel("Electric Field")
        plt.legend()
        plt.title("Electric Field with Dispersion")
        plt.show()
