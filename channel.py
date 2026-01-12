import numpy as np
from typing import Optional


def fractional_delay(signal: np.ndarray, delay_ns: float, fs: float) -> np.ndarray:
    """
    分数阶延时（时域实现）

    Parameters:
        signal: 输入信号
        delay_ns: 延时 (ns)
        fs: 采样频率 (Hz)

    Returns:
        延时后的信号
    """
    n = len(signal)
    delay_samples = delay_ns * 1e-9 * fs
    eps = 1e-10

    result = np.zeros(n)

    for i in range(n):
        src_idx = i - delay_samples

        if src_idx < -eps:
            result[i] = 0
        elif src_idx >= n - 1 + eps:
            result[i] = 0
        else:
            src_idx = max(0, min(src_idx, n - 1 - eps))
            idx0 = int(np.floor(src_idx))
            idx1 = min(idx0 + 1, n - 1)
            t = src_idx - idx0
            result[i] = (1 - t) * signal[idx0] + t * signal[idx1]

    return result


def add_awgn(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    添加高斯白噪声
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
    return signal + noise


def channel(tx_signal: np.ndarray, delay_ns: float, fs: float, snr_db: Optional[float] = None) -> np.ndarray:
    """
    中间延时处理：应用分数阶延时并可选添加噪声
    """
    rx_signal = fractional_delay(tx_signal, delay_ns, fs)
    if snr_db is not None:
        rx_signal = add_awgn(rx_signal, snr_db)
    return rx_signal
