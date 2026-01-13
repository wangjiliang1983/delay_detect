import numpy as np
from typing import Optional


def fractional_delay(signal: np.ndarray, delay_ns: float, fs: float) -> np.ndarray:
    """
    Fractional Delay (Time Domain Implementation)

    Parameters:
        signal: Input signal
        delay_ns: Delay (ns)
        fs: Sampling frequency (Hz)

    Returns:
        Signal after delay
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
    Add Additive White Gaussian Noise (AWGN)
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
    return signal + noise


def channel(tx_signal: np.ndarray, delay_ns: float, fs: float, snr_db: Optional[float] = None) -> np.ndarray:
    """
    Intermediate delay processing: apply fractional delay and optionally add noise
    """
    rx_signal = fractional_delay(tx_signal, delay_ns, fs)
    if snr_db is not None:
        rx_signal = add_awgn(rx_signal, snr_db)
    return rx_signal
