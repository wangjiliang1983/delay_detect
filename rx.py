import numpy as np


def cross_correlation(sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    """
    Cross-correlation calculation (using FFT acceleration)

    Parameters:
        sig1: Signal 1
        sig2: Signal 2

    Returns:
        Cross-correlation result
    """
    n = len(sig1) + len(sig2) - 1
    F1 = np.fft.fft(sig1, n)
    F2 = np.fft.fft(sig2, n)
    corr_fft = F1 * np.conj(F2)
    correlation = np.fft.ifft(corr_fft)
    return np.real(correlation)


def find_peak_index(correlation: np.ndarray) -> int:
    """
    Find the peak position of the cross-correlation

    Parameters:
        correlation: Cross-correlation result

    Returns:
        Peak index
    """
    return int(np.argmax(correlation))


def parabolic_interpolation(y_left: float, y_peak: float, y_right: float) -> float:
    """
    Parabolic interpolation to estimate peak offset

    Parameters:
        y_left: Amplitude of the point to the left of the peak (index -1)
        y_peak: Amplitude of the peak point (index 0)
        y_right: Amplitude of the point to the right of the peak (index +1)

    Returns:
        Offset delta (-0.5 ~ +0.5 sampling period)
    """
    numerator = y_left - y_right
    denominator = 2 * (y_left - 2 * y_peak + y_right)

    if np.abs(denominator) < 1e-10:
        return 0.0

    delta = numerator / denominator
    delta = np.clip(delta, -0.5, 0.5)
    return delta


def rx(rx_signal: np.ndarray, tx_signal: np.ndarray, fs: float) -> float:
    """
    Receiver: Estimate signal delay

    Parameters:
        rx_signal: Received signal
        tx_signal: Transmitted signal (reference signal)
        fs: Sampling frequency (Hz)

    Returns:
        Estimated delay (seconds)
    """
    correlation = cross_correlation(rx_signal, tx_signal)
    k_peak = find_peak_index(correlation)

    n = len(correlation)
    tx_len = len(tx_signal)

    effective_k = k_peak
    if k_peak > tx_len:
        effective_k = k_peak - n

    if effective_k <= 0:
        y_left = 0
        y_peak = correlation[0]
        y_right = correlation[1]
    elif effective_k >= n - 1:
        y_left = correlation[n - 2]
        y_peak = correlation[n - 1]
        y_right = 0
    else:
        y_left = correlation[k_peak - 1]
        y_peak = correlation[k_peak]
        y_right = correlation[k_peak + 1]

    delta = parabolic_interpolation(y_left, y_peak, y_right)
    ts = 1 / fs
    total_delay = (effective_k + delta) * ts
    return total_delay
