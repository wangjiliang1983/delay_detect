import numpy as np


def cross_correlation(sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    """
    互相关计算（使用 FFT 加速）

    Parameters:
        sig1: 信号 1
        sig2: 信号 2

    Returns:
        互相关结果
    """
    n = len(sig1) + len(sig2) - 1
    F1 = np.fft.fft(sig1, n)
    F2 = np.fft.fft(sig2, n)
    corr_fft = F1 * np.conj(F2)
    correlation = np.fft.ifft(corr_fft)
    return np.real(correlation)


def find_peak_index(correlation: np.ndarray) -> int:
    """
    查找互相关峰值位置

    Parameters:
        correlation: 互相关结果

    Returns:
        峰值索引
    """
    return int(np.argmax(correlation))


def parabolic_interpolation(y_left: float, y_peak: float, y_right: float) -> float:
    """
    抛物线插值估计峰值偏移

    Parameters:
        y_left: 峰值左侧点幅值 (索引 -1)
        y_peak: 峰值点幅值 (索引 0)
        y_right: 峰值右侧点幅值 (索引 +1)

    Returns:
        偏移量 δ (-0.5 ~ +0.5 采样周期)
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
    接收机：估计信号延时

    Parameters:
        rx_signal: 接收信号
        tx_signal: 发射信号（参考信号）
        fs: 采样频率 (Hz)

    Returns:
        估计延时 (秒)
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
