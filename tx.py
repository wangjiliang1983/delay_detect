import numpy as np


def generate_m_sequence(order: int = 10, seed: int = 1) -> np.ndarray:
    """
    生成 m 序列（最大长度序列）

    Parameters:
        order: 阶数，默认 10
        seed: 初始状态种子，默认 1

    Returns:
        m 序列数组，长度为 2^order - 1，值为 +1/-1
    """
    n = (1 << order) - 1
    state = seed & ((1 << order) - 1)
    sequence = []

    for _ in range(n):
        sequence.append(state & 1)
        feedback = 0
        for tap in [10, 3]:
            if (state >> (tap - 1)) & 1:
                feedback ^= 1
        state = (state >> 1) | (feedback << (order - 1))

    symbols = 2 * np.array(sequence, dtype=np.float64) - 1
    return symbols


def bpsk_modulate(bits: np.ndarray, fc: float, fs: float) -> np.ndarray:
    """
    BPSK 调制

    Parameters:
        bits: 输入符号序列 (+1/-1)
        fc: 载波频率 (Hz)
        fs: 采样频率 (Hz)

    Returns:
        调制后的实数信号
    """
    n = np.arange(len(bits))
    carrier = np.cos(2 * np.pi * fc / fs * n)
    return bits * carrier


def tx(num_periods: int = 10, fc: float = 200e6, fs: float = 1e9, order: int = 10) -> np.ndarray:
    """
    发射机：生成 m 序列并进行 BPSK 调制

    Parameters:
        num_periods: m 序列周期数
        fc: 载波频率 (Hz)
        fs: 采样频率 (Hz)
        order: m 序列阶数

    Returns:
        发射信号数组
    """
    m_seq = generate_m_sequence(order)
    one_period = m_seq
    signal = np.tile(one_period, num_periods)
    tx_signal = bpsk_modulate(signal, fc, fs)
    return tx_signal
