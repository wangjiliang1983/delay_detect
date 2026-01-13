import numpy as np


def generate_m_sequence(order: int = 10, seed: int = 1) -> np.ndarray:
    """
    Generate m-sequence (Maximum Length Sequence)

    Parameters:
        order: Order of the sequence, default is 10
        seed: Initial state seed, default is 1

    Returns:
        m-sequence array, length is 2^order - 1, values are +1/-1
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
    BPSK Modulation

    Parameters:
        bits: Input symbol sequence (+1/-1)
        fc: Carrier frequency (Hz)
        fs: Sampling frequency (Hz)

    Returns:
        Modulated real-valued signal
    """
    n = np.arange(len(bits))
    carrier = np.cos(2 * np.pi * fc / fs * n)
    return bits * carrier


def tx(num_periods: int = 10, fc: float = 200e6, fs: float = 1e9, order: int = 10) -> np.ndarray:
    """
    Transmitter: Generate m-sequence and perform BPSK modulation

    Parameters:
        num_periods: Number of periods of the m-sequence
        fc: Carrier frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Order of the m-sequence

    Returns:
        Transmitted signal array
    """
    m_seq = generate_m_sequence(order)
    one_period = m_seq
    signal = np.tile(one_period, num_periods)
    tx_signal = bpsk_modulate(signal, fc, fs)
    return tx_signal
