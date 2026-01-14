import numpy as np


def cross_correlation(sig1: np.ndarray, sig2: np.ndarray, method: str = 'standard', upsample_factor: int = 16) -> np.ndarray:
    """
    Cross-correlation calculation (using FFT acceleration)

    Parameters:
        sig1: Signal 1
        sig2: Signal 2
        method: 'standard' or 'gcc_phat'
        upsample_factor: Oversampling factor for frequency domain interpolation

    Returns:
        Upsampled cross-correlation result
    """
    n = len(sig1) + len(sig2) - 1
    F1 = np.fft.fft(sig1, n)
    F2 = np.fft.fft(sig2, n)
    corr_fft = F1 * np.conj(F2)
    
    if method == 'gcc_phat':
        # GCC-PHAT Normalization
        corr_fft = corr_fft / (np.abs(corr_fft) + 1e-10)
    
    # Frequency domain zero-padding for oversampling
    if upsample_factor > 1:
        n_up = n * upsample_factor
        corr_fft_up = np.zeros(n_up, dtype=complex)
        
        # Handle odd/even length cases for correct frequency mapping
        if n % 2 == 1:
            half = (n + 1) // 2
            corr_fft_up[:half] = corr_fft[:half]
            corr_fft_up[- (n - half):] = corr_fft[half:]
        else:
            half = n // 2
            corr_fft_up[:half] = corr_fft[:half]
            # Nyquist component splitting
            corr_fft_up[half] = corr_fft[half] / 2
            corr_fft_up[- (n - half - 1):] = corr_fft[half+1:]
            corr_fft_up[n_up - n//2] = corr_fft[half] / 2
            
        corr_fft = corr_fft_up * upsample_factor # maintain energy scaling
    
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


def rx(rx_signal: np.ndarray, tx_signal: np.ndarray, fs: float, strategy: str = 'oversampling', upsample_factor: int = 16) -> float:
    """
    Receiver: Estimate signal delay

    Parameters:
        rx_signal: Received signal
        tx_signal: Transmitted signal (reference signal)
        fs: Sampling frequency (Hz)
        strategy: 'parabolic', 'oversampling', or 'gcc_phat'
        upsample_factor: Oversampling factor (only used for 'oversampling' and 'gcc_phat')

    Returns:
        Estimated delay (seconds)
    """
    if strategy == 'parabolic':
        # Standard Parabolic Interpolation (No Oversampling here)
        correlation = cross_correlation(rx_signal, tx_signal, method='standard', upsample_factor=1)
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

    elif strategy == 'oversampling':
        # Pure Oversampling
        correlation = cross_correlation(rx_signal, tx_signal, method='standard', upsample_factor=upsample_factor)
        k_peak = find_peak_index(correlation)
        current_upsample = upsample_factor
        
    elif strategy == 'gcc_phat':
        # GCC-PHAT (usually benefits from oversampling to find sharp peak location)
        correlation = cross_correlation(rx_signal, tx_signal, method='gcc_phat', upsample_factor=upsample_factor)
        k_peak = find_peak_index(correlation)
        current_upsample = upsample_factor
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Standard delay extraction for oversampled signals
    n_up = len(correlation)
    
    # Calculate delay in samples (upsampled)
    if k_peak > n_up // 2:
        shift_samples = k_peak - n_up
    else:
        shift_samples = k_peak
        
    # Convert to time
    fs_up = fs * current_upsample
    total_delay = shift_samples / fs_up
    
    return total_delay
