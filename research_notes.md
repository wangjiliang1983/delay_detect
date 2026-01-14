# Research: Improved Delay Detection Strategies for SSTDR

The project is used for **SSTDR (Spread Spectrum Time Domain Reflectometry)** to detect cable faults ("wounds"). In this context, the goal is to precisely locate impedance discontinuities (faults) by measuring the time delay of reflected spread spectrum signals.

## Key Requirements for SSTDR
1.  **High Resolution**: Converting time delay to distance ($d = v \cdot t / 2$). Small time errors lead to large distance errors.
    -   *Example*: 1 ns error $\approx$ 10-15 cm error (depending on velocity factor).
2.  **Sensitivity**: Detecting small reflections (soft faults) amidst noise and stronger reflections.

## Recommended Strategies

### 1. Frequency Domain Oversampling (Recommended)
-   **Relevance to SSTDR**: Crucial for pinpointing the *exact* peak location between samples. This directly improves the spatial resolution of fault location without increasing the ADC sampling rate.
-   **Mechanism**: Zero-pad the cross-correlation in the frequency domain before IFFT.
-   **Why it's better than Parabolic**: Parabolic interpolation assumes a specific peak shape spreading over only 3 samples. Oversampling reconstructs the band-limited signal shape globally, providing superior accuracy for multipath discrimination.

### 2. Background Subtraction (Calibration)
-   **Relevance to SSTDR**: Essential for removing static reflections (e.g., from the connection point/connector) to reveal small faults on the line.
-   **Mechanism**: Record a baseline `rx_signal` (no fault) and subtract it from the active `rx_signal` before correlation, or subtract the baseline *correlation* from the active *correlation*.

## Proposed Plan
We will focus on **Method 1 (Oversampling)** first to improve the core measurement engine. This will make the distance measurement of any detected fault more accurate.

### Implementation Details for Oversampling
1.  Modify `cross_correlation` to accept an `upsample_factor` (e.g., 16x).
2.  In frequency domain:
    -   Compute `Corr_Freq = FFT(Rx) * Conj(FFT(Tx))`
    -   Create a new larger array of size $N \times M$.
    -   Place positive frequencies of `Corr_Freq` at the start, negative frequencies at the end, and zeros in the middle.
3.  Perform IFFT on the expanded array.
4.  Find peak index on the upsampled grid.
5.  Delay = `peak_index / (fs * upsample_factor)`.
