# Implement Oversampling for High-Resolution SSTDR

## Goal Description
To improve the accuracy of cable fault location (SSTDR), we will implement frequency-domain oversampling in the receiver. This replaces the basic parabolic interpolation with a more theoretically robust method for band-limited signals, allowing for sub-sample delay estimation with high precision.

## Proposed Changes

### [delay_detect]

#### [MODIFY] [rx.py](file:///e:/misc/20260113_delay_detect/delay_detect/rx.py)
-   Update `cross_correlation` to support an `upsample_factor`.
-   Implement zero-padding in the frequency domain within `cross_correlation` (or a helper function).
-   Update `rx` function to use the upsampled correlation for peak finding.
-   Remove `parabolic_interpolation` as it will be superseded by high-factor oversampling (or used as a final refinement, but oversampling usually suffices).

#### [MODIFY] [simulation.py](file:///e:/misc/20260113_delay_detect/delay_detect/simulation.py)
-   Update usage of `rx` if the signature changes (optional, likely can keep default arguments).
-   (Optional) Add a new test case in `simulation.py` to compare precision with and without oversampling? For now, we will rely on existing benchmarks.

## Verification Plan

### Automated Tests
-   Run `python simulation.py` which already calculates RMSE.
-   **Expectation**: RMSE should decrease significantly (currently ~38 ps). With 16x oversampling, it should be lower or comparable but more robust.
-   *Note*: Parabolic interpolation is actually quite good for single peaks. Oversampling shines when peaks are distorted. For the ideal simulation, we expect similar or slightly better performance.
