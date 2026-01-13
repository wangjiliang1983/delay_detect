import numpy as np
import matplotlib.pyplot as plt
from tx import tx
from channel import channel
from rx import rx


def run_demo():
    """Demonstrate single delay measurement"""
    fs = 1e9
    fc = 200e6
    delay_true = 100.3

    tx_signal = tx(num_periods=10, fc=fc, fs=fs)
    rx_signal = channel(tx_signal, delay_true, fs, snr_db=20)
    delay_measured = rx(rx_signal, tx_signal, fs)

    error = (delay_measured * 1e9 - delay_true) * 1e3

    print(f"True Delay: {delay_true:.3f} ns")
    print(f"Measured Delay: {delay_measured * 1e9:.6f} ns")
    print(f"Measurement Error: {error:.2f} ps")


def run_delay_sweep():
    """Delay sweep test"""
    fs = 1e9
    fc = 200e6

    tx_signal = tx(num_periods=10, fc=fc, fs=fs)

    delay_start = 100.0
    delay_stop = 105.0
    delay_step = 0.05

    delays_true = np.arange(delay_start, delay_stop + delay_step, delay_step)
    delays_measured = []

    for delay_ns in delays_true:
        rx_signal = channel(tx_signal, delay_ns, fs, snr_db=20)
        delay_est = rx(rx_signal, tx_signal, fs) * 1e9
        delays_measured.append(delay_est)

    delays_measured = np.array(delays_measured)
    errors = (delays_measured - delays_true) * 1e3

    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(np.abs(errors))
    bias = np.mean(errors)

    print(f"Delay Sweep Test Results (SNR=20dB)")
    print(f"  RMSE: {rmse:.2f} ps")
    print(f"  Max Error: {max_error:.2f} ps")
    print(f"  Bias: {bias:.2f} ps")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(delays_true, errors, 'b-', linewidth=1)
    plt.xlabel('True Delay (ns)')
    plt.ylabel('Error (ps)')
    plt.title('Delay Measurement Error')
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Error (ps)')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    plt.show()


def run_noise_test():
    """Noise robustness test"""
    fs = 1e9
    fc = 200e6

    tx_signal = tx(num_periods=10, fc=fc, fs=fs)
    delay_true = 100.3

    snr_values = np.arange(-20, 25, 5)
    rmse_values = []

    for snr_db in snr_values:
        errors = []
        for _ in range(50):
            rx_signal = channel(tx_signal, delay_true, fs, snr_db=float(snr_db))
            delay_est = rx(rx_signal, tx_signal, fs) * 1e9
            errors.append((delay_est - delay_true) * 1e3)
        rmse_values.append(np.sqrt(np.mean(np.array(errors) ** 2)))

    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_values, rmse_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('SNR (dB)')
    plt.ylabel('RMSE (ps)')
    plt.title('RMSE vs SNR')
    plt.grid(True)
    plt.savefig('noise_test.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    print("=" * 50)
    print("AD/DA Link Delay Measurement System Simulation")
    print("=" * 50)
    print()

    print("1. Demonstrate single delay measurement")
    print("-" * 50)
    run_demo()
    print()

    print("2. Delay sweep test")
    print("-" * 50)
    run_delay_sweep()
    print()

    print("3. Noise robustness test")
    print("-" * 50)
    run_noise_test()
