import numpy as np
import matplotlib.pyplot as plt
import pywt


def plot_cwt(signal: np.ndarray, dt: float, T: float) -> None:
    """Plots the Continuous Wavelet Transform (CWT) of a 1D signal.

    Args:
        signal: The 1D time-series signal.
        dt: The time interval between samples (in seconds).
        T: The total duration of the signal (in seconds).
    """
    t = np.arange(0, T, dt)

    scales = np.linspace(1, 128, 1000)
    wavelet = "cmor1.5-1.0"
    frequencies = pywt.scale2frequency(wavelet, scales) / dt

    coefficients, _ = pywt.cwt(signal, scales, wavelet, dt)

    num_lin_points = 500
    f_min = frequencies[-1]
    f_max = frequencies[0]
    f_lin = np.linspace(f_min, f_max, num_lin_points)

    freqs_sorted = frequencies[::-1]
    coeffs_sorted = coefficients[::-1, :]

    cwt_lin = np.empty(
        (num_lin_points, coefficients.shape[1]), dtype=coefficients.dtype
    )
    for i in range(coefficients.shape[1]):
        cwt_lin[:, i] = np.interp(f_lin, freqs_sorted, coeffs_sorted[:, i])

    fig, ax = plt.subplots(figsize=(10, 6))

    img = ax.imshow(
        np.abs(cwt_lin[::-1, :]),
        extent=[t[0], t[-1], f_lin[-1], f_lin[0]],
        aspect="auto",
        cmap="jet",
        interpolation="bilinear",
    )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")

    num_yticks = 10
    ytick_values = np.linspace(f_lin[0], f_lin[-1], num_yticks)
    ax.set_yticks(ytick_values)
    ax.set_yticklabels([f"{val:.1f}" for val in ytick_values[::-1]])

    fig.colorbar(img, ax=ax, label="Coefficient magnitude")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    T = 100.0
    dt = 0.01
    t = np.arange(0, T, dt)

    f0 = 10.0
    f1 = 20.0

    inst_freq = f0 + (f1 - f0) * t / T
    signal = 0.4 * np.sin(2 * np.pi * t * (f0 + (f1 - f0) * t**1.3 / (2 * T)))

    plot_cwt(signal, dt, T)
