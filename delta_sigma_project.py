"""
Delta-Sigma Modulator (DSM) Project - Single-file Python Implementation
======================================================================

This script implements a first-order Delta-Sigma Modulator (DSM) with:
  - 1-bit quantizer and 3-bit uniform quantizer
  - Manual PSD estimation (averaged periodogram)
  - FFT amplitude spectrum (dBFS)
  - Automatic in-band SNDR / SNR / THD / Dynamic Range estimation
  - FIR low-pass filter design using windowed-sinc (implemented from scratch)
  - Manual downsampling (decimation)
  - Full analysis before and after FIR + downsampling
  - Two runs: free-choice amplitude and group-based amplitude

Toolbox restrictions (project rules):
  - No advanced signal-processing toolboxes/functions (e.g., scipy.signal, welch, fir1, filter, decimate)
  - FFT/IFFT allowed for spectral analysis
  - Basic math, array ops, convolution (np.convolve), and plotting are allowed

Outputs:
  - Prints detailed metrics to console
  - Saves all figures to per-run folders:
      figs_<label>/

Notes:
  - The DSM is a simplified discrete-time first-order modulator:
        v[n] = v[n-1] + (x[n] - y[n-1])
        y[n] = Q(v[n])
  - Coherent input frequency is used: fin = k * fs / N (integer number of cycles)
    to reduce spectral leakage and stabilize metric estimation.
"""

from __future__ import annotations

import os
import re
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Global configuration
# =============================================================================

SHOW_PLOTS = False  # If True: keep figures open and call plt.show() at the end


# =============================================================================
# Utilities: directories, filenames, figure saving, and metric printing
# =============================================================================

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def slugify(text: str) -> str:
    """Convert a string to a safe filename-like slug."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def save_and_close(fig_dir: str, filename: str, dpi: int = 200) -> None:
    """
    Save current Matplotlib figure into fig_dir/filename.
    Close it if SHOW_PLOTS is False to avoid memory warnings.
    """
    ensure_dir(fig_dir)
    full_path = os.path.join(fig_dir, filename)
    plt.savefig(full_path, dpi=dpi, bbox_inches="tight")
    if not SHOW_PLOTS:
        plt.close()


def print_metrics_block(title: str, metrics: dict) -> None:
    """Print a detailed metric report for a signal in the selected in-band."""
    print("\n--------------------------------------")
    print(title)
    print("--------------------------------------")

    print(f"NFFT: {metrics['nfft']}")
    print(f"df (Hz/bin): {metrics['df']}")
    print(f"In-band max bin kB: {metrics['kB']}")
    print(f"Fundamental bin k1: {metrics['fund_bin']}")
    print(f"Fundamental Frequency (Hz): {metrics['fund_freq']}")
    print(f"Guard bins: ±{metrics['guard_bins']}  |  Fund guard range: {metrics['fund_guard_range']}")

    print("\n[Powers in-band 0..fB (DC excluded)]")
    print(f"P_signal: {metrics['P_signal']}")
    print(f"P_harm:   {metrics['P_harm']}")
    print(f"P_noise:  {metrics['P_noise']}")
    print(f"P_inband: {metrics['P_inband']}")

    print("\n[Contribution in-band]")
    print(f"Signal %:  {metrics['sig_pct']:.6f}%")
    print(f"Harm %:    {metrics['harm_pct']:.6f}%")
    print(f"Noise %:   {metrics['noise_pct']:.6f}%")

    print("\n[Metrics]")
    print(f"SNDR (dB): {metrics['SNDR_dB']}")
    print(f"SNR  (dB): {metrics['SNR_dB']}")
    print(f"THD  (dB): {metrics['THD_dB']}")
    print(f"Dynamic Range (dB): {metrics['DR_dB']}")

    print(f"\nHarmonic bins: {metrics['harmonic_bins']}")
    print("--------------------------------------")


# =============================================================================
# Windows (implemented from scratch)
# =============================================================================

def hann_window(N: int) -> np.ndarray:
    """Hann window, length N."""
    n = np.arange(N)
    return 0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))


def hamming_window(N: int) -> np.ndarray:
    """Hamming window, length N."""
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))


def blackman_window(N: int) -> np.ndarray:
    """Blackman window, length N."""
    n = np.arange(N)
    return (
        0.42
        - 0.5 * np.cos(2 * np.pi * n / (N - 1))
        + 0.08 * np.cos(4 * np.pi * n / (N - 1))
    )


# =============================================================================
# Quantizers (1-bit and uniform multi-bit)
# =============================================================================

def quantizer_1bit(v: float, vref: float = 1.0) -> float:
    """
    1-bit quantizer output levels: +vref or -vref.
    """
    return vref if v >= 0 else -vref


def quantizer_uniform(v: float, bits: int = 3, vref: float = 1.0) -> float:
    """
    Mid-rise uniform quantizer with full-scale range [-vref, +vref).
    Output is clipped to avoid overflow.
    """
    L = 2 ** bits
    delta = 2 * vref / L
    v_clipped = np.clip(v, -vref, vref - delta)
    code = np.floor((v_clipped + vref) / delta)
    y = code * delta - vref + delta / 2
    return float(y)


# =============================================================================
# First-order DSM (simple discrete-time integrator + quantizer)
# =============================================================================

def dsm_first_order(x: np.ndarray, mode: str = "1bit", vref: float = 1.0) -> np.ndarray:
    """
    First-order delta-sigma modulator:

        v[n] = v[n-1] + (x[n] - y[n-1])
        y[n] = Q(v[n])

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    mode : str
        "1bit" or "3bit".
    vref : float
        Full-scale reference level for quantizer output.

    Returns
    -------
    y : np.ndarray
        Quantized DSM output sequence.
    """
    N = len(x)
    y = np.zeros(N)

    v = 0.0
    y_prev = 0.0

    for n in range(N):
        v = v + (x[n] - y_prev)

        if mode == "1bit":
            y[n] = quantizer_1bit(v, vref=vref)
        elif mode == "3bit":
            y[n] = quantizer_uniform(v, bits=3, vref=vref)
        else:
            raise ValueError("mode must be '1bit' or '3bit'")

        y_prev = y[n]

    return y


# =============================================================================
# FIR Low-pass design using windowed-sinc (from scratch)
# =============================================================================

def fir_lowpass_windowed_sinc(fc: float, fs: float, num_taps: int, window_type: str = "blackman") -> np.ndarray:
    """
    Design a low-pass FIR filter using windowed-sinc method.

    Parameters
    ----------
    fc : float
        Cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    num_taps : int
        Number of FIR taps (filter length). Must be odd for integer group delay.
    window_type : str
        "hann", "hamming", "blackman", or "rect".

    Returns
    -------
    h : np.ndarray
        FIR filter coefficients.
    """
    if num_taps < 3:
        raise ValueError("num_taps must be >= 3")
    if num_taps % 2 == 0:
        raise ValueError("num_taps must be odd to have integer sample delay")

    n = np.arange(num_taps)
    M = num_taps - 1
    m = n - M / 2

    # Normalize cutoff by fs (not by Nyquist): np.sinc uses pi-normalized form internally.
    norm_fc = fc / fs
    h_ideal = 2 * norm_fc * np.sinc(2 * norm_fc * m)

    if window_type == "hann":
        w = hann_window(num_taps)
    elif window_type == "hamming":
        w = hamming_window(num_taps)
    elif window_type == "blackman":
        w = blackman_window(num_taps)
    elif window_type == "rect":
        w = np.ones(num_taps)
    else:
        raise ValueError("window_type must be 'hann', 'hamming', 'blackman', or 'rect'")

    h = h_ideal * w

    # Normalize DC gain to 1.0 (sum of taps).
    h = h / (np.sum(h) + 1e-30)
    return h


# =============================================================================
# Convolution and downsampling (manual decimation)
# =============================================================================

def convolve_conv(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """
    Linear convolution using numpy (allowed by project rules).
    Returns 'full' convolution.
    """
    return np.convolve(x, h, mode="full")


def downsample(x: np.ndarray, D: int) -> np.ndarray:
    """
    Downsample by integer factor D using slicing (manual decimation).
    No anti-aliasing is done here; filtering must occur before this step.
    """
    if D < 1:
        raise ValueError("Downsample factor D must be >= 1")
    return x[::D]


# =============================================================================
# PSD estimation: averaged periodogram (manual Welch-like)
# =============================================================================

def averaged_periodogram_psd(
    x: np.ndarray,
    fs: float,
    block_len: int,
    nfft: int | None = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD using averaged periodogram (no scipy/welch used).

    Steps:
      1) Split signal into K non-overlapping blocks of length block_len
      2) Window each block
      3) Compute FFT of each block
      4) Average magnitude-squared spectra and scale to get PSD (power/Hz)

    Returns
    -------
    f : np.ndarray
        Frequency axis for rFFT (0..fs/2).
    psd_avg : np.ndarray
        PSD estimate (linear units).
    """
    if len(x) == 0:
        raise ValueError("Empty signal")
    if block_len <= 0:
        raise ValueError("block_len must be > 0")

    if len(x) < block_len:
        block_len = len(x)

    if nfft is None:
        nfft = block_len
    if nfft < block_len:
        raise ValueError("nfft must be >= block_len")

    if window == "hann":
        w = hann_window(block_len)
    elif window == "hamming":
        w = hamming_window(block_len)
    elif window == "rect":
        w = np.ones(block_len)
    else:
        raise ValueError("window must be 'hann', 'hamming', or 'rect'")

    K = len(x) // block_len
    if K == 0:
        # This only happens if len(x) == 0 or block_len == 0 (already guarded)
        K = 1
        x = x[:block_len]
    else:
        x = x[:K * block_len]

    # Window power normalization factor
    U = np.sum(w ** 2)

    psd_acc = np.zeros(nfft // 2 + 1)

    for i in range(K):
        xb = x[i * block_len:(i + 1) * block_len] * w
        X = np.fft.rfft(xb, nfft)
        Pxx = (np.abs(X) ** 2) / (fs * U + 1e-30)  # power spectral density
        psd_acc += Pxx

    psd_avg = psd_acc / max(K, 1)
    f = np.fft.rfftfreq(nfft, d=1 / fs)
    return f, psd_avg


# =============================================================================
# FFT amplitude spectrum in dBFS
# =============================================================================

def amplitude_spectrum_dbfs(
    x: np.ndarray,
    fs: float,
    nfft: int | None = None,
    window: str = "hann",
    full_scale_peak: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a single-sided amplitude spectrum in dBFS.

    dBFS uses full_scale_peak as 0 dBFS reference:
      A_dbfs = 20 * log10( A_peak / full_scale_peak )

    Returns
    -------
    f : np.ndarray
        Frequency axis (0..fs/2).
    A_dbfs : np.ndarray
        Single-sided amplitude spectrum (dBFS).
    """
    if len(x) == 0:
        raise ValueError("Empty signal")

    if nfft is None:
        nfft = len(x)
    nfft = min(nfft, len(x))
    x = x[:nfft]

    if window == "hann":
        w = hann_window(nfft)
    elif window == "hamming":
        w = hamming_window(nfft)
    elif window == "rect":
        w = np.ones(nfft)
    else:
        raise ValueError("window must be 'hann', 'hamming', or 'rect'")

    # Coherent gain for amplitude correction
    CG = np.sum(w) / nfft

    X = np.fft.rfft(x * w, nfft)
    f = np.fft.rfftfreq(nfft, d=1 / fs)

    # Convert FFT bins to peak amplitude (single-sided)
    A = (2.0 * np.abs(X)) / (nfft * CG + 1e-30)

    # Correct DC and Nyquist (if exists) which should not be doubled
    A[0] *= 0.5
    if nfft % 2 == 0:
        A[-1] *= 0.5

    A_dbfs = 20.0 * np.log10((A / (full_scale_peak + 1e-30)) + 1e-30)
    return f, A_dbfs


# =============================================================================
# SNDR / SNR / THD / DR estimation in-band (0..fB) with bin separation
# =============================================================================

def compute_sndr_and_dr(
    x: np.ndarray,
    fs: float,
    fB: float,
    nfft: int | None = None,
    num_harmonics: int = 5,
    guard_bins: int = 2,
    window: str = "hann",
    full_scale_peak: float = 1.0,
) -> dict:
    """
    Compute in-band SNDR, SNR, THD, and Dynamic Range by separating:
      - Fundamental (signal)
      - Harmonics (distortion)
      - Remaining in-band bins (noise)

    The algorithm:
      1) Window + rFFT
      2) Convert to one-sided power spectrum (PSD-like scaling)
      3) Convert to per-bin power by multiplying by df
      4) Restrict to 0..fB
      5) Find fundamental bin as the largest non-DC bin
      6) Sum power in ±guard_bins around fundamental as signal power
      7) For harmonics h=2..num_harmonics, find bins near h*k1 (if in-band)
         and sum power in their guard bands
      8) The remaining in-band power is noise
      9) Compute metrics:
            SNDR = P_signal / (P_noise + P_harm)
            SNR  = P_signal / P_noise
            THD  = P_harm / P_signal
            DR   = P_FS / P_noise

    Returns a dictionary with detailed intermediate values for debugging.
    """
    if len(x) == 0:
        raise ValueError("Empty signal")

    if nfft is None:
        nfft = len(x)
    nfft = min(nfft, len(x))
    x = x[:nfft]

    if window == "hann":
        w = hann_window(nfft)
    elif window == "hamming":
        w = hamming_window(nfft)
    elif window == "rect":
        w = np.ones(nfft)
    else:
        raise ValueError("window must be 'hann', 'hamming', or 'rect'")

    U = np.sum(w ** 2)          # window power
    df = fs / nfft              # bin spacing (Hz per bin)

    X = np.fft.rfft(x * w, nfft)

    # One-sided power spectral density estimate (power/Hz)
    Pxx = (np.abs(X) ** 2) / (fs * U + 1e-30)
    if len(Pxx) > 2:
        Pxx[1:-1] *= 2.0  # account for folding to one-sided (except DC, Nyquist)

    # Convert PSD to power per bin by multiplying by df
    Pbin = Pxx * df

    # In-band bin limit
    kB = int(np.floor(fB * nfft / fs))
    kB = min(kB, len(Pbin) - 1)
    Pbin_in = Pbin[:kB + 1]

    # Fundamental bin search (exclude DC)
    search = Pbin_in.copy()
    if len(search) > 0:
        search[0] = 0.0
    k1 = int(np.argmax(search))
    f1 = k1 * fs / nfft

    def sum_guard(center_bin: int) -> tuple[float, tuple[int, int]]:
        """Sum power in a guard band around a bin index."""
        b0 = max(center_bin - guard_bins, 0)
        b1 = min(center_bin + guard_bins, len(Pbin_in) - 1)
        return float(np.sum(Pbin_in[b0:b1 + 1])), (b0, b1)

    # Signal power (fundamental)
    P_signal, fund_rng = sum_guard(k1)

    # Harmonics power
    harmonic_bins: list[int] = []
    harm_ranges: list[tuple[int, int]] = []
    P_harm = 0.0

    for h in range(2, num_harmonics + 1):
        kh = int(round(h * k1))
        if 0 < kh <= kB and kh not in harmonic_bins:
            harmonic_bins.append(kh)
            Ph, rr = sum_guard(kh)
            P_harm += Ph
            harm_ranges.append(rr)

    # Noise mask (exclude DC, fund guard, harmonic guards)
    mask = np.ones_like(Pbin_in, dtype=bool)
    if len(mask) > 0:
        mask[0] = False  # exclude DC from "noise"

    b0, b1 = fund_rng
    mask[b0:b1 + 1] = False
    for (bh0, bh1) in harm_ranges:
        mask[bh0:bh1 + 1] = False

    P_noise = float(np.sum(Pbin_in[mask]))
    P_inband = float(np.sum(Pbin_in[1:]))  # in-band total excluding DC

    # Metrics
    P_nd = P_noise + P_harm
    SNDR_dB = 10.0 * np.log10(P_signal / (P_nd + 1e-30))
    SNR_dB = 10.0 * np.log10(P_signal / (P_noise + 1e-30))
    THD_dB = 10.0 * np.log10(P_harm / (P_signal + 1e-30)) if P_harm > 0 else -np.inf

    # Dynamic range referenced to full-scale sine power: P_FS = (A_FS_peak^2)/2
    P_FS = (full_scale_peak ** 2) / 2.0
    DR_dB = 10.0 * np.log10(P_FS / (P_noise + 1e-30))

    # Percent contributions (for sanity check)
    sig_pct = 100.0 * P_signal / (P_inband + 1e-30)
    harm_pct = 100.0 * P_harm / (P_inband + 1e-30)
    noise_pct = 100.0 * P_noise / (P_inband + 1e-30)

    return {
        "nfft": nfft,
        "df": df,
        "kB": kB,
        "fund_bin": k1,
        "fund_freq": f1,
        "guard_bins": guard_bins,
        "fund_guard_range": fund_rng,
        "P_signal": P_signal,
        "P_harm": P_harm,
        "P_noise": P_noise,
        "P_inband": P_inband,
        "sig_pct": sig_pct,
        "harm_pct": harm_pct,
        "noise_pct": noise_pct,
        "SNDR_dB": SNDR_dB,
        "DR_dB": DR_dB,
        "SNR_dB": SNR_dB,
        "THD_dB": THD_dB,
        "harmonic_bins": harmonic_bins,
    }


# =============================================================================
# OSR estimation for first-order DSM (approximate theory-based)
# =============================================================================

def osr_required_first_order(sndr_target_db: float, A: float = 0.5, delta: float = 2.0) -> float:
    """
    Approximate OSR requirement for a first-order DSM.

    This uses a simplified relation where in-band quantization noise decreases ~ OSR^3.
    It is meant as a design starting point, not a strict guarantee.

    Parameters
    ----------
    sndr_target_db : float
        Target SNDR in dB (e.g., 10-bit -> 6.02*10 + 1.76).
    A : float
        Input sine peak amplitude.
    delta : float
        Quantizer step size. For 1-bit with output levels ±Vref, delta ~ 2*Vref.

    Returns
    -------
    osr_est : float
        Estimated OSR (not necessarily integer).
    """
    Ps = A ** 2 / 2.0
    Pn_coeff = (delta ** 2 / 12.0) * (np.pi ** 2 / 3.0)
    osr_cubed = (Pn_coeff / (Ps + 1e-30)) * (10.0 ** (sndr_target_db / 10.0))
    return osr_cubed ** (1.0 / 3.0)


def next_power_of_two(x: float) -> int:
    """Return the smallest power-of-two integer >= x."""
    p = 1
    while p < x:
        p *= 2
    return p


# =============================================================================
# FIR response analysis (passband ripple and stopband attenuation estimate)
# =============================================================================

def analyze_fir_response(h: np.ndarray, fs: float, fp: float, fsb: float, nfft: int = 65536) -> dict:
    """
    Evaluate FIR magnitude response and approximate:
      - Passband min/max (0..fp)
      - Stopband max magnitude (fsb..Nyquist)
      - Stopband attenuation estimate
    """
    H = np.fft.rfft(h, nfft)
    f = np.fft.rfftfreq(nfft, d=1 / fs)
    mag_db = 20.0 * np.log10(np.abs(H) + 1e-30)

    pb = np.where(f <= fp)[0]
    sb = np.where(f >= fsb)[0]

    pb_min_db = float(np.min(mag_db[pb])) if len(pb) else None
    pb_max_db = float(np.max(mag_db[pb])) if len(pb) else None

    sb_max_db = float(np.max(mag_db[sb])) if len(sb) else None
    stop_atten_db = -sb_max_db if sb_max_db is not None else None

    return {
        "pb_min_db": pb_min_db,
        "pb_max_db": pb_max_db,
        "sb_max_db": sb_max_db,
        "stop_atten_db": stop_atten_db,
    }


# =============================================================================
# Full signal analysis: time, FFT, PSD, and metrics + figure saving
# =============================================================================

def full_signal_analysis(
    tag_title: str,
    y: np.ndarray,
    fs: float,
    fB: float,
    fig_dir: str,
    nfft_fft: int | None = None,
    psd_block_len: int = 2048,
    full_scale_peak: float = 1.0,
) -> dict:
    """
    Run the complete analysis required by the project on a signal:
      - Time-domain waveform plot
      - FFT amplitude spectrum plot (full + zoom)
      - PSD estimate plot (full + zoom)
      - In-band SNDR/SNR/THD/DR metrics printed
    """
    tag = slugify(tag_title)

    # --- Time-domain plot (first samples only) ---
    show_N = min(400, len(y))
    t = np.arange(len(y)) / fs

    plt.figure()
    plt.plot(t[:show_N] * 1e6, y[:show_N])
    plt.xlabel("Time (µs)")
    plt.ylabel("Amplitude")
    plt.title(tag_title + " - Time Waveform")
    plt.grid(True)
    save_and_close(fig_dir, f"{tag}_time.png")

    # --- FFT amplitude spectrum (dBFS) ---
    if nfft_fft is None:
        nfft_fft = min(len(y), 32768)

    f_fft, A_dbfs = amplitude_spectrum_dbfs(
        y, fs, nfft=nfft_fft, window="hann", full_scale_peak=full_scale_peak
    )

    plt.figure()
    plt.plot(f_fft, A_dbfs)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dBFS)")
    plt.title(tag_title + " - FFT (Full)")
    plt.grid(True)
    plt.xlim(0, fs / 2)
    save_and_close(fig_dir, f"{tag}_fft_full.png")

    plt.figure()
    plt.plot(f_fft, A_dbfs)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dBFS)")
    plt.title(tag_title + " - FFT (Zoomed 0..5*fB)")
    plt.grid(True)
    plt.xlim(0, 5 * fB)
    save_and_close(fig_dir, f"{tag}_fft_zoom.png")

    # --- PSD (averaged periodogram) ---
    block_len = min(psd_block_len, len(y))
    f_psd, psd = averaged_periodogram_psd(y, fs, block_len=block_len, nfft=block_len, window="hann")

    plt.figure()
    plt.plot(f_psd, 10.0 * np.log10(psd + 1e-30))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title(tag_title + " - PSD (Full)")
    plt.grid(True)
    plt.xlim(0, fs / 2)
    save_and_close(fig_dir, f"{tag}_psd_full.png")

    plt.figure()
    plt.plot(f_psd, 10.0 * np.log10(psd + 1e-30))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (dB/Hz)")
    plt.title(tag_title + " - PSD (Zoomed 0..5*fB)")
    plt.grid(True)
    plt.xlim(0, 5 * fB)
    save_and_close(fig_dir, f"{tag}_psd_zoom.png")

    # --- Metrics ---
    metrics = compute_sndr_and_dr(
        y, fs, fB,
        nfft=min(len(y), 32768),
        num_harmonics=5,
        guard_bins=2,
        window="hann",
        full_scale_peak=full_scale_peak,
    )
    print_metrics_block(tag_title + " - METRICS (In-band 0..fB)", metrics)
    return metrics


# =============================================================================
# FIR + Downsampling chain and re-analysis (project requirement)
# =============================================================================

def fir_decimation_and_reanalysis(
    y1: np.ndarray,
    y3: np.ndarray,
    fs: float,
    fB: float,
    OSR: int,
    N: int,
    fig_dir: str,
    full_scale_peak: float,
) -> None:
    """
    Design FIR, filter both 1-bit and 3-bit DSM outputs, downsample, and re-analyze.
    """
    print("\n======================================")
    print("FIR DESIGN + DOWNSAMPLING (PROJECT SECTION)")
    print("======================================")

    # Decimation factor choice:
    # The project wants "a suitable" factor. Here we use D = OSR/4 as a reasonable multi-stage-like step.
    D = OSR // 4
    D = max(D, 1)

    fs_out = fs / D
    nyq_out = fs_out / 2.0

    # Filter specification:
    # - Passband: 0..fB
    # - Stopband should begin below the post-downsample Nyquist to suppress aliasing.
    fp = fB
    fsb = 0.90 * nyq_out
    fc = 0.5 * (fp + fsb)  # mid-point cutoff for windowed-sinc design

    # Strong stopband attenuation needs many taps with windowed-sinc.
    # Blackman is chosen for good sidelobe suppression.
    num_taps = 1201  # must be odd
    order = num_taps - 1
    window_type = "blackman"

    h = fir_lowpass_windowed_sinc(fc=fc, fs=fs, num_taps=num_taps, window_type=window_type)
    fir_info = analyze_fir_response(h=h, fs=fs, fp=fp, fsb=fsb, nfft=65536)

    print("\n--- FIR FILTER SPECIFICATIONS ---")
    print(f"FIR Order: {order}")
    print(f"Number of Taps: {num_taps}")
    print(f"Window Type: {window_type}")
    print(f"Passband Edge fp (Hz): {fp}")
    print(f"Stopband Start fsb (Hz): {fsb}")
    print(f"Cutoff fc (Hz): {fc}")
    print(f"Downsampling Factor D: {D}")
    print(f"Output Sampling Rate fs_out (Hz): {fs_out}")
    print(f"Output Nyquist (Hz): {nyq_out}")

    print("\n--- MEASURED FIR RESPONSE (Approx) ---")
    print(f"Passband min magnitude (dB): {fir_info['pb_min_db']}")
    print(f"Passband max magnitude (dB): {fir_info['pb_max_db']}")
    print(f"Stopband max magnitude (dB): {fir_info['sb_max_db']}")
    print(f"Estimated Stopband Attenuation (dB): {fir_info['stop_atten_db']}")

    # Plot FIR magnitude response
    H = np.fft.rfft(h, 65536)
    fH = np.fft.rfftfreq(65536, d=1 / fs)
    Hdb = 20.0 * np.log10(np.abs(H) + 1e-30)

    plt.figure()
    plt.plot(fH, Hdb)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FIR Frequency Response |H(f)| (dB) - Full")
    plt.grid(True)
    plt.xlim(0, fs / 2)
    save_and_close(fig_dir, "fir_response_full.png")

    plt.figure()
    plt.plot(fH, Hdb)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("FIR Frequency Response |H(f)| (dB) - Zoom (0..2*Nyquist_out)")
    plt.grid(True)
    plt.xlim(0, 2 * nyq_out)
    save_and_close(fig_dir, "fir_response_zoom.png")

    # Filter using allowed convolution
    y1_f = convolve_conv(y1, h)
    y3_f = convolve_conv(y3, h)

    # Align to compensate linear-phase FIR group delay
    delay = (num_taps - 1) // 2
    y1_aligned = y1_f[delay:delay + N]
    y3_aligned = y3_f[delay:delay + N]

    # Downsample
    y1_ds = downsample(y1_aligned, D)
    y3_ds = downsample(y3_aligned, D)

    print("\n======================================")
    print("RE-ANALYSIS AFTER FIR + DOWNSAMPLING")
    print("======================================")

    full_signal_analysis(
        tag_title="FINAL 1-bit Output (After FIR + Downsampling)",
        y=y1_ds,
        fs=fs_out,
        fB=fB,
        fig_dir=fig_dir,
        nfft_fft=min(len(y1_ds), 8192),
        psd_block_len=min(len(y1_ds), 1024),
        full_scale_peak=full_scale_peak,
    )

    full_signal_analysis(
        tag_title="FINAL 3-bit Output (After FIR + Downsampling)",
        y=y3_ds,
        fs=fs_out,
        fB=fB,
        fig_dir=fig_dir,
        nfft_fft=min(len(y3_ds), 8192),
        psd_block_len=min(len(y3_ds), 1024),
        full_scale_peak=full_scale_peak,
    )


# =============================================================================
# Group-based amplitude rule
# =============================================================================

def amplitude_from_group(d: int) -> float:
    """
    Project rule (even d):
        A = 0.6 + 0.05 * (d mod 4)
    """
    return 0.6 + 0.05 * (d % 4)


# =============================================================================
# Run the full project for a given amplitude and label
# =============================================================================

def run_project_for_amplitude(
    A: float,
    group_label: str,
    vref: float = 1.0,
    dc: float = 0.55,
    Ksine: int = 1,
    Kdc: int = 0,
) -> None:
    """
    Run the entire pipeline for a selected input amplitude A:
      - Compute approximate OSR for target (10-bit equivalent SNDR)
      - Choose OSR as power-of-two >= estimate (minimum 256)
      - Set fs = 2 * OSR * fB
      - Choose coherent fin = k * fs / N
      - Simulate 1-bit and 3-bit first-order DSM
      - Analyze raw DSM outputs (time/FFT/PSD/metrics)
      - Design FIR, filter and downsample, re-analyze final outputs
      - Save all figures to figs_<group_label>/
    """
    fig_dir = f"figs_{group_label}"
    ensure_dir(fig_dir)

    # Target: 10-bit equivalent SNDR (ideal ADC relation)
    target_bits = 10
    sndr_target = 6.02 * target_bits + 1.76  # dB

    # In-band bandwidth (project design choice)
    fB = 10_000.0  # Hz

    # Approximate OSR requirement (first-order DSM)
    # For 1-bit: delta ~ 2*vref
    osr_est = osr_required_first_order(sndr_target_db=sndr_target, A=A, delta=2.0 * vref)
    OSR = next_power_of_two(osr_est)
    OSR = max(OSR, 256)

    # Sampling frequency
    fs = int(2.0 * OSR * fB)

    # Simulation length and coherent input frequency selection
    N = 32768
    k = 7  # small bin index => low in-band tone, coherent
    fin = k * fs / N  # coherent tone (integer number of cycles in N)

    print("\n============================================================")
    print(f"RUN FOR AMPLITUDE A = {A}   |   {group_label}")
    print("============================================================")
    print(f"Vref = {vref} (Full-scale ±Vref)  |  Vfs = {2*vref}")
    print(f"Ksine={Ksine}, Kdc={Kdc}, dc={dc}")
    print("Estimated OSR:", osr_est)
    print("Chosen OSR:", OSR)
    print("Required fs (Hz):", fs)
    print(f"fin = {fin:.2f} Hz , fB = {fB:.2f} Hz , N = {N}")
    print(f"Figures folder: {fig_dir}/")

    # Simple overload check (peak must remain safely below vref)
    peak_in = abs(Ksine) * abs(A) + abs(Kdc) * abs(dc)
    if peak_in > 0.9 * vref:
        print("WARNING: Input peak is close to Vref. Potential overload risk.")
    else:
        print("OK: Input peak is safely below Vref (no overload expected).")

    # Generate input (matching the provided MATLAB flags idea)
    t = np.arange(N) / fs
    x = (Ksine * A * np.sin(2.0 * np.pi * fin * t)) + (Kdc * dc)

    # DSM simulation
    y1 = dsm_first_order(x, mode="1bit", vref=vref)
    y3 = dsm_first_order(x, mode="3bit", vref=vref)

    print("\n======================================")
    print("RAW DSM OUTPUT ANALYSIS (TIME / FFT / PSD / SNDR / DR)")
    print("======================================")

    full_signal_analysis(
        tag_title="RAW DSM 1-bit Output",
        y=y1,
        fs=fs,
        fB=fB,
        fig_dir=fig_dir,
        nfft_fft=32768,
        psd_block_len=2048,
        full_scale_peak=vref,
    )

    full_signal_analysis(
        tag_title="RAW DSM 3-bit Output",
        y=y3,
        fs=fs,
        fB=fB,
        fig_dir=fig_dir,
        nfft_fft=32768,
        psd_block_len=2048,
        full_scale_peak=vref,
    )

    # FIR + downsampling + re-analysis
    fir_decimation_and_reanalysis(
        y1=y1,
        y3=y3,
        fs=fs,
        fB=fB,
        OSR=OSR,
        N=N,
        fig_dir=fig_dir,
        full_scale_peak=vref,
    )

    print(f"\nAll figures saved under '{fig_dir}/'.")


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    """
    Run the project for:
      1) A = 0.5 (free choice)
      2) A = amplitude_from_group(d=4) => 0.6
    """
    # Run 1: free choice amplitude
    A_free = 0.5

    # Run 2: group-based amplitude (d = 4)
    d = 4
    A_group = amplitude_from_group(d)

    # Parameters based on the provided MATLAB snippet
    Vref = 1.0
    dc = 0.55
    Ksine = 1
    Kdc = 0  # keep DC disabled by default (matches your runs)

    run_project_for_amplitude(A_free, "A0p50_free_choice", vref=Vref, dc=dc, Ksine=Ksine, Kdc=Kdc)
    run_project_for_amplitude(A_group, "A0p60_group_4", vref=Vref, dc=dc, Ksine=Ksine, Kdc=Kdc)

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
