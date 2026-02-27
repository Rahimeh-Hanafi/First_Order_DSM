# First-Order Delta–Sigma Modulator (DSM) Simulation + Digital Decimation (Python)

This repository contains a **from-scratch, single-file Python** implementation of a **first-order Delta–Sigma Modulator (DSM)** with:

- **1-bit quantizer** and **3-bit uniform quantizer**
- **Noise shaping** via DSM feedback
- **Digital low-pass FIR filter** (windowed-sinc design)
- **Downsampling (decimation)**
- **Spectral analysis**: FFT amplitude in dBFS + averaged periodogram PSD
- **In-band performance metrics**: **SNDR**, **SNR**, **THD**, **Dynamic Range**

The implementation targets typical course/project requirements where you must compare:
- **Before vs after** FIR filtering & downsampling
- **1-bit vs 3-bit** quantization performance

---

## Project Goal

1. **Simulate a first-order DSM**
   - Show how a very low-resolution quantizer (especially **1-bit**) can still achieve high **in-band resolution** using **oversampling + noise shaping**.

2. **Compare system performance**
   - **Before vs after** digital filtering & downsampling  
   - **1-bit vs 3-bit** quantizer outputs

3. **Quantify results**
   - Compute and report **SNDR/SNR/THD/DR** within the desired signal band (0..fB).

4. **Generate figures**
   - Time-domain waveform, FFT spectrum, PSD plots for each stage.

---

## Constraints / Tooling Limitations

This project intentionally avoids “one-line DSP solutions” and uses only allowed primitives.

### Allowed
- `numpy` for array operations and FFTs (`np.fft.rfft`)
- `matplotlib` for plotting
- `np.convolve` for FIR filtering (explicitly used as manual convolution)

### Not Used / Avoided
- **No** `scipy.signal` (no `firwin`, `welch`, `lfilter`, `decimate`, etc.)
- **No** toolbox-style DSP functions
- **No** black-box metric calculators

**Reason:** The aim is to demonstrate understanding of DSM, FIR design, spectral estimation, and metric computation by implementing them directly.

---

## Repository Structure

- `delta_sigma_project.py` (single script)
- `docs`
  - `Communication Systems Final Project Rules _ 1.pdf`
  - `Communication Systems Final Project Rules _ 2.pdf`
  - `Communication Systems Final Project.pdf`
  - `Communication Systems Final Project Report.pdf`
  - `DSM_Simulink_Model.png`
  - `DSmod_v2.0.zip` (Matlab Files of Project Definitions)
- Output folders auto-created:
  - `figs_A0p50_free_choice/`
  - `figs_A0p60_group_4/`

Each folder contains:
- Time waveform plots
- FFT amplitude plots (full + zoom)
- PSD plots (full + zoom)
- FIR frequency response plots

---

## How to Run

```bash
python delta_sigma_project.py
```

The script runs two cases:
1. **A = 0.5** (free choice)
2. **A = 0.6** (group formula with `d = 4`)

You can modify parameters in `main()`:
- `A_free`, `d`
- `Vref`, `Ksine`, `Kdc`, `dc`

---

## High-Level Pipeline

### Stage 1 — Input Generation

A coherent sine tone is generated:

- `fin = k * fs / N`

**Why?**
- The sine lands exactly on an FFT bin (integer number of cycles in the record).
- This reduces spectral leakage and makes SNDR/SNR estimates more stable.

**Effect**
- Cleaner FFT peak and more reliable noise/distortion separation.

---

### Stage 2 — First-Order DSM Core

The DSM follows the discrete-time loop:

\[
v[n] = v[n-1] + (x[n] - y[n-1]), \quad y[n] = Q(v[n])
\]

**Why this structure?**
- Standard first-order DSM topology:
  - Integrator accumulates error
  - Feedback forces **quantization noise shaping** (pushes noise to higher frequencies)

**Effect**
- Lower in-band noise compared to a direct low-rate quantizer.
- Higher out-of-band noise (visible in PSD increasing with frequency).

---

### Stage 3 — Quantizers (1-bit vs 3-bit)

#### 1-bit quantizer
Outputs ±Vref.

**Why 1-bit?**
- Classic DSM case: simplest DAC/feedback and strong linearity.
- Demonstrates how noise shaping compensates for low resolution.

#### 3-bit uniform quantizer
A mid-rise uniform quantizer over ±Vref.

**Why include 3-bit?**
- Shows how increasing bit-depth reduces step size (Δ),
  improving raw SNDR/SNR and typically lowering distortion.

**Effect**
- Raw DSM output often shows higher SNDR for 3-bit.
- Post-decimation results depend on the filter/decimation design choices.

---

### Stage 4 — Spectral Estimation Choices

#### Windowing (Hann/Hamming/Blackman)
FFT and PSD use a window (default: Hann).

**Why?**
- Windows reduce leakage when signals are not perfectly coherent.
- Even with coherence, windows improve robustness to numerical effects.

**Effect**
- Cleaner noise-floor estimate and fewer skirts around the tone.

#### PSD via Averaged Periodogram (Welch-like without overlap)
Signal is split into blocks, windowed, FFT’d, and averaged.

**Why?**
- A single FFT estimate can have high variance.
- Averaging reduces variance and improves interpretability.

**Effect**
- Smoother PSD curves and more stable noise-floor comparisons.

---

### Stage 5 — In-band Metric Computation (SNDR/SNR/THD/DR)

Power is computed per FFT bin and separated into:

- **Signal power**: fundamental bin ± guard bins
- **Harmonics**: several harmonics (up to a selected count), each ± guard bins
- **Noise**: all remaining bins inside 0..fB (excluding DC)

**Why guard bins?**
- Windowing spreads tone energy across nearby bins.
- Guard bins ensure tone energy is counted as signal, not noise.

**Effect**
- More accurate SNDR/SNR, especially when using Hann window.

---

### Stage 6 — FIR Low-Pass Filter (Windowed-Sinc)

A low-pass FIR is built using:
- Ideal sinc impulse response
- A window (default: Blackman)
- DC gain normalized to 1

**Why FIR + windowed-sinc?**
- FIR is stable and linear-phase (predictable delay).
- Windowed-sinc is a standard “from-scratch” approach.
- Blackman provides strong stopband attenuation.

**Effect**
- Removes shaped out-of-band noise before downsampling.
- Prevents aliasing in decimation.
- Low passband ripple with enough taps.

---

### Stage 7 — Downsampling (Decimation)

The output is downsampled using:
- `D = OSR / 4` (integer)
- `fs_out = fs / D`

**Why choose D = OSR/4?**
- Practical compromise:
  - Significant rate reduction
  - Output Nyquist stays high enough to preserve the band of interest
- Avoids very aggressive decimation that would require a much sharper FIR.

**Effect**
- Output data rate is reduced.
- In-band metrics should remain close if aliasing is properly controlled.

---

## Why You Might Observe “Raw SNDR > Final SNDR”

Ideally, after proper decimation, in-band performance is maintained or improves.
However, FFT-based measurement can show a slightly lower final SNDR because:

- Post-decimation analysis uses smaller `NFFT` → fewer bins and coarser resolution
- Guard-bin and windowing settings influence how power is partitioned
- Some shaped noise may sit near the edge of fB and is treated differently after filtering

This does not automatically mean the decimation is wrong unless you clearly observe aliasing artifacts.

---

## Expected Outcomes

Typical observations:
- **Noise shaping**: DSM PSD increases with frequency (high-frequency noise)
- **1-bit DSM**: decent in-band SNDR due to oversampling + noise shaping
- **3-bit DSM**: higher raw SNDR/SNR and lower THD
- **After FIR + decimation**:
  - High-frequency noise strongly reduced
  - Output spectrum dominated by the tone with a low in-band noise floor

---

## Key Design Decisions Summary

| Component | Solution Chosen | Why | Expected Effect |
|---|---|---|---|
| Input frequency | Coherent tone (`fin = k*fs/N`) | Stable FFT metrics, low leakage | Cleaner tone, reliable SNDR |
| DSM structure | Standard first-order loop | Matches theory + project | Noise shaping visible |
| Quantizers | 1-bit + 3-bit uniform | Required comparison | 3-bit improves raw SNDR |
| PSD estimation | Averaged periodogram | Lower variance than single FFT | Smoother noise floor |
| FIR design | Windowed-sinc + Blackman | From-scratch, strong stopband | Prevent aliasing |
| Decimation factor | `D = OSR/4` | Balanced reduction + feasible FIR | Lower fs, preserved band |
| Metric separation | Guard bins + harmonic search | Correct power accounting | More accurate SNDR/SNR |

---

## Customization Tips

- Change analysis bandwidth:
  - `fB = 10_000` in `run_project_for_amplitude()`
- Enable DC input (similar to MATLAB template behavior):
  - Set `Kdc = 1` in `main()`
- Change filter strength:
  - Increase `num_taps` for higher stopband attenuation
  - Try different window types to explore tradeoffs
- Improve post-decimation metric stability:
  - Increase simulation length `N` so the decimated signal has more samples
