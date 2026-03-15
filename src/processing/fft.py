"""
src/processing/fft.py
 
FFT processing module.
Takes a raw float32 audio chunk from MicInput and returns the
frequency-domain magnitude spectrum in decibels — ready for display
or further analysis (peak detection, harmonic analysis, filtering).
 
Theory recap
------------
- FFT converts time-domain samples → frequency-domain complex numbers.
- We only use the first half of the output (Nyquist symmetry).
- A window function is applied before the FFT to suppress spectral leakage.
- Magnitude is converted to dB so quiet and loud signals are both visible.
 
The single public function you'll call from everywhere:
 
    spectrum, freqs = compute_fft(samples, sample_rate)
 
That's the whole interface. Everything else in this file supports that.
"""

import numpy as np
from numpy.typing import NDArray
from enum import Enum

# ─── Window types ─────────────────────────────────────────────────────────────

class Window(Enum):
    """
    Available window functions.
    Pass one of these to compute_fft() via the window= argument.
 
    HANN     — general purpose, good leakage suppression. Default.
    HAMMING  — slightly sharper peaks than Hann, marginally more leakage.
    BLACKMAN — strongest leakage suppression, wider peaks. Good for weak
               signals sitting next to strong ones (common in SDR work).
    FLAT_TOP — most accurate amplitude values. Widest peaks. Use when
               you care about measuring exact signal strength.
    RECT     — no windowing (rectangular). Sharpest peaks, worst leakage.
               Only correct when signal is perfectly periodic in the chunk.
    """
    HANN     = "hann"
    HAMMING  = "hamming"
    BLACKMAN = "blackman"
    FLAT_TOP = "flattop"
    RECT     = "rect"

# ─── Window cache ─────────────────────────────────────────────────────────────
# Window arrays are expensive to recompute every frame (we'd be calling
# np.hanning(1024) ~40 times per second).  We cache them by (size, type)
# so each unique combination is only computed once.

_window_cache: dict[tuple[int, Window], NDArray[np.float32]] = {}

def _get_window(size: int, window_type: Window) -> NDArray[np.float32]:
    """
    Return a cached window array of the requested size and type.
    """
    key = (size, window_type)
    if key not in _window_cache:
        if window_type == Window.HANN:
            w = np.hanning(size)
        elif window_type == Window.HAMMING:
            w = np.hamming(size)
        elif window_type == Window.BLACKMAN:
            w = np.blackman(size)
        elif window_type == Window.FLAT_TOP:
            # NumPy doesn't have a built-in flat-top, but SciPy does.
            # We construct it manually here so fft.py has no SciPy dependency.
            # Coefficients from the standard flat-top definition:
            a0, a1, a2, a3, a4 = 0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368
            n = np.arange(size)
            w = (a0
                 - a1 * np.cos(2 * np.pi * n / size)
                 + a2 * np.cos(4 * np.pi * n / size)
                 - a3 * np.cos(6 * np.pi * n / size)
                 + a4 * np.cos(8 * np.pi * n / size))
        else:  # RECT
            w = np.ones(size)
 
        _window_cache[key] = w.astype(np.float32)
 
    return _window_cache[key]

# ─── Frequency axis cache ──────────────────────────────────────────────────────
# Similarly, the frequency axis (the array of Hz values for each bin)
# only depends on chunk_size and sample_rate, so we cache it too.

_freq_cache: dict[tuple[int, int], NDArray[np.float32]] = {}

def _get_freq_axis(chunk_size: int, sample_rate: int) -> NDArray[np.float32]:
    """
    Return the frequency value (in Hz) for each output bin.
 
    Output shape: (chunk_size // 2,)
    Output range: 0 Hz → sample_rate / 2  (the Nyquist limit)
    """
    key = (chunk_size, sample_rate)
    if key not in _freq_cache:
        # np.fft.rfftfreq returns the positive frequencies for a real FFT.
        # It already knows to stop at Nyquist — cleaner than slicing manually.
        freqs = np.fft.rfftfreq(chunk_size, d=1.0 / sample_rate)
        _freq_cache[key] = freqs.astype(np.float32)
    return _freq_cache[key]

# ─── Core FFT function ─────────────────────────────────────────────────────────
def compute_fft(
    samples:     NDArray[np.float32],
    sample_rate: int   = 44100,
    window:      Window = Window.HANN,
    db_floor:    float  = -90.0,     # dB value to clamp the floor to
    db_ceil:     float  =   0.0,     # dB value to clamp the ceiling to
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Compute the magnitude spectrum of an audio chunk.
 
    Parameters
    ----------
    samples     : float32 array from MicInput.read(), shape (N,), range [-1, 1]
    sample_rate : samples per second (must match what MicInput was opened with)
    window      : which window function to apply (default: Hann)
    db_floor    : minimum dB value in output (clips noise floor)
    db_ceil     : maximum dB value in output (clips peaks)
 
    Returns
    -------
    magnitude_db : float32 array, shape (N//2 + 1,), values in [db_floor, db_ceil]
                   Each element is the signal strength at the corresponding frequency.
    freqs        : float32 array, shape (N//2 + 1,), values in Hz
                   freqs[i] tells you what frequency bin i represents.
 
    Usage
    -----
        spectrum, freqs = compute_fft(chunk, sample_rate=44100)
        # spectrum[0]  → power at 0 Hz (DC component)
        # spectrum[10] → power at freqs[10] Hz
        # np.argmax(spectrum) → index of the loudest frequency bin
    """
    chunk_size = len(samples)
 
    # ── Step 1: Apply window ───────────────────────────────────────────────
    # Multiply the samples element-wise by the window curve.
    # This tapers the signal smoothly to zero at both ends of the chunk,
    # eliminating the discontinuity that causes spectral leakage.
    #
    # Think of it as: instead of abruptly cutting off the audio at the
    # chunk boundary, we gently fade it in and out.
    window_array = _get_window(chunk_size, window)
    windowed     = samples * window_array
 
    # ── Step 2: FFT ────────────────────────────────────────────────────────
    # np.fft.rfft is the right call for real-valued input (not complex).
    # It automatically exploits the Nyquist symmetry and returns only
    # the positive-frequency half — shape (N//2 + 1,) of complex numbers.
    #
    # Using rfft instead of fft:
    #   - Avoids computing the redundant mirrored half
    #   - Output is half the size, which matters at high frame rates
    #   - No manual slicing needed
    fft_complex = np.fft.rfft(windowed)
 
    # ── Step 3: Magnitude ──────────────────────────────────────────────────
    # Each bin is a complex number (real + imaginary).
    # np.abs computes the magnitude: sqrt(real² + imag²)
    # This is the "length" of the complex vector — the signal amplitude
    # at that frequency, regardless of phase.
    magnitude = np.abs(fft_complex)
 
    # ── Step 4: Normalise for window energy ───────────────────────────────
    # Applying the window reduces the total energy in the signal
    # (because we multiplied most samples by values < 1).
    # We compensate by dividing by the sum of window coefficients.
    # Without this, windowed spectra would appear artificially quieter
    # than rectangular ones — inconsistent and confusing.
    magnitude /= np.sum(window_array)
 
    # ── Step 5: Convert to decibels ───────────────────────────────────────
    # 20 * log10(amplitude) converts linear amplitude to dB.
    # The +1e-10 prevents log10(0) = -infinity, which would corrupt the display.
    # 1e-10 is small enough to be well below any real signal (~-200 dB).
    magnitude_db = 20.0 * np.log10(magnitude + 1e-10)
 
    # ── Step 6: Clamp to display range ────────────────────────────────────
    # Clip values to [db_floor, db_ceil].
    # This prevents extreme values from collapsing the display scale.
    # -90 dB is a sensible noise floor for 16-bit audio (theoretical
    # dynamic range of 16-bit PCM is ~96 dB).
    magnitude_db = np.clip(magnitude_db, db_floor, db_ceil)
 
    # ── Step 7: Frequency axis ─────────────────────────────────────────────
    freqs = _get_freq_axis(chunk_size, sample_rate)
 
    return magnitude_db.astype(np.float32), freqs

# ─── Helper utilities ──────────────────────────────────────────────────────────
# Small pure functions the display and analysis modules will use.

def bin_to_hz(bin_index: int, chunk_size: int, sample_rate: int) -> float:
    """Convert an FFT bin index to its frequency in Hz."""
    return bin_index * sample_rate / chunk_size

def hz_to_bin(frequency_hz: float, chunk_size: int, sample_rate: int) -> int:
    """Convert a frequency in Hz to the nearest FFT bin index."""
    return int(round(frequency_hz * chunk_size / sample_rate))

def peak_frequency(
    magnitude_db: NDArray[np.float32],
    freqs:        NDArray[np.float32],
    min_hz:       float = 80.0,    # ignore below this (mic handling noise)
    max_hz:       float = 8000.0,  # ignore above this (focus on voice range)
) -> tuple[float, float]:
    """
    Find the frequency with the highest magnitude in a given range.
 
    Returns
    -------
    (frequency_hz, magnitude_db) of the peak bin.
 
    The min/max defaults focus on the human voice range (80 Hz – 8 kHz).
    Adjust for other signals — e.g. for full audio set max_hz=20000.
    """
    # Build a mask for the frequency range we care about
    mask = (freqs >= min_hz) & (freqs <= max_hz)
 
    # Apply mask — we only search within the valid range
    masked_magnitude = magnitude_db.copy()
    masked_magnitude[~mask] = -np.inf   # make out-of-range bins invisible to argmax
 
    peak_bin = int(np.argmax(masked_magnitude))
    return float(freqs[peak_bin]), float(magnitude_db[peak_bin])

# ─── Smoke test ───────────────────────────────────────────────────────────────
# Run directly to verify FFT output is sane on a synthetic signal:
#   python src/processing/fft.py
 
if __name__ == "__main__":
    import matplotlib.pyplot as plt
 
    SR         = 4410
    CHUNK      = 1024
    TEST_FREQ  = 440.0   # A4 — concert pitch, easy to verify
 
    print(f"Smoke test — synthetic {TEST_FREQ} Hz sine wave\n")
 
    # Generate a clean 440 Hz sine wave (this is what a tuning fork sounds like)
    t       = np.arange(CHUNK) / SR
    sine    = np.sin(2 * np.pi * TEST_FREQ * t).astype(np.float32)
 
    # Run FFT with each window type so you can see the difference
    fig, axes = plt.subplots(len(Window), 1, figsize=(12, 14), sharex=True)
    fig.suptitle(f"Window comparison — {TEST_FREQ} Hz sine wave", fontsize=13)
 
    for ax, win in zip(axes, Window):
        spectrum, freqs = compute_fft(sine, sample_rate=SR, window=win)
        peak_hz, peak_db = peak_frequency(spectrum, freqs)
 
        ax.plot(freqs, spectrum, linewidth=0.8, color='#00cfff')
        ax.axvline(TEST_FREQ, color='#ff3355', linewidth=1, linestyle='--',
                   label=f'True freq: {TEST_FREQ} Hz')
        ax.set_xlim(0, 2000)
        ax.set_ylim(-90, 5)
        ax.set_ylabel('dB')
        ax.set_title(f'{win.name}  —  detected peak: {peak_hz:.1f} Hz @ {peak_db:.1f} dB',
                     fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
 
    axes[-1].set_xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()
 
    # Numeric check
    spectrum, freqs = compute_fft(sine, sample_rate=SR, window=Window.HANN)
    peak_hz, peak_db = peak_frequency(spectrum, freqs)
    freq_res = SR / CHUNK
 
    print(f"  Chunk size       : {CHUNK} samples")
    print(f"  Frequency res    : {freq_res:.1f} Hz/bin")
    print(f"  True frequency   : {TEST_FREQ} Hz")
    print(f"  Detected peak    : {peak_hz:.1f} Hz  ({abs(peak_hz - TEST_FREQ):.1f} Hz error)")
    print(f"  Peak magnitude   : {peak_db:.1f} dB")
    print(f"  Output shape     : {spectrum.shape}")
    print(f"  Output dtype     : {spectrum.dtype}")
    print(f"\nIf detected peak is within {freq_res:.0f} Hz of {TEST_FREQ} Hz, FFT is working ✓")
 
 
 