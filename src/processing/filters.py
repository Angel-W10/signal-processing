"""
src/processing/filters.py

Filter bank module.
Designs, stores, and applies digital filters to audio chunks in real time.
Handles filter state correctly across chunk boundaries so there are no
click artifacts at chunk edges.

Key design decisions
--------------------
- All filters use SOS (Second Order Sections) form for numerical stability.
- Each filter carries its own zi (state) array between chunks.
- Filters are stored in an ordered dict so they apply in insertion order.
- Parameters (cutoff, bandwidth) are re-designable at runtime without
  restarting the stream — important for interactive UI controls.
- The same FilterBank class works identically on SDR IQ data. You only
  change the sample_rate when constructing it.

Public interface
----------------
    bank = FilterBank(sample_rate=44100)

    bank.add_lowpass ("lp",    cutoff_hz=3000)
    bank.add_highpass("hp",    cutoff_hz=80)
    bank.add_bandpass("voice", lo_hz=300,  hi_hz=3400)
    bank.add_notch   ("hum",   center_hz=60, q=30)

    bank.enable("lp")
    bank.disable("hp")
    bank.set_cutoff("lp", new_cutoff_hz=1500)   # redesigns on the fly

    filtered = bank.process(chunk)              # np.ndarray float32 in, out
    info     = bank.info()                      # dict summary for display
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum, auto


# ─── Filter type enum ─────────────────────────────────────────────────────────

class FilterType(Enum):
    LOWPASS  = auto()
    HIGHPASS = auto()
    BANDPASS = auto()
    NOTCH    = auto()


# ─── Internal filter record ───────────────────────────────────────────────────

@dataclass
class _Filter:
    """
    Everything needed to apply one filter to an ongoing audio stream.

    sos : np.ndarray
        Second-order sections array, shape (n_sections, 6).
        This encodes the filter's frequency response.
        Computed once by scipy.signal.butter / iirnotch, stored here.

    zi : np.ndarray
        Filter state — the "memory" of the filter.
        Shape: (n_sections, 2).
        Carries over from one chunk to the next so the filter sounds
        continuous across chunk boundaries.
        Re-initialised whenever the filter is redesigned.

    enabled : bool
        Whether this filter is currently applied in process().

    filter_type : FilterType
        Stored for reference and for re-design on parameter change.

    params : dict
        The human-readable parameters this filter was designed with
        (cutoff_hz, lo_hz, hi_hz, center_hz, q, order).
        Stored so we can redesign with one parameter changed.
    """
    sos:         NDArray[np.float64]
    zi:          NDArray[np.float64]
    enabled:     bool
    filter_type: FilterType
    params:      dict = field(default_factory=dict)


# ─── FilterBank class ─────────────────────────────────────────────────────────

class FilterBank:
    """
    A named, ordered collection of digital filters applied in series.

    'In series' means the output of filter 1 becomes the input of filter 2,
    and so on.  Order matters — a highpass before a lowpass is a bandpass.

    Parameters
    ----------
    sample_rate : int
        Must match the sample rate of the audio stream (default 44100 Hz).
        All cutoff frequencies are interpreted relative to this.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        # OrderedDict preserves insertion order — filters apply in the order
        # you add them, which is intuitive and predictable.
        self._filters: OrderedDict[str, _Filter] = OrderedDict()

    # ── Private helpers ────────────────────────────────────────────────────

    def _nyquist(self) -> float:
        """
        Nyquist frequency in Hz.
        This is the highest frequency we can represent at this sample rate.
        scipy.signal.butter expects cutoff as a fraction of Nyquist (0–1),
        so we divide all Hz values by this before passing to scipy.
        """
        return self.sample_rate / 2.0

    def _normalise(self, hz: float) -> float:
        """
        Convert a frequency in Hz to a normalised Wn value for scipy.
        scipy butter() expects Wn in (0, 1) where 1.0 = Nyquist.
        Clamps to (0.001, 0.999) to avoid the mathematically degenerate
        edge cases of exactly 0 or exactly Nyquist.
        """
        return float(np.clip(hz / self._nyquist(), 0.001, 0.999))

    def _make_filter(
        self,
        sos:         NDArray,
        filter_type: FilterType,
        params:      dict,
        enabled:     bool = True,
    ) -> _Filter:
        """
        Wrap a raw SOS array into a _Filter record with initialised state.

        sosfilt_zi computes the initial state for a step response —
        meaning the filter starts "settled" rather than ringing.
        We scale it by samples[0] in process() to avoid a transient
        at the very start of the stream.
        """
        # zi shape: (n_sections, 2)
        # This is the template state for a unit-amplitude signal.
        zi_template = sosfilt_zi(sos)
        return _Filter(
            sos         = sos,
            zi          = zi_template,   # will be scaled on first process() call
            enabled     = enabled,
            filter_type = filter_type,
            params      = params,
        )

    # ── Filter constructors ────────────────────────────────────────────────

    def add_lowpass(
        self,
        name:      str,
        cutoff_hz: float,
        order:     int   = 5,
        enabled:   bool  = True,
    ) -> "FilterBank":
        """
        Add a Butterworth low-pass filter.

        Passes frequencies below cutoff_hz, attenuates above.
        cutoff_hz is the -3 dB point (half-power frequency).

        order : higher = sharper rolloff, more computation, more ringing.
                4–6 is the sweet spot for audio.

        Returns self so calls can be chained:
            bank.add_lowpass(...).add_highpass(...)
        """
        wn  = self._normalise(cutoff_hz)
        sos = butter(order, wn, btype='low', output='sos')
        self._filters[name] = self._make_filter(
            sos, FilterType.LOWPASS,
            params=dict(cutoff_hz=cutoff_hz, order=order),
            enabled=enabled,
        )
        return self

    def add_highpass(
        self,
        name:      str,
        cutoff_hz: float,
        order:     int  = 5,
        enabled:   bool = True,
    ) -> "FilterBank":
        """
        Add a Butterworth high-pass filter.

        Passes frequencies above cutoff_hz, attenuates below.

        Common uses:
        - cutoff_hz=80   : remove mic handling rumble
        - cutoff_hz=20   : remove subsonic DC drift
        - cutoff_hz=100  : remove RTL-SDR DC spike (when you get the hardware)
        """
        wn  = self._normalise(cutoff_hz)
        sos = butter(order, wn, btype='high', output='sos')
        self._filters[name] = self._make_filter(
            sos, FilterType.HIGHPASS,
            params=dict(cutoff_hz=cutoff_hz, order=order),
            enabled=enabled,
        )
        return self

    def add_bandpass(
        self,
        name:    str,
        lo_hz:   float,
        hi_hz:   float,
        order:   int  = 4,
        enabled: bool = True,
    ) -> "FilterBank":
        """
        Add a Butterworth band-pass filter.

        Passes only frequencies between lo_hz and hi_hz.

        Note: scipy doubles the filter order for bandpass designs
        internally — an order=4 bandpass is actually an 8th-order filter.
        Keep order at 2–4 for bandpass to avoid instability.

        Common uses:
        - lo=300,  hi=3400 : telephone/voice band
        - lo=80,   hi=1200 : male fundamental + first few harmonics
        - lo=1000, hi=4000 : consonant intelligibility range
        """
        lo  = self._normalise(lo_hz)
        hi  = self._normalise(hi_hz)
        sos = butter(order, [lo, hi], btype='band', output='sos')
        self._filters[name] = self._make_filter(
            sos, FilterType.BANDPASS,
            params=dict(lo_hz=lo_hz, hi_hz=hi_hz, order=order),
            enabled=enabled,
        )
        return self

    def add_notch(
        self,
        name:      str,
        center_hz: float,
        q:         float = 30.0,
        enabled:   bool  = True,
    ) -> "FilterBank":
        """
        Add a notch (band-stop) filter at a specific frequency.

        Surgically removes center_hz while leaving everything else intact.

        q : Quality factor — controls notch width.
            Higher Q = narrower notch.
            Q=30 removes ~±10 Hz around center. Good for mains hum removal.
            Q=10 is wider, Q=100 is very tight.

        Common uses:
        - center_hz=50,  q=30 : remove EU mains hum (50 Hz)
        - center_hz=60,  q=30 : remove US mains hum (60 Hz)
        - center_hz=100, q=50 : remove 2nd harmonic of mains
        """
        # iirnotch returns (b, a) coefficients — convert to SOS for stability
        b, a = iirnotch(center_hz, q, fs=self.sample_rate)

        # Convert ba → sos manually (scipy doesn't have iirnotch_sos directly)
        from scipy.signal import tf2sos
        sos = tf2sos(b, a)

        self._filters[name] = self._make_filter(
            sos, FilterType.NOTCH,
            params=dict(center_hz=center_hz, q=q),
            enabled=enabled,
        )
        return self

    # ── Runtime controls ───────────────────────────────────────────────────

    def enable(self, name: str) -> None:
        """Enable a filter by name. Raises KeyError if name not found."""
        self._filters[name].enabled = True

    def disable(self, name: str) -> None:
        """Disable a filter by name. Signal passes through unaffected."""
        self._filters[name].enabled = False

    def toggle(self, name: str) -> bool:
        """Toggle a filter on/off. Returns the new enabled state."""
        f = self._filters[name]
        f.enabled = not f.enabled
        return f.enabled

    def remove(self, name: str) -> None:
        """Remove a filter entirely from the bank."""
        del self._filters[name]

    def set_cutoff(self, name: str, cutoff_hz: float) -> None:
        """
        Redesign a lowpass or highpass filter with a new cutoff frequency.
        The filter state is reset — expect a very brief transient.
        The enabled state is preserved.
        """
        f = self._filters[name]
        enabled = f.enabled
        params  = {**f.params, "cutoff_hz": cutoff_hz}

        if f.filter_type == FilterType.LOWPASS:
            self.add_lowpass(name, cutoff_hz, order=params.get("order", 5), enabled=enabled)
        elif f.filter_type == FilterType.HIGHPASS:
            self.add_highpass(name, cutoff_hz, order=params.get("order", 5), enabled=enabled)
        else:
            raise ValueError(f"set_cutoff only works on lowpass/highpass filters. "
                             f"'{name}' is a {f.filter_type.name}.")

    def set_bandpass(self, name: str, lo_hz: float, hi_hz: float) -> None:
        """Redesign a bandpass filter with new bounds."""
        f = self._filters[name]
        self.add_bandpass(name, lo_hz, hi_hz,
                          order=f.params.get("order", 4),
                          enabled=f.enabled)

    def set_notch(self, name: str, center_hz: float, q: float = None) -> None:
        """Redesign a notch filter with a new center frequency (and optionally new Q)."""
        f = self._filters[name]
        new_q = q if q is not None else f.params.get("q", 30.0)
        self.add_notch(name, center_hz, q=new_q, enabled=f.enabled)

    # ── Core processing ────────────────────────────────────────────────────

    def process(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Apply all enabled filters to a chunk of samples, in insertion order.

        Parameters
        ----------
        samples : float32 array, shape (N,), range [-1.0, +1.0]
                  Typically the output of MicInput.read().

        Returns
        -------
        filtered : float32 array, same shape as input.
                   If no filters are enabled, returns the input unchanged.

        State management
        ----------------
        Each filter's zi is updated in-place by sosfilt.
        The updated zi is stored back so the next call continues seamlessly.
        This is what prevents the click artifacts at chunk boundaries.
        """
        # Work in float64 internally for numerical precision.
        # We convert back to float32 at the end.
        out = samples.astype(np.float64)

        for name, filt in self._filters.items():
            if not filt.enabled:
                continue

            # Scale the initial state by the first sample value.
            # This settles the filter immediately rather than ringing
            # for the first few samples of a new stream.
            # Only matters on the very first call — after that zi carries
            # real state from the previous chunk.
            zi_scaled = filt.zi * out[0]

            # sosfilt applies the filter and returns:
            #   out      — the filtered signal
            #   zi_new   — the updated state to pass into the next chunk
            out, zi_new = sosfilt(filt.sos, out, zi=zi_scaled)

            # Store updated state for next chunk
            filt.zi = zi_new / (out[-1] if out[-1] != 0 else 1.0)

        return out.astype(np.float32)

    def process_copy(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Apply filters WITHOUT updating state.
        Useful for previewing what a filter will sound like before enabling it,
        or for running the same chunk through multiple filter configurations.
        State is completely unchanged after this call.
        """
        out = samples.astype(np.float64)

        for name, filt in self._filters.items():
            if not filt.enabled:
                continue
            zi_scaled = filt.zi * out[0]
            out, _ = sosfilt(filt.sos, out, zi=zi_scaled)
            # _ discards the new state — original filt.zi is untouched

        return out.astype(np.float32)

    # ── Inspection ────────────────────────────────────────────────────────

    def info(self) -> dict:
        """
        Return a summary of all filters — useful for display panels.

        Returns
        -------
        dict mapping filter name → {type, enabled, params}
        """
        return {
            name: {
                "type":    filt.filter_type.name,
                "enabled": filt.enabled,
                "params":  filt.params,
            }
            for name, filt in self._filters.items()
        }

    def __len__(self) -> int:
        return len(self._filters)

    def __contains__(self, name: str) -> bool:
        return name in self._filters

    def __repr__(self) -> str:
        lines = [f"FilterBank(sample_rate={self.sample_rate}, filters=["]
        for name, f in self._filters.items():
            status = "ON " if f.enabled else "OFF"
            lines.append(f"  {status}  {name:20s}  {f.filter_type.name:10s}  {f.params}")
        lines.append("])")
        return "\n".join(lines)
    
    def _get(self, name: str) -> _Filter:
        if name not in self._filters:
            raise KeyError(f"No filter named '{name}'. Available: {list(self._filters.keys())}")
        return self._filters[name]

    @property
    def filter_names(self) -> list[str]:
        return list(self._filters.keys())

    @property
    def active_count(self) -> int:
        return sum(1 for f in self._filters.values() if f.enabled)


# ─── Smoke test ───────────────────────────────────────────────────────────────
# python src/processing/filters.py

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.signal import freqz_sos

    SR = 44100

    # Build a bank with all four filter types
    bank = (
        FilterBank(sample_rate=SR)
        .add_lowpass ("lp",    cutoff_hz=3000)
        .add_highpass("hp",    cutoff_hz=80)
        .add_bandpass("voice", lo_hz=300, hi_hz=3400)
        .add_notch   ("hum",   center_hz=60, q=30)
    )

    print(bank)
    print()

    # ── Plot 1: frequency response of each filter ──────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Filter frequency responses", fontsize=13)
    axes = axes.flatten()

    colors = ['#00cfff', '#00ff88', '#ffaa00', '#ff3355']

    for ax, (name, filt), color in zip(axes, bank._filters.items(), colors):
        w, h = freqz_sos(filt.sos, worN=4096, fs=SR)
        db   = 20 * np.log10(np.abs(h) + 1e-10)

        ax.plot(w, db, color=color, linewidth=1.5)
        ax.axhline(-3,  color='white', linewidth=0.6, linestyle='--', alpha=0.4,
                   label='-3 dB point')
        ax.axhline(-60, color='gray',  linewidth=0.6, linestyle=':',  alpha=0.3)
        ax.set_title(f"{name}  ({filt.filter_type.name})  params={filt.params}",
                     fontsize=9)
        ax.set_xlim(0, SR / 2)
        ax.set_ylim(-80, 5)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#080f16')

    fig.patch.set_facecolor('#040a0f')
    plt.tight_layout()

    # ── Plot 2: before/after on a synthetic noisy signal ──────────────────
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig2.suptitle("FilterBank.process() — before vs after (voice bandpass only)",
                  fontsize=12)

    t       = np.arange(SR) / SR    # 1 second
    # Synthesise: voice-like 200 Hz fundamental + hum at 60 Hz + high freq hiss
    signal  = (
        0.4 * np.sin(2 * np.pi * 200  * t) +   # voice fundamental
        0.4 * np.sin(2 * np.pi * 400  * t) +   # 1st harmonic
        0.4 * np.sin(2 * np.pi * 60   * t) +   # mains hum
        0.2 * np.random.randn(SR)               # broadband noise
    ).astype(np.float32)

    # Only enable voice bandpass for this demo
    bank.disable("lp")
    bank.disable("hp")
    bank.disable("hum")
    bank.enable("voice")

    filtered = bank.process(signal)

    ax1.plot(t[:500], signal[:500],   color='#00cfff', linewidth=0.8, label='Original')
    ax2.plot(t[:500], filtered[:500], color='#00ff88', linewidth=0.8, label='Bandpass filtered (300–3400 Hz)')

    for ax in (ax1, ax2):
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.set_facecolor('#080f16')

    ax2.set_xlabel("Time (s)")
    fig2.patch.set_facecolor('#040a0f')
    plt.tight_layout()

    plt.show()

    # ── Runtime control demo ───────────────────────────────────────────────
    print("Runtime control demo:")
    print(f"  'lp' enabled before toggle : {bank._filters['lp'].enabled}")
    bank.toggle("lp")
    print(f"  'lp' enabled after  toggle : {bank._filters['lp'].enabled}")

    bank.set_cutoff("lp", 1000)
    print(f"  'lp' cutoff after set_cutoff(1000) : "
          f"{bank._filters['lp'].params['cutoff_hz']} Hz")

    print("\ninfo() output:")
    for name, meta in bank.info().items():
        print(f"  {name}: {meta}")