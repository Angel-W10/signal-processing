# SIGNAL MONITOR

> A retro phosphor oscilloscope audio analyzer — built as a learning project for DSP and SDR fundamentals.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-2.6-green?style=flat-square)
![NumPy](https://img.shields.io/badge/NumPy-DSP-013243?style=flat-square&logo=numpy)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## What It Does

Captures live audio from your microphone and displays it as a real-time phosphor oscilloscope with four panels:

- **Waveform** — main panel, triggered time-domain oscilloscope trace with amplitude and time zoom
- **Waterfall** — scrolling spectrogram showing signal history over time
- **Spectrum Analyzer** — FFT magnitude plot with peak hold and log/linear frequency axis
- **Filter Controls** — interactive filter bank with real-time parameter adjustment

Four phosphor themes: **Green** (P31), **Amber** (P3), **Blue** (P7), **Night Vision** (red).

This project is the first half of a larger SDR (Software Defined Radio) build. When a hardware RTL-SDR dongle arrives, the mic input module swaps for an SDR input module — the entire processing and display pipeline stays identical.

---

## Layout

```
┌─────────────────────────────────────────────┐
│  SIGNAL MONITOR          SR:44kHz  FPS:60   │
├─────────────────────────────────────────────┤
│                                             │
│              WAVEFORM          (full width) │
│         triggered trace + RMS meter         │
│                                             │
├─────────────────────────────────────────────┤
│         WATERFALL / SPECTROGRAM (full width)│
│      (time flows downward, freq = x)        │
├──────────────────────┬──────────────────────┤
│   SPECTRUM ANALYZER  │   FILTER BANK        │
│   FFT + peak hold    │   [1][2][3][4]       │
└──────────────────────┴──────────────────────┘
```

---

## The Signal Pipeline

```
Microphone
    ↓
PyAudio          — raw sample buffer (bytes → float32)
    ↓
FilterBank       — lowpass / highpass / bandpass / notch (Butterworth IIR)
    ↓
compute_fft      — windowed FFT → magnitude in dB
    ↓
Pygame display   — waveform + waterfall + spectrum + controls at 60fps
```

When RTL-SDR hardware arrives, `MicInput` → `RtlSdrInput`. Everything downstream is unchanged.

---

## Project Structure

```
signal-monitor/
│
├── src/
│   ├── input/
│   │   ├── mic.py          # PyAudio mic capture (callback mode)
│   │   └── sdr.py          # RTL-SDR input (placeholder)
│   │
│   ├── processing/
│   │   ├── fft.py          # Windowed FFT, dB conversion, peak detection
│   │   └── filters.py      # Butterworth filter bank (SOS form, stateful)
│   │
│   ├── display/
│   │   ├── theme.py        # Phosphor themes, glow renderer, scan lines
│   │   ├── waveform.py     # Time-domain oscilloscope trace (main panel)
│   │   ├── waterfall.py    # Scrolling spectrogram (numpy pixel buffer)
│   │   ├── spectrum.py     # FFT magnitude panel
│   │   └── controls.py     # Filter bank UI + key handling
│   │
│   └── main.py             # Entry point — wires everything together
│
├── assets/
│   └── fonts/
│       ├── digital-7.ttf
│       └── ShareTechMono-Regular.ttf
│
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/signal-monitor.git
cd signal-monitor
```

### 2. Create environment

```bash
conda create -n signal-monitor python=3.11
conda activate signal-monitor
```

### 3. Install dependencies

```bash
pip install numpy scipy matplotlib pygame librosa pyaudio
```

> **macOS:** Install PortAudio first: `brew install portaudio`
>
> **Windows:** If `pip install pyaudio` fails: `pip install pipwin && pipwin install pyaudio`
>
> **Linux:** `sudo apt-get install portaudio19-dev && pip install pyaudio`

### 4. Run

```bash
python -m src.main
```

Allow microphone access when prompted.

---

## Keybindings

### Global

| Key | Action |
|-----|--------|
| `ESC` / `Q` | Quit |
| `F1` | Cycle phosphor theme |

### Waveform (main panel)

| Key | Action |
|-----|--------|
| `A` | Amplitude zoom in — boost Y scale, makes quiet signals fill the panel |
| `S` | Amplitude zoom out — reduce Y scale |
| `Z` | Time zoom in — fewer samples shown, more detail per cycle |
| `X` | Time zoom out — more samples shown, more context |
| `T` | Toggle trigger (stabilises periodic signals) |
| `G` | Toggle amplitude grid |

> **Tip:** Start talking and hit `A` until the waveform fills ~60% of the panel. That's the sweet spot for voice.

### Spectrum

| Key | Action |
|-----|--------|
| `L` | Toggle log / linear frequency axis |
| `P` | Toggle peak hold |

### Waterfall

| Key | Action |
|-----|--------|
| `L` | Toggle log / linear frequency axis |
| `+` / `-` | Scroll speed |

### Filter Controls

| Key | Action |
|-----|--------|
| `1` `2` `3` `4` | Select filter slot |
| `SPACE` | Toggle selected filter on/off |
| `↑` `↓` | Adjust selected filter frequency |
| `SHIFT` + `↑` `↓` | Adjust frequency (fast step) |
| `B` | Bypass all filters |
| `H` | Toggle help panel |

---

## Filters

Four pre-wired slots, all disabled by default:

| Slot | Type | Default | Purpose |
|------|------|---------|---------| 
| `[1]` | Highpass | 80 Hz | Remove mic rumble / DC offset |
| `[2]` | Lowpass | 3400 Hz | Cut high-frequency hiss |
| `[3]` | Bandpass | 80–3400 Hz | Telephone-quality voice band |
| `[4]` | Notch | 50 Hz | Remove mains hum (change to 60 Hz for US) |

Enable a filter with `1-4` then `SPACE`. Adjust cutoff with `↑↓`. All filters maintain state across chunks — no click artifacts at chunk boundaries.

---

## Themes

| Theme | Phosphor | Inspired By |
|-------|----------|-------------|
| **Green** | P31 | Tektronix lab scopes, 1960s–90s |
| **Amber** | P3 | IBM 3270 terminals |
| **Blue** | P7 | Radar displays, storage scopes |
| **Night Vision** | Red | Astronomy / dark room use |

Cycle with `F1`. All panels update simultaneously including the waterfall colormap.

---

## Individual Panel Tests

Each panel has a standalone smoke test with synthetic animated data:

```bash
python -m src.display.theme     # all four themes side by side
python -m src.display.waveform  # triggered trace, clipping test, zoom
python -m src.display.waterfall # drifting carrier + harmonic series
python -m src.display.spectrum  # animated FFT with drifting peaks
python -m src.display.controls  # filter bank UI
```

---

## DSP Concepts Covered

This project was built as a learning exercise. Concepts implemented from scratch:

- **Sampling theorem** and Nyquist limit
- **FFT** (fast Fourier transform) — time domain → frequency domain
- **Window functions** — Hann, Hamming, Blackman, Flat-top (spectral leakage suppression)
- **IQ data** and why it matters (groundwork for SDR)
- **Butterworth IIR filters** in SOS (Second Order Sections) form
- **Filter state** (`zi`) — continuous filtering across chunk boundaries
- **Spectrogram** / waterfall display — numpy pixel buffer + `surfarray.blit_array`
- **Oscilloscope triggering** — rising zero-crossing sync for stable waveform display
- **Amplitude and time zoom** — independent Y and X axis scaling for waveform analysis
- **Phosphor CRT simulation** — persistence, glow, scan lines, vignette

---

## Roadmap

- [ ] RTL-SDR input module (`src/input/sdr.py`)
- [ ] FM demodulation
- [ ] NOAA APT weather satellite decoder
- [ ] Pitch / harmonic detection (HPS algorithm)
- [ ] ESP32 / OLED display output
- [ ] Raspberry Pi framebuffer renderer

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | All signal processing math |
| `scipy` | Filter design (`butter`, `sosfilt`), FFT helpers |
| `pyaudio` | Microphone capture |
| `pygame` | Real-time display at 60fps |
| `librosa` | Audio analysis (pitch detection, future use) |
| `matplotlib` | Offline analysis and filter smoke tests |

---

## License

MIT — do whatever you want with it.

---

*Built while waiting for an RTL-SDR dongle to arrive in the post.*
