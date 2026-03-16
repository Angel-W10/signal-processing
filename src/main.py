"""
src/main.py

Entry point for the Signal Monitor.
Wires together every module we've built:

    MicInput → FilterBank → compute_fft → [SpectrumPanel,
                                            WaterfallPanel,
                                            WaveformPanel,
                                            ControlsPanel]

Run with:
    python -m src.main

Layout
------
    ┌─────────────────────────────────────────────┐
    │            HEADER BAR                       │
    ├─────────────────────────────────────────────┤
    │                                             │
    │              WAVEFORM                       │  38% height
    │                                             │
    ├─────────────────────────────────────────────┤
    │              WATERFALL                      │  28% height
    ├──────────────────────┬──────────────────────┤
    │   SPECTRUM           │   FILTER CONTROLS    │  34% height
    └──────────────────────┴──────────────────────┘

Keybindings (global)
--------------------
    ESC / Q     quit
    F1          cycle theme (all panels update simultaneously)
    L           toggle log/linear axis  (spectrum + waterfall)
    P           toggle peak hold        (spectrum)
    T           toggle trigger          (waveform)
    G           toggle grid             (waveform)
    1-4         select filter slot      (controls)
    SPACE       toggle selected filter  (controls)
    UP / DOWN   adjust filter freq      (controls)
    B           bypass all filters      (controls)
    H           toggle help             (controls)
"""

import sys
import os
import time
import numpy as np
import pygame

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensures src/ is on the path regardless of how the file is invoked.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.input.mic          import MicInput, SAMPLE_RATE, CHUNK_SIZE
from src.processing.fft     import compute_fft, peak_frequency, Window
from src.processing.filters import FilterBank
from src.display.theme      import get_theme, THEMES
from src.display.spectrum   import SpectrumPanel
from src.display.waterfall  import WaterfallPanel
from src.display.waveform   import WaveformPanel
from src.display.controls   import ControlsPanel


# ─── Window configuration ─────────────────────────────────────────────────────

WINDOW_W = 1280
WINDOW_H = 780
WINDOW_TITLE = "SIGNAL MONITOR  //  PHOSPHOR AUDIO ANALYZER"
TARGET_FPS   = 60

# Layout proportions (fractions of usable height below header)
# Waveform is the main panel — sits at the top and gets the most space.
# Spectrum moves to the bottom strip. Waterfall stays in the middle.
HEADER_H      = 32
WAVEFORM_FRAC = 0.38
WATERFALL_FRAC= 0.28
SPECTRUM_FRAC = 1.0 - WAVEFORM_FRAC - WATERFALL_FRAC


# ─── Layout builder ───────────────────────────────────────────────────────────

def build_rects(w: int, h: int) -> dict[str, pygame.Rect]:
    """
    Compute panel rects from window dimensions.
    Called on startup and on window resize.

    Layout (top to bottom):
      [        WAVEFORM          ]   full width  — 38%
      [        WATERFALL         ]   full width  — 28%
      [ SPECTRUM  |  CONTROLS ]      half + half — 34%
    """
    usable_h = h - HEADER_H

    waveform_h  = int(usable_h * WAVEFORM_FRAC)
    waterfall_h = int(usable_h * WATERFALL_FRAC)
    bottom_h    = usable_h - waveform_h - waterfall_h

    y_wave  = HEADER_H
    y_water = y_wave  + waveform_h
    y_bot   = y_water + waterfall_h

    half_w = w // 2

    return {
        "header"   : pygame.Rect(0,      0,       w,          HEADER_H),
        "waveform" : pygame.Rect(0,      y_wave,  w,          waveform_h),
        "waterfall": pygame.Rect(0,      y_water, w,          waterfall_h),
        "spectrum" : pygame.Rect(0,      y_bot,   half_w,     bottom_h),
        "controls" : pygame.Rect(half_w, y_bot,   w - half_w, bottom_h),
    }


# ─── Header renderer ──────────────────────────────────────────────────────────

def draw_header(
    surface : pygame.Surface,
    rect    : pygame.Rect,
    theme,
    fps     : float,
    sr      : int,
    theme_name: str,
    mic_active: bool,
):
    """
    Draw the top header bar with title, status, FPS, and sample rate.
    """
    pygame.draw.rect(surface, theme.BG_PANEL, rect)
    pygame.draw.line(surface, theme.GRID,
                     (rect.x, rect.bottom - 1), (rect.right, rect.bottom - 1), 1)

    # Left — title
    theme.draw_text(surface, "SIGNAL MONITOR",
                    (rect.x + 12, rect.centery),
                    font="mono_large", color=theme.ACCENT,
                    anchor="midleft", glow=True)

    # Center — theme name
    theme.draw_text(surface, f"[ {theme_name.upper()} ]",
                    (rect.centerx, rect.centery),
                    font="digit_main", color=theme.TEXT_DIM, anchor="center")

    # Right — status cluster
    rx = rect.right - 12

    # FPS
    fps_color = theme.PHOSPHOR_MID if fps >= 50 else theme.WARNING
    theme.draw_text(surface, f"{fps:.0f} FPS",
                    (rx, rect.centery),
                    font="digit_main", color=fps_color, anchor="midright")
    rx -= 90

    # Sample rate
    theme.draw_text(surface, f"{sr // 1000}kHz",
                    (rx, rect.centery),
                    font="digit_main", color=theme.TEXT_DIM, anchor="midright")
    rx -= 70

    # Mic indicator
    if mic_active:
        dot_color = theme.PHOSPHOR_MID
        label     = "● MIC"
    else:
        dot_color = theme.WARNING
        label     = "○ MIC"

    theme.draw_text(surface, label,
                    (rx, rect.centery),
                    font="digit_main", color=dot_color,
                    anchor="midright", glow=mic_active)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode(
        (WINDOW_W, WINDOW_H),
        pygame.RESIZABLE,
    )
    pygame.display.set_caption(WINDOW_TITLE)
    clock = pygame.time.Clock()

    # ── Theme ────────────────────────────────────────────────────────────────
    theme_idx = 0
    theme     = get_theme(theme_idx)

    # ── Layout ───────────────────────────────────────────────────────────────
    rects = build_rects(WINDOW_W, WINDOW_H)

    # ── Audio input ───────────────────────────────────────────────────────────
    mic = MicInput(sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE)
    mic.start()

    # ── Filter bank ───────────────────────────────────────────────────────────
    filter_bank = FilterBank(sample_rate=SAMPLE_RATE)

    # ── Display panels ────────────────────────────────────────────────────────
    spectrum_panel  = SpectrumPanel(
        screen, rects["spectrum"], theme, sample_rate=SAMPLE_RATE
    )
    waterfall_panel = WaterfallPanel(
        screen, rects["waterfall"], theme, sample_rate=SAMPLE_RATE
    )
    waveform_panel  = WaveformPanel(
        screen, rects["waveform"], theme,
        zoom=0.4,   # show 40% of chunk = ~2.5x zoom on the time axis
    )

    def on_theme_change():
        """Cycle theme and push it to every panel simultaneously."""
        nonlocal theme_idx, theme
        theme_idx = (theme_idx + 1) % len(THEMES)
        theme     = get_theme(theme_idx)

        spectrum_panel.theme  = theme
        waterfall_panel.theme = theme
        waterfall_panel._build_colormap()   # rebuild color LUT for new theme
        waveform_panel.theme  = theme
        controls_panel.theme  = theme
        print(f"[main] theme → {theme.NAME}")

    controls_panel = ControlsPanel(
        screen, rects["controls"], theme, filter_bank,
        sample_rate     = SAMPLE_RATE,
        chunk_size      = CHUNK_SIZE,
        on_theme_change = on_theme_change,
    )

    # ── All panels in one list for easy routing ────────────────────────────
    all_panels = [spectrum_panel, waterfall_panel, waveform_panel, controls_panel]

    print(f"[main] running — {SAMPLE_RATE}Hz  chunk={CHUNK_SIZE}  target={TARGET_FPS}fps")
    print(f"[main] window  {WINDOW_W}×{WINDOW_H}  theme={theme.NAME}")
    print("[main] ESC or Q to quit")

    running   = True
    frame     = 0

    while running:

        # ── Events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                # Rebuild layout and update every panel's rect
                W, H  = event.w, event.h
                rects = build_rects(W, H)
                waveform_panel.rect  = rects["waveform"]
                waterfall_panel.rect = rects["waterfall"]
                waterfall_panel._pixel_buf = None   # force buffer rebuild
                spectrum_panel.rect  = rects["spectrum"]
                controls_panel.rect  = rects["controls"]

            elif event.type == pygame.KEYDOWN:

                # Global quit
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

                # Global theme cycle — F1
                elif event.key == pygame.K_F1:
                    on_theme_change()

                # Route to controls first (1-4, SPACE, B, H, UP, DOWN)
                # If controls doesn't consume it, route to other panels
                else:
                    consumed = controls_panel.handle_key(event)
                    if not consumed:
                        spectrum_panel.handle_key(event)   # L, P
                        waveform_panel.handle_key(event)   # T, G, Z, X
                        waterfall_panel.handle_key(event)  # L, +, -

        # ── Audio processing ──────────────────────────────────────────────────
        raw_chunk = mic.read()

        # Apply filters (all disabled by default — user enables via controls)
        filtered_chunk = filter_bank.process(raw_chunk)

        # FFT on the filtered signal
        spectrum_db, freqs = compute_fft(
            filtered_chunk,
            sample_rate = SAMPLE_RATE,
            window      = Window.HANN,
        )

        # Peak frequency for stats readout
        peak_hz, peak_db = peak_frequency(spectrum_db, freqs)
        rms              = float(np.sqrt(np.mean(filtered_chunk ** 2)))
        fps              = clock.get_fps()

        # Push stats to controls panel
        controls_panel.update_stats(fps, peak_hz, peak_db, rms)

        # ── Feed panels ───────────────────────────────────────────────────────
        spectrum_panel.update(spectrum_db, freqs)
        waterfall_panel.update(spectrum_db, freqs)
        waveform_panel.update(filtered_chunk)

        # ── Render ────────────────────────────────────────────────────────────
        # begin_frame does the phosphor persistence fade instead of hard clear
        theme.begin_frame(screen)

        # Dark background fill for the full window
        screen.fill(theme.BG)

        # Header
        draw_header(
            screen, rects["header"], theme,
            fps        = fps,
            sr         = SAMPLE_RATE,
            theme_name = theme.NAME,
            mic_active = mic.running,
        )

        # Panels
        spectrum_panel.draw()
        waterfall_panel.draw()
        waveform_panel.draw()
        controls_panel.draw()

        # Divider between waveform and controls
        pygame.draw.line(
            screen, theme.GRID,
            (rects["controls"].x, rects["controls"].y),
            (rects["controls"].x, rects["controls"].bottom),
            1
        )

        # Scan lines + vignette on top of everything
        theme.end_frame(screen)

        pygame.display.flip()
        clock.tick(TARGET_FPS)
        frame += 1

    # ── Cleanup ───────────────────────────────────────────────────────────────
    print("[main] shutting down...")
    mic.stop()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()