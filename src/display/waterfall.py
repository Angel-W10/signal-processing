"""
src/display/waterfall.py

Waterfall (spectrogram) display panel.
Scrolls FFT magnitude history downward — newest data at top, oldest at bottom.
Each row is one FFT snapshot, each pixel's color represents signal strength at
that frequency at that moment in time.

This is the primary display in all SDR software — it makes intermittent
signals, frequency drift, and interference patterns immediately visible
in a way a static spectrum never could.

Performance approach
--------------------
Rather than drawing individual rectangles per bin (catastrophically slow),
we maintain a numpy uint8 RGB array the same size as the plot in pixels.
Each frame we shift the entire array down one row (np.roll), write the new
FFT row as colors into row 0, then blit the whole array to a pygame Surface
in one operation via pygame.surfarray.blit_array.

This is O(width × height) numpy operations per frame — fast enough for 60fps
even at high resolutions.

Interface
---------
    panel = WaterfallPanel(surface, rect, theme, sample_rate=44100)
    panel.update(magnitude_db, freqs)
    panel.draw()

Keybindings
---------
    L   toggle log / linear frequency axis
    +   increase scroll speed (more history visible)
    -   decrease scroll speed
"""

import pygame
import numpy as np
from numpy.typing import NDArray

try:
    from display.theme import BaseTheme
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.display.theme import BaseTheme


# ─── Constants ────────────────────────────────────────────────────────────────

PAD_LEFT   = 36
PAD_RIGHT  = 8
PAD_TOP    = 24
PAD_BOTTOM = 22

FREQ_MARKERS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]


# ─── WaterfallPanel ───────────────────────────────────────────────────────────

class WaterfallPanel:
    """
    Scrolling spectrogram — time flows downward, frequency is the x axis,
    color encodes signal strength.

    The pixel buffer
    ----------------
    self._pixel_buf is a numpy array of shape (plot_h, plot_w, 3) uint8.
    It IS the image — each element is an [R, G, B] value.

    Every frame:
      1. np.roll shifts all rows down by `scroll_rows` pixels
      2. New FFT data is mapped to colors and written into the top rows
      3. pygame.surfarray.blit_array blits the whole buffer in one call

    The surface
    -----------
    self._waterfall_surf is a pygame.Surface the same size as the plot.
    We blit_array into it, then blit it onto self.surface at plot position.
    Kept as an instance variable so we don't recreate it every frame.
    """

    def __init__(
        self,
        surface     : pygame.Surface,
        rect        : pygame.Rect,
        theme       : BaseTheme,
        sample_rate : int   = 44100,
        db_floor    : float = -90.0,
        db_ceil     : float =  0.0,
        scroll_rows : int   = 1,
    ):
        self.surface     = surface
        self.rect        = rect
        self.theme       = theme
        self.sample_rate = sample_rate
        self.db_floor    = db_floor
        self.db_ceil     = db_ceil
        self.db_range    = db_ceil - db_floor
        self.scroll_rows = scroll_rows   # rows to advance per frame

        # Feature toggles
        self.log_axis = True   # L key

        # Latest FFT data
        self._spectrum : NDArray[np.float32] = np.full(512, db_floor, dtype=np.float32)
        self._freqs    : NDArray[np.float32] = np.linspace(0, sample_rate / 2, 512)

        # Pixel buffer and surface — built on first draw or resize
        self._pixel_buf       : NDArray[np.uint8] | None = None
        self._waterfall_surf  : pygame.Surface    | None = None
        self._last_plot_size  : tuple             | None = None

        # Colormap lookup table — precomputed for speed
        # Maps 0–255 intensity to an RGB color using the theme's signal_color()
        self._colormap : NDArray[np.uint8] | None = None

    # ── Public interface ──────────────────────────────────────────────────────

    def update(
        self,
        magnitude_db : NDArray[np.float32],
        freqs        : NDArray[np.float32],
    ):
        """Feed new FFT data. Call once per frame before draw()."""
        self._spectrum = magnitude_db
        self._freqs    = freqs

    def draw(self):
        """Render the waterfall to self.surface."""
        plot = self._plot_rect()
        self._ensure_buffers(plot)
        self._scroll_and_write(plot)
        self._blit_waterfall(plot)
        self._draw_freq_markers(plot)
        self._draw_time_axis(plot)
        self._draw_labels()
        self.theme.draw_panel_border(self.surface, self.rect)

    def handle_key(self, event: pygame.event.Event):
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_l:
            self.log_axis = not self.log_axis
            # Reset buffer so old history with different axis doesn't confuse
            self._pixel_buf = None
        elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
            self.scroll_rows = min(8, self.scroll_rows + 1)
        elif event.key == pygame.K_MINUS:
            self.scroll_rows = max(1, self.scroll_rows - 1)

    # ── Buffer management ─────────────────────────────────────────────────────

    def _ensure_buffers(self, plot: pygame.Rect):
        """
        Create or recreate the pixel buffer and waterfall surface if
        the plot size has changed (window resize) or first call.

        The pixel buffer is (height, width, 3) — height rows, width columns,
        3 channels (RGB). This matches how pygame.surfarray expects data
        AFTER we transpose (surfarray uses (width, height) order).
        """
        size = (plot.width, plot.height)
        if self._last_plot_size == size and self._pixel_buf is not None:
            return

        self._pixel_buf      = np.zeros((plot.height, plot.width, 3), dtype=np.uint8)
        self._waterfall_surf = pygame.Surface((plot.width, plot.height))
        self._last_plot_size = size

        # Rebuild colormap for this theme
        self._build_colormap()

        print(f"[WaterfallPanel] buffer {plot.width}×{plot.height} px")

    def _build_colormap(self):
        """
        Precompute 256 RGB colors from the theme's signal_color() function.

        signal_color(norm) is called once per frame per pixel if we call it
        naively — at 800×200 that's 160,000 calls/frame. Instead we build
        a lookup table of 256 entries once and use numpy indexing to map
        the entire row in one operation.
        """
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            r, g, b = self.theme.signal_color(i / 255.0)
            lut[i] = [r, g, b]
        self._colormap = lut

    # ── Core scroll + write ───────────────────────────────────────────────────

    def _scroll_and_write(self, plot: pygame.Rect):
        """
        The heart of the waterfall.

        Step 1 — scroll: shift all rows down by scroll_rows.
          np.roll along axis=0 moves every row downward.
          The rows that "fall off" the bottom wrap to the top —
          we immediately overwrite them with new data so it doesn't matter.

        Step 2 — write: map the new FFT snapshot to a row of pixel colors.
          We resample the spectrum to match the plot width, normalise to
          0–255, then use the colormap LUT to get RGB values.
          All of this is numpy — no Python loops.
        """
        buf = self._pixel_buf
        w   = plot.width

        # ── Step 1: Scroll ───────────────────────────────────────────────
        # Shift buffer down by scroll_rows. New rows appear at the top (row 0).
        buf[:] = np.roll(buf, self.scroll_rows, axis=0)

        # ── Step 2: Resample spectrum to plot width ───────────────────────
        # The FFT output has N bins (e.g. 512). The plot might be 800px wide.
        # We need to map bins → pixels, respecting log/linear axis setting.
        row_colors = self._spectrum_to_row(plot.width)

        # Write into the top scroll_rows rows
        for r in range(self.scroll_rows):
            buf[r, :, :] = row_colors

    def _spectrum_to_row(self, width: int) -> NDArray[np.uint8]:
        """
        Convert the current spectrum to a 1D array of RGB colors,
        one per pixel column of the waterfall.

        Returns shape (width, 3) uint8.

        This is the critical performance path — must be fast.
        We use numpy indexing throughout, no Python loops.
        """
        spectrum = self._spectrum
        freqs    = self._freqs
        nyquist  = self.sample_rate / 2.0
        n_bins   = len(spectrum)

        # Build an array of frequency values for each pixel column
        pixel_indices = np.arange(width)

        if self.log_axis:
            # Log frequency mapping: pixel → Hz
            min_log   = np.log10(max(1.0, float(freqs[1])))
            max_log   = np.log10(nyquist)
            log_freqs = min_log + (pixel_indices / width) * (max_log - min_log)
            hz_per_px = 10.0 ** log_freqs
        else:
            # Linear frequency mapping: pixel → Hz
            hz_per_px = (pixel_indices / width) * nyquist

        # Map Hz values to bin indices
        # bin = hz * n_bins / nyquist, clamped to valid range
        bin_indices = np.clip(
            (hz_per_px * n_bins / nyquist).astype(np.int32),
            0, n_bins - 1
        )

        # Gather spectrum values for each pixel column
        pixel_db = spectrum[bin_indices]

        # Normalise dB values to 0–255 for colormap lookup
        norm      = (pixel_db - self.db_floor) / self.db_range
        norm      = np.clip(norm, 0.0, 1.0)
        lut_idx   = (norm * 255).astype(np.uint8)

        # Colormap lookup — maps each index to [R, G, B]
        # self._colormap shape: (256, 3)
        # lut_idx shape: (width,)
        # result shape: (width, 3)
        return self._colormap[lut_idx]

    # ── Blit to screen ────────────────────────────────────────────────────────

    def _blit_waterfall(self, plot: pygame.Rect):
        """
        Blit the pixel buffer to the waterfall surface, then to screen.

        pygame.surfarray.blit_array expects (width, height) order,
        but our buffer is (height, width, 3). We transpose axes 0 and 1.
        The transpose is a view (no copy) so it's essentially free.
        """
        # Transpose from (height, width, 3) → (width, height, 3)
        transposed = np.transpose(self._pixel_buf, (1, 0, 2))

        pygame.surfarray.blit_array(self._waterfall_surf, transposed)

        # Blit waterfall surface onto main surface at plot position
        self.surface.blit(self._waterfall_surf, (plot.x, plot.y))

    # ── Axis overlays ─────────────────────────────────────────────────────────

    def _draw_freq_markers(self, plot: pygame.Rect):
        """
        Draw vertical frequency reference lines on top of the waterfall.
        Drawn with a dim line so they're visible but don't obscure the signal.
        """
        nyquist      = self.sample_rate / 2.0
        last_label_x = -999
        min_spacing  = 38

        for freq in FREQ_MARKERS:
            if freq >= nyquist:
                continue

            x = self._freq_to_x(freq, plot)
            if x < plot.x or x > plot.right:
                continue

            # Dim vertical line — semi-transparent so waterfall shows through
            line_surf = pygame.Surface((1, plot.height), pygame.SRCALPHA)
            line_surf.fill((*self.theme.GRID_CENTER, 90))
            self.surface.blit(line_surf, (x, plot.y))

            if x - last_label_x < min_spacing:
                continue
            last_label_x = x

            label = f"{freq // 1000}k" if freq >= 1000 else str(freq)
            self.theme.draw_text(
                self.surface, label,
                (x, plot.bottom + 4),
                font="digit_small",
                color=self.theme.TEXT_DIM,
                anchor="midtop",
            )

    def _draw_time_axis(self, plot: pygame.Rect):
        """
        Draw horizontal time reference lines every N rows.
        Gives a sense of the time scale — how many seconds of history are shown.
        """
        # Every 50 pixels of height = a time marker
        step = 50
        for y_offset in range(step, plot.height, step):
            y = plot.y + y_offset
            line_surf = pygame.Surface((plot.width, 1), pygame.SRCALPHA)
            line_surf.fill((*self.theme.GRID, 60))
            self.surface.blit(line_surf, (plot.x, y))

    # ── Labels ────────────────────────────────────────────────────────────────

    def _draw_labels(self):
        """Draw panel title and axis info."""
        axis_label   = "LOG" if self.log_axis else "LIN"
        speed_label  = f"SPD:{self.scroll_rows}"
        self.theme.draw_panel_label(
            self.surface, self.rect,
            "WATERFALL",
            f"/ SPECTROGRAM  [{axis_label}]  [{speed_label}]  [L=axis  ±=speed]"
        )

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _plot_rect(self) -> pygame.Rect:
        return pygame.Rect(
            self.rect.x      + PAD_LEFT,
            self.rect.y      + PAD_TOP,
            self.rect.width  - PAD_LEFT - PAD_RIGHT,
            self.rect.height - PAD_TOP  - PAD_BOTTOM,
        )

    def _freq_to_x(self, freq_hz: float, plot: pygame.Rect) -> int:
        """Map Hz to pixel x within plot — same logic as spectrum panel."""
        nyquist = self.sample_rate / 2.0
        freq_hz = max(1.0, freq_hz)

        if self.log_axis:
            min_log   = np.log10(max(1.0, float(self._freqs[1])))
            max_log   = np.log10(nyquist)
            log_range = max_log - min_log
            if log_range <= 0:
                return plot.x
            norm = (np.log10(freq_hz) - min_log) / log_range
        else:
            norm = freq_hz / nyquist

        return plot.x + int(np.clip(norm, 0.0, 1.0) * plot.width)


# ─── Smoke test ───────────────────────────────────────────────────────────────
# python -m src.display.waterfall

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.display.theme import get_theme, THEMES

    pygame.init()
    W, H   = 1100, 380
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("WaterfallPanel  —  L=log/lin  ±=speed  N=theme  ESC=quit")
    clock  = pygame.time.Clock()

    SR      = 44100
    N_BINS  = 512
    freqs   = np.linspace(0, SR / 2, N_BINS, dtype=np.float32)

    theme_idx = 0
    theme     = get_theme(theme_idx)
    panel     = WaterfallPanel(screen, pygame.Rect(10, 10, W - 20, H - 20), theme, SR)

    frame   = 0
    running = True

    print("Smoke test running.")
    print("  L   → toggle log/linear axis")
    print("  +/- → scroll speed")
    print("  N   → cycle themes")
    print("  ESC → quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_n:
                    theme_idx   = (theme_idx + 1) % len(THEMES)
                    theme       = get_theme(theme_idx)
                    panel.theme = theme
                    panel._build_colormap()   # rebuild LUT for new theme colors
                    print(f"Theme → {theme.NAME}")
                else:
                    panel.handle_key(event)

        # ── Animated fake spectrum ────────────────────────────────────────
        # Simulate: drifting carrier, sweeping signal, noise floor
        t        = frame * 0.016
        spectrum = np.full(N_BINS, -80.0, dtype=np.float32)

        # Drifting carrier — like a satellite doppler shift
        carrier_hz = 1200 + 800 * np.sin(t * 0.15)
        carrier_bin = int(carrier_hz / (SR / 2) * N_BINS)
        for i in range(N_BINS):
            dist = abs(i - carrier_bin)
            spectrum[i] = max(spectrum[i], -10.0 - dist * 0.8)

        # Periodic burst — like a pager or IoT device transmitting
        if int(t * 2) % 5 == 0:
            burst_hz  = 3000 + 500 * np.sin(t)
            burst_bin = int(burst_hz / (SR / 2) * N_BINS)
            for i in range(N_BINS):
                dist = abs(i - burst_bin)
                spectrum[i] = max(spectrum[i], -15.0 - dist * 1.2)

        # Harmonic series — like a voice or instrument
        fund_hz  = 300 + 100 * np.sin(t * 0.4)
        for harmonic in range(1, 6):
            h_hz  = fund_hz * harmonic
            h_bin = int(h_hz / (SR / 2) * N_BINS)
            amp   = -20.0 - harmonic * 8.0
            for i in range(N_BINS):
                dist = abs(i - h_bin)
                spectrum[i] = max(spectrum[i], amp - dist * 2.0)

        # Noise floor variation
        spectrum += np.random.normal(0, 2.0, N_BINS).astype(np.float32)
        np.clip(spectrum, -90, 0, out=spectrum)

        # ── Draw ─────────────────────────────────────────────────────────
        theme.begin_frame(screen)
        pygame.draw.rect(screen, theme.BG, screen.get_rect())

        panel.update(spectrum, freqs)
        panel.draw()

        theme.end_frame(screen)
        pygame.display.flip()
        clock.tick(60)
        frame += 1

    pygame.quit()