"""
src/display/spectrum.py

Spectrum analyzer panel.
Draws a live FFT magnitude plot with phosphor glow, gradient fill,
peak hold, and togglable log/linear frequency axis.

Receives data from compute_fft() and renders inside a pygame.Rect.
Uses the shared Theme for all colors and glow effects.

Interface
---------
    panel = SpectrumPanel(surface, rect, theme, sample_rate=44100)
    panel.update(magnitude_db, freqs)   # call with new FFT data each frame
    panel.draw()                         # render to surface

Keybindings (call panel.handle_key(event) from main loop)
---------
    L   toggle log / linear frequency axis
    P   toggle peak hold line
"""

import pygame
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

try:
    from display.theme import BaseTheme
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.display.theme import BaseTheme


# ─── Constants ────────────────────────────────────────────────────────────────

# Frequencies to mark on the x axis — acoustically meaningful set
FREQ_MARKERS = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# Peak hold timing
PEAK_HOLD_FRAMES = 90    # frames to hold before decay starts (~1.5s at 60fps)
PEAK_DECAY_RATE  = 0.4   # dB to subtract per frame once decay begins

# dB axis reference lines
DB_MARKERS = [-80, -60, -40, -20, 0]

# Padding inside the rect for the plot area
PAD_LEFT   = 36   # room for dB labels
PAD_RIGHT  = 8
PAD_TOP    = 28   # room for panel label
PAD_BOTTOM = 22   # room for freq labels


# ─── SpectrumPanel ────────────────────────────────────────────────────────────

class SpectrumPanel:
    """
    Renders a live FFT magnitude spectrum with phosphor aesthetics.

    Visual layers (drawn bottom to top)
    ------------------------------------
    1. Grid (graticule lines + dB markers + freq markers)
    2. Gradient fill under spectrum line
    3. Spectrum glow polyline
    4. Peak hold glow line
    5. Labels (panel title, dB scale, freq axis)
    """

    def __init__(
        self,
        surface     : pygame.Surface,
        rect        : pygame.Rect,
        theme       : BaseTheme,
        sample_rate : int   = 44100,
        db_floor    : float = -90.0,
        db_ceil     : float =  0.0,
    ):
        self.surface     = surface
        self.rect        = rect
        self.theme       = theme
        self.sample_rate = sample_rate
        self.db_floor    = db_floor
        self.db_ceil     = db_ceil
        self.db_range    = db_ceil - db_floor

        # Feature toggles
        self.log_axis    = True    # L key
        self.show_peak   = True    # P key

        # Latest FFT data (initialised to silence)
        self._spectrum   : NDArray[np.float32] = np.full(512, db_floor, dtype=np.float32)
        self._freqs      : NDArray[np.float32] = np.linspace(0, sample_rate / 2, 512, dtype=np.float32)

        # Peak hold state
        self._peak_hold  : NDArray[np.float32] = np.full(512, db_floor, dtype=np.float32)
        self._peak_decay : NDArray[np.int32]   = np.zeros(512, dtype=np.int32)

        # Pre-rendered gradient fill surface (built on first draw or resize)
        self._gradient_surf : pygame.Surface | None = None
        self._last_plot_rect: pygame.Rect | None    = None

        # Cached screen-space points (rebuilt each frame)
        self._signal_points : list = []
        self._peak_points   : list = []

    # ── Public interface ──────────────────────────────────────────────────────

    def update(
        self,
        magnitude_db : NDArray[np.float32],
        freqs        : NDArray[np.float32],
    ):
        """
        Feed new FFT data into the panel.
        Call once per frame before draw().

        Also updates peak hold state here — separating data update from
        rendering keeps draw() purely about pixels.
        """
        self._spectrum = magnitude_db
        self._freqs    = freqs
        self._update_peak_hold(magnitude_db)

    def draw(self):
        """Render the full panel to self.surface inside self.rect."""
        plot = self._plot_rect()
        self._ensure_gradient(plot)

        # ── Grid ────────────────────────────────────────────────────────────
        self.theme.draw_grid(self.surface, plot, cols=10, rows=8)
        self._draw_db_markers(plot)
        self._draw_freq_markers(plot)

        # ── Build screen-space point lists ──────────────────────────────────
        self._build_points(plot)

        # ── Gradient fill ───────────────────────────────────────────────────
        self._draw_gradient_fill(plot)

        # ── Signal line ─────────────────────────────────────────────────────
        if len(self._signal_points) > 1:
            # Average signal level drives glow intensity
            avg_norm = float(np.mean(
                (self._spectrum - self.db_floor) / self.db_range
            ))
            intensity = 0.4 + avg_norm * 0.6   # map to 0.4–1.0 range
            self.theme.draw_glow_polyline(
                self.surface, self._signal_points, intensity=intensity
            )

        # ── Peak hold line ───────────────────────────────────────────────────
        if self.show_peak and len(self._peak_points) > 1:
            # Draw peak hold as a dim amber line — same glow system, low intensity
            old_core = self.theme.PHOSPHOR_CORE
            old_mid  = self.theme.PHOSPHOR_MID
            old_out  = self.theme.PHOSPHOR_OUTER
            # Temporarily redirect glow colors to peak hold color
            self.theme.PHOSPHOR_CORE  = self.theme.PEAK_HOLD
            self.theme.PHOSPHOR_MID   = tuple(int(c * 0.6) for c in self.theme.PEAK_HOLD)
            self.theme.PHOSPHOR_OUTER = tuple(int(c * 0.2) for c in self.theme.PEAK_HOLD)
            self.theme.draw_glow_polyline(
                self.surface, self._peak_points, intensity=0.35
            )
            # Restore colors
            self.theme.PHOSPHOR_CORE  = old_core
            self.theme.PHOSPHOR_MID   = old_mid
            self.theme.PHOSPHOR_OUTER = old_out

        # ── Labels & border ──────────────────────────────────────────────────
        self._draw_labels(plot)
        self.theme.draw_panel_border(self.surface, self.rect)

    def handle_key(self, event: pygame.event.Event):
        """Process keydown events. Call from your main event loop."""
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_l:
            self.log_axis = not self.log_axis
        elif event.key == pygame.K_p:
            self.show_peak = not self.show_peak

    # ── Peak hold ─────────────────────────────────────────────────────────────

    def _update_peak_hold(self, spectrum: NDArray[np.float32]):
        """
        Update peak hold arrays.

        For each bin:
          - If new value exceeds stored peak → update peak, reset decay counter
          - Otherwise → increment decay counter
          - Once decay counter exceeds PEAK_HOLD_FRAMES → subtract PEAK_DECAY_RATE/frame
        """
        # Resize arrays if FFT size changed
        if len(self._peak_hold) != len(spectrum):
            self._peak_hold  = np.full(len(spectrum), self.db_floor, dtype=np.float32)
            self._peak_decay = np.zeros(len(spectrum), dtype=np.int32)

        # Where new value exceeds peak — update and reset decay
        new_peak_mask = spectrum > self._peak_hold
        self._peak_hold[new_peak_mask]  = spectrum[new_peak_mask]
        self._peak_decay[new_peak_mask] = 0

        # Where new value does NOT exceed peak — increment decay
        self._peak_decay[~new_peak_mask] += 1

        # Where decay has exceeded hold time — start falling
        decaying = self._peak_decay > PEAK_HOLD_FRAMES
        self._peak_hold[decaying] -= PEAK_DECAY_RATE

        # Clamp to floor
        np.clip(self._peak_hold, self.db_floor, self.db_ceil, out=self._peak_hold)

    # ── Coordinate mapping ────────────────────────────────────────────────────

    def _plot_rect(self) -> pygame.Rect:
        """
        The inner rect where the signal is actually drawn —
        inset from self.rect to leave room for axis labels.
        """
        return pygame.Rect(
            self.rect.x      + PAD_LEFT,
            self.rect.y      + PAD_TOP,
            self.rect.width  - PAD_LEFT - PAD_RIGHT,
            self.rect.height - PAD_TOP  - PAD_BOTTOM,
        )

    def _freq_to_x(self, freq_hz: float, plot: pygame.Rect) -> int:
        """
        Map a frequency in Hz to a pixel x position within plot.

        Linear mode: x scales linearly with frequency.
        Log mode: x scales with log10(freq), so octaves are equal width.
        """
        nyquist  = self.sample_rate / 2.0
        freq_hz  = max(1.0, freq_hz)   # avoid log(0)

        if self.log_axis:
            min_log = np.log10(max(1.0, self._freqs[1]))  # skip 0 Hz bin
            max_log = np.log10(nyquist)
            log_range = max_log - min_log
            if log_range <= 0:
                return plot.x
            norm = (np.log10(freq_hz) - min_log) / log_range
        else:
            norm = freq_hz / nyquist

        return plot.x + int(np.clip(norm, 0.0, 1.0) * plot.width)

    def _db_to_y(self, db: float, plot: pygame.Rect) -> int:
        """
        Map a dB value to a pixel y position within plot.

        db_floor → bottom of plot (y = plot.bottom)
        db_ceil  → top of plot    (y = plot.top)
        Pygame y increases downward, so we flip.
        """
        norm = (db - self.db_floor) / self.db_range
        norm = np.clip(norm, 0.0, 1.0)
        return plot.bottom - int(norm * plot.height)

    # ── Point building ────────────────────────────────────────────────────────

    def _build_points(self, plot: pygame.Rect):
        """
        Convert the spectrum and peak_hold arrays into lists of (x, y) tuples
        in screen space. Called every frame before drawing.

        We subsample or skip bins that map to the same x pixel to avoid
        drawing thousands of overlapping points (wasteful and looks the same).
        """
        spectrum   = self._spectrum
        peak_hold  = self._peak_hold
        freqs      = self._freqs
        n          = len(spectrum)

        signal_pts = []
        peak_pts   = []
        last_x     = -1

        for i in range(n):
            freq = float(freqs[i])
            if freq <= 0:
                continue

            x = self._freq_to_x(freq, plot)

            # Skip if this bin maps to the same x pixel as the last one
            # (log axis compresses many bins into a few pixels at high freq)
            if x == last_x:
                continue
            last_x = x

            if x < plot.x or x > plot.right:
                continue

            y_sig  = self._db_to_y(float(spectrum[i]),  plot)
            y_peak = self._db_to_y(float(peak_hold[i]), plot)

            signal_pts.append((x, y_sig))
            peak_pts.append((x, y_peak))

        self._signal_points = signal_pts
        self._peak_points   = peak_pts

    # ── Gradient fill ─────────────────────────────────────────────────────────

    def _ensure_gradient(self, plot: pygame.Rect):
        """
        Build the vertical gradient Surface if it doesn't exist or plot changed.
        The gradient goes from semi-opaque phosphor color at top to transparent
        at bottom — blitted over the filled polygon to give it depth.
        """
        if self._gradient_surf is not None and self._last_plot_rect == plot:
            return

        w, h = plot.width, plot.height
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))

        # Draw horizontal strips from top (bright) to bottom (transparent)
        r, g, b = self.theme.PHOSPHOR_MID
        for y in range(h):
            alpha = int(60 * (1.0 - y / h) ** 1.8)   # power curve — fast fade
            pygame.draw.line(surf, (r, g, b, alpha), (0, y), (w, y))

        self._gradient_surf    = surf
        self._last_plot_rect   = pygame.Rect(plot)

    def _draw_gradient_fill(self, plot: pygame.Rect):
        """
        Fill the area under the spectrum line with a gradient.

        1. Build polygon: signal points + bottom-right corner + bottom-left corner
        2. Fill with dim phosphor color
        3. Blit gradient surface over it (BLEND_RGBA_ADD for additive glow feel)
        """
        if len(self._signal_points) < 2:
            return

        # Close the polygon along the bottom edge
        polygon = (
            self._signal_points
            + [(self._signal_points[-1][0], plot.bottom)]
            + [(self._signal_points[0][0],  plot.bottom)]
        )

        # Create a temporary surface for the fill (so we can control alpha)
        fill_surf = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        fill_surf.fill((0, 0, 0, 0))

        # Offset polygon points to fill_surf local coordinates
        ox, oy = self.rect.x, self.rect.y
        local_poly = [(px - ox, py - oy) for px, py in polygon]

        r, g, b = self.theme.PHOSPHOR_OUTER
        pygame.draw.polygon(fill_surf, (r, g, b, 55), local_poly)

        self.surface.blit(fill_surf, (ox, oy))

        # Gradient overlay — adds the bright-top fade-to-dark effect
        self.surface.blit(
            self._gradient_surf,
            (plot.x, plot.y),
            special_flags=pygame.BLEND_RGBA_ADD,
        )

    # ── Axis labels & markers ─────────────────────────────────────────────────

    def _draw_db_markers(self, plot: pygame.Rect):
        """
        Draw horizontal dB reference lines and labels on the left edge.
        """
        for db in DB_MARKERS:
            y = self._db_to_y(db, plot)
            if y < plot.top or y > plot.bottom:
                continue

            # Faint horizontal reference line
            pygame.draw.line(
                self.surface, self.theme.GRID,
                (plot.x, y), (plot.right, y), 1
            )

            # dB label — left of plot area, digit font
            label = f"{db:+d}"
            self.theme.draw_text(
                self.surface, label,
                (plot.x - 4, y),
                font="digit_small",
                color=self.theme.TEXT_DIM,
                anchor="midright",
            )

    def _draw_freq_markers(self, plot: pygame.Rect):
        """
        Draw vertical frequency reference lines and Hz labels on the bottom.
        Skips labels that would overlap.
        """
        nyquist    = self.sample_rate / 2.0
        last_label_x = -999
        min_spacing  = 36   # minimum pixels between labels

        for freq in FREQ_MARKERS:
            if freq >= nyquist:
                continue

            x = self._freq_to_x(freq, plot)
            if x < plot.x or x > plot.right:
                continue

            # Faint vertical reference line
            pygame.draw.line(
                self.surface, self.theme.GRID,
                (x, plot.top), (x, plot.bottom), 1
            )

            # Skip label if too close to previous one
            if x - last_label_x < min_spacing:
                continue
            last_label_x = x

            # Format: "500" below 1000, "2k" at 2000+
            label = f"{freq // 1000}k" if freq >= 1000 else str(freq)
            self.theme.draw_text(
                self.surface, label,
                (x, plot.bottom + 4),
                font="digit_small",
                color=self.theme.TEXT_DIM,
                anchor="midtop",
            )

    def _draw_labels(self, plot: pygame.Rect):
        """Draw panel title, axis mode indicator, and peak hold toggle state."""

        # Panel title
        axis_label = "LOG" if self.log_axis else "LIN"
        self.theme.draw_panel_label(
            self.surface, self.rect,
            "SPECTRUM",
            f"/ FFT  [{axis_label}]  [L=axis  P=peak]"
        )

        # Peak frequency readout — top right of panel
        if len(self._spectrum) > 0:
            peak_bin  = int(np.argmax(self._spectrum))
            peak_freq = float(self._freqs[peak_bin]) if peak_bin < len(self._freqs) else 0.0
            peak_db   = float(self._spectrum[peak_bin])

            if peak_freq >= 1000:
                freq_str = f"{peak_freq/1000:.2f}kHz"
            else:
                freq_str = f"{peak_freq:.0f}Hz"

            self.theme.draw_text(
                self.surface,
                f"{freq_str}  {peak_db:.1f}dB",
                (self.rect.right - 8, self.rect.y + 6),
                font="digit_main",
                color=self.theme.TEXT,
                anchor="topright",
                glow=True,
            )

        # Peak hold indicator
        if self.show_peak:
            self.theme.draw_text(
                self.surface, "PEAK",
                (plot.x + 4, self.rect.bottom - 18),
                font="digit_small",
                color=self.theme.PEAK_HOLD,
            )


# ─── Smoke test ───────────────────────────────────────────────────────────────
# python src/display/spectrum.py

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.display.theme import PhosphorGreenTheme, AmberTheme, THEMES, get_theme

    pygame.init()
    W, H   = 1100, 420
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("SpectrumPanel smoke test  —  L=log/lin  P=peak  T=theme")
    clock  = pygame.time.Clock()

    SR       = 44100
    N_BINS   = 512
    freqs    = np.linspace(0, SR / 2, N_BINS, dtype=np.float32)

    theme_idx = 0
    theme     = get_theme(theme_idx)
    panel     = SpectrumPanel(screen, pygame.Rect(10, 10, W - 20, H - 20), theme, SR)

    frame   = 0
    running = True

    print("Smoke test running.")
    print("  L → toggle log/linear axis")
    print("  P → toggle peak hold")
    print("  T → cycle themes")
    print("  ESC → quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_t:
                    theme_idx = (theme_idx + 1) % len(THEMES)
                    theme     = get_theme(theme_idx)
                    panel.theme = theme
                    print(f"Theme → {theme.NAME}")
                else:
                    panel.handle_key(event)

        # ── Synthesise a fake spectrum that animates ──────────────────────
        # Several Gaussian peaks drifting around — mimics real signal content
        t        = frame * 0.02
        spectrum = np.full(N_BINS, -82.0, dtype=np.float32)

        peaks = [
            (200,   18, 30 + 18 * np.sin(t * 0.7)),    # voice fundamental
            (440,   12, 22 + 12 * np.sin(t * 1.1)),    # A4
            (880,   10, 18 + 10 * np.sin(t * 0.9)),    # A5
            (1760,  8,  14 + 8  * np.sin(t * 1.3)),    # A6
            (3500,  6,  10 + 6  * np.sin(t * 0.6)),    # upper harmonic
            (8000,  4,  -5 + 4  * np.sin(t * 1.7)),    # high freq noise
        ]

        for center_hz, width_hz, amplitude_db in peaks:
            center_bin = int(center_hz / (SR / 2) * N_BINS)
            width_bins = max(1, int(width_hz / (SR / 2) * N_BINS))
            for i in range(N_BINS):
                dist = (i - center_bin) ** 2 / (2 * width_bins ** 2)
                spectrum[i] = max(spectrum[i], float(amplitude_db) - 80 * dist)

        # Add noise floor variation
        spectrum += np.random.normal(0, 1.5, N_BINS).astype(np.float32)
        np.clip(spectrum, -90, 0, out=spectrum)

        # ── Update + draw ─────────────────────────────────────────────────
        theme.begin_frame(screen)
        pygame.draw.rect(screen, theme.BG_PANEL, pygame.Rect(10, 10, W - 20, H - 20))

        panel.update(spectrum, freqs)
        panel.draw()

        theme.end_frame(screen)
        pygame.display.flip()
        clock.tick(60)
        frame += 1

    pygame.quit()