"""
src/display/waveform.py

Waveform (time-domain) display panel.
Draws raw audio samples as a phosphor oscilloscope trace with:
  - Rising-edge trigger for stable display
  - Clipping indicators (flash when signal hits ±1.0)
  - RMS level bar on the right edge
  - Center zero line and amplitude grid

Interface
---------
    panel = WaveformPanel(surface, rect, theme)
    panel.update(samples)   # float32 array from MicInput.read()
    panel.draw()

Keybindings (call panel.handle_key(event))
---------
    T   toggle trigger on/off
    G   toggle amplitude grid lines
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

PAD_LEFT         = 10
PAD_RIGHT        = 28    # room for RMS bar on the right
PAD_TOP          = 24    # room for panel label
PAD_BOTTOM       = 8

CLIP_FLASH_FRAMES = 12   # how many frames the clipping indicator stays lit
AMPLITUDE_SCALE   = 0.82 # fraction of half-height used — leaves a small margin

# Amplitude reference lines drawn across the panel
AMPLITUDE_MARKERS = [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]

RMS_BAR_WIDTH     = 10   # pixels wide for the RMS bar


# ─── WaveformPanel ────────────────────────────────────────────────────────────

class WaveformPanel:
    """
    Renders a triggered oscilloscope waveform trace.

    Visual layers (drawn bottom to top)
    ------------------------------------
    1. Amplitude grid lines
    2. Center zero line (brighter than grid)
    3. Waveform glow polyline
    4. Clipping flash (top/bottom edge, WARNING color)
    5. RMS bar (right edge)
    6. Labels
    """

    def __init__(
        self,
        surface : pygame.Surface,
        rect    : pygame.Rect,
        theme   : BaseTheme,
    ):
        self.surface = surface
        self.rect    = rect
        self.theme   = theme

        # Feature toggles
        self.use_trigger  = True   # T key
        self.show_grid    = True   # G key

        # Latest sample buffer
        self._samples : NDArray[np.float32] = np.zeros(1024, dtype=np.float32)

        # Clipping state — counts down each frame
        self._clip_top_frames    = 0
        self._clip_bottom_frames = 0

        # RMS history for smoothing (avoid jumpy bar)
        self._rms_smooth = 0.0

    # ── Public interface ──────────────────────────────────────────────────────

    def update(self, samples: NDArray[np.float32]):
        """
        Feed a new chunk of audio samples into the panel.
        Call once per frame before draw().

        samples : float32 array, range [-1.0, +1.0], any length
        """
        self._samples = samples

        # Check for clipping — any sample at or beyond the ADC ceiling
        if np.any(samples >= 0.999):
            self._clip_top_frames = CLIP_FLASH_FRAMES
        if np.any(samples <= -0.999):
            self._clip_bottom_frames = CLIP_FLASH_FRAMES

        # Count down clip flash timers
        self._clip_top_frames    = max(0, self._clip_top_frames    - 1)
        self._clip_bottom_frames = max(0, self._clip_bottom_frames - 1)

    def draw(self):
        """Render the full panel to self.surface inside self.rect."""
        plot = self._plot_rect()

        # ── Grid ────────────────────────────────────────────────────────────
        if self.show_grid:
            self._draw_amplitude_grid(plot)

        # ── Zero center line ────────────────────────────────────────────────
        center_y = plot.y + plot.height // 2
        pygame.draw.line(
            self.surface, self.theme.GRID_CENTER,
            (plot.x, center_y), (plot.right, center_y), 1
        )

        # ── Waveform trace ───────────────────────────────────────────────────
        points = self._build_points(plot)
        if len(points) > 1:
            # Intensity driven by RMS — quiet signal glows dimly,
            # loud signal glows brighter
            rms       = float(np.sqrt(np.mean(self._samples ** 2)))
            intensity = 0.3 + min(rms * 3.0, 0.7)
            self.theme.draw_glow_polyline(self.surface, points, intensity=intensity)

        # ── Clipping indicators ──────────────────────────────────────────────
        self._draw_clip_indicators(plot)

        # ── RMS bar ──────────────────────────────────────────────────────────
        self._draw_rms_bar(plot)

        # ── Labels & border ──────────────────────────────────────────────────
        self._draw_labels(plot)
        self.theme.draw_panel_border(self.surface, self.rect)

    def handle_key(self, event: pygame.event.Event):
        """Process keydown events. Call from your main event loop."""
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_t:
            self.use_trigger = not self.use_trigger
        elif event.key == pygame.K_g:
            self.show_grid = not self.show_grid

    # ── Trigger ───────────────────────────────────────────────────────────────

    def _find_trigger(self, samples: NDArray[np.float32]) -> int:
        """
        Find the first rising zero crossing in the sample buffer.

        A rising zero crossing is where the signal goes from negative
        to positive — sample[i-1] < 0 and sample[i] >= 0.

        This synchronises successive frames to the same phase of the
        waveform, making periodic signals appear stable rather than
        scrolling. Without this a 440 Hz sine looks like random noise.

        Returns the trigger index, or 0 if no crossing is found.
        """
        if not self.use_trigger or len(samples) < 2:
            return 0

        # Search only the first half — leave room to draw a full window
        search_end = len(samples) // 2

        for i in range(1, search_end):
            if samples[i - 1] < 0.0 and samples[i] >= 0.0:
                return i

        # No crossing found — fall back to start
        return 0

    # ── Point building ────────────────────────────────────────────────────────

    def _build_points(self, plot: pygame.Rect) -> list:
        """
        Convert the sample buffer to screen-space (x, y) points.

        Starts from the trigger index.
        Maps sample index linearly across plot width.
        Maps amplitude to y with center=0, flipped for Pygame coords.
        """
        samples    = self._samples
        trig_idx   = self._find_trigger(samples)

        # How many samples fit in the display window
        n_display  = min(len(samples) - trig_idx, len(samples))
        if n_display < 2:
            return []

        display_samples = samples[trig_idx : trig_idx + n_display]
        n               = len(display_samples)

        half_h  = plot.height / 2
        center_y = plot.y + half_h
        x_scale  = plot.width / n

        points = []
        for i, sample in enumerate(display_samples):
            x = int(plot.x + i * x_scale)
            # Flip: positive amplitude → upward → smaller y in Pygame
            y = int(center_y - sample * half_h * AMPLITUDE_SCALE)
            # Clamp to plot bounds — prevents drawing outside the panel
            y = max(plot.top, min(plot.bottom, y))
            points.append((x, y))

        return points

    # ── Amplitude grid ────────────────────────────────────────────────────────

    def _draw_amplitude_grid(self, plot: pygame.Rect):
        """
        Draw faint horizontal lines at fixed amplitude reference levels.
        The 0.0 line is handled separately (brighter, drawn always).
        """
        half_h   = plot.height / 2
        center_y = plot.y + half_h

        for amp in AMPLITUDE_MARKERS:
            if amp == 0.0:
                continue   # drawn separately as center line
            y = int(center_y - amp * half_h * AMPLITUDE_SCALE)
            if y < plot.top or y > plot.bottom:
                continue

            pygame.draw.line(
                self.surface, self.theme.GRID,
                (plot.x, y), (plot.right, y), 1
            )

            # Amplitude label on left edge
            label = f"{amp:+.2f}"
            self.theme.draw_text(
                self.surface, label,
                (plot.x + 2, y - 1),
                font="digit_small",
                color=self.theme.TEXT_DIM,
                anchor="bottomleft",
            )

    # ── Clipping indicators ───────────────────────────────────────────────────

    def _draw_clip_indicators(self, plot: pygame.Rect):
        """
        Flash a bright WARNING-colored bar at the top and/or bottom edge
        when the signal is clipping (hitting the ADC ±1.0 ceiling).

        Clipping means your mic gain is too high — the signal is being
        truncated and information is lost. This makes it immediately visible.
        """
        bar_height = 4

        if self._clip_top_frames > 0:
            # Fade alpha based on remaining frames
            alpha = int(255 * self._clip_top_frames / CLIP_FLASH_FRAMES)
            flash = pygame.Surface((plot.width, bar_height), pygame.SRCALPHA)
            r, g, b = self.theme.WARNING
            flash.fill((r, g, b, alpha))
            self.surface.blit(flash, (plot.x, plot.top))

        if self._clip_bottom_frames > 0:
            alpha = int(255 * self._clip_bottom_frames / CLIP_FLASH_FRAMES)
            flash = pygame.Surface((plot.width, bar_height), pygame.SRCALPHA)
            r, g, b = self.theme.WARNING
            flash.fill((r, g, b, alpha))
            self.surface.blit(flash, (plot.x, plot.bottom - bar_height))

    # ── RMS bar ───────────────────────────────────────────────────────────────

    def _draw_rms_bar(self, plot: pygame.Rect):
        """
        Draw a vertical level meter on the right edge of the plot.

        RMS (root mean square) is the standard measure of signal "loudness"
        for audio. It's the square root of the mean of squared samples —
        a physically meaningful measure of signal power.

        We smooth it with a simple exponential moving average to avoid
        a jittery bar. Attack is fast (loud sounds register immediately),
        release is slow (bar falls gradually).
        """
        rms = float(np.sqrt(np.mean(self._samples ** 2)))

        # Exponential moving average — fast attack, slow release
        attack  = 0.6
        release = 0.08
        if rms > self._rms_smooth:
            self._rms_smooth += (rms - self._rms_smooth) * attack
        else:
            self._rms_smooth += (rms - self._rms_smooth) * release

        rms_norm  = min(1.0, self._rms_smooth * 3.0)   # scale up — RMS is usually < 0.3
        bar_h     = int(rms_norm * plot.height)
        bar_x     = plot.right + 4
        bar_y     = plot.bottom - bar_h

        if bar_h <= 0:
            return

        # Background track
        pygame.draw.rect(
            self.surface, self.theme.GRID,
            (bar_x, plot.top, RMS_BAR_WIDTH, plot.height), 0
        )

        # Filled level — color shifts from phosphor mid → warning at high levels
        color = self.theme.signal_color(rms_norm)
        if rms_norm > 0.85:
            color = self.theme.WARNING

        pygame.draw.rect(
            self.surface, color,
            (bar_x, bar_y, RMS_BAR_WIDTH, bar_h), 0
        )

        # Glow on the bar top edge
        pygame.draw.line(
            self.surface, self.theme.PHOSPHOR_CORE,
            (bar_x, bar_y), (bar_x + RMS_BAR_WIDTH, bar_y), 2
        )

        # RMS label below bar
        self.theme.draw_text(
            self.surface, "RMS",
            (bar_x + RMS_BAR_WIDTH // 2, plot.bottom + 2),
            font="digit_small",
            color=self.theme.TEXT_DIM,
            anchor="midtop",
        )

    # ── Labels ────────────────────────────────────────────────────────────────

    def _draw_labels(self, plot: pygame.Rect):
        """Draw panel title, trigger state, and live readouts."""

        trig_label = "TRIG:ON" if self.use_trigger else "TRIG:OFF"
        self.theme.draw_panel_label(
            self.surface, self.rect,
            "WAVEFORM",
            f"/ TIME DOMAIN  [{trig_label}]  [T=trig  G=grid]"
        )

        # Live sample count and peak amplitude readout
        peak_amp = float(np.max(np.abs(self._samples)))
        rms      = float(np.sqrt(np.mean(self._samples ** 2)))

        self.theme.draw_text(
            self.surface,
            f"PEAK {peak_amp:.3f}   RMS {rms:.3f}",
            (self.rect.right - PAD_RIGHT - 4, self.rect.y + 6),
            font="digit_main",
            color=self.theme.TEXT,
            anchor="topright",
            glow=True,
        )

    # ── Coordinate helpers ────────────────────────────────────────────────────

    def _plot_rect(self) -> pygame.Rect:
        """Inner rect inset from self.rect to leave room for labels."""
        return pygame.Rect(
            self.rect.x      + PAD_LEFT,
            self.rect.y      + PAD_TOP,
            self.rect.width  - PAD_LEFT - PAD_RIGHT,
            self.rect.height - PAD_TOP  - PAD_BOTTOM,
        )


# ─── Smoke test ───────────────────────────────────────────────────────────────
# python -m src.display.waveform

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.display.theme import get_theme, THEMES

    pygame.init()
    W, H   = 900, 280
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("WaveformPanel  —  T=trigger  G=grid  K=clip test  Theme=N")
    clock  = pygame.time.Clock()

    SR        = 44100
    CHUNK     = 1024
    t_arr     = np.arange(CHUNK) / SR

    theme_idx = 0
    theme     = get_theme(theme_idx)
    panel     = WaveformPanel(screen, pygame.Rect(10, 10, W - 20, H - 20), theme)

    frame   = 0
    running = True
    clip_test = False

    print("Smoke test running.")
    print("  T → toggle trigger (watch waveform stabilise/scroll)")
    print("  G → toggle grid")
    print("  K → toggle clipping test (force samples to ±1.0)")
    print("  N → cycle themes")
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
                    print(f"Theme → {theme.NAME}")
                elif event.key == pygame.K_k:
                    clip_test = not clip_test
                    print(f"Clip test → {'ON' if clip_test else 'OFF'}")
                else:
                    panel.handle_key(event)

        # ── Synthesise test signal ────────────────────────────────────────
        t      = frame * 0.016
        freq   = 220 + 180 * np.sin(t * 0.3)        # slowly varying pitch

        # Mix of fundamental + harmonics — like a voice
        sig = (
              0.45 * np.sin(2 * np.pi * freq       * t_arr + t)
            + 0.20 * np.sin(2 * np.pi * freq * 2   * t_arr + t * 1.1)
            + 0.10 * np.sin(2 * np.pi * freq * 3   * t_arr + t * 0.9)
            + 0.05 * np.random.normal(0, 0.1, CHUNK)
        ).astype(np.float32)

        if clip_test:
            sig *= 2.5    # force clipping

        np.clip(sig, -1.0, 1.0, out=sig)

        # ── Draw ─────────────────────────────────────────────────────────
        theme.begin_frame(screen)
        pygame.draw.rect(screen, theme.BG_PANEL, pygame.Rect(10, 10, W - 20, H - 20))

        panel.update(sig)
        panel.draw()

        theme.end_frame(screen)
        pygame.display.flip()
        clock.tick(60)
        frame += 1

    pygame.quit()