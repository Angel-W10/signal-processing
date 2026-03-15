"""
src/display/theme.py

Shared visual theme for the oscilloscope display.
All panels (spectrum, waterfall, waveform, controls) receive a Theme instance
and use its methods to draw — one place to control the entire look.

Switching themes is one line in main.py:
    theme = PhosphorGreenTheme()   # classic green CRT
    theme = AmberTheme()           # vintage amber monitor
    theme = BlueTheme()            # modern blue phosphor

Adding a new theme: subclass BaseTheme, override the color constants,
done — all rendering logic is inherited.

Effects implemented
-------------------
- Phosphor persistence  : semi-transparent clear instead of hard clear
- Glow / bloom          : multi-pass line drawing at varying width + alpha
- Scan lines            : pre-rendered horizontal stripe overlay
- CRT vignette          : corner darkening
- Signal heat mapping   : bright signals shift toward white (hot phosphor)
- Subtle flicker        : ±2% brightness variation per frame
"""

import pygame
import numpy as np
import random
import os
from dataclasses import dataclass
from typing import Tuple

# Type aliases
Color  = Tuple[int, int, int]
ColorA = Tuple[int, int, int, int]


# ─── Asset path helper ────────────────────────────────────────────────────────

def _asset(*parts) -> str:
    """Return absolute path to a file in the assets/ directory."""
    root = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
    return os.path.abspath(os.path.join(root, *parts))


# ─── BaseTheme ────────────────────────────────────────────────────────────────

class BaseTheme:
    """
    Base class for all themes.

    Subclasses override the color constants at the top.
    All rendering methods are inherited and use self.COLOR_NAME,
    so changing colors in a subclass automatically affects all rendering.

    Color constants every subclass must define
    ------------------------------------------
    BG              : window background (slightly tinted, not pure black)
    BG_PANEL        : panel background (slightly lighter than BG)
    PHOSPHOR_CORE   : brightest signal color (center of glow)
    PHOSPHOR_MID    : mid glow color
    PHOSPHOR_OUTER  : outermost glow (barely visible)
    PHOSPHOR_DIM    : very dim — used for ghost/inactive elements
    GRID            : grid line color
    GRID_CENTER     : center crosshair (slightly brighter)
    PEAK_HOLD       : peak hold line color
    TEXT            : primary label text
    TEXT_DIM        : secondary / inactive text
    ACCENT          : highlight / selected element
    WARNING         : clipping / over-range indicator
    """

    # ── Colors — override in subclasses ──────────────────────────────────────
    BG             : Color = (3,   13,  5)
    BG_PANEL       : Color = (5,   18,  7)
    PHOSPHOR_CORE  : Color = (200, 255, 180)
    PHOSPHOR_MID   : Color = (57,  255, 110)
    PHOSPHOR_OUTER : Color = (10,  74,  30)
    PHOSPHOR_DIM   : Color = (8,   40,  16)
    GRID           : Color = (13,  46,  16)
    GRID_CENTER    : Color = (20,  70,  25)
    PEAK_HOLD      : Color = (255, 200, 50)
    TEXT           : Color = (140, 255, 160)
    TEXT_DIM       : Color = (40,  100, 50)
    ACCENT         : Color = (0,   255, 136)
    WARNING        : Color = (255, 60,  60)

    # ── Effect parameters — tweak per theme ──────────────────────────────────
    PERSISTENCE_ALPHA : int   = 40     # 0=hard clear, 255=no clear. 30-60 is good.
    SCANLINE_ALPHA    : int   = 25     # opacity of scan line overlay
    VIGNETTE_ALPHA    : int   = 140    # corner darkening strength
    FLICKER_AMOUNT    : float = 0.02   # ±fraction of brightness variation
    GLOW_PASSES       : int   = 3      # number of glow render passes
    FONT_MAIN_SIZE    : int   = 13     # primary label font size
    FONT_SMALL_SIZE   : int   = 10     # secondary label font size
    FONT_LARGE_SIZE   : int   = 20     # header / value readout size
    NAME              : str   = "Base"

    def __init__(self):
        pygame.font.init()
        self._load_fonts()
        self._scanline_surface = None   # built lazily on first draw
        self._vignette_surface = None
        self._flicker_offset   = 0.0

    # ── Font loading ──────────────────────────────────────────────────────────

    def _load_fonts(self):
        """Load theme fonts. Falls back to pygame built-in if files missing."""
        digital_path    = _asset("fonts", "digital-7.ttf")
        mono_path       = _asset("fonts", "ShareTechMono-Regular.ttf")

        try:
            self.font_digit_large = pygame.font.Font(digital_path, self.FONT_LARGE_SIZE + 8)
            self.font_digit_main  = pygame.font.Font(digital_path, self.FONT_MAIN_SIZE + 4)
            self.font_digit_small = pygame.font.Font(digital_path, self.FONT_SMALL_SIZE + 2)
            self._digital_ok = True
        except FileNotFoundError:
            fallback = pygame.font.match_font('monospace')
            self.font_digit_large = pygame.font.Font(fallback, self.FONT_LARGE_SIZE)
            self.font_digit_main  = pygame.font.Font(fallback, self.FONT_MAIN_SIZE)
            self.font_digit_small = pygame.font.Font(fallback, self.FONT_SMALL_SIZE)
            self._digital_ok = False

        try:
            self.font_mono_main  = pygame.font.Font(mono_path, self.FONT_MAIN_SIZE)
            self.font_mono_small = pygame.font.Font(mono_path, self.FONT_SMALL_SIZE)
            self.font_mono_large = pygame.font.Font(mono_path, self.FONT_LARGE_SIZE)
        except FileNotFoundError:
            fallback = pygame.font.match_font('monospace')
            self.font_mono_main  = pygame.font.Font(fallback, self.FONT_MAIN_SIZE)
            self.font_mono_small = pygame.font.Font(fallback, self.FONT_SMALL_SIZE)
            self.font_mono_large = pygame.font.Font(fallback, self.FONT_LARGE_SIZE)

    # ── Frame-level effects ───────────────────────────────────────────────────

    def begin_frame(self, surface: pygame.Surface):
        """
        Call at the start of every frame INSTEAD of surface.fill(BG).

        Draws a semi-transparent black rectangle over the previous frame.
        This is phosphor persistence — old signal content fades gradually
        rather than disappearing instantly.

        PERSISTENCE_ALPHA controls the feel:
          Low  (20-30) : long trails, dreamy, slow response
          Mid  (40-60) : classic oscilloscope feel
          High (80+)   : snappy, almost no trails
        """
        # Update flicker for this frame — tiny random brightness shift
        self._flicker_offset = random.uniform(
            -self.FLICKER_AMOUNT, self.FLICKER_AMOUNT
        )

        # Semi-transparent black overlay — fades previous frame content
        fade = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        fade.fill((0, 0, 0, self.PERSISTENCE_ALPHA))
        surface.blit(fade, (0, 0))

    def end_frame(self, surface: pygame.Surface):
        """
        Call at the end of every frame, after all panels have drawn.

        Applies scan lines and vignette on top of everything.
        These are overlay effects — they sit above the signal content.
        """
        w, h = surface.get_size()
        self._draw_scanlines(surface, w, h)
        self._draw_vignette(surface, w, h)

    # ── Scan lines ────────────────────────────────────────────────────────────

    def _draw_scanlines(self, surface: pygame.Surface, w: int, h: int):
        """
        Blit a pre-rendered horizontal scan line pattern over the entire surface.

        The pattern is a surface with alternating transparent and semi-opaque
        rows — every other row is slightly darkened, giving the CRT line feel.

        We build this once and cache it. If the window is resized we rebuild.
        """
        if (self._scanline_surface is None or
                self._scanline_surface.get_size() != (w, h)):
            sl = pygame.Surface((w, h), pygame.SRCALPHA)
            sl.fill((0, 0, 0, 0))
            for y in range(0, h, 2):
                pygame.draw.line(
                    sl,
                    (0, 0, 0, self.SCANLINE_ALPHA),
                    (0, y), (w, y)
                )
            self._scanline_surface = sl

        surface.blit(self._scanline_surface, (0, 0))

    # ── Vignette ──────────────────────────────────────────────────────────────

    def _draw_vignette(self, surface: pygame.Surface, w: int, h: int):
        """
        Darken the corners to simulate CRT tube curvature.
        Pre-rendered as a radial gradient from transparent center to dark edges.
        """
        if (self._vignette_surface is None or
                self._vignette_surface.get_size() != (w, h)):
            vig = pygame.Surface((w, h), pygame.SRCALPHA)
            vig.fill((0, 0, 0, 0))

            cx, cy = w // 2, h // 2
            max_r  = (cx**2 + cy**2) ** 0.5

            # Draw concentric ellipses from edge inward — builds up a gradient
            steps = 60
            for i in range(steps):
                t     = i / steps                      # 0 = center, 1 = edge
                alpha = int(self.VIGNETTE_ALPHA * (t ** 2.5))
                rx    = int(cx * (1 - t * 0.3))
                ry    = int(cy * (1 - t * 0.3))
                if rx <= 0 or ry <= 0:
                    continue
                s = pygame.Surface((rx * 2, ry * 2), pygame.SRCALPHA)
                s.fill((0, 0, 0, 0))
                pygame.draw.ellipse(s, (0, 0, 0, 3), s.get_rect())
                vig.blit(s, (cx - rx, cy - ry))

            # Solid dark corners
            corner_size = min(w, h) // 5
            for corner_x, corner_y in [
                (0, 0), (w - corner_size, 0),
                (0, h - corner_size), (w - corner_size, h - corner_size)
            ]:
                corner = pygame.Surface((corner_size, corner_size), pygame.SRCALPHA)
                corner.fill((0, 0, 0, self.VIGNETTE_ALPHA // 2))
                vig.blit(corner, (corner_x, corner_y))

            self._vignette_surface = vig

        surface.blit(self._vignette_surface, (0, 0))

    # ── Glow line drawing ─────────────────────────────────────────────────────

    def draw_glow_line(
        self,
        surface : pygame.Surface,
        p1      : Tuple[int, int],
        p2      : Tuple[int, int],
        intensity: float = 1.0,    # 0.0 → 1.0, modulates glow width and brightness
    ):
        """
        Draw a single line segment with phosphor glow effect.

        Three passes:
          Pass 1 — wide line,   outer glow color,  low alpha
          Pass 2 — medium line, mid glow color,    medium alpha
          Pass 3 — thin line,   core color,        full alpha

        intensity modulates how bright/wide the glow is.
        Use intensity=spectrum_db_norm to make louder signals glow harder.
        """
        intensity = max(0.0, min(1.0, intensity))
        flicker   = 1.0 + self._flicker_offset

        # Pass 1 — outer glow (wide, dim)
        outer_w = max(1, int(6 * intensity))
        outer_c = self._tint(self.PHOSPHOR_OUTER, flicker * 0.8)
        pygame.draw.line(surface, outer_c, p1, p2, outer_w + 4)

        # Pass 2 — mid glow
        mid_w = max(1, int(3 * intensity))
        mid_c = self._tint(self.PHOSPHOR_MID, flicker * 0.9)
        pygame.draw.line(surface, mid_c, p1, p2, mid_w + 2)

        # Pass 3 — bright core
        core_c = self._tint(self.PHOSPHOR_CORE, flicker)
        pygame.draw.line(surface, core_c, p1, p2, 1)

    def draw_glow_polyline(
        self,
        surface  : pygame.Surface,
        points   : list,
        intensity: float = 1.0,
    ):
        """
        Draw a connected polyline with glow.
        More efficient than calling draw_glow_line for each segment
        when you have many points (spectrum line, waveform).
        """
        if len(points) < 2:
            return

        intensity = max(0.0, min(1.0, intensity))
        flicker   = 1.0 + self._flicker_offset

        # Pass 1 — outer glow
        outer_c = self._tint(self.PHOSPHOR_OUTER, flicker * 0.8)
        pygame.draw.lines(surface, outer_c, False, points, max(1, int(5 * intensity) + 3))

        # Pass 2 — mid glow
        mid_c = self._tint(self.PHOSPHOR_MID, flicker * 0.9)
        pygame.draw.lines(surface, mid_c, False, points, max(1, int(3 * intensity) + 1))

        # Pass 3 — bright core
        core_c = self._tint(self.PHOSPHOR_CORE, flicker)
        pygame.draw.lines(surface, core_c, False, points, 1)

    def signal_color(self, norm: float) -> Color:
        """
        Map a normalised signal value (0.0→1.0) to a phosphor color.

        Low values  → dim outer glow color  (cold, barely visible)
        Mid values  → mid phosphor color    (normal signal)
        High values → core color            (hot phosphor, near white)

        This is what makes the waterfall feel like a real phosphor display —
        stronger signals look literally hotter/brighter.
        """
        norm = max(0.0, min(1.0, norm))
        if norm < 0.4:
            t = norm / 0.4
            return self._lerp_color(self.PHOSPHOR_DIM, self.PHOSPHOR_OUTER, t)
        elif norm < 0.75:
            t = (norm - 0.4) / 0.35
            return self._lerp_color(self.PHOSPHOR_OUTER, self.PHOSPHOR_MID, t)
        else:
            t = (norm - 0.75) / 0.25
            return self._lerp_color(self.PHOSPHOR_MID, self.PHOSPHOR_CORE, t)

    # ── Grid drawing ──────────────────────────────────────────────────────────

    def draw_grid(
        self,
        surface    : pygame.Surface,
        rect       : pygame.Rect,
        cols       : int = 10,
        rows       : int = 8,
        minor_ticks: bool = True,
    ):
        """
        Draw an oscilloscope-style graticule inside rect.

        Major lines divide the panel into cols × rows cells.
        Minor tick marks appear at the midpoints of each cell.
        The center horizontal line is brighter (the zero/reference line).
        """
        x0, y0, w, h = rect.x, rect.y, rect.width, rect.height

        col_w = w / cols
        row_h = h / rows

        # Major vertical lines
        for i in range(1, cols):
            x = int(x0 + i * col_w)
            color = self.GRID_CENTER if i == cols // 2 else self.GRID
            pygame.draw.line(surface, color, (x, y0), (x, y0 + h), 1)

        # Major horizontal lines
        for i in range(1, rows):
            y     = int(y0 + i * row_h)
            color = self.GRID_CENTER if i == rows // 2 else self.GRID
            pygame.draw.line(surface, color, (x0, y), (x0 + w, y), 1)

        if not minor_ticks:
            return

        # Minor tick marks — small dashes at cell midpoints
        tick_len = 4
        for i in range(cols + 1):
            x = int(x0 + i * col_w)
            for j in range(rows + 1):
                y     = int(y0 + j * row_h)
                my    = int(y0 + (j + 0.5) * row_h)
                mx    = int(x0 + (i + 0.5) * col_w)
                # Vertical tick at horizontal midpoint
                if 0 <= mx <= x0 + w:
                    pygame.draw.line(surface, self.GRID, (mx, y - tick_len // 2), (mx, y + tick_len // 2), 1)
                # Horizontal tick at vertical midpoint
                if 0 <= my <= y0 + h:
                    pygame.draw.line(surface, self.GRID, (x - tick_len // 2, my), (x + tick_len // 2, my), 1)

    # ── Text rendering ────────────────────────────────────────────────────────

    def draw_text(
        self,
        surface : pygame.Surface,
        text    : str,
        pos     : Tuple[int, int],
        font    : str   = "mono_small",   # "mono_main", "mono_large", "digit_main" etc
        color   : Color = None,
        anchor  : str   = "topleft",      # "topleft", "topright", "center", "bottomleft"
        glow    : bool  = False,
    ):
        """
        Render text onto surface with optional phosphor glow.

        font parameter selects from loaded fonts:
            "mono_small"   "mono_main"   "mono_large"
            "digit_small"  "digit_main"  "digit_large"
        """
        color  = color or self.TEXT
        f      = self._get_font(font)
        surf   = f.render(text, True, color)
        rect   = surf.get_rect(**{anchor: pos})

        if glow:
            # Render dim glow behind text
            glow_surf = f.render(text, True, self.PHOSPHOR_OUTER)
            for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
                surface.blit(glow_surf, rect.move(dx, dy))

        surface.blit(surf, rect)
        return rect

    def draw_panel_label(
        self,
        surface : pygame.Surface,
        rect    : pygame.Rect,
        title   : str,
        subtitle: str = "",
    ):
        """
        Draw the panel title in the top-left corner of rect.
        Title in mono_small + accent color. Subtitle in TEXT_DIM.
        """
        x, y = rect.x + 8, rect.y + 6
        self.draw_text(surface, title, (x, y),
                       font="mono_small", color=self.ACCENT)
        if subtitle:
            tw = self._get_font("mono_small").size(title)[0]
            self.draw_text(surface, f"  {subtitle}", (x + tw, y),
                           font="mono_small", color=self.TEXT_DIM)

    def draw_panel_border(self, surface: pygame.Surface, rect: pygame.Rect):
        """Draw a dim border around a panel rect."""
        pygame.draw.rect(surface, self.GRID, rect, 1)

    # ── Utility ───────────────────────────────────────────────────────────────

    def _get_font(self, name: str) -> pygame.font.Font:
        fonts = {
            "mono_small"  : self.font_mono_small,
            "mono_main"   : self.font_mono_main,
            "mono_large"  : self.font_mono_large,
            "digit_small" : self.font_digit_small,
            "digit_main"  : self.font_digit_main,
            "digit_large" : self.font_digit_large,
        }
        return fonts.get(name, self.font_mono_small)

    @staticmethod
    def _lerp_color(c1: Color, c2: Color, t: float) -> Color:
        """Linear interpolate between two RGB colors."""
        return (
            int(c1[0] + (c2[0] - c1[0]) * t),
            int(c1[1] + (c2[1] - c1[1]) * t),
            int(c1[2] + (c2[2] - c1[2]) * t),
        )

    @staticmethod
    def _tint(color: Color, factor: float) -> Color:
        """Multiply all channels by factor, clamp to [0, 255]."""
        return (
            max(0, min(255, int(color[0] * factor))),
            max(0, min(255, int(color[1] * factor))),
            max(0, min(255, int(color[2] * factor))),
        )


# ─── Concrete themes ──────────────────────────────────────────────────────────

class PhosphorGreenTheme(BaseTheme):
    """
    Classic P31 phosphor — the iconic green oscilloscope look.
    Most laboratory oscilloscopes from the 1960s–1990s used this phosphor.
    """
    NAME              = "Phosphor Green"
    BG                = (3,   13,  5)
    BG_PANEL          = (5,   18,  7)
    PHOSPHOR_CORE     = (200, 255, 180)
    PHOSPHOR_MID      = (57,  255, 110)
    PHOSPHOR_OUTER    = (10,  74,  30)
    PHOSPHOR_DIM      = (8,   40,  16)
    GRID              = (13,  46,  16)
    GRID_CENTER       = (20,  70,  25)
    PEAK_HOLD         = (255, 200, 50)
    TEXT              = (140, 255, 160)
    TEXT_DIM          = (40,  100, 50)
    ACCENT            = (0,   255, 136)
    WARNING           = (255, 60,  60)
    PERSISTENCE_ALPHA = 40
    SCANLINE_ALPHA    = 28


class AmberTheme(BaseTheme):
    """
    P3 amber phosphor — used in many early computer terminals (IBM 3270 etc).
    Warmer, more orange feel. Often considered easier on the eyes for long sessions.
    """
    NAME              = "Amber"
    BG                = (12,  7,   0)
    BG_PANEL          = (16,  10,  0)
    PHOSPHOR_CORE     = (255, 240, 180)
    PHOSPHOR_MID      = (255, 160, 20)
    PHOSPHOR_OUTER    = (80,  40,  0)
    PHOSPHOR_DIM      = (40,  20,  0)
    GRID              = (50,  28,  0)
    GRID_CENTER       = (75,  42,  0)
    PEAK_HOLD         = (255, 255, 120)
    TEXT              = (255, 200, 100)
    TEXT_DIM          = (100, 60,  10)
    ACCENT            = (255, 180, 0)
    WARNING           = (255, 80,  80)
    PERSISTENCE_ALPHA = 35
    SCANLINE_ALPHA    = 22


class BluePhosphorTheme(BaseTheme):
    """
    P7 blue-white phosphor — used in some radar displays and storage scopes.
    High contrast, cool and clinical feel.
    """
    NAME              = "Blue Phosphor"
    BG                = (0,   4,   14)
    BG_PANEL          = (0,   6,   18)
    PHOSPHOR_CORE     = (180, 220, 255)
    PHOSPHOR_MID      = (40,  140, 255)
    PHOSPHOR_OUTER    = (5,   30,  80)
    PHOSPHOR_DIM      = (3,   15,  40)
    GRID              = (8,   22,  55)
    GRID_CENTER       = (12,  35,  80)
    PEAK_HOLD         = (255, 220, 80)
    TEXT              = (100, 180, 255)
    TEXT_DIM          = (30,  60,  120)
    ACCENT            = (0,   200, 255)
    WARNING           = (255, 80,  80)
    PERSISTENCE_ALPHA = 45
    SCANLINE_ALPHA    = 20


class RedTheme(BaseTheme):
    """
    Dark room / night vision mode.
    Red light doesn't ruin dark-adapted vision — used in astronomy setups.
    Fitting for an SDR project tracking satellites in the dark.
    """
    NAME              = "Night Vision"
    BG                = (10,  0,   0)
    BG_PANEL          = (14,  2,   2)
    PHOSPHOR_CORE     = (255, 180, 160)
    PHOSPHOR_MID      = (220, 40,  20)
    PHOSPHOR_OUTER    = (70,  5,   0)
    PHOSPHOR_DIM      = (35,  3,   0)
    GRID              = (50,  8,   5)
    GRID_CENTER       = (75,  12,  8)
    PEAK_HOLD         = (255, 255, 100)
    TEXT              = (220, 100, 80)
    TEXT_DIM          = (90,  30,  20)
    ACCENT            = (255, 80,  40)
    WARNING           = (255, 255, 0)
    PERSISTENCE_ALPHA = 38
    SCANLINE_ALPHA    = 25


# ─── Theme registry ───────────────────────────────────────────────────────────
# In main.py you can cycle through these with a keypress.

THEMES = [
    PhosphorGreenTheme,
    AmberTheme,
    BluePhosphorTheme,
    RedTheme,
]

def get_theme(index: int) -> BaseTheme:
    """Return an instantiated theme by index. Wraps around."""
    return THEMES[index % len(THEMES)]()


# ─── Smoke test ───────────────────────────────────────────────────────────────
# python src/display/theme.py
# Opens a Pygame window showing all themes side by side with glow effects.

if __name__ == "__main__":
    pygame.init()
    W, H   = 1200, 400
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Theme preview")
    clock  = pygame.time.Clock()

    themes     = [cls() for cls in THEMES]
    panel_w    = W // len(themes)
    running    = True
    frame      = 0

    # Fake spectrum data — a sine sweep for visual interest
    num_bins   = 256
    fake_freqs = np.linspace(0, 22050, num_bins)

    print("Theme preview — press ESC or close window to exit")
    print(f"Loaded themes: {[t.NAME for t in themes]}")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        screen.fill((0, 0, 0))
        frame += 1

        for idx, theme in enumerate(themes):
            panel_rect = pygame.Rect(idx * panel_w, 0, panel_w, H)

            # Panel background
            pygame.draw.rect(screen, theme.BG_PANEL, panel_rect)

            # Grid
            inner = pygame.Rect(panel_rect.x + 10, 40, panel_w - 20, H - 80)
            theme.draw_grid(screen, inner, cols=8, rows=6)

            # Fake waveform — animated sine wave with glow
            t      = frame * 0.03
            points = []
            for i in range(inner.width):
                x   = inner.x + i
                val = np.sin(i * 0.04 + t) * 0.4 + np.sin(i * 0.015 + t * 0.7) * 0.3
                y   = int(inner.centery + val * inner.height * 0.35)
                points.append((x, y))

            if len(points) > 1:
                theme.draw_glow_polyline(screen, points, intensity=0.85)

            # Peak hold line — flat line across
            ph_y = inner.y + int(inner.height * 0.25)
            theme.draw_glow_line(
                screen,
                (inner.x, ph_y), (inner.right, ph_y),
                intensity=0.5
            )

            # Panel border
            theme.draw_panel_border(screen, panel_rect)

            # Labels
            theme.draw_panel_label(screen, panel_rect, theme.NAME, "/ PREVIEW")
            theme.draw_text(
                screen,
                f"1337.42 Hz",
                (panel_rect.centerx, H - 30),
                font="digit_main",
                color=theme.TEXT,
                anchor="center",
                glow=True,
            )

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()