"""
src/display/controls.py

Filter controls panel.
Displays the active FilterBank state and handles keyboard input
for toggling, selecting, and retuning filters in real time.

Keybindings
-----------
    1-4     select filter slot
    SPACE   toggle selected filter on/off
    UP/DOWN adjust selected filter cutoff (SHIFT = fast step)
    B       bypass ALL filters
    F1      cycle themes
    H       toggle help display
"""

import pygame
import numpy as np
from numpy.typing import NDArray

try:
    from display.theme import BaseTheme
    from processing.filters import FilterBank, FilterType
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.display.theme import BaseTheme
    from src.processing.filters import FilterBank, FilterType


# ─── Constants ────────────────────────────────────────────────────────────────

PAD            = 10
PAD_TOP        = 24
ROW_HEIGHT     = 28
FREQ_STEP_FINE = 10.0
FREQ_STEP_FAST = 100.0
MIN_CUTOFF_HZ  = 20.0
MAX_CUTOFF_HZ  = 20000.0


# ─── ControlsPanel ────────────────────────────────────────────────────────────

class ControlsPanel:

    def __init__(
        self,
        surface         : pygame.Surface,
        rect            : pygame.Rect,
        theme           : BaseTheme,
        filter_bank     : FilterBank,
        sample_rate     : int = 44100,
        chunk_size      : int = 1024,
        on_theme_change = None,
    ):
        self.surface         = surface
        self.rect            = rect
        self.theme           = theme
        self.filter_bank     = filter_bank
        self.sample_rate     = sample_rate
        self.chunk_size      = chunk_size
        self.on_theme_change = on_theme_change

        self.selected_slot = 0
        self.master_bypass = False
        self.show_help     = True

        self._fps       = 0.0
        self._peak_freq = 0.0
        self._peak_db   = -90.0
        self._rms       = 0.0

        self._setup_default_filters()

    # ── Default filters ───────────────────────────────────────────────────────

    def _setup_default_filters(self):
        """
        Four pre-wired slots. Uses the actual filters.py API:
          add_highpass / add_lowpass  — retune with set_cutoff(name, hz)
          add_bandpass(name, lo, hi)  — retune with set_bandpass(name, lo, hi)
          add_notch(name, hz, q)      — retune with set_notch(name, hz)
        All start disabled.
        """
        fb = self.filter_bank
        fb.add_highpass("HP: DC REMOVE",  cutoff_hz=80.0,   order=4, enabled=False)
        fb.add_lowpass ("LP: VOICE TOP",  cutoff_hz=3400.0, order=4, enabled=False)
        fb.add_bandpass("BP: VOICE BAND", 80.0, 3400.0,     order=3, enabled=False)
        fb.add_notch   ("NOTCH: MAINS",   center_hz=50.0,   q=30.0,  enabled=False)

    # ── Public interface ──────────────────────────────────────────────────────

    def update_stats(self, fps, peak_freq, peak_db, rms):
        self._fps       = fps
        self._peak_freq = peak_freq
        self._peak_db   = peak_db
        self._rms       = rms

    def draw(self):
        pygame.draw.rect(self.surface, self.theme.BG_PANEL, self.rect)

        x = self.rect.x + PAD
        y = self.rect.y + PAD_TOP + PAD

        self.theme.draw_panel_label(self.surface, self.rect, "FILTER BANK", "/ CONTROLS")

        y = self._draw_bypass_indicator(x, y); y += 6
        y = self._draw_filter_slots(x, y);     y += 8
        y = self._draw_freq_readout(x, y);     y += 8

        pygame.draw.line(self.surface, self.theme.GRID,
                         (x, y), (self.rect.right - PAD, y), 1)
        y += 6
        y = self._draw_stats(x, y); y += 6

        if self.show_help:
            pygame.draw.line(self.surface, self.theme.GRID,
                             (x, y), (self.rect.right - PAD, y), 1)
            y += 6
            self._draw_help(x, y)

        self.theme.draw_panel_border(self.surface, self.rect)

    def handle_key(self, event: pygame.event.Event) -> bool:
        if event.type != pygame.KEYDOWN:
            return False

        names = self.filter_bank.filter_names

        # Select slot 1-4
        if event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
            slot = event.key - pygame.K_1
            if slot < len(names):
                self.selected_slot = slot
            return True

        # Toggle selected on/off
        if event.key == pygame.K_SPACE:
            if names and self.selected_slot < len(names):
                self.filter_bank.toggle(names[self.selected_slot])
            return True

        # Master bypass
        if event.key == pygame.K_b:
            self.master_bypass = not self.master_bypass
            if self.master_bypass:
                for name in names:
                    self.filter_bank.disable(name)
            return True

        # Adjust frequency
        if event.key in (pygame.K_UP, pygame.K_DOWN):
            mods      = pygame.key.get_mods()
            step      = FREQ_STEP_FAST if (mods & pygame.KMOD_SHIFT) else FREQ_STEP_FINE
            direction = 1 if event.key == pygame.K_UP else -1
            self._adjust_selected_filter(direction * step)
            return True

        # Toggle help
        if event.key == pygame.K_h:
            self.show_help = not self.show_help
            return True

        # Theme cycle
        if event.key == pygame.K_F1:
            if self.on_theme_change:
                self.on_theme_change()
            return True

        return False

    # ── Filter adjustment ─────────────────────────────────────────────────────

    def _adjust_selected_filter(self, delta_hz: float):
        """
        Retune the selected filter using the actual FilterBank API:
          LOWPASS/HIGHPASS → set_cutoff(name, hz)
          BANDPASS         → set_bandpass(name, lo, hi)
          NOTCH            → set_notch(name, hz)
        """
        names = self.filter_bank.filter_names
        if not names or self.selected_slot >= len(names):
            return

        name = names[self.selected_slot]
        f    = self.filter_bank._filters[name]
        nyq  = self.sample_rate / 2.0 - 1.0
        p    = f.params

        if f.filter_type == FilterType.LOWPASS:
            new_hz = float(np.clip(p["cutoff_hz"] + delta_hz, MIN_CUTOFF_HZ, min(MAX_CUTOFF_HZ, nyq)))
            self.filter_bank.set_cutoff(name, new_hz)

        elif f.filter_type == FilterType.HIGHPASS:
            new_hz = float(np.clip(p["cutoff_hz"] + delta_hz, MIN_CUTOFF_HZ, min(MAX_CUTOFF_HZ, nyq)))
            self.filter_bank.set_cutoff(name, new_hz)

        elif f.filter_type == FilterType.BANDPASS:
            new_lo = float(np.clip(p["lo_hz"] + delta_hz, MIN_CUTOFF_HZ, nyq))
            new_hi = float(np.clip(p["hi_hz"] + delta_hz, MIN_CUTOFF_HZ, nyq))
            if new_lo < new_hi:
                self.filter_bank.set_bandpass(name, new_lo, new_hi)

        elif f.filter_type == FilterType.NOTCH:
            new_hz = float(np.clip(p["center_hz"] + delta_hz, MIN_CUTOFF_HZ, min(MAX_CUTOFF_HZ, nyq)))
            self.filter_bank.set_notch(name, new_hz)

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_bypass_indicator(self, x, y):
        label = "[ BYPASS: ALL FILTERS OFF ]" if self.master_bypass else "[ BYPASS: OFF ]"
        color = self.theme.WARNING if self.master_bypass else self.theme.TEXT_DIM
        self.theme.draw_text(self.surface, label, (x, y),
                             font="digit_small", color=color, glow=self.master_bypass)
        return y + 16

    def _draw_filter_slots(self, x, y):
        names = self.filter_bank.filter_names

        for idx, name in enumerate(names):
            f           = self.filter_bank._filters[name]
            is_selected = idx == self.selected_slot
            is_enabled  = f.enabled and not self.master_bypass

            # Row highlight for selected slot
            row_rect = pygame.Rect(x - 4, y - 2, self.rect.width - PAD * 2, ROW_HEIGHT - 4)
            if is_selected:
                pygame.draw.rect(self.surface, self.theme.GRID,   row_rect, 0)
                pygame.draw.rect(self.surface, self.theme.ACCENT, row_rect, 1)

            # [N] slot number
            self.theme.draw_text(self.surface, f"[{idx + 1}]", (x, y),
                                 font="digit_main",
                                 color=self.theme.ACCENT if is_selected else self.theme.TEXT_DIM)

            # ON / OFF
            self.theme.draw_text(self.surface,
                                 "ON " if is_enabled else "OFF",
                                 (x + 36, y),
                                 font="digit_main",
                                 color=self.theme.PHOSPHOR_MID if is_enabled else self.theme.TEXT_DIM,
                                 glow=is_enabled)

            # Name + freq params — use the correct param keys for each type
            p = f.params
            if f.filter_type == FilterType.BANDPASS:
                freq_str = f"{p['lo_hz']:.0f}-{p['hi_hz']:.0f}Hz"
            elif f.filter_type == FilterType.NOTCH:
                freq_str = f"{p['center_hz']:.0f}Hz Q{p['q']:.0f}"
            else:
                freq_str = f"{p['cutoff_hz']:.0f}Hz"

            self.theme.draw_text(self.surface, f"{name}  {freq_str}", (x + 76, y),
                                 font="mono_small",
                                 color=self.theme.TEXT if is_enabled else self.theme.TEXT_DIM)
            y += ROW_HEIGHT

        return y

    def _draw_freq_readout(self, x, y):
        names = self.filter_bank.filter_names
        if not names or self.selected_slot >= len(names):
            return y

        name = names[self.selected_slot]
        f    = self.filter_bank._filters[name]
        p    = f.params

        if f.filter_type == FilterType.BANDPASS:
            freq_str = f"{p['lo_hz']:.1f} — {p['hi_hz']:.1f} Hz"
        elif f.filter_type == FilterType.NOTCH:
            freq_str = f"{p['center_hz']:.1f} Hz  Q={p['q']:.0f}"
        else:
            freq_str = f"{p['cutoff_hz']:.1f} Hz"

        self.theme.draw_text(self.surface, f"SEL FREQ: {freq_str}", (x, y),
                             font="digit_main", color=self.theme.ACCENT, glow=True)
        y += 18
        self.theme.draw_text(self.surface,
                             f"UP/DN={FREQ_STEP_FINE:.0f}Hz  SHIFT+UP/DN={FREQ_STEP_FAST:.0f}Hz",
                             (x, y), font="mono_small", color=self.theme.TEXT_DIM)
        return y + 16

    def _draw_stats(self, x, y):
        sr_khz   = self.sample_rate / 1000.0
        chunk_ms = 1000.0 * self.chunk_size / self.sample_rate
        freq_str = (f"{self._peak_freq/1000:.2f}kHz"
                    if self._peak_freq >= 1000 else f"{self._peak_freq:.0f}Hz")

        for label, value in [
            ("FPS",   f"{self._fps:.1f}"),
            ("SR",    f"{sr_khz:.1f}kHz"),
            ("CHUNK", f"{self.chunk_size}  ({chunk_ms:.1f}ms)"),
            ("PEAK",  f"{freq_str}  {self._peak_db:.1f}dB"),
            ("RMS",   f"{self._rms:.4f}"),
        ]:
            self.theme.draw_text(self.surface, f"{label:<6}", (x, y),
                                 font="digit_small", color=self.theme.TEXT_DIM)
            self.theme.draw_text(self.surface, value, (x + 48, y),
                                 font="digit_main", color=self.theme.TEXT)
            y += 16
        return y

    def _draw_help(self, x, y):
        self.theme.draw_text(self.surface, "KEYBINDINGS", (x, y),
                             font="mono_small", color=self.theme.TEXT_DIM)
        y += 14
        for key, desc in [
            ("1-4",    "select filter"),
            ("SPACE",  "toggle on/off"),
            ("UP/DN",  "adjust freq"),
            ("SHF+UP", "adjust fast"),
            ("B",      "bypass all"),
            ("F1",     "cycle theme"),
            ("H",      "toggle help"),
            ("L",      "log/lin axis"),
            ("P",      "peak hold"),
        ]:
            self.theme.draw_text(self.surface, f"{key:<10}", (x, y),
                                 font="digit_small", color=self.theme.ACCENT)
            self.theme.draw_text(self.surface, desc, (x + 70, y),
                                 font="mono_small", color=self.theme.TEXT_DIM)
            y += 13
        return y


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from src.display.theme import get_theme, THEMES
    from src.processing.filters import FilterBank

    pygame.init()
    W, H   = 420, 600
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("ControlsPanel — 1-4 select  SPACE toggle  UP/DN freq  B bypass  F1 theme")
    clock  = pygame.time.Clock()

    theme_idx = 0
    theme     = get_theme(theme_idx)
    fb        = FilterBank(sample_rate=44100)

    def next_theme():
        global theme_idx, theme
        theme_idx   = (theme_idx + 1) % len(THEMES)
        theme       = get_theme(theme_idx)
        panel.theme = theme
        print(f"Theme → {theme.NAME}")

    panel = ControlsPanel(
        screen, pygame.Rect(10, 10, W - 20, H - 20),
        theme, fb, on_theme_change=next_theme
    )

    frame = 0
    running = True
    print("Controls smoke test — ESC to quit")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    panel.handle_key(event)

        t = frame * 0.016
        panel.update_stats(
            fps       = clock.get_fps(),
            peak_freq = 440 + 200 * np.sin(t * 0.5),
            peak_db   = -20 + 5  * np.sin(t),
            rms       = 0.08 + 0.04 * np.sin(t * 1.3),
        )

        theme.begin_frame(screen)
        screen.fill(theme.BG)
        panel.draw()
        theme.end_frame(screen)
        pygame.display.flip()
        clock.tick(60)
        frame += 1

    pygame.quit()