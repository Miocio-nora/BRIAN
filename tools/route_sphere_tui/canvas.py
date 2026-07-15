from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from rich.style import Style
from rich.text import Text

RGB = tuple[int, int, int]

_BRAILLE_BITS = np.asarray(
    [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ],
    dtype=np.uint16,
)


@dataclass(frozen=True)
class Glyph:
    char: str
    color: RGB
    bold: bool
    priority: float


class BrailleCanvas:
    """True-color 2x4 sub-cell canvas with ASCII glyph overlays."""

    def __init__(self, width: int, height: int) -> None:
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.pixel_width = self.width * 2
        self.pixel_height = self.height * 4
        self.intensity = np.zeros((self.pixel_height, self.pixel_width), dtype=np.float32)
        self.color = np.zeros((self.pixel_height, self.pixel_width, 3), dtype=np.float32)
        self.glyphs: dict[tuple[int, int], Glyph] = {}

    def point(
        self,
        x: float,
        y: float,
        *,
        color: RGB,
        intensity: float = 1.0,
        radius: float = 0.0,
    ) -> None:
        px = int(round(x * 2.0))
        py = int(round(y * 4.0))
        pixel_radius = max(0, int(round(radius * 2.0)))
        for dy in range(-pixel_radius, pixel_radius + 1):
            for dx in range(-pixel_radius, pixel_radius + 1):
                distance = math.sqrt(dx * dx + (dy * 0.5) ** 2)
                if distance > max(0.5, pixel_radius):
                    continue
                fade = 1.0 if pixel_radius == 0 else max(0.0, 1.0 - distance / (pixel_radius + 0.5))
                self._pixel(px + dx, py + dy, color, intensity * fade)

    def line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        color: RGB,
        intensity: float = 1.0,
        dashed: bool = False,
        start: float = 0.0,
        end: float = 1.0,
    ) -> None:
        start = min(1.0, max(0.0, float(start)))
        end = min(1.0, max(start, float(end)))
        ax = x0 + (x1 - x0) * start
        ay = y0 + (y1 - y0) * start
        bx = x0 + (x1 - x0) * end
        by = y0 + (y1 - y0) * end
        samples = max(2, int(max(abs(bx - ax) * 2.0, abs(by - ay) * 4.0)) + 1)
        for index, value in enumerate(np.linspace(0.0, 1.0, samples)):
            if dashed and (index // 2) % 2:
                continue
            x = ax + (bx - ax) * float(value)
            y = ay + (by - ay) * float(value)
            self._pixel(int(round(x * 2.0)), int(round(y * 4.0)), color, intensity)

    def polyline(
        self,
        points: Iterable[tuple[float, float]],
        *,
        color: RGB,
        intensity: float,
        closed: bool = False,
        dashed: bool = False,
    ) -> None:
        values = list(points)
        if len(values) < 2:
            return
        pairs = list(zip(values[:-1], values[1:]))
        if closed:
            pairs.append((values[-1], values[0]))
        for first, second in pairs:
            self.line(*first, *second, color=color, intensity=intensity, dashed=dashed)

    def glyph(
        self,
        x: float,
        y: float,
        char: str,
        *,
        color: RGB,
        bold: bool = False,
        priority: float = 1.0,
    ) -> None:
        cx = int(round(x))
        cy = int(round(y))
        if not (0 <= cx < self.width and 0 <= cy < self.height):
            return
        previous = self.glyphs.get((cx, cy))
        if previous is None or priority >= previous.priority:
            self.glyphs[(cx, cy)] = Glyph(char[:1], color, bold, priority)

    def text(self) -> Text:
        output = Text(no_wrap=True, overflow="crop")
        for cell_y in range(self.height):
            for cell_x in range(self.width):
                glyph = self.glyphs.get((cell_x, cell_y))
                if glyph is not None:
                    output.append(
                        glyph.char,
                        Style(color=_rich_color(glyph.color), bold=glyph.bold),
                    )
                    continue
                y0 = cell_y * 4
                x0 = cell_x * 2
                block = self.intensity[y0 : y0 + 4, x0 : x0 + 2]
                active = block > 0.008
                if not bool(active.any()):
                    output.append(" ")
                    continue
                code = 0
                for dot_y in range(4):
                    for dot_x in range(2):
                        if active[dot_y, dot_x]:
                            code |= int(_BRAILLE_BITS[dot_y, dot_x])
                weighted = block[..., None] * self.color[y0 : y0 + 4, x0 : x0 + 2]
                denominator = max(1e-6, float(block.sum()))
                rgb = np.clip(weighted.sum(axis=(0, 1)) / denominator, 0.0, 255.0)
                strength = min(1.0, max(0.18, float(block.max())))
                visible = tuple(int(round(float(channel) * (0.45 + 0.55 * strength))) for channel in rgb)
                output.append(chr(0x2800 + code), Style(color=_rich_color(visible)))
            if cell_y + 1 < self.height:
                output.append("\n")
        return output

    def _pixel(self, px: int, py: int, color: RGB, intensity: float) -> None:
        if not (0 <= px < self.pixel_width and 0 <= py < self.pixel_height):
            return
        value = min(1.0, max(0.0, float(intensity)))
        previous = float(self.intensity[py, px])
        if value >= previous:
            self.color[py, px] = color
        elif previous > 0.0:
            mix = value / (previous + value)
            self.color[py, px] = self.color[py, px] * (1.0 - mix) + np.asarray(color) * mix
        self.intensity[py, px] = max(previous, value)


def _rich_color(color: RGB) -> str:
    return f"rgb({color[0]},{color[1]},{color[2]})"
