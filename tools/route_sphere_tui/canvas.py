from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal

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

_QUADRANT_CHARS = (
    " ",
    "▘",
    "▝",
    "▀",
    "▖",
    "▌",
    "▞",
    "▛",
    "▗",
    "▚",
    "▐",
    "▜",
    "▄",
    "▙",
    "▟",
    "█",
)

_NORTH = 0x1
_EAST = 0x2
_SOUTH = 0x4
_WEST = 0x8
_STROKE_CHARS = {
    _NORTH: "╵",
    _EAST: "╴",
    _SOUTH: "╷",
    _WEST: "╶",
    _NORTH | _SOUTH: "│",
    _EAST | _WEST: "─",
    _NORTH | _EAST: "└",
    _EAST | _SOUTH: "┌",
    _SOUTH | _WEST: "┐",
    _WEST | _NORTH: "┘",
    _NORTH | _EAST | _SOUTH: "├",
    _NORTH | _EAST | _WEST: "┴",
    _NORTH | _SOUTH | _WEST: "┤",
    _EAST | _SOUTH | _WEST: "┬",
    _NORTH | _EAST | _SOUTH | _WEST: "┼",
}


@dataclass(frozen=True)
class Glyph:
    char: str
    color: RGB
    bold: bool
    priority: float


@dataclass(frozen=True)
class Stroke:
    mask: int
    color: RGB
    bold: bool
    priority: float


class BrailleCanvas:
    """True-color sub-cell canvas with foreground glyph overlays."""

    def __init__(
        self,
        width: int,
        height: int,
        *,
        render_mode: Literal["braille", "quadrant"] = "braille",
    ) -> None:
        if render_mode not in ("braille", "quadrant"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.width = max(1, int(width))
        self.height = max(1, int(height))
        self.render_mode = render_mode
        self.subpixel_width = 2
        self.subpixel_height = 4 if render_mode == "braille" else 2
        self.pixel_width = self.width * self.subpixel_width
        self.pixel_height = self.height * self.subpixel_height
        self.intensity = np.zeros((self.pixel_height, self.pixel_width), dtype=np.float32)
        self.color = np.zeros((self.pixel_height, self.pixel_width, 3), dtype=np.float32)
        self.glyphs: dict[tuple[int, int], Glyph] = {}
        self.strokes: dict[tuple[int, int], Stroke] = {}

    def point(
        self,
        x: float,
        y: float,
        *,
        color: RGB,
        intensity: float = 1.0,
        radius: float = 0.0,
    ) -> None:
        px = int(round(x * self.subpixel_width))
        py = int(round(y * self.subpixel_height))
        pixel_radius = max(0, int(round(radius * 2.0)))
        for dy in range(-pixel_radius, pixel_radius + 1):
            for dx in range(-pixel_radius, pixel_radius + 1):
                y_scale = self.subpixel_width / self.subpixel_height
                distance = math.sqrt(dx * dx + (dy * y_scale) ** 2)
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
        samples = max(
            2,
            int(
                max(
                    abs(bx - ax) * self.subpixel_width,
                    abs(by - ay) * self.subpixel_height,
                )
            )
            + 1,
        )
        for index, value in enumerate(np.linspace(0.0, 1.0, samples)):
            if dashed and (index // 2) % 2:
                continue
            x = ax + (bx - ax) * float(value)
            y = ay + (by - ay) * float(value)
            self._pixel(
                int(round(x * self.subpixel_width)),
                int(round(y * self.subpixel_height)),
                color,
                intensity,
            )

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

    def thin_line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        color: RGB,
        priority: float = 1.0,
        bold: bool = False,
        start: float = 0.0,
        end: float = 1.0,
    ) -> None:
        """Draw a connected one-cell stroke using light box-drawing glyphs."""
        start = min(1.0, max(0.0, float(start)))
        end = min(1.0, max(start, float(end)))
        ax = x0 + (x1 - x0) * start
        ay = y0 + (y1 - y0) * start
        bx = x0 + (x1 - x0) * end
        by = y0 + (y1 - y0) * end
        path = _orthogonal_path(
            int(round(ax)),
            int(round(ay)),
            int(round(bx)),
            int(round(by)),
        )
        self._add_stroke_path(path, color=color, priority=priority, bold=bold)

    def thin_polyline(
        self,
        points: Iterable[tuple[float, float]],
        *,
        color: RGB,
        priority: float = 1.0,
        bold: bool = False,
        closed: bool = False,
    ) -> None:
        values = list(points)
        if len(values) < 2:
            return
        pairs = list(zip(values[:-1], values[1:]))
        if closed:
            pairs.append((values[-1], values[0]))
        for first, second in pairs:
            self.thin_line(
                *first,
                *second,
                color=color,
                priority=priority,
                bold=bold,
            )

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
                stroke = self.strokes.get((cell_x, cell_y))
                if stroke is not None:
                    output.append(
                        _STROKE_CHARS[stroke.mask],
                        Style(color=_rich_color(stroke.color), bold=stroke.bold),
                    )
                    continue
                y0 = cell_y * self.subpixel_height
                x0 = cell_x * self.subpixel_width
                block = self.intensity[
                    y0 : y0 + self.subpixel_height,
                    x0 : x0 + self.subpixel_width,
                ]
                active = block > 0.008
                if not bool(active.any()):
                    output.append(" ")
                    continue
                char = self._subcell_char(active)
                weighted = block[..., None] * self.color[
                    y0 : y0 + self.subpixel_height,
                    x0 : x0 + self.subpixel_width,
                ]
                denominator = max(1e-6, float(block.sum()))
                rgb = np.clip(weighted.sum(axis=(0, 1)) / denominator, 0.0, 255.0)
                strength = min(1.0, max(0.18, float(block.max())))
                visible = tuple(int(round(float(channel) * (0.45 + 0.55 * strength))) for channel in rgb)
                output.append(char, Style(color=_rich_color(visible)))
            if cell_y + 1 < self.height:
                output.append("\n")
        return output

    def _add_stroke_path(
        self,
        path: list[tuple[int, int]],
        *,
        color: RGB,
        priority: float,
        bold: bool,
    ) -> None:
        for first, second in zip(path[:-1], path[1:]):
            dx = second[0] - first[0]
            dy = second[1] - first[1]
            if (dx, dy) == (1, 0):
                first_bit, second_bit = _EAST, _WEST
            elif (dx, dy) == (-1, 0):
                first_bit, second_bit = _WEST, _EAST
            elif (dx, dy) == (0, 1):
                first_bit, second_bit = _SOUTH, _NORTH
            elif (dx, dy) == (0, -1):
                first_bit, second_bit = _NORTH, _SOUTH
            else:
                continue
            self._add_stroke_cell(first, first_bit, color=color, priority=priority, bold=bold)
            self._add_stroke_cell(second, second_bit, color=color, priority=priority, bold=bold)

    def _add_stroke_cell(
        self,
        cell: tuple[int, int],
        bit: int,
        *,
        color: RGB,
        priority: float,
        bold: bool,
    ) -> None:
        if not (0 <= cell[0] < self.width and 0 <= cell[1] < self.height):
            return
        previous = self.strokes.get(cell)
        if previous is None or priority > previous.priority:
            self.strokes[cell] = Stroke(bit, color, bold, priority)
            return
        if priority == previous.priority:
            merged_color = tuple(max(a, b) for a, b in zip(previous.color, color))
            self.strokes[cell] = Stroke(
                previous.mask | bit,
                merged_color,
                previous.bold or bold,
                priority,
            )

    def _subcell_char(self, active: np.ndarray) -> str:
        if self.render_mode == "quadrant":
            mask = (
                int(active[0, 0])
                | (int(active[0, 1]) << 1)
                | (int(active[1, 0]) << 2)
                | (int(active[1, 1]) << 3)
            )
            return _QUADRANT_CHARS[mask]
        code = 0
        for dot_y in range(4):
            for dot_x in range(2):
                if active[dot_y, dot_x]:
                    code |= int(_BRAILLE_BITS[dot_y, dot_x])
        return chr(0x2800 + code)

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


def _orthogonal_path(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    diagonal = _bresenham_path(x0, y0, x1, y1)
    if len(diagonal) < 2:
        return diagonal
    path = [diagonal[0]]
    for target in diagonal[1:]:
        current = path[-1]
        if current[0] != target[0] and current[1] != target[1]:
            horizontal = (target[0], current[1])
            vertical = (current[0], target[1])
            horizontal_error = _line_error(horizontal, (x0, y0), (x1, y1))
            vertical_error = _line_error(vertical, (x0, y0), (x1, y1))
            if horizontal_error == vertical_error:
                intermediate = horizontal if len(path) % 2 else vertical
            else:
                intermediate = horizontal if horizontal_error < vertical_error else vertical
            if intermediate != path[-1]:
                path.append(intermediate)
        if target != path[-1]:
            path.append(target)
    return path


def _bresenham_path(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    while True:
        path.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return path
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def _line_error(
    point: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> int:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    return abs(dy * (point[0] - start[0]) - dx * (point[1] - start[1]))
