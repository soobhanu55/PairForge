"""Render the captured `python main.py` output as a typewriter-style terminal
recording GIF, using the actual real output text (no fabricated content).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_TXT = Path(r"C:\Users\bhanu\AppData\Local\Temp\demo_output.txt")
OUT_GIF = ROOT / "docs" / "demo.gif"

FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_SIZE = 16
PADDING = 24
LINE_SPACING = 6
BG = (13, 17, 23)          # GitHub-dark-style terminal background
FG = (201, 209, 217)       # default text
GREEN = (63, 185, 80)      # command prompt line
YELLOW = (230, 192, 90)    # headers / section markers
CHAR_STEP = 3               # characters revealed per animation frame (typewriter speed)
HOLD_FRAMES_END = 40        # frames to hold on the final, fully-revealed state
MS_PER_FRAME = 18           # base frame duration


def load_lines() -> list[str]:
    prompt = "PS D:\\Quant Projects\\Q1> python main.py"
    body = OUTPUT_TXT.read_text(encoding="utf-8").splitlines()
    return [prompt, ""] + body


def line_color(line: str) -> tuple[int, int, int]:
    if line.startswith("PS "):
        return GREEN
    if line.strip().startswith("[") or line.endswith(":") or "walk-forward windows" in line or "cointegrated pair" in line:
        return YELLOW
    return FG


def measure_canvas(font: ImageFont.FreeTypeFont, lines: list[str]) -> tuple[int, int]:
    tmp = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(tmp)
    max_width = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line if line else " ", font=font)
        max_width = max(max_width, bbox[2] - bbox[0])
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
    total_height = len(lines) * (line_height + LINE_SPACING)
    return max_width + PADDING * 2, total_height + PADDING * 2


def render_frame(font: ImageFont.FreeTypeFont, lines: list[str], visible_chars: int, size: tuple[int, int]) -> Image.Image:
    img = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(img)
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]

    remaining = visible_chars
    y = PADDING
    for line in lines:
        if remaining <= 0:
            break
        shown = line[: min(len(line), remaining)]
        remaining -= len(line) + 1  # +1 for the implicit newline
        draw.text((PADDING, y), shown, font=font, fill=line_color(line))
        y += line_height + LINE_SPACING

    return img


def main() -> None:
    OUT_GIF.parent.mkdir(exist_ok=True)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    lines = load_lines()
    size = measure_canvas(font, lines)

    total_chars = sum(len(line) + 1 for line in lines)
    frames: list[Image.Image] = []
    durations: list[int] = []

    visible = 0
    while visible < total_chars:
        frames.append(render_frame(font, lines, visible, size))
        durations.append(MS_PER_FRAME)
        visible += CHAR_STEP

    final_frame = render_frame(font, lines, total_chars, size)
    for _ in range(HOLD_FRAMES_END):
        frames.append(final_frame)
        durations.append(60)

    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"Wrote {OUT_GIF} ({len(frames)} frames, {size[0]}x{size[1]})")


if __name__ == "__main__":
    main()
