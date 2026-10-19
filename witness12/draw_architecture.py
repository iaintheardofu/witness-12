"""Exact architecture figure. Text is drawn, not guessed by an image model."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "architecture.png"

W, H = 1920, 1080
BG = (14, 18, 24)
PANEL = (22, 32, 44)
INK = (232, 238, 244)
MUTED = (154, 170, 186)
GOLD = (232, 176, 74)
TEAL = (78, 163, 168)
RESERVE = (196, 112, 74)
LINE = (64, 86, 104)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        trial = word if not cur else cur + " " + word
        if draw.textlength(trial, font=fnt) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def panel(draw, box, fill, outline=LINE, radius=16):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def center_text(draw, box, text, fnt, fill=INK):
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    heights = [draw.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    gap = 4
    total = sum(heights) + gap * (len(lines) - 1)
    y = y0 + (y1 - y0 - total) / 2
    for line, h in zip(lines, heights):
        tw = draw.textlength(line, font=fnt)
        draw.text((x0 + (x1 - x0 - tw) / 2, y), line, font=fnt, fill=fill)
        y += h + gap


def main() -> None:
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)
    title = font(40, True)
    sub = font(20)
    head = font(18, True)
    body = font(16)
    small = font(14)
    chip_f = font(15, True)
    chip_s = font(13)

    draw.text((64, 28), "WITNESS-12", font=title, fill=GOLD)
    draw.text(
        (430, 42),
        "Twelve AKD1000s. One spacecraft ledger. Speech only when the prediction breaks.",
        font=sub,
        fill=INK,
    )

    panel(draw, (48, 110, 1872, 210), PANEL)
    center_text(
        draw,
        (48, 118, 1872, 202),
        "Synthetic CubeSat bus\nvoltage, battery temperature, wheel, radio, heater command, eclipse, solar current",
        head,
    )

    draw.polygon([(940, 214), (980, 214), (960, 236)], fill=GOLD)

    panel(draw, (48, 244, 1872, 330), PANEL)
    center_text(
        draw,
        (48, 250, 1872, 324),
        "Host encoder\nresidual = observed minus predicted. The chip packet is sparse: magnitude bins and command flags. Quiet is all zeros.",
        head,
    )

    draw.polygon([(940, 334), (980, 334), (960, 356)], fill=GOLD)

    labels = [
        ("0", "Eclipse in", False),
        ("1", "Eclipse out", False),
        ("2", "Wheel desat", False),
        ("3", "Heater, commanded", False),
        ("4", "Radio pass", False),
        ("5", "Payload on", False),
        ("6", "Safe hold", False),
        ("7", "Gyro tumble", False),
        ("8", "Array surge", False),
        ("9", "Coolant swing", False),
        ("10", "Vibration", False),
        ("11", "Reserve\nlearns the new fault", True),
    ]
    left, top, gap = 48, 368, 12
    box_w = (1872 - 48 - gap * 5) // 6
    box_h = 92
    for i, (idx, name, reserve) in enumerate(labels):
        col, row = i % 6, i // 6
        x = left + col * (box_w + gap)
        y = top + row * (box_h + gap)
        fill = (92, 48, 32) if reserve else (18, 58, 66)
        outline = RESERVE if reserve else TEAL
        panel(draw, (x, y, x + box_w, y + box_h), fill, outline)
        center_text(draw, (x + 8, y + 6, x + box_w - 8, y + box_h - 6), f"AKD1000  {idx}\n{name}", chip_f)

    draw.polygon([(940, 572), (980, 572), (960, 594)], fill=GOLD)

    panel(draw, (48, 604, 930, 760), PANEL)
    draw.text((72, 620), "CPU ledger", font=head, fill=GOLD)
    for i, line in enumerate(
        [
            "Predict the next bus state from the last accepted mode.",
            "Store the residual. Assimilate a mode only if one specialist fires.",
            "After the reserved chip takes the new class, the same bus goes quiet.",
        ]
    ):
        draw.text((72, 658 + i * 28), line, font=body, fill=INK)

    panel(draw, (954, 604, 1872, 760), PANEL)
    draw.text((978, 620), "Four sentences, and no others", font=head, fill=GOLD)
    claims = [("OBSERVED", TEAL), ("PREDICTED", TEAL), ("RESIDUAL", GOLD), ("HYPOTHESIS", RESERVE)]
    for i, (name, color) in enumerate(claims):
        x = 978 + (i % 2) * 430
        y = 668 + (i // 2) * 40
        draw.rounded_rectangle((x, y, x + 400, y + 32), radius=8, fill=(28, 40, 54), outline=color)
        draw.text((x + 16, y + 6), name, font=chip_f, fill=color)

    draw.text(
        (64, 788),
        "The reserved chip is fit after it is mapped. Eight windows. The label names those windows. It is not a diagnosed cause.",
        font=body,
        fill=INK,
    )
    draw.text(
        (64, 824),
        "Chips that do not separate their class stay in the table as misses. The report does not send a command.",
        font=body,
        fill=INK,
    )

    panel(draw, (48, 870, 1872, 1024), (18, 24, 32))
    notes = [
        "Not a language model. The sentences are filled from the ledger. A number that is not in the ledger cannot be printed.",
        "Not a watt claim. The on-chip power sensor is not read. The Razer Core X V2 is on USB. The RTX is not on the PCI bus, so it is not in this result.",
        "Not the first on-chip class. BrainChip already teaches a last-layer class on one AKD1000. Two-board keyword federation has been published.",
        "The open measurement is the crew: twelve resident specialists, one ledger, one new lookalike fault, then silence.",
    ]
    y = 888
    for line in notes:
        draw.text((72, y), line, font=small, fill=MUTED)
        y += 32

    img.save(OUT, "PNG")
    print(OUT)


if __name__ == "__main__":
    main()
