"""Render the frozen Phase 3.5 development comparisons from public evidence."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SOURCE = Path("docs/evidence/phase_3_5_results.json")
OUTPUT = Path("docs/evidence/phase_3_5_screening.png")
COLORS = ["#314f7e", "#18a58e", "#f3a953", "#7856a6", "#b84c6d", "#5a9a4d", "#48a9cc"]
INK, MUTED, GRID = "#17243b", "#51617a", "#dce3ed"


def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def label(draw, xy, text, *, size=18, color=INK, bold=False, anchor=None):
    draw.text(xy, str(text), font=font(size, bold), fill=color, anchor=anchor)


def panel(draw, x, y, title):
    draw.rounded_rectangle((x, y, x + 770, y + 370), radius=14, fill="white", outline=GRID, width=2)
    label(draw, (x + 24, y + 20), title, size=22, bold=True)
    return x + 50, y + 85, 675, 230


def bars(draw, box, names, series, maximum, fmt, *, legend=None):
    x, y, width, height = box
    count = len(names)
    groups = len(series)
    base = y + height
    draw.line((x, base, x + width, base), fill=MUTED, width=2)
    draw.line((x, y, x, base), fill=MUTED, width=2)
    for t in range(1, 5):
        yy = base - height * t / 4
        draw.line((x, yy, x + width, yy), fill=GRID, width=1)
        label(draw, (x - 8, yy), fmt(maximum * t / 4), size=13, color=MUTED, anchor="rm")
    stride = width / count
    bw = min(35, stride * 0.65 / groups)
    for i, name in enumerate(names):
        cx = x + stride * (i + 0.5)
        for j, (values, color) in enumerate(series):
            value = max(0, values[i])
            h = height * value / maximum if maximum else 0
            left = cx + (j - (groups - 1) / 2) * (bw + 3) - bw / 2
            draw.rectangle((left, base - h, left + bw, base), fill=color)
        label(draw, (cx, base + 12), name, size=14, color=MUTED, anchor="mt")
    if legend:
        left = x + 3
        for text, color in legend:
            draw.rectangle((left, y - 25, left + 12, y - 13), fill=color)
            label(draw, (left + 17, y - 29), text, size=14, color=MUTED)
            left += 170


def render():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = data["screening_runs"]
    if len(rows) != 6:
        raise ValueError("Expected the six frozen screening arms")
    names = [
        f"{r['seed']} "
        + {"normal": "N", "expanded_5": "5%", "expanded_10": "10%"}[r["candidate_id"]]
        for r in rows
    ]
    image = Image.new("RGB", (1600, 1720), "#f5f7fb")
    draw = ImageDraw.Draw(image)
    label(draw, (35, 25), "SIRFEX Phase 3.5 | Development screening", size=32, bold=True)
    label(
        draw,
        (35, 69),
        "Matched seeds 3 and 17; 51,200 PPO interactions per arm. Final holdouts unopened.",
        size=19,
        color=MUTED,
    )
    positions = [(25 + 790 * (i % 2), 110 + 390 * (i // 2)) for i in range(8)]

    box = panel(draw, *positions[0], "1. Target versus realized stress exposure")
    targets = [(r["target_stress_interaction_fraction"] or 0) * 100 for r in rows]
    realized = [r["exposure"]["stress_interaction_fraction"] * 100 for r in rows]
    bars(
        draw,
        box,
        names,
        [(targets, COLORS[0]), (realized, COLORS[1])],
        12,
        lambda x: f"{x:.0f}%",
        legend=[("Target", COLORS[0]), ("Realized", COLORS[1])],
    )

    box = panel(draw, *positions[1], "2. Realized stress interaction fraction")
    bars(draw, box, names, [(realized, COLORS[1])], 12, lambda x: f"{x:.0f}%")

    x, y, width, height = panel(draw, *positions[2], "3. Stress interactions per discovery")
    stressed = [r for r in rows if r["candidate_id"] != "normal"]
    discoveries = sorted(
        {key for r in stressed for key in r["exposure"]["per_discovery_interactions"]}
    )
    if len(discoveries) != 7:
        raise ValueError("Expected seven discovered scenarios")
    max_total = max(r["exposure"]["stress_interactions"] for r in stressed)
    base, stride = y + height, width / len(stressed)
    draw.line((x, base, x + width, base), fill=MUTED, width=2)
    for i, row in enumerate(stressed):
        left = x + stride * (i + 0.5) - 42
        cursor = base
        for j, discovery in enumerate(discoveries):
            value = row["exposure"]["per_discovery_interactions"].get(discovery, 0)
            h = height * value / (max_total * 1.05)
            draw.rectangle((left, cursor - h, left + 84, cursor), fill=COLORS[j])
            cursor -= h
        label(
            draw,
            (left + 42, base + 12),
            f"{row['seed']} {row['candidate_id'][-2:]}%",
            size=14,
            color=MUTED,
            anchor="mt",
        )
    for j in range(7):
        lx = x + j * 96
        draw.rectangle((lx, y - 23, lx + 11, y - 11), fill=COLORS[j])
        label(draw, (lx + 16, y - 27), f"D{j + 1}", size=13, color=MUTED)
    label(draw, (x + width, y + 5), f"max {max_total:,}", size=14, color=MUTED, anchor="ra")

    box = panel(draw, *positions[3], "4. Nominal MAE (lower is better)")
    bars(
        draw,
        box,
        names,
        [([r["development_nominal"]["mean_tracking_error"] for r in rows], COLORS[2])],
        0.06,
        lambda x: f"{x:.3f}",
    )

    box = panel(draw, *positions[4], "5. Development feasible-stress MAE")
    bars(
        draw,
        box,
        names,
        [([r["development_stress"]["mean_tracking_error"] for r in rows], COLORS[3])],
        0.14,
        lambda x: f"{x:.2f}",
    )

    box = panel(draw, *positions[5], "6. Physical violation frequency")
    values = [r["development_stress"]["constraint_violation_frequency"] for r in rows]
    bars(draw, box, names, [(values, COLORS[4])], 1, lambda x: f"{x:.1f}")
    label(
        draw,
        (box[0] + box[2] / 2, box[1] + 80),
        "0 in every development stress arm",
        size=19,
        bold=True,
        anchor="mm",
    )

    x, y, width, height = panel(draw, *positions[6], "7. Tracking MAE by physical difficulty")
    strata = ["nominal_like", "moderate", "high", "near_boundary"]
    cw = width / 4
    rh = height / 6
    for j, s in enumerate(strata):
        label(
            draw,
            (x + j * cw + cw / 2, y - 7),
            s.replace("_", " "),
            size=13,
            anchor="ms",
            color=MUTED,
        )
    for i, row in enumerate(rows):
        for j, s in enumerate(strata):
            v = row["development_stress"]["difficulty_strata"][s]["mean_tracking_error"]
            intensity = min(1, v / 0.20)
            color = (
                int(225 - 120 * intensity),
                int(246 - 145 * intensity),
                int(242 - 120 * intensity),
            )
            left, top = x + j * cw + 3, y + i * rh + 2
            draw.rounded_rectangle((left, top, left + cw - 6, top + rh - 4), radius=3, fill=color)
            label(draw, (left + cw / 2 - 3, top + rh / 2 - 2), f"{v:.3f}", size=14, anchor="mm")
        label(draw, (x - 8, y + (i + 0.5) * rh), names[i], size=12, color=MUTED, anchor="rm")

    box = panel(draw, *positions[7], "8. Physical command and action saturation")
    actions = [r["development_stress"]["action_statistics"] for r in rows]
    means = [a["mean"] for a in actions]
    saturation = [a["lower_bound_fraction"] + a["upper_bound_fraction"] for a in actions]
    bars(
        draw,
        box,
        names,
        [(means, COLORS[0]), (saturation, COLORS[4])],
        1,
        lambda x: f"{x:.2f}",
        legend=[("Mean physical command", COLORS[0]), ("Bound fraction", COLORS[4])],
    )
    label(
        draw,
        (40, 1680),
        "D1–D7 are the seven frozen discovery IDs; exact mapping and counts are in the public JSON."
        " Seed is the independent policy replicate.",
        size=16,
        color=MUTED,
    )
    image.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    render()
