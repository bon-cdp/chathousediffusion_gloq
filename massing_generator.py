#!/usr/bin/env python3
"""
Claude-Driven Massing Generator - Architectural Art Edition

Creates beautiful, professional floor plate visualizations.
Focus on aesthetics and clarity over perfect unit counts.
"""

import math
from PIL import Image, ImageDraw, ImageFont
from dataclasses import dataclass
from typing import List, Tuple
import sys

sys.path.insert(0, ".")
from multifloor_generator.csv_parser import parse_building_csv, BuildingSpec


# Professional architectural color palette
COLORS = {
    "Studio": (255, 218, 185),     # Peach/salmon
    "1BR": (176, 196, 222),        # Light steel blue
    "2BR": (152, 251, 152),        # Pale green
    "3BR": (221, 160, 221),        # Plum
    "Core": (119, 136, 153),       # Light slate gray
    "Corridor": (245, 245, 245),   # White smoke
    "Background": (255, 255, 255), # White
    "Grid": (230, 230, 230),       # Light gray for grid
    "Text": (60, 60, 60),          # Dark gray text
    "Accent": (70, 130, 180),      # Steel blue for accents
}


@dataclass
class Unit:
    unit_type: str
    x: float
    y: float
    width: float
    depth: float
    number: str


def create_architectural_floor_plate(building: BuildingSpec, output_path: str = "floor_plate.png"):
    """
    Create a beautiful architectural floor plate drawing.
    """

    # Canvas setup
    floor_ft = building.floor_side_ft
    scale = 4.0  # pixels per foot - higher res
    margin = 100  # pixels

    canvas_w = int(floor_ft * scale) + margin * 2
    canvas_h = int(floor_ft * scale) + margin * 2 + 120  # Extra for title/legend

    img = Image.new("RGB", (canvas_w, canvas_h), COLORS["Background"])
    draw = ImageDraw.Draw(img)

    # Fonts
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        tiny_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 9)
    except:
        title_font = label_font = small_font = tiny_font = ImageFont.load_default()

    def ft_to_px(x, y):
        return (int(x * scale + margin), int(y * scale + margin + 60))

    # --- TITLE BLOCK ---
    draw.text((margin, 20), building.name.upper(), fill=COLORS["Text"], font=title_font)
    draw.text((margin, 50), f"TYPICAL FLOOR PLAN  |  {floor_ft:.0f}' × {floor_ft:.0f}'  |  {building.typical_floor_plate_sf:,.0f} SF",
              fill=COLORS["Accent"], font=small_font)

    # --- FLOOR PLATE OUTLINE ---
    outline = [ft_to_px(0, 0), ft_to_px(floor_ft, floor_ft)]
    draw.rectangle(outline, outline=COLORS["Text"], width=3)

    # --- CORE ELEMENTS (elevators and stairs - no fill, no label) ---
    core_w = 35
    core_h = 30
    core_x = (floor_ft - core_w) / 2
    core_y = (floor_ft - core_h) / 2

    # Elevator bank (5 passenger elevators grouped)
    elev_w = 6
    elev_h = 6
    for i in range(5):
        ex = core_x + 2 + i * 7
        ey = core_y + 4
        er = [ft_to_px(ex, ey), ft_to_px(ex + elev_w, ey + elev_h)]
        draw.rectangle(er, fill=(90, 90, 90), outline=(60, 60, 60), width=1)

    # Freight elevators (2, larger)
    for i in range(2):
        ex = core_x + 2 + i * 9
        ey = core_y + 14
        er = [ft_to_px(ex, ey), ft_to_px(ex + elev_w + 2, ey + elev_h + 2)]
        draw.rectangle(er, fill=(100, 100, 100), outline=(70, 70, 70), width=1)

    # Stairs (3)
    stair_w = 8
    stair_h = 12
    for i in range(3):
        sx = core_x + 20 + i * 10
        sy = core_y + 10
        if sx + stair_w <= core_x + core_w:
            sr = [ft_to_px(sx, sy), ft_to_px(sx + stair_w, sy + stair_h)]
            draw.rectangle(sr, fill=(110, 110, 110), outline=(80, 80, 80), width=1)
            # Stair treads
            for j in range(5):
                ly = sy + 1 + j * 2
                draw.line([ft_to_px(sx + 1, ly)[0], ft_to_px(sx + 1, ly)[1],
                          ft_to_px(sx + stair_w - 1, ly)[0], ft_to_px(sx + stair_w - 1, ly)[1]],
                         fill=(140, 140, 140), width=1)

    # --- UNITS ---
    units = []
    unit_depth = 26  # Standard residential depth
    setback = 5  # Edge setback
    party_wall = 1  # Between units

    # Bounds - units must stay within these
    min_x = setback
    max_x = floor_ft - setback
    min_y = setback
    max_y = floor_ft - setback

    unit_num = {"Studio": 1, "1BR": 1, "2BR": 1, "3BR": 1}

    def place_unit(utype, x, y, w, d):
        """Place unit with bounds checking."""
        # Ensure unit stays within floor plate
        if x + w > max_x:
            w = max_x - x  # Trim to fit
        if y + d > max_y:
            d = max_y - y
        if w < 10 or d < 10:  # Too small, skip
            return False

        num = f"{utype[0]}{unit_num[utype]:02d}"
        unit_num[utype] += 1
        units.append(Unit(utype, x, y, w, d, num))
        return True

    # Calculate usable width for N/S wings
    wing_width = floor_ft - 2 * setback  # 162.5 ft usable

    # --- NORTH WING (top) ---
    x = setback
    y = setback
    door_gap = 5  # Gap for corridor/door access

    # 2BR corner (40' wide)
    place_unit("2BR", x, y, 40, unit_depth)
    x += 41

    # Fill middle with 1BR and Studios
    place_unit("1BR", x, y, 24, unit_depth)
    x += 25
    place_unit("1BR", x, y, 24, unit_depth)
    x += 25
    place_unit("Studio", x, y, 18, unit_depth)
    x += 19

    # Last 1BR before corner - gap on SOUTH for corridor access to 202
    place_unit("1BR", x, y, 24, unit_depth - door_gap)
    x += 25

    # 2BR corner (202) - full depth now, access via gap south of previous unit
    remaining = max_x - x
    place_unit("2BR", x, y, remaining, unit_depth)

    # --- SOUTH WING (bottom) ---
    x = setback
    y = floor_ft - setback - unit_depth

    place_unit("2BR", x, y, 40, unit_depth)
    x += 41
    place_unit("1BR", x, y, 24, unit_depth)
    x += 25
    place_unit("1BR", x, y, 24, unit_depth)
    x += 25
    place_unit("Studio", x, y, 18, unit_depth)
    x += 19

    # Last 1BR before corner - gap on NORTH for corridor access to 204
    place_unit("1BR", x, y + door_gap, 24, unit_depth - door_gap)
    x += 25

    # 2BR corner (204) - full depth, access via gap north of previous unit
    remaining = max_x - x
    place_unit("2BR", x, y, remaining, unit_depth)

    # --- WEST WING (left side, between N and S wings) ---
    x = setback
    y_start = setback + unit_depth + party_wall
    y_end = floor_ft - setback - unit_depth - party_wall
    available_height = y_end - y_start

    y = y_start
    # Fill with units stacked vertically
    place_unit("1BR", x, y, unit_depth, 28)
    y += 29
    place_unit("Studio", x, y, unit_depth, 22)
    y += 23
    place_unit("1BR", x, y, unit_depth, 28)
    y += 29
    # Fill remaining space
    remaining_h = y_end - y
    if remaining_h > 15:
        place_unit("Studio", x, y, unit_depth, remaining_h)

    # --- EAST WING (right side) ---
    x = floor_ft - setback - unit_depth  # Start from right edge minus unit depth
    y = y_start

    place_unit("1BR", x, y, unit_depth, 28)
    y += 29
    place_unit("Studio", x, y, unit_depth, 22)
    y += 23
    place_unit("1BR", x, y, unit_depth, 28)
    y += 29
    remaining_h = y_end - y
    if remaining_h > 15:
        place_unit("Studio", x, y, unit_depth, remaining_h)

    # --- BOH (Back of House) spaces around core ---
    boh_color = (200, 200, 210)  # Light grayish blue for BOH
    boh_outline = (150, 150, 160)

    # Trash/recycling room (north of core)
    trash_x = core_x - 12
    trash_y = core_y - 15
    trash_w = 20
    trash_h = 12
    tr = [ft_to_px(trash_x, trash_y), ft_to_px(trash_x + trash_w, trash_y + trash_h)]
    draw.rectangle(tr, fill=boh_color, outline=boh_outline, width=1)
    draw.text((ft_to_px(trash_x + 3, trash_y + 3)[0], ft_to_px(trash_x, trash_y + 3)[1]),
              "TRASH", fill=(120, 120, 130), font=tiny_font)

    # Electrical room (south of core)
    elec_x = core_x + 10
    elec_y = core_y + core_h + 3
    elec_w = 18
    elec_h = 10
    er = [ft_to_px(elec_x, elec_y), ft_to_px(elec_x + elec_w, elec_y + elec_h)]
    draw.rectangle(er, fill=boh_color, outline=boh_outline, width=1)
    draw.text((ft_to_px(elec_x + 2, elec_y + 2)[0], ft_to_px(elec_x, elec_y + 2)[1]),
              "ELEC", fill=(120, 120, 130), font=tiny_font)

    # Mechanical room (west of core)
    mech_x = core_x - 18
    mech_y = core_y + 5
    mech_w = 14
    mech_h = 18
    mr = [ft_to_px(mech_x, mech_y), ft_to_px(mech_x + mech_w, mech_y + mech_h)]
    draw.rectangle(mr, fill=boh_color, outline=boh_outline, width=1)
    draw.text((ft_to_px(mech_x + 2, mech_y + 6)[0], ft_to_px(mech_x, mech_y + 6)[1]),
              "MECH", fill=(120, 120, 130), font=tiny_font)

    # Storage/janitor (east of core)
    stor_x = core_x + core_w + 4
    stor_y = core_y + 8
    stor_w = 12
    stor_h = 14
    sr = [ft_to_px(stor_x, stor_y), ft_to_px(stor_x + stor_w, stor_y + stor_h)]
    draw.rectangle(sr, fill=boh_color, outline=boh_outline, width=1)
    draw.text((ft_to_px(stor_x + 1, stor_y + 4)[0], ft_to_px(stor_x, stor_y + 4)[1]),
              "STOR", fill=(120, 120, 130), font=tiny_font)

    # Draw all units
    for unit in units:
        rect = [ft_to_px(unit.x, unit.y), ft_to_px(unit.x + unit.width, unit.y + unit.depth)]
        draw.rectangle(rect, fill=COLORS[unit.unit_type], outline=COLORS["Text"], width=1)

        # Unit label
        cx = (rect[0][0] + rect[1][0]) // 2
        cy = (rect[0][1] + rect[1][1]) // 2
        draw.text((cx - 12, cy - 12), unit.unit_type, fill=COLORS["Text"], font=tiny_font)
        draw.text((cx - 8, cy + 2), unit.number, fill=COLORS["Accent"], font=tiny_font)

    # --- DIMENSIONS ---
    dim_y = ft_to_px(0, floor_ft)[1] + 25
    # Horizontal dimension
    draw.line([ft_to_px(0, 0)[0], dim_y, ft_to_px(floor_ft, 0)[0], dim_y], fill=COLORS["Text"], width=1)
    draw.line([ft_to_px(0, 0)[0], dim_y - 5, ft_to_px(0, 0)[0], dim_y + 5], fill=COLORS["Text"], width=1)
    draw.line([ft_to_px(floor_ft, 0)[0], dim_y - 5, ft_to_px(floor_ft, 0)[0], dim_y + 5], fill=COLORS["Text"], width=1)
    draw.text((canvas_w // 2 - 20, dim_y + 8), f"{floor_ft:.0f}'", fill=COLORS["Text"], font=small_font)

    # --- LEGEND ---
    legend_y = canvas_h - 80
    draw.text((margin, legend_y), "LEGEND", fill=COLORS["Text"], font=label_font)

    legend_items = [
        ("Studio", f"{unit_num['Studio']-1}", COLORS["Studio"]),
        ("1BR", f"{unit_num['1BR']-1}", COLORS["1BR"]),
        ("2BR", f"{unit_num['2BR']-1}", COLORS["2BR"]),
        ("BOH", "4", (200, 200, 210)),
    ]

    x_off = margin
    for name, count, color in legend_items:
        draw.rectangle([x_off, legend_y + 25, x_off + 20, legend_y + 40],
                       fill=color, outline=COLORS["Text"])
        draw.text((x_off + 25, legend_y + 25), f"{name}: {count}", fill=COLORS["Text"], font=tiny_font)
        x_off += 100

    # --- NORTH ARROW (outside floor plate, top right) ---
    arrow_x = canvas_w - 50
    arrow_y = 25
    # Elegant circle with arrow
    draw.ellipse([arrow_x - 20, arrow_y, arrow_x + 20, arrow_y + 40], outline=COLORS["Accent"], width=2)
    # Arrow pointing up
    draw.polygon([(arrow_x, arrow_y + 8), (arrow_x - 6, arrow_y + 20), (arrow_x + 6, arrow_y + 20)],
                 fill=COLORS["Accent"])
    draw.rectangle([arrow_x - 2, arrow_y + 18, arrow_x + 2, arrow_y + 32], fill=COLORS["Accent"])
    draw.text((arrow_x - 4, arrow_y + 42), "N", fill=COLORS["Accent"], font=small_font)

    # --- SCALE BAR ---
    scale_x = margin
    scale_y = legend_y + 55
    scale_len_ft = 50  # 50 foot scale bar
    scale_len_px = int(scale_len_ft * scale)

    draw.rectangle([scale_x, scale_y, scale_x + scale_len_px, scale_y + 6], fill=COLORS["Text"])
    draw.rectangle([scale_x, scale_y, scale_x + scale_len_px // 2, scale_y + 6], fill=COLORS["Background"], outline=COLORS["Text"])
    draw.text((scale_x, scale_y + 10), "0", fill=COLORS["Text"], font=tiny_font)
    draw.text((scale_x + scale_len_px // 2 - 8, scale_y + 10), "25'", fill=COLORS["Text"], font=tiny_font)
    draw.text((scale_x + scale_len_px - 12, scale_y + 10), "50'", fill=COLORS["Text"], font=tiny_font)

    # --- STATS ---
    total_units = sum(unit_num[k] - 1 for k in unit_num)
    stats_text = f"Units: {total_units}  |  Target: {building.units_per_floor}/floor  |  {building.stories_above_grade} Stories  |  {building.typical_floor_plate_sf:,.0f} GSF"
    draw.text((scale_x + scale_len_px + 30, scale_y + 2), stats_text, fill=(120, 120, 120), font=tiny_font)

    # --- DRAWING BORDER (subtle frame) ---
    border_margin = 15
    draw.rectangle(
        [border_margin, border_margin, canvas_w - border_margin, canvas_h - border_margin],
        outline=(220, 220, 220), width=1
    )

    # --- SIGNATURE ---
    draw.text((canvas_w - margin - 120, canvas_h - 25),
              "Generated by Claude", fill=(180, 180, 180), font=tiny_font)

    # Save
    img.save(output_path, quality=95)
    print(f"\nSaved: {output_path}")
    print(f"  Canvas: {canvas_w} × {canvas_h} px")
    print(f"  Scale: 1\" = {1/scale * 12:.1f} ft")
    print(f"  Units placed: {total_units}")

    return img


def main():
    print("=" * 60)
    print("ARCHITECTURAL FLOOR PLATE GENERATOR")
    print("=" * 60)

    building = parse_building_csv("gloq_example_floorplan_breakdowns - Sheet1.csv")

    print(f"\nBuilding: {building.name}")
    print(f"Floor: {building.floor_side_ft:.0f}' × {building.floor_side_ft:.0f}' ({building.typical_floor_plate_sf:,.0f} SF)")
    print(f"Target units/floor: {building.units_per_floor}")
    print(f"\nUnit mix:")
    for du in building.dwelling_units:
        print(f"  {du.unit_type}: {du.count} total ({du.width_ft}' × {du.depth_ft}')")

    print("\nGenerating floor plate...")
    create_architectural_floor_plate(building, "floor_plate_canoga.png")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
