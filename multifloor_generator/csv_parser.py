"""
CSV Parser for Building Program Data.

Parses the detailed building specification CSV into structured data classes
for use in the multi-floor generator pipeline.
"""

import csv
import re
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path


@dataclass
class DwellingUnit:
    """Specification for a dwelling unit type."""
    unit_type: str           # "Studio", "1BR", "2BR", "3BR"
    sf: float               # Total square footage
    width_ft: float         # Width in feet
    depth_ft: float         # Depth in feet
    count: int              # Number of units of this type
    bathroom_count: int = 1  # Number of bathrooms

    @property
    def rooms_count(self) -> int:
        """Estimate number of rooms for this unit type."""
        room_counts = {
            "Studio": 4,  # Bedroom, Kitchen, Bathroom, Entry
            "1BR": 5,     # Living, Bedroom, Kitchen, Bathroom, Entry
            "2BR": 7,     # Living, 2 Beds, Kitchen, 2 Baths, Entry
            "3BR": 8,     # Living, 3 Beds, Kitchen, 2 Baths, Entry
        }
        return room_counts.get(self.unit_type, 5)


@dataclass
class CoreElement:
    """Specification for a vertical core element."""
    element_type: str       # "elevator_passenger", "elevator_freight", "stair", "vestibule"
    sf_per_floor: float     # Square feet per floor
    count: int              # Number of this element
    total_sf: float = 0.0   # Total SF across all floors


@dataclass
class CirculationSpec:
    """Circulation specifications."""
    corridor_width_in: float        # Corridor width in inches
    corridor_sf: float              # Total corridor area
    elevator_lobby_sf: float        # Elevator lobby area
    stair_vestibule_sf_per_floor: float  # Stair vestibule per floor


@dataclass
class FloorSpec:
    """Specification for a single floor."""
    floor_number: int
    floor_area_sf: float            # Total floor area
    floor_side_ft: float            # Side length (sqrt of area for square)
    dwelling_units: List[DwellingUnit] = field(default_factory=list)
    unit_counts: Dict[str, int] = field(default_factory=dict)
    core_elements: List[CoreElement] = field(default_factory=list)
    corridor_width_ft: float = 5.0  # Default 60 inches
    is_typical: bool = True         # Is this a typical residential floor


@dataclass
class BuildingSpec:
    """Complete building specification."""
    name: str
    address: str

    # Overall dimensions
    lot_size_sf: float
    far: float
    gfa_sf: float
    gba_sf: float

    # Stories
    stories_above_grade: int
    stories_below_grade: int
    stories_total: int

    # Floor plate
    typical_floor_plate_sf: float
    floor_side_ft: float            # Computed from floor plate

    # Dwelling units
    total_units: int
    units_per_floor: int
    dwelling_units: List[DwellingUnit] = field(default_factory=list)

    # Core elements
    core_elements: List[CoreElement] = field(default_factory=list)

    # Circulation
    circulation: Optional[CirculationSpec] = None

    # Per-floor specs
    floor_specs: List[FloorSpec] = field(default_factory=list)


def parse_sf_value(value: str) -> float:
    """Parse a square footage value from CSV (handles commas, 'SF' suffix)."""
    if not value or value == "-":
        return 0.0
    # Remove SF suffix and commas
    cleaned = re.sub(r'[,\s]*(SF|sf)?', '', str(value))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_dimension(value: str) -> float:
    """Parse a dimension value (handles 'FT', 'IN' suffixes)."""
    if not value or value == "-":
        return 0.0
    cleaned = re.sub(r'[,\s]*(FT|ft|IN|in)?', '', str(value))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_int_value(value: str) -> int:
    """Parse an integer value from CSV."""
    if not value or value == "-":
        return 0
    cleaned = re.sub(r'[,\s]', '', str(value))
    try:
        return int(float(cleaned))
    except ValueError:
        return 0


def find_value_in_rows(
    rows: List[List[str]],
    search_term: str,
    value_col: int = 6,
    start_row: int = 0
) -> Tuple[Optional[str], int]:
    """
    Find a value in the CSV rows by searching for a term.

    Returns (value, row_index) or (None, -1) if not found.
    """
    search_lower = search_term.lower()

    for i, row in enumerate(rows[start_row:], start=start_row):
        # Search in first 8 columns for the term
        for j, cell in enumerate(row[:8]):
            cell_str = str(cell).strip().lower()

            # Skip if the cell contains multiple different terms (likely a header)
            # e.g., "Gross building area (GBA) & gross floor area (GFA) analysis"
            if '&' in cell_str or 'analysis' in cell_str:
                continue

            # Match if the search term is a substantial part of the cell
            # (not just a substring of a much longer text)
            if search_lower in cell_str:
                # Verify this is the right cell - should be close in length
                # or the cell should start with the term
                if len(cell_str) < len(search_lower) * 2 or cell_str.startswith(search_lower):
                    # Found the search term - now find the value
                    # Try columns in order of likelihood
                    for vc in [6, 7, 5, 8, 9, 10, 11, 12]:
                        if vc < len(row):
                            val = str(row[vc]).strip()
                            # Check if this looks like a numeric value
                            if val and val != "-" and val.lower() != "true" and val.lower() != "false":
                                # Check if it contains digits (likely a value)
                                if any(c.isdigit() for c in val):
                                    return val, i
                    # If no numeric value found, return the specified column
                    if value_col < len(row):
                        return row[value_col], i
    return None, -1


def find_value_in_row_after_label(
    rows: List[List[str]],
    search_term: str,
    start_row: int = 0
) -> Tuple[Optional[str], int]:
    """
    Find a value in the same row after the label column.

    Specifically for circulation data where values are at column 10.
    """
    search_lower = search_term.lower()

    for i, row in enumerate(rows[start_row:], start=start_row):
        for j, cell in enumerate(row[:8]):
            if search_lower in str(cell).lower():
                # Search remaining columns for a value with SF
                for vc in range(j + 1, min(len(row), 20)):
                    val = str(row[vc]).strip()
                    if val and 'SF' in val and any(c.isdigit() for c in val):
                        return val, i
                # Try columns 9-12 specifically (common for circulation data)
                for vc in [10, 9, 11, 12]:
                    if vc < len(row):
                        val = str(row[vc]).strip()
                        if val and val != "-" and any(c.isdigit() for c in val):
                            return val, i
    return None, -1


def parse_building_csv(csv_path: str, building_index: int = 0) -> BuildingSpec:
    """
    Parse building specification CSV file.

    Args:
        csv_path: Path to CSV file
        building_index: Index of building to parse (0 for first building)

    Returns:
        BuildingSpec object with parsed data
    """
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Find the start of the target building (look for building headers)
    # Buildings are separated by project names at the start of sections
    building_starts = []
    for i, row in enumerate(rows):
        # Look for rows that start a new building section
        if row and row[0] and not row[0].startswith(','):
            first_cell = str(row[0]).strip()
            # Check if it looks like a building/project identifier
            if first_cell and (
                'canoga' in first_cell.lower() or
                'cloverfield' in first_cell.lower() or
                first_cell[0].isdigit()  # Starts with number like "7."
            ):
                building_starts.append(i)

    # Determine row range for target building
    start_row = building_starts[building_index] if building_index < len(building_starts) else 0
    end_row = building_starts[building_index + 1] if building_index + 1 < len(building_starts) else len(rows)

    # Subset rows to target building
    building_rows = rows[start_row:end_row]

    # Find project name (first non-empty cell in first row)
    name = ""
    for cell in building_rows[0]:
        if cell and cell.strip():
            name = cell.strip()
            # Clean up name
            if name.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                name = name[2:].strip()
            break

    # Parse building overview from subset
    lot_size_val, _ = find_value_in_rows(building_rows, "Lot Size")
    far_val, _ = find_value_in_rows(building_rows, "FAR")
    gfa_val, _ = find_value_in_rows(building_rows, "Gross Floor Area (GFA)")
    gba_val, _ = find_value_in_rows(building_rows, "Gross Building Area (GBA)")
    stories_val, _ = find_value_in_rows(building_rows, "Stories")
    floor_plate_val, _ = find_value_in_rows(building_rows, "Typical floor plate")

    lot_size = parse_sf_value(lot_size_val)
    try:
        far = float(far_val) if far_val else 0.0
    except ValueError:
        far = 0.0
    gfa = parse_sf_value(gfa_val)
    gba = parse_sf_value(gba_val)
    stories_total = parse_int_value(stories_val)
    typical_floor_plate = parse_sf_value(floor_plate_val)

    # Parse story breakdown
    above_grade_val, _ = find_value_in_rows(building_rows, "Stories (Above Grade)")
    below_grade_val, _ = find_value_in_rows(building_rows, "Stories (Below Grade)")
    stories_above = parse_int_value(above_grade_val)
    stories_below = parse_int_value(below_grade_val)

    # If stories_above is 0 but total is set, assume all above grade
    if stories_above == 0 and stories_total > 0:
        stories_above = stories_total - stories_below

    # Calculate floor side (assume square)
    floor_side = math.sqrt(typical_floor_plate) if typical_floor_plate > 0 else 0

    # Parse dwelling unit counts
    total_units_val, _ = find_value_in_rows(building_rows, "Qty of dwelling unit")
    units_per_floor_val, _ = find_value_in_rows(building_rows, "dwelling unit/floor")
    total_units = parse_int_value(total_units_val)
    units_per_floor = parse_int_value(units_per_floor_val)

    # Parse unit types - ONLY from building_rows
    dwelling_units = []

    # Find unit type rows - look for "Studio + 1 Bath", "1 Bedroom + 1 Bath", etc.
    for i, row in enumerate(building_rows):
        row_text = ' '.join(str(c) for c in row[:3]).lower()

        # Skip rows that don't look like unit definitions
        # (must have bath info and either typical or TRUE in row)
        has_bath = 'bath' in row_text
        row_full = ' '.join(str(c) for c in row[:15]).lower()
        has_data_marker = 'typical' in row_text or 'true' in row_full

        if not has_bath or not has_data_marker:
            continue

        # Skip rows with powder (these are variants, not main types)
        if 'powder' in row_text:
            continue

        # Studio
        if 'studio' in row_text and '1 bath' in row_text:
            unit = _parse_unit_row(row, "Studio", 1)
            if unit and unit.count > 0:
                dwelling_units.append(unit)

        # 1 Bedroom
        elif '1 bedroom' in row_text and '1 bath' in row_text:
            unit = _parse_unit_row(row, "1BR", 1)
            if unit and unit.count > 0:
                dwelling_units.append(unit)

        # 2 Bedroom (check for both 1 bath and 2 bath variants)
        elif '2 bedroom' in row_text and ('2 bath' in row_text or '1 bath' in row_text):
            bath_count = 2 if '2 bath' in row_text else 1
            unit = _parse_unit_row(row, "2BR", bath_count)
            if unit and unit.count > 0:
                dwelling_units.append(unit)

        # 3 Bedroom
        elif '3 bedroom' in row_text and '2 bath' in row_text:
            unit = _parse_unit_row(row, "3BR", 2)
            if unit and unit.count > 0:
                dwelling_units.append(unit)

    # Parse core elements (circulation)
    core_elements = []

    # Passenger elevators
    elev_pass_qty, _ = find_value_in_rows(building_rows, "Qty of Elevator - Passenger")
    elev_pass_sf, _ = find_value_in_rows(building_rows, "Elevator -Passenger SF/floor")
    if elev_pass_qty or elev_pass_sf:
        core_elements.append(CoreElement(
            element_type="elevator_passenger",
            sf_per_floor=parse_sf_value(elev_pass_sf) or 306,  # Default
            count=parse_int_value(elev_pass_qty) or 5  # Default
        ))

    # Freight elevators
    elev_freight_qty, _ = find_value_in_rows(building_rows, "Qty of Elevator - Freight")
    elev_freight_sf, _ = find_value_in_rows(building_rows, "Elevator - Freight SF/floor")
    if elev_freight_qty or elev_freight_sf:
        core_elements.append(CoreElement(
            element_type="elevator_freight",
            sf_per_floor=parse_sf_value(elev_freight_sf) or 252,  # Default
            count=parse_int_value(elev_freight_qty) or 2  # Default
        ))

    # Stairs
    stair_qty, _ = find_value_in_rows(building_rows, "Qty of Stair")
    stair_sf, _ = find_value_in_rows(building_rows, "Stair SF/floor")
    if stair_qty or stair_sf:
        core_elements.append(CoreElement(
            element_type="stair",
            sf_per_floor=parse_sf_value(stair_sf) or 254,  # Default
            count=parse_int_value(stair_qty) or 3  # Default
        ))

    # Stair vestibules
    vestibule_sf, _ = find_value_in_rows(building_rows, "Vestibule - Stair SF/FL")
    if vestibule_sf or stair_qty:
        core_elements.append(CoreElement(
            element_type="vestibule",
            sf_per_floor=parse_sf_value(vestibule_sf) or 120,  # Default
            count=parse_int_value(stair_qty) if stair_qty else 3
        ))

    # Parse circulation specs
    corridor_width_val, _ = find_value_in_rows(building_rows, "Residential corridor width")
    # For corridor SF and lobby SF, use the specialized function that searches further right
    corridor_sf_val, _ = find_value_in_row_after_label(building_rows, "Corridor - Residential")
    elev_lobby_val, _ = find_value_in_row_after_label(building_rows, "Vestibule - Elevator Lobby")

    circulation = CirculationSpec(
        corridor_width_in=parse_dimension(corridor_width_val) or 60,
        corridor_sf=parse_sf_value(corridor_sf_val),
        elevator_lobby_sf=parse_sf_value(elev_lobby_val),
        stair_vestibule_sf_per_floor=parse_sf_value(vestibule_sf) if vestibule_sf else 120
    )

    # Create building spec
    building = BuildingSpec(
        name=name,
        address=name,  # Use name as address
        lot_size_sf=lot_size,
        far=far,
        gfa_sf=gfa,
        gba_sf=gba,
        stories_above_grade=stories_above,
        stories_below_grade=stories_below,
        stories_total=stories_total,
        typical_floor_plate_sf=typical_floor_plate,
        floor_side_ft=floor_side,
        total_units=total_units,
        units_per_floor=units_per_floor,
        dwelling_units=dwelling_units,
        core_elements=core_elements,
        circulation=circulation
    )

    # Generate floor specs
    building.floor_specs = distribute_units_to_floors(building)

    return building


def _parse_unit_row(row: List[str], unit_type: str, bath_count: int) -> Optional[DwellingUnit]:
    """Parse a dwelling unit row from CSV."""
    try:
        # This CSV has a specific structure for unit rows
        # Look for: Has?, % SF, NSF, Unit Count, Room Size (SF), Width (FT), Depth (FT)

        count = 0
        sf = 0
        width = 0
        depth = 0

        # Scan through row for specific patterns
        for i, cell in enumerate(row):
            cell_str = str(cell).strip()

            # Look for unit count (integer in a reasonable range)
            if cell_str.isdigit():
                val = int(cell_str)
                if 1 <= val <= 500 and count == 0:
                    count = val

            # Look for SF values
            if 'SF' in cell_str:
                sf_val = parse_sf_value(cell_str)
                if sf_val > 0:
                    # Smaller SF values are individual unit sizes
                    if sf_val < 2000 and sf == 0:
                        sf = sf_val
                    elif sf_val >= 2000:
                        # This is likely total NSF, skip
                        pass

            # Look for FT dimensions
            if 'FT' in cell_str:
                ft_val = parse_dimension(cell_str)
                if ft_val > 0:
                    if width == 0:
                        width = ft_val
                    elif depth == 0:
                        depth = ft_val

        # Alternative: look at specific column positions (based on CSV structure)
        # The CSV has columns roughly at positions:
        # Col 9: Has?
        # Col 10: % SF
        # Col 12: NSF total
        # Col 15: Unit Count
        # Col 17: Room Size (SF)
        # Col 19: Width (FT)
        # Col 21: Depth (FT)

        if count == 0 and len(row) > 15:
            for idx in [15, 14, 16]:
                if idx < len(row) and str(row[idx]).strip().isdigit():
                    count = int(row[idx])
                    if count > 0:
                        break

        if sf == 0 and len(row) > 17:
            for idx in [17, 18, 19]:
                if idx < len(row):
                    sf = parse_sf_value(str(row[idx]))
                    if sf > 0 and sf < 2000:
                        break

        if width == 0 and len(row) > 19:
            for idx in [19, 20, 21]:
                if idx < len(row):
                    width = parse_dimension(str(row[idx]))
                    if width > 0:
                        break

        if depth == 0 and len(row) > 21:
            for idx in [21, 22, 23]:
                if idx < len(row):
                    depth = parse_dimension(str(row[idx]))
                    if depth > 0:
                        break

        if count > 0:
            return DwellingUnit(
                unit_type=unit_type,
                sf=sf,
                width_ft=width,
                depth_ft=depth,
                count=count,
                bathroom_count=bath_count
            )

    except Exception as e:
        print(f"Warning: Failed to parse unit row for {unit_type}: {e}")

    return None


def distribute_units_to_floors(building: BuildingSpec) -> List[FloorSpec]:
    """
    Distribute dwelling units across floors.

    Creates FloorSpec objects for each floor with proportional unit distribution.
    """
    floor_specs = []
    num_res_floors = building.stories_above_grade

    if num_res_floors <= 0 or not building.dwelling_units:
        return floor_specs

    # Calculate units per floor for each type
    units_per_floor_by_type = {}
    for unit in building.dwelling_units:
        per_floor = unit.count / num_res_floors
        units_per_floor_by_type[unit.unit_type] = round(per_floor)

    corridor_width_ft = 5.0  # Default 60 inches
    if building.circulation:
        corridor_width_ft = building.circulation.corridor_width_in / 12.0

    # Create floor specs
    for floor_num in range(1, num_res_floors + 1):
        floor_units = []
        unit_counts = {}

        for unit in building.dwelling_units:
            # Get count for this floor (handle rounding)
            floor_count = units_per_floor_by_type.get(unit.unit_type, 0)

            # Adjust for last floor to match totals
            if floor_num == num_res_floors:
                already_assigned = floor_count * (num_res_floors - 1)
                floor_count = unit.count - already_assigned

            if floor_count > 0:
                floor_unit = DwellingUnit(
                    unit_type=unit.unit_type,
                    sf=unit.sf,
                    width_ft=unit.width_ft,
                    depth_ft=unit.depth_ft,
                    count=floor_count,
                    bathroom_count=unit.bathroom_count
                )
                floor_units.append(floor_unit)
                unit_counts[unit.unit_type] = floor_count

        floor_specs.append(FloorSpec(
            floor_number=floor_num,
            floor_area_sf=building.typical_floor_plate_sf,
            floor_side_ft=building.floor_side_ft,
            dwelling_units=floor_units,
            unit_counts=unit_counts,
            core_elements=building.core_elements.copy(),
            corridor_width_ft=corridor_width_ft,
            is_typical=True
        ))

    return floor_specs


def print_building_summary(building: BuildingSpec):
    """Print a summary of the building specification."""
    print(f"\n{'='*60}")
    print(f"Building: {building.name}")
    print(f"{'='*60}")
    print(f"\nOverall Metrics:")
    print(f"  Lot Size: {building.lot_size_sf:,.0f} SF")
    print(f"  FAR: {building.far}")
    print(f"  GFA: {building.gfa_sf:,.0f} SF")
    print(f"  GBA: {building.gba_sf:,.0f} SF")
    print(f"\nStories:")
    print(f"  Above Grade: {building.stories_above_grade}")
    print(f"  Below Grade: {building.stories_below_grade}")
    print(f"  Total: {building.stories_total}")
    print(f"\nFloor Plate:")
    print(f"  Typical: {building.typical_floor_plate_sf:,.0f} SF")
    print(f"  Side (square): {building.floor_side_ft:.1f} ft")
    print(f"\nDwelling Units:")
    print(f"  Total: {building.total_units}")
    print(f"  Per Floor: {building.units_per_floor}")
    for unit in building.dwelling_units:
        print(f"  {unit.unit_type}: {unit.count} units @ {unit.sf:.0f} SF ({unit.width_ft:.1f}' x {unit.depth_ft:.1f}')")
    print(f"\nCore Elements:")
    for elem in building.core_elements:
        print(f"  {elem.element_type}: {elem.count} x {elem.sf_per_floor:.0f} SF/floor")
    if building.circulation:
        print(f"\nCirculation:")
        print(f"  Corridor Width: {building.circulation.corridor_width_in:.0f} in")
        print(f"  Corridor SF: {building.circulation.corridor_sf:,.0f} SF")


if __name__ == "__main__":
    # Test with example CSV
    import sys

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # Default to expected location
        csv_path = "../gloq_example_floorplan_breakdowns - Sheet1.csv"

    try:
        building = parse_building_csv(csv_path)
        print_building_summary(building)
    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
    except Exception as e:
        print(f"Error parsing CSV: {e}")
        raise
