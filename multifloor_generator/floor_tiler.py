"""
Floor Zone Tiler.

Divides each floor into zones for diffusion-based generation.
Each zone contains a subset of dwelling units that fits within
the model's 10-room constraint.
"""

import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

from .config import MAX_ROOMS_PER_ZONE, DEFAULT_GRID_SIZE, MODEL_OUTPUT_SIZE
from .csv_parser import FloorSpec, DwellingUnit
from .core_template import CoreTemplate
from .utils.room_mapper import RoomMapper, RoomSpec
from .utils.coordinate_mapper import CoordinateMapper


class ZoneType(Enum):
    """Types of zones in a floor layout."""
    CORE = "core"
    WING = "wing"
    CORRIDOR = "corridor"


@dataclass
class Zone:
    """
    Represents a zone within a floor layout.

    Each zone corresponds to one diffusion model generation,
    containing up to MAX_ROOMS_PER_ZONE rooms.
    """
    zone_id: str
    zone_type: ZoneType
    grid_position: Tuple[int, int]  # (row, col) in grid
    direction: str                   # Cardinal direction (north, northeast, etc.)
    pixel_bounds: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in output image

    # Room assignments
    rooms: List[RoomSpec] = field(default_factory=list)
    unit_assignments: List[Tuple[str, int]] = field(default_factory=list)  # (unit_type, count)

    # Connections
    connects_to_core: bool = True
    corridor_direction: str = "center"  # Direction where corridor connects

    def room_count(self) -> int:
        """Get total number of rooms in this zone."""
        return len(self.rooms)

    def can_add_unit(self, rooms_count: int) -> bool:
        """Check if a unit with given room count can be added."""
        return self.room_count() + rooms_count <= MAX_ROOMS_PER_ZONE

    def to_json(self) -> str:
        """Convert zone rooms to JSON specification for diffusion model."""
        if not self.rooms:
            return "{}"

        grouped = {}
        for room in self.rooms:
            if room.room_type not in grouped:
                grouped[room.room_type] = {"rooms": []}
            grouped[room.room_type]["rooms"].append(room.to_dict())

        return json.dumps(grouped, indent=2)


class FloorTiler:
    """
    Divides a floor into zones for diffusion-based generation.

    Strategy:
    1. Place core zone at center of grid
    2. Create wing zones around core
    3. Distribute dwelling units to wing zones
    4. Split units into multiple zones if needed (respecting room limit)
    """

    def __init__(
        self,
        floor_spec: FloorSpec,
        grid_size: int = DEFAULT_GRID_SIZE,
        zone_size: int = MODEL_OUTPUT_SIZE
    ):
        """
        Initialize floor tiler.

        Args:
            floor_spec: Floor specification from CSV
            grid_size: Grid dimension (e.g., 3 for 3x3)
            zone_size: Size of each zone in pixels
        """
        self.floor_spec = floor_spec
        self.grid_size = grid_size
        self.zone_size = zone_size
        self.output_size = grid_size * zone_size

        self.coord_mapper = CoordinateMapper(
            floor_spec.floor_area_sf,
            self.output_size
        )
        self.room_mapper = RoomMapper()

        self.zones: List[Zone] = []
        self.core_zone: Optional[Zone] = None
        self.wing_zones: List[Zone] = []

    def create_zones(self) -> List[Zone]:
        """
        Create all zones for the floor.

        Returns:
            List of Zone objects
        """
        self.zones = []
        self.room_mapper.reset_counter()

        # Create core zone at center
        self.core_zone = self._create_core_zone()
        self.zones.append(self.core_zone)

        # Create wing zones
        self.wing_zones = self._create_wing_zones()
        self.zones.extend(self.wing_zones)

        # Distribute units to wing zones
        self._distribute_units()

        return self.zones

    def _create_core_zone(self) -> Zone:
        """Create the central core zone."""
        center = self.grid_size // 2
        bounds = self._get_zone_bounds(center, center)

        return Zone(
            zone_id="core",
            zone_type=ZoneType.CORE,
            grid_position=(center, center),
            direction="center",
            pixel_bounds=bounds,
            connects_to_core=False,  # Core doesn't connect to itself
            corridor_direction="center"
        )

    def _create_wing_zones(self) -> List[Zone]:
        """Create wing zones around the core."""
        wings = []
        center = self.grid_size // 2

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Skip center (core) zone
                if row == center and col == center:
                    continue

                direction = self.coord_mapper.get_cardinal_direction(row, col, self.grid_size)
                bounds = self._get_zone_bounds(row, col)

                # Determine corridor connection direction
                corridor_dir = self._get_corridor_direction(row, col, center)

                zone = Zone(
                    zone_id=f"wing_{direction}",
                    zone_type=ZoneType.WING,
                    grid_position=(row, col),
                    direction=direction,
                    pixel_bounds=bounds,
                    connects_to_core=True,
                    corridor_direction=corridor_dir
                )
                wings.append(zone)

        return wings

    def _get_zone_bounds(self, row: int, col: int) -> Tuple[int, int, int, int]:
        """Get pixel bounds for a zone position."""
        x1 = col * self.zone_size
        y1 = row * self.zone_size
        x2 = x1 + self.zone_size
        y2 = y1 + self.zone_size
        return (x1, y1, x2, y2)

    def _get_corridor_direction(self, row: int, col: int, center: int) -> str:
        """Determine which direction the corridor should face (toward core)."""
        if row < center:
            ns = "south"
        elif row > center:
            ns = "north"
        else:
            ns = ""

        if col < center:
            ew = "east"
        elif col > center:
            ew = "west"
        else:
            ew = ""

        if ns and ew:
            return ns + ew
        elif ns:
            return ns
        elif ew:
            return ew
        else:
            return "center"

    def _distribute_units(self):
        """Distribute dwelling units to wing zones."""
        if not self.floor_spec.dwelling_units:
            return

        # Flatten all units to distribute
        units_to_distribute = []
        for unit in self.floor_spec.dwelling_units:
            for i in range(unit.count):
                units_to_distribute.append((unit, i))

        # Distribute units round-robin to wing zones
        zone_idx = 0
        num_wings = len(self.wing_zones)

        for unit, instance in units_to_distribute:
            unit_id = f"{unit.unit_type}_{instance}"

            # Get rooms for this unit
            rooms = self.room_mapper.unit_to_rooms(
                unit.unit_type,
                unit_id,
                total_sf=unit.sf,
                location=self.wing_zones[zone_idx].direction
            )

            # Try to add to current zone
            attempts = 0
            while attempts < num_wings:
                zone = self.wing_zones[zone_idx]

                if zone.can_add_unit(len(rooms)):
                    zone.rooms.extend(rooms)
                    zone.unit_assignments.append((unit.unit_type, 1))
                    break

                # Move to next zone
                zone_idx = (zone_idx + 1) % num_wings
                attempts += 1

            if attempts == num_wings:
                # All zones full, need to create sub-zones or handle overflow
                # For now, just add to least-full zone
                least_full = min(self.wing_zones, key=lambda z: z.room_count())
                # Truncate rooms to fit
                available = MAX_ROOMS_PER_ZONE - least_full.room_count()
                if available > 0:
                    least_full.rooms.extend(rooms[:available])
                    least_full.unit_assignments.append((unit.unit_type, 1))

            # Advance to next zone for distribution
            zone_idx = (zone_idx + 1) % num_wings

    def get_core_template(self) -> CoreTemplate:
        """Get core template for the core zone."""
        return CoreTemplate.from_core_elements(
            self.floor_spec.core_elements,
            zone_side_ft=self.coord_mapper.get_zone_size_ft(self.grid_size),
            target_size=self.zone_size
        )

    def get_wing_zone_mask(self, zone: Zone) -> "Image":
        """
        Create mask for wing zone generation.

        The mask indicates:
        - White: Buildable area
        - Black: Corridor/boundary (where units cannot be placed)
        """
        from PIL import Image, ImageDraw

        mask = Image.new("L", (self.zone_size, self.zone_size), 255)
        draw = ImageDraw.Draw(mask)

        # Add corridor region based on connection direction
        corridor_width = max(3, int(self.floor_spec.corridor_width_ft / self.coord_mapper.ft_per_pixel))

        center = self.zone_size // 2

        if "north" in zone.corridor_direction or "south" in zone.corridor_direction:
            # Vertical corridor
            x1 = center - corridor_width // 2
            y1 = 0 if "north" in zone.corridor_direction else center
            x2 = center + corridor_width // 2
            y2 = center if "north" in zone.corridor_direction else self.zone_size
            draw.rectangle([x1, y1, x2, y2], fill=128)  # Gray for corridor

        if "east" in zone.corridor_direction or "west" in zone.corridor_direction:
            # Horizontal corridor
            x1 = 0 if "west" in zone.corridor_direction else center
            y1 = center - corridor_width // 2
            x2 = center if "west" in zone.corridor_direction else self.zone_size
            y2 = center + corridor_width // 2
            draw.rectangle([x1, y1, x2, y2], fill=128)

        return mask

    def get_zone_summary(self) -> str:
        """Get a summary of zone assignments."""
        lines = [f"Floor {self.floor_spec.floor_number} Zone Summary"]
        lines.append("=" * 40)

        if self.core_zone:
            lines.append(f"\nCore Zone: {self.core_zone.zone_id}")
            lines.append(f"  Position: {self.core_zone.grid_position}")

        lines.append(f"\nWing Zones ({len(self.wing_zones)}):")
        for zone in self.wing_zones:
            unit_str = ", ".join(f"{t}:{c}" for t, c in zone.unit_assignments)
            lines.append(f"  {zone.zone_id}: {zone.room_count()} rooms ({unit_str})")

        return "\n".join(lines)

    def visualize_grid(self) -> "Image":
        """
        Create a visualization of the zone grid.

        Shows zone positions and unit distributions.
        """
        from PIL import Image, ImageDraw

        cell_size = 100
        img_size = self.grid_size * cell_size
        img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        for zone in self.zones:
            row, col = zone.grid_position
            x1 = col * cell_size + 5
            y1 = row * cell_size + 5
            x2 = (col + 1) * cell_size - 5
            y2 = (row + 1) * cell_size - 5

            # Color based on zone type
            if zone.zone_type == ZoneType.CORE:
                color = (200, 200, 255)  # Light blue
            else:
                # Color intensity based on room count
                intensity = min(255, 150 + zone.room_count() * 10)
                color = (255, 255 - zone.room_count() * 20, 200)

            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(100, 100, 100))

            # Add labels
            label = f"{zone.direction}\n{zone.room_count()}r"
            draw.text((x1 + 10, y1 + 10), label, fill=(0, 0, 0))

        return img


def create_floor_zones(
    floor_spec: FloorSpec,
    grid_size: int = DEFAULT_GRID_SIZE
) -> Tuple[List[Zone], CoreTemplate]:
    """
    Convenience function to create zones for a floor.

    Args:
        floor_spec: Floor specification
        grid_size: Grid dimension

    Returns:
        Tuple of (zones list, core template)
    """
    tiler = FloorTiler(floor_spec, grid_size)
    zones = tiler.create_zones()
    core_template = tiler.get_core_template()

    return zones, core_template


if __name__ == "__main__":
    # Test with sample floor spec
    from .csv_parser import DwellingUnit, CoreElement

    test_floor = FloorSpec(
        floor_number=3,
        floor_area_sf=29743,
        floor_side_ft=172.4,
        dwelling_units=[
            DwellingUnit("Studio", 447, 17.2, 26.1, 10, 1),
            DwellingUnit("1BR", 552, 22.3, 24.7, 25, 1),
            DwellingUnit("2BR", 867, 41.6, 20.8, 12, 2),
        ],
        unit_counts={"Studio": 10, "1BR": 25, "2BR": 12},
        core_elements=[
            CoreElement("elevator_passenger", 306, 5),
            CoreElement("elevator_freight", 252, 2),
            CoreElement("stair", 254, 3),
        ],
        corridor_width_ft=5.0
    )

    tiler = FloorTiler(test_floor)
    zones = tiler.create_zones()

    print(tiler.get_zone_summary())

    # Save visualization
    vis = tiler.visualize_grid()
    vis.save("floor_grid_test.png")
    print("\nSaved floor_grid_test.png")
