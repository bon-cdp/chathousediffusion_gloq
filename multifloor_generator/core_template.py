"""
Core Zone Template Generator.

Creates fixed templates for building core elements (elevators, stairs, lobbies)
to ensure vertical alignment across all floors.
"""

import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw

from .config import CORE_ELEMENTS, ROOM_COLORS, MODEL_OUTPUT_SIZE
from .csv_parser import CoreElement


@dataclass
class CoreElementPlacement:
    """Placement information for a core element."""
    element_type: str
    position: Tuple[int, int]  # (x, y) center position in pixels
    size: Tuple[int, int]      # (width, height) in pixels
    room_type: str             # Model room type for JSON
    room_size: str             # Size category (XS, S, M, L, XL)
    name: str                  # Unique name for room spec
    color: Tuple[int, int, int]  # RGB color for visualization


@dataclass
class CoreTemplate:
    """
    Template for building core zone.

    The core zone contains all vertical circulation elements and is
    rendered identically on each floor to ensure alignment.
    """
    size: int = MODEL_OUTPUT_SIZE  # 64x64 pixels
    placements: List[CoreElementPlacement] = field(default_factory=list)
    corridor_pixels: List[Tuple[int, int, int, int]] = field(default_factory=list)  # Corridor regions

    @classmethod
    def from_core_elements(
        cls,
        core_elements: List[CoreElement],
        zone_side_ft: float,
        target_size: int = MODEL_OUTPUT_SIZE
    ) -> "CoreTemplate":
        """
        Create core template from building core elements.

        Args:
            core_elements: List of CoreElement from building spec
            zone_side_ft: Size of the zone in feet
            target_size: Output image size in pixels
        """
        template = cls(size=target_size)
        ft_per_pixel = zone_side_ft / target_size

        # Calculate total core area to determine layout
        total_elements = sum(e.count for e in core_elements)

        # Layout strategy: arrange elements in a grid pattern around central lobby
        center = target_size // 2

        # Place elevator lobby at center
        lobby_size_px = max(8, int(15 / ft_per_pixel))  # ~15 ft lobby
        template.placements.append(CoreElementPlacement(
            element_type="lobby",
            position=(center, center),
            size=(lobby_size_px, lobby_size_px),
            room_type="LivingRoom",
            room_size="M",
            name="ElevatorLobby",
            color=CORE_ELEMENTS["lobby"]["color"]
        ))

        # Place passenger elevators in a row above lobby
        passenger_elev = next((e for e in core_elements if e.element_type == "elevator_passenger"), None)
        if passenger_elev:
            elev_sf = passenger_elev.sf_per_floor
            elev_side_ft = math.sqrt(elev_sf)
            elev_size_px = max(3, int(elev_side_ft / ft_per_pixel))

            # Arrange in a row
            num_elev = passenger_elev.count
            spacing = elev_size_px + 2
            start_x = center - (num_elev * spacing) // 2 + spacing // 2

            for i in range(num_elev):
                x = start_x + i * spacing
                y = center - lobby_size_px // 2 - elev_size_px // 2 - 2
                template.placements.append(CoreElementPlacement(
                    element_type="elevator_passenger",
                    position=(x, y),
                    size=(elev_size_px, elev_size_px),
                    room_type="Storage",
                    room_size="S",
                    name=f"PassengerElev_{i+1}",
                    color=CORE_ELEMENTS["elevator_passenger"]["color"]
                ))

        # Place freight elevators on sides
        freight_elev = next((e for e in core_elements if e.element_type == "elevator_freight"), None)
        if freight_elev:
            elev_sf = freight_elev.sf_per_floor
            elev_side_ft = math.sqrt(elev_sf)
            elev_size_px = max(4, int(elev_side_ft / ft_per_pixel))

            positions = [
                (center - lobby_size_px - elev_size_px, center),
                (center + lobby_size_px + elev_size_px, center),
            ]

            for i in range(min(freight_elev.count, len(positions))):
                x, y = positions[i]
                template.placements.append(CoreElementPlacement(
                    element_type="elevator_freight",
                    position=(x, y),
                    size=(elev_size_px, elev_size_px + 2),
                    room_type="Storage",
                    room_size="M",
                    name=f"FreightElev_{i+1}",
                    color=CORE_ELEMENTS["elevator_freight"]["color"]
                ))

        # Place stairs at corners
        stairs = next((e for e in core_elements if e.element_type == "stair"), None)
        if stairs:
            stair_sf = stairs.sf_per_floor
            stair_side_ft = math.sqrt(stair_sf)
            stair_size_px = max(4, int(stair_side_ft / ft_per_pixel))

            # Position stairs at three corners (or in a triangle)
            offset = lobby_size_px + stair_size_px + 4
            positions = [
                (center - offset, center - offset),  # Top-left
                (center + offset, center - offset),  # Top-right
                (center, center + offset),           # Bottom-center
            ]

            for i in range(min(stairs.count, len(positions))):
                x, y = positions[i]
                template.placements.append(CoreElementPlacement(
                    element_type="stair",
                    position=(x, y),
                    size=(stair_size_px, stair_size_px),
                    room_type="Entrance",
                    room_size="S",
                    name=f"Stair_{i+1}",
                    color=CORE_ELEMENTS["stair"]["color"]
                ))

        # Place vestibules adjacent to stairs
        vestibules = next((e for e in core_elements if e.element_type == "vestibule"), None)
        if vestibules:
            vest_sf = vestibules.sf_per_floor
            vest_side_ft = math.sqrt(vest_sf)
            vest_size_px = max(2, int(vest_side_ft / ft_per_pixel))

            # Place vestibule next to each stair
            stair_placements = [p for p in template.placements if p.element_type == "stair"]
            for i, stair in enumerate(stair_placements):
                x = stair.position[0] + stair.size[0] // 2 + vest_size_px // 2 + 1
                y = stair.position[1]
                template.placements.append(CoreElementPlacement(
                    element_type="vestibule",
                    position=(x, y),
                    size=(vest_size_px, vest_size_px),
                    room_type="Entrance",
                    room_size="XS",
                    name=f"StairVestibule_{i+1}",
                    color=CORE_ELEMENTS["vestibule"]["color"]
                ))

        # Define corridor regions connecting elements to edges
        corridor_width_px = max(3, int(5 / ft_per_pixel))  # ~5 ft corridor
        template.corridor_pixels = [
            # Horizontal corridor through center
            (0, center - corridor_width_px // 2, target_size, center + corridor_width_px // 2),
            # Vertical corridor through center
            (center - corridor_width_px // 2, 0, center + corridor_width_px // 2, target_size),
        ]

        return template

    @classmethod
    def create_default(cls, target_size: int = MODEL_OUTPUT_SIZE) -> "CoreTemplate":
        """
        Create a default core template for typical apartment building.

        Based on 6464 Canoga Ave specs:
        - 5 passenger elevators
        - 2 freight elevators
        - 3 stairs with vestibules
        """
        default_elements = [
            CoreElement("elevator_passenger", 306, 5),
            CoreElement("elevator_freight", 252, 2),
            CoreElement("stair", 254, 3),
            CoreElement("vestibule", 120, 3),
        ]

        # Assume zone is about 57 ft (172 ft floor / 3 zones)
        return cls.from_core_elements(default_elements, zone_side_ft=57.5, target_size=target_size)

    def render(self) -> Image.Image:
        """
        Render the core template to an image.

        Returns:
            PIL Image with core layout visualization
        """
        # Create base image (white background)
        img = Image.new("RGB", (self.size, self.size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Draw corridors first (as background)
        corridor_color = (240, 240, 240)  # Light gray
        for x1, y1, x2, y2 in self.corridor_pixels:
            draw.rectangle([x1, y1, x2, y2], fill=corridor_color)

        # Draw each element
        for placement in self.placements:
            x, y = placement.position
            w, h = placement.size

            # Calculate bounding box
            x1 = x - w // 2
            y1 = y - h // 2
            x2 = x1 + w
            y2 = y1 + h

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], fill=placement.color, outline=(100, 100, 100))

        return img

    def render_grayscale(self) -> Image.Image:
        """
        Render core template as grayscale room labels.

        Returns:
            Grayscale image with room type labels (0-17)
        """
        # Room type to label mapping (from image_process.py)
        type_to_label = {
            "LivingRoom": 0,
            "Storage": 11,
            "Entrance": 10,
        }

        img = Image.new("L", (self.size, self.size), 17)  # Default to interior door
        draw = ImageDraw.Draw(img)

        # Draw corridors as exterior (label 13)
        for x1, y1, x2, y2 in self.corridor_pixels:
            draw.rectangle([x1, y1, x2, y2], fill=10)  # Entrance/corridor

        # Draw each element
        for placement in self.placements:
            x, y = placement.position
            w, h = placement.size
            x1 = x - w // 2
            y1 = y - h // 2
            x2 = x1 + w
            y2 = y1 + h

            label = type_to_label.get(placement.room_type, 11)  # Default to Storage
            draw.rectangle([x1, y1, x2, y2], fill=label)

        return img

    def to_json(self) -> str:
        """
        Convert core template to JSON specification for diffusion model.

        Returns:
            JSON string in format expected by graph_encoder.py
        """
        grouped = {}

        for placement in self.placements:
            room_type = placement.room_type
            if room_type not in grouped:
                grouped[room_type] = {"rooms": []}

            room_spec = {
                "name": placement.name,
                "location": "center",  # Core is always center
                "size": placement.room_size,
                "link": []
            }

            # Add links based on adjacency
            # Lobby connects to all elevators
            if placement.element_type == "lobby":
                for p in self.placements:
                    if "Elev" in p.name:
                        room_spec["link"].append(p.name)

            # Elevators connect to lobby
            elif "Elev" in placement.name:
                room_spec["link"].append("ElevatorLobby")

            # Stairs connect to vestibules
            elif placement.element_type == "stair":
                vest_name = placement.name.replace("Stair", "StairVestibule")
                room_spec["link"].append(vest_name)

            # Vestibules connect to stairs
            elif placement.element_type == "vestibule":
                stair_name = placement.name.replace("StairVestibule", "Stair")
                room_spec["link"].append(stair_name)

            grouped[room_type]["rooms"].append(room_spec)

        return json.dumps(grouped, indent=2)

    def get_connection_points(self) -> Dict[str, Tuple[int, int]]:
        """
        Get connection points where wing zones should connect to core.

        Returns:
            Dictionary mapping direction to (x, y) pixel coordinates
        """
        center = self.size // 2
        edge_offset = 2  # Pixels from edge

        return {
            "north": (center, edge_offset),
            "south": (center, self.size - edge_offset),
            "east": (self.size - edge_offset, center),
            "west": (edge_offset, center),
            "northeast": (self.size - edge_offset, edge_offset),
            "northwest": (edge_offset, edge_offset),
            "southeast": (self.size - edge_offset, self.size - edge_offset),
            "southwest": (edge_offset, self.size - edge_offset),
        }

    def get_corridor_mask(self) -> Image.Image:
        """
        Get mask showing corridor regions (for wing zone generation).

        Returns:
            Binary mask (white = corridor/open, black = blocked)
        """
        mask = Image.new("L", (self.size, self.size), 0)  # Black background
        draw = ImageDraw.Draw(mask)

        # Mark corridor regions as open
        for x1, y1, x2, y2 in self.corridor_pixels:
            draw.rectangle([x1, y1, x2, y2], fill=255)

        return mask


def create_core_json_from_elements(core_elements: List[CoreElement]) -> str:
    """
    Create JSON specification from core elements without full template.

    Useful for quick testing or when template rendering isn't needed.
    """
    rooms = []

    # Add elevator lobby
    rooms.append({
        "name": "ElevatorLobby",
        "type": "LivingRoom",
        "location": "center",
        "size": "M",
        "link": []
    })

    # Add passenger elevators
    passenger = next((e for e in core_elements if e.element_type == "elevator_passenger"), None)
    if passenger:
        for i in range(passenger.count):
            rooms.append({
                "name": f"PassengerElev_{i+1}",
                "type": "Storage",
                "location": "center",
                "size": "S",
                "link": ["ElevatorLobby"]
            })
            rooms[0]["link"].append(f"PassengerElev_{i+1}")

    # Add freight elevators
    freight = next((e for e in core_elements if e.element_type == "elevator_freight"), None)
    if freight:
        for i in range(freight.count):
            rooms.append({
                "name": f"FreightElev_{i+1}",
                "type": "Storage",
                "location": "center",
                "size": "M",
                "link": ["ElevatorLobby"]
            })
            rooms[0]["link"].append(f"FreightElev_{i+1}")

    # Add stairs and vestibules
    stairs = next((e for e in core_elements if e.element_type == "stair"), None)
    if stairs:
        for i in range(stairs.count):
            stair_name = f"Stair_{i+1}"
            vest_name = f"StairVestibule_{i+1}"
            rooms.append({
                "name": stair_name,
                "type": "Entrance",
                "location": "center",
                "size": "S",
                "link": [vest_name]
            })
            rooms.append({
                "name": vest_name,
                "type": "Entrance",
                "location": "center",
                "size": "XS",
                "link": [stair_name]
            })

    # Group by type
    grouped = {}
    for room in rooms:
        room_type = room.pop("type")
        if room_type not in grouped:
            grouped[room_type] = {"rooms": []}
        grouped[room_type]["rooms"].append(room)

    return json.dumps(grouped, indent=2)


if __name__ == "__main__":
    # Test core template generation
    template = CoreTemplate.create_default()

    print("Core Template Elements:")
    for p in template.placements:
        print(f"  {p.name}: {p.element_type} at {p.position}, size {p.size}")

    print("\nJSON Specification:")
    print(template.to_json())

    # Save rendered image
    img = template.render()
    img.save("core_template_test.png")
    print("\nSaved core_template_test.png")
