"""
Room type mapping utilities.

Maps apartment unit types to diffusion model room specifications.
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from ..config import UNIT_ROOM_MAPPINGS, SIZE_THRESHOLDS, MAX_ROOMS_PER_ZONE


@dataclass
class RoomSpec:
    """Specification for a single room."""
    name: str
    room_type: str
    location: str = "Unknown"
    size: str = "M"
    links: List[str] = field(default_factory=list)
    unit_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "type": self.room_type,
            "location": self.location,
            "size": self.size,
            "link": self.links
        }


class RoomMapper:
    """Maps apartment units to room specifications."""

    def __init__(self):
        self.room_counter = {}

    def reset_counter(self):
        """Reset room name counters."""
        self.room_counter = {}

    def _get_unique_name(self, base_name: str, unit_id: str) -> str:
        """Generate unique room name."""
        key = f"{unit_id}_{base_name}"
        return key

    def sf_to_size(self, sf: float) -> str:
        """Convert square footage to size category."""
        for size, (min_sf, max_sf) in SIZE_THRESHOLDS.items():
            if min_sf <= sf < max_sf:
                return size
        return "M"

    def unit_to_rooms(
        self,
        unit_type: str,
        unit_id: str,
        total_sf: Optional[float] = None,
        location: str = "Unknown"
    ) -> List[RoomSpec]:
        """
        Convert a dwelling unit to room specifications.

        Args:
            unit_type: "Studio", "1BR", "2BR", "3BR"
            unit_id: Unique identifier for the unit
            total_sf: Total square footage (for size adjustment)
            location: Cardinal direction for room placement

        Returns:
            List of RoomSpec objects
        """
        if unit_type not in UNIT_ROOM_MAPPINGS:
            raise ValueError(f"Unknown unit type: {unit_type}")

        template = UNIT_ROOM_MAPPINGS[unit_type]
        rooms = []
        room_names = []

        for room_def in template:
            name = self._get_unique_name(room_def["name"], unit_id)
            room_names.append(name)

            # Adjust size based on total SF if provided
            size = room_def["size"]
            if total_sf:
                # Scale size based on unit total
                if total_sf > 700:
                    size = self._upsize(size)
                elif total_sf < 400:
                    size = self._downsize(size)

            rooms.append(RoomSpec(
                name=name,
                room_type=room_def["type"],
                location=location,
                size=size,
                links=[],
                unit_id=unit_id
            ))

        # Add standard links within unit
        self._add_unit_links(rooms, unit_type)

        return rooms

    def _upsize(self, size: str) -> str:
        """Increase size category."""
        sizes = ["XS", "S", "M", "L", "XL"]
        idx = sizes.index(size) if size in sizes else 2
        return sizes[min(idx + 1, len(sizes) - 1)]

    def _downsize(self, size: str) -> str:
        """Decrease size category."""
        sizes = ["XS", "S", "M", "L", "XL"]
        idx = sizes.index(size) if size in sizes else 2
        return sizes[max(idx - 1, 0)]

    def _add_unit_links(self, rooms: List[RoomSpec], unit_type: str):
        """Add logical room connections within a unit."""
        room_by_type = {r.room_type: r for r in rooms}
        room_by_name = {r.name: r for r in rooms}

        # Entry connects to Living/Master
        entry = next((r for r in rooms if r.room_type == "Entrance"), None)
        if entry:
            living = next((r for r in rooms if r.room_type == "LivingRoom"), None)
            master = next((r for r in rooms if r.room_type == "MasterRoom"), None)

            if living:
                entry.links.append(living.name)
                living.links.append(entry.name)
            elif master:
                entry.links.append(master.name)
                master.links.append(entry.name)

        # Living connects to Kitchen
        living = next((r for r in rooms if r.room_type == "LivingRoom"), None)
        kitchen = next((r for r in rooms if r.room_type == "Kitchen"), None)
        if living and kitchen:
            living.links.append(kitchen.name)
            kitchen.links.append(living.name)

        # Master connects to its bathroom
        master = next((r for r in rooms if r.room_type == "MasterRoom"), None)
        bathrooms = [r for r in rooms if r.room_type == "Bathroom"]
        if master and bathrooms:
            master_bath = bathrooms[0]
            master.links.append(master_bath.name)
            master_bath.links.append(master.name)

        # Second bedroom connects to second bathroom (if exists)
        second = next((r for r in rooms if r.room_type == "SecondRoom"), None)
        if second and len(bathrooms) > 1:
            second_bath = bathrooms[1]
            second.links.append(second_bath.name)
            second_bath.links.append(second.name)

    def rooms_to_json(self, rooms: List[RoomSpec]) -> str:
        """
        Convert rooms to JSON format expected by diffusion model.

        Format matches graph_encoder.py expectations:
        {
            "RoomType": {
                "rooms": [{"name": ..., "location": ..., "size": ..., "link": [...]}]
            }
        }
        """
        grouped = {}
        for room in rooms:
            if room.room_type not in grouped:
                grouped[room.room_type] = {"rooms": []}
            grouped[room.room_type]["rooms"].append(room.to_dict())

        return json.dumps(grouped, indent=2)

    def can_fit_in_zone(self, rooms: List[RoomSpec]) -> bool:
        """Check if rooms fit within zone limit."""
        return len(rooms) <= MAX_ROOMS_PER_ZONE

    def split_rooms_for_zones(
        self,
        rooms: List[RoomSpec],
        max_per_zone: int = MAX_ROOMS_PER_ZONE
    ) -> List[List[RoomSpec]]:
        """
        Split rooms into zone-compatible groups.

        Tries to keep units together when possible.
        """
        if len(rooms) <= max_per_zone:
            return [rooms]

        # Group by unit
        units = {}
        for room in rooms:
            uid = room.unit_id or "unknown"
            if uid not in units:
                units[uid] = []
            units[uid].append(room)

        # Pack units into zones
        zones = []
        current_zone = []
        current_count = 0

        for unit_id, unit_rooms in units.items():
            if current_count + len(unit_rooms) <= max_per_zone:
                current_zone.extend(unit_rooms)
                current_count += len(unit_rooms)
            else:
                if current_zone:
                    zones.append(current_zone)
                current_zone = unit_rooms.copy()
                current_count = len(unit_rooms)

        if current_zone:
            zones.append(current_zone)

        return zones
