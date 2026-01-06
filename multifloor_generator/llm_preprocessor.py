"""
LLM Preprocessor using DashScope.

Enhances room specifications with spatial relationships using LLM.
"""

import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .config import DASHSCOPE_CONFIG, ROOM_LOCATIONS, ROOM_SIZES
from .floor_tiler import Zone


@dataclass
class EnhancedSpec:
    """Enhanced room specification from LLM."""
    original_json: str
    enhanced_json: str
    changes_made: List[str]
    success: bool
    error: Optional[str] = None


class LLMPreprocessor:
    """
    Uses LLM to enhance room specifications with spatial relationships.

    Adds:
    - Cardinal locations based on corridor position
    - Adjacency links based on typical apartment layouts
    - Size adjustments based on room function
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize LLM preprocessor.

        Args:
            api_key: DashScope API key
            base_url: API base URL
            model: Model name (e.g., 'qwen-turbo')
        """
        self.api_key = api_key or DASHSCOPE_CONFIG["api_key"]
        self.base_url = base_url or DASHSCOPE_CONFIG["base_url"]
        self.model = model or DASHSCOPE_CONFIG["model"]

        self.client = None
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize the OpenAI client for DashScope.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        try:
            from openai import OpenAI

            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
            self._initialized = True
            return True

        except ImportError:
            print("OpenAI package not installed. Run: pip install openai")
            return False
        except Exception as e:
            print(f"Failed to initialize LLM client: {e}")
            return False

    def enhance_zone_spec(
        self,
        zone: Zone,
        floor_context: Optional[str] = None
    ) -> EnhancedSpec:
        """
        Enhance zone specification with LLM.

        Args:
            zone: Zone with room assignments
            floor_context: Additional context about the floor

        Returns:
            EnhancedSpec with improved room relationships
        """
        original_json = zone.to_json()

        if not original_json or original_json == "{}":
            return EnhancedSpec(
                original_json=original_json,
                enhanced_json=original_json,
                changes_made=[],
                success=True
            )

        if not self._initialized and not self.initialize():
            # Fall back to rule-based enhancement
            return self._enhance_with_rules(zone, original_json)

        try:
            prompt = self._create_enhancement_prompt(zone, original_json, floor_context)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=2000
            )

            content = response.choices[0].message.content

            # Extract JSON from response
            enhanced_json = self._extract_json(content)

            # Validate enhanced JSON
            if not self._validate_json(enhanced_json):
                return self._enhance_with_rules(zone, original_json)

            changes = self._identify_changes(original_json, enhanced_json)

            return EnhancedSpec(
                original_json=original_json,
                enhanced_json=enhanced_json,
                changes_made=changes,
                success=True
            )

        except Exception as e:
            # Fall back to rule-based enhancement
            result = self._enhance_with_rules(zone, original_json)
            result.error = str(e)
            return result

    def _get_system_prompt(self) -> str:
        """Get system prompt for LLM."""
        return """You are an expert in residential apartment floor plan design.
Your task is to enhance room specifications with appropriate spatial relationships.

Rules:
1. Locations must be one of: north, northwest, west, southwest, south, southeast, east, northeast, center, Unknown
2. Sizes must be one of: XS, S, M, L, XL, Unknown
3. Links should reflect typical apartment layouts:
   - Entry connects to Living/Hallway
   - Living connects to Kitchen and Dining
   - Bedrooms connect to their bathrooms
   - Kitchen may connect to Dining
4. Keep existing room names and types unchanged
5. Return ONLY valid JSON matching the input structure

Output ONLY the JSON, no explanations."""

    def _create_enhancement_prompt(
        self,
        zone: Zone,
        original_json: str,
        floor_context: Optional[str]
    ) -> str:
        """Create prompt for LLM enhancement."""
        prompt = f"""Enhance this apartment zone specification:

Zone Direction: {zone.direction}
Corridor connects from: {zone.corridor_direction}

Current specification:
{original_json}

Context: This is a wing zone in a multi-family apartment building.
The zone connects to the main corridor on the {zone.corridor_direction} side.

Please:
1. Assign appropriate locations based on zone direction ({zone.direction})
2. Rooms near corridor should face {zone.corridor_direction}
3. Bedrooms should be away from corridor for privacy
4. Add logical room connections (links)

Return ONLY the enhanced JSON."""

        if floor_context:
            prompt += f"\n\nAdditional context: {floor_context}"

        return prompt

    def _extract_json(self, content: str) -> str:
        """Extract JSON from LLM response."""
        # Try to find JSON block
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            return json_match.group()

        # Try markdown code block
        code_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
        if code_match:
            return code_match.group(1).strip()

        return content

    def _validate_json(self, json_str: str) -> bool:
        """Validate that JSON is properly formatted."""
        try:
            data = json.loads(json_str)

            # Check structure
            for room_type, type_data in data.items():
                if not isinstance(type_data, dict):
                    return False
                if "rooms" not in type_data:
                    return False
                for room in type_data["rooms"]:
                    if "name" not in room:
                        return False
                    # Validate location
                    if "location" in room and room["location"] not in ROOM_LOCATIONS + ["Unknown"]:
                        return False
                    # Validate size
                    if "size" in room and room["size"] not in ROOM_SIZES:
                        return False

            return True
        except:
            return False

    def _identify_changes(self, original: str, enhanced: str) -> List[str]:
        """Identify what changes were made."""
        changes = []
        try:
            orig_data = json.loads(original)
            enh_data = json.loads(enhanced)

            for room_type in enh_data:
                if room_type not in orig_data:
                    changes.append(f"Added room type: {room_type}")
                    continue

                for i, room in enumerate(enh_data[room_type].get("rooms", [])):
                    if i >= len(orig_data[room_type].get("rooms", [])):
                        continue

                    orig_room = orig_data[room_type]["rooms"][i]
                    name = room.get("name", "?")

                    if room.get("location") != orig_room.get("location"):
                        changes.append(f"{name}: location -> {room.get('location')}")

                    if room.get("size") != orig_room.get("size"):
                        changes.append(f"{name}: size -> {room.get('size')}")

                    orig_links = set(orig_room.get("link", []))
                    new_links = set(room.get("link", []))
                    added_links = new_links - orig_links
                    if added_links:
                        changes.append(f"{name}: added links -> {list(added_links)}")

        except:
            pass

        return changes

    def _enhance_with_rules(self, zone: Zone, original_json: str) -> EnhancedSpec:
        """
        Enhance specification using rule-based logic.

        Fallback when LLM is not available.
        """
        try:
            data = json.loads(original_json)
        except:
            return EnhancedSpec(
                original_json=original_json,
                enhanced_json=original_json,
                changes_made=[],
                success=True
            )

        changes = []

        # Location mapping based on zone direction
        location_map = self._get_location_map(zone.direction, zone.corridor_direction)

        for room_type, type_data in data.items():
            for room in type_data.get("rooms", []):
                name = room.get("name", "")

                # Assign location based on room type
                if room.get("location", "Unknown") == "Unknown":
                    new_loc = location_map.get(room_type, zone.direction)
                    room["location"] = new_loc
                    changes.append(f"{name}: location -> {new_loc}")

                # Add links based on room type
                self._add_room_links(room, room_type, type_data, data)

        enhanced_json = json.dumps(data, indent=2)

        return EnhancedSpec(
            original_json=original_json,
            enhanced_json=enhanced_json,
            changes_made=changes,
            success=True
        )

    def _get_location_map(self, zone_dir: str, corridor_dir: str) -> Dict[str, str]:
        """Get room type to location mapping based on zone position."""
        # Rooms that should be near corridor
        corridor_rooms = ["Entrance", "LivingRoom", "Kitchen"]

        # Rooms that should be away from corridor
        private_rooms = ["MasterRoom", "SecondRoom", "ChildRoom", "Bathroom"]

        # Opposite direction mapping
        opposite = {
            "north": "south", "south": "north",
            "east": "west", "west": "east",
            "northeast": "southwest", "southwest": "northeast",
            "northwest": "southeast", "southeast": "northwest",
        }

        location_map = {}

        # Assign corridor-facing rooms
        for room_type in corridor_rooms:
            location_map[room_type] = corridor_dir

        # Assign private rooms to opposite side
        away_dir = opposite.get(corridor_dir, zone_dir)
        for room_type in private_rooms:
            location_map[room_type] = away_dir

        # Central rooms
        location_map["DiningRoom"] = "center"
        location_map["Storage"] = zone_dir
        location_map["Balcony"] = away_dir

        return location_map

    def _add_room_links(
        self,
        room: Dict,
        room_type: str,
        type_data: Dict,
        all_data: Dict
    ):
        """Add logical room links based on room type."""
        current_links = set(room.get("link", []))

        # Entry should connect to Living
        if room_type == "Entrance":
            living_rooms = all_data.get("LivingRoom", {}).get("rooms", [])
            for lr in living_rooms:
                if lr.get("name") not in current_links:
                    current_links.add(lr.get("name"))

        # Living should connect to Kitchen
        if room_type == "LivingRoom":
            kitchen_rooms = all_data.get("Kitchen", {}).get("rooms", [])
            for kr in kitchen_rooms:
                if kr.get("name") not in current_links:
                    current_links.add(kr.get("name"))

        # Bedrooms should connect to bathrooms
        if room_type in ["MasterRoom", "SecondRoom", "ChildRoom"]:
            bath_rooms = all_data.get("Bathroom", {}).get("rooms", [])
            for br in bath_rooms:
                if br.get("name") not in current_links:
                    current_links.add(br.get("name"))
                    break  # Only link to one bathroom

        room["link"] = list(current_links)

    def enhance_multiple_zones(
        self,
        zones: List[Zone],
        floor_context: Optional[str] = None
    ) -> Dict[str, EnhancedSpec]:
        """
        Enhance specifications for multiple zones.

        Args:
            zones: List of zones to enhance
            floor_context: Shared floor context

        Returns:
            Dictionary mapping zone_id to EnhancedSpec
        """
        results = {}

        for zone in zones:
            result = self.enhance_zone_spec(zone, floor_context)
            results[zone.zone_id] = result
            print(f"Enhanced {zone.zone_id}: {len(result.changes_made)} changes")

        return results


class RuleBasedPreprocessor(LLMPreprocessor):
    """
    Preprocessor that only uses rule-based enhancement.

    For testing or when LLM API is not available.
    """

    def __init__(self):
        super().__init__()

    def initialize(self) -> bool:
        return True

    def enhance_zone_spec(
        self,
        zone: Zone,
        floor_context: Optional[str] = None
    ) -> EnhancedSpec:
        """Use only rule-based enhancement."""
        original_json = zone.to_json()
        return self._enhance_with_rules(zone, original_json)


def create_preprocessor(use_llm: bool = True) -> LLMPreprocessor:
    """
    Factory function to create appropriate preprocessor.

    Args:
        use_llm: If True, use LLM enhancement; otherwise use rules only

    Returns:
        LLMPreprocessor instance
    """
    if use_llm:
        return LLMPreprocessor()
    return RuleBasedPreprocessor()


if __name__ == "__main__":
    # Test preprocessor
    from .floor_tiler import Zone, ZoneType
    from .utils.room_mapper import RoomSpec

    test_zone = Zone(
        zone_id="wing_north",
        zone_type=ZoneType.WING,
        grid_position=(0, 1),
        direction="north",
        pixel_bounds=(64, 0, 128, 64),
        corridor_direction="south",
        rooms=[
            RoomSpec("Living_1", "LivingRoom", "Unknown", "M", []),
            RoomSpec("Kitchen_1", "Kitchen", "Unknown", "S", []),
            RoomSpec("Bedroom_1", "MasterRoom", "Unknown", "M", []),
            RoomSpec("Bath_1", "Bathroom", "Unknown", "XS", []),
        ]
    )

    # Test with rules only
    preprocessor = RuleBasedPreprocessor()
    result = preprocessor.enhance_zone_spec(test_zone)

    print(f"Original:\n{result.original_json}")
    print(f"\nEnhanced:\n{result.enhanced_json}")
    print(f"\nChanges: {result.changes_made}")
