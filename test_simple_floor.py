#!/usr/bin/env python3
"""
Simple test: Generate a small floor with ring corridor around core.

Run with: source venv/bin/activate && python3 test_simple_floor.py
"""

import sys
import os

# Add paths
sys.path.insert(0, "./chathousediffusion_gloq")

from PIL import Image, ImageDraw, ImageFont
import json

# ============================================
# STEP 1: Define a simple floor spec
# ============================================
print("=" * 60)
print("STEP 1: Realistic floor - 8 units around core (units as blocks!)")
print("=" * 60)

# SEMANTIC TRANSLATION: Tell the model what it understands!
#
# The model knows:
# - Room types: LivingRoom, MasterRoom, Kitchen, Bathroom, Storage, Entrance, etc.
# - Positions: north, south, east, west, center, corners
# - Sizes: XS, S, M, L, XL
# - Links: which rooms connect to which
#
# Our translation:
# - Elevator/stair CORE → Storage (center) - model knows what storage/utility is
# - Large units (2BR) → LivingRoom (exterior walls, large)
# - Medium units (1BR) → MasterRoom (exterior walls, medium)
# - Small units (Studio) → SecondRoom (exterior walls, small)
# - All units connect to the CORE (that's how you access them!)

simple_floor_spec = {
    # CORE: Elevator and stair wells in the CENTER
    # Using Storage type - model understands utility/mechanical spaces
    "Storage": {
        "rooms": [
            {"name": "ElevatorCore", "location": "center", "size": "M",
             "link": ["Unit_N", "Unit_S", "Unit_E", "Unit_W", "Unit_NE", "Unit_NW", "Unit_SE", "Unit_SW"]},
        ]
    },
    # LARGE UNITS (2BR) on corners - best views, largest spaces
    # Using LivingRoom type for largest residential spaces
    "LivingRoom": {
        "rooms": [
            {"name": "Unit_NE", "location": "northeast", "size": "L", "link": ["ElevatorCore"]},
            {"name": "Unit_NW", "location": "northwest", "size": "L", "link": ["ElevatorCore"]},
        ]
    },
    # MEDIUM UNITS (1BR) on exterior walls
    # Using MasterRoom type for medium residential spaces
    "MasterRoom": {
        "rooms": [
            {"name": "Unit_N", "location": "north", "size": "M", "link": ["ElevatorCore"]},
            {"name": "Unit_S", "location": "south", "size": "M", "link": ["ElevatorCore"]},
            {"name": "Unit_E", "location": "east", "size": "M", "link": ["ElevatorCore"]},
            {"name": "Unit_W", "location": "west", "size": "M", "link": ["ElevatorCore"]},
        ]
    },
    # SMALL UNITS (Studio) on remaining corners
    # Using SecondRoom type for smaller residential spaces
    "SecondRoom": {
        "rooms": [
            {"name": "Unit_SE", "location": "southeast", "size": "S", "link": ["ElevatorCore"]},
            {"name": "Unit_SW", "location": "southwest", "size": "S", "link": ["ElevatorCore"]},
        ]
    },
}

# What we're telling the model:
# - There's a storage/utility core in the CENTER
# - Large living spaces (2BR) in the CORNERS (NE, NW)
# - Medium bedrooms (1BR) on the SIDES (N, S, E, W)
# - Smaller rooms (Studio) in remaining CORNERS (SE, SW)
# - Everything LINKS to the central core
#
# The model should place:
# +--------+--------+--------+
# | 2BR    | 1BR    | 2BR    |   <- Large corners, medium sides
# | (NW)   | (N)    | (NE)   |
# +--------+--------+--------+
# | 1BR    | CORE   | 1BR    |   <- Core in center, units around
# | (W)    |(elev)  | (E)    |
# +--------+--------+--------+
# | Studio | 1BR    | Studio |   <- Small corners, medium sides
# | (SW)   | (S)    | (SE)   |
# +--------+--------+--------+

# Count rooms
total_rooms = sum(len(v["rooms"]) for v in simple_floor_spec.values())
print(f"Total rooms: {total_rooms}")
print(f"Room types: {list(simple_floor_spec.keys())}")

# ============================================
# STEP 2: Convert to JSON string for model
# ============================================
print("\n" + "=" * 60)
print("STEP 2: JSON spec for diffusion model")
print("=" * 60)

json_spec = json.dumps(simple_floor_spec, indent=2)
print(json_spec[:500] + "..." if len(json_spec) > 500 else json_spec)

# ============================================
# STEP 3: Try LLM enhancement (optional)
# ============================================
print("\n" + "=" * 60)
print("STEP 3: LLM Enhancement (adding spatial context)")
print("=" * 60)

try:
    from multifloor_generator.llm_preprocessor import LLMPreprocessor
    from multifloor_generator.config import DASHSCOPE_CONFIG

    llm = LLMPreprocessor(
        api_key=DASHSCOPE_CONFIG["api_key"],
        base_url=DASHSCOPE_CONFIG["base_url"],
        model=DASHSCOPE_CONFIG["model"]
    )

    # Create a simple prompt for enhancement
    enhancement_prompt = """
    This is a simple apartment floor with 4 dwelling units arranged around a central core.
    The core contains elevators and stairs.
    A ring corridor wraps around the core, and each unit's entrance opens onto this corridor.
    Each unit has: Living room, Bedroom, Kitchen, Bathroom, and Entrance.
    Units are positioned at North, South, East, and West of the core.
    """

    print("Calling LLM for spatial enhancement...")
    # Note: We'd call llm.enhance() here but let's skip for now to test diffusion first
    print("(Skipping LLM for initial test - using raw spec)")

except Exception as e:
    print(f"LLM not available: {e}")
    print("Continuing with raw spec...")

# ============================================
# STEP 4: Create input mask (boundary constraint)
# ============================================
print("\n" + "=" * 60)
print("STEP 4: Create boundary mask")
print("=" * 60)

# Create a simple mask - white interior, could add black borders for walls
mask = Image.new("L", (64, 64), 255)  # All white = all valid space
mask.save("test_mask.png")
print("Saved test_mask.png (64x64 white = all valid)")

# ============================================
# STEP 5: Generate with MOCK first
# ============================================
print("\n" + "=" * 60)
print("STEP 5: Mock generation (no GPU needed)")
print("=" * 60)

from multifloor_generator.unit_zone_generator import MockGenerator
from multifloor_generator.floor_tiler import Zone, ZoneType

# Create a dummy zone with our spec
mock_zone = Zone(
    zone_id="simple_floor",
    zone_type=ZoneType.WING,
    grid_position=(1, 1),
    direction="center",
    pixel_bounds=(0, 0, 64, 64),
    rooms=[]  # We'll pass JSON directly
)

mock_gen = MockGenerator()
mock_result = mock_gen.generate_zone(mock_zone, json_override=json_spec)
mock_result.image.save("test_mock_result.png")
print(f"Mock generation: {'SUCCESS' if mock_result.success else 'FAILED'}")
print("Saved test_mock_result.png")

# ============================================
# STEP 6: Generate with REAL diffusion model
# ============================================
print("\n" + "=" * 60)
print("STEP 6: Real diffusion model generation")
print("=" * 60)

try:
    from multifloor_generator.unit_zone_generator import UnitZoneGenerator

    print("Initializing diffusion model...")
    real_gen = UnitZoneGenerator()

    if real_gen.load_model():
        print("\nGenerating floor plan (this takes ~5 min on CPU)...")
        print("Watch for output...\n")

        real_result = real_gen.generate_zone(mock_zone, mask=mask, json_override=json_spec)

        if real_result.success:
            real_result.image.save("test_real_result.png")
            print(f"\nSUCCESS! Saved test_real_result.png")
        else:
            print(f"\nFailed: {real_result.error}")
    else:
        print("Model failed to load - check paths and dependencies")
        print("You can still view test_mock_result.png for the mock version")

except Exception as e:
    print(f"Error with real model: {e}")
    import traceback
    traceback.print_exc()
    print("\nYou can still view test_mock_result.png for the mock version")

print("\n" + "=" * 60)
print("DONE! Check the generated images:")
print("  - test_mask.png (input boundary)")
print("  - test_mock_result.png (mock visualization)")
print("  - test_real_result.png (diffusion output, if successful)")
print("=" * 60)
