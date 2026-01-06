"""
Configuration for Multi-Floor Generator

Contains API keys, model paths, and constants.
"""

import os
from dataclasses import dataclass
from typing import Optional

# DashScope API Configuration
DASHSCOPE_CONFIG = {
    "api_key": os.getenv("DASHSCOPE_API_KEY", "sk-3154176795dd40969654a6efb517ab0a"),
    "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "model": "qwen-turbo"
}

# ChatHouseDiffusion Model Configuration
MODEL_CONFIG = {
    "results_folder": "./chathousediffusion_gloq/predict_model",
    "milestone": 98,
    "inject_step": 40,
    "cond_scale": 1.0,
    "sampling_timesteps": 50
}

# Room Type Mappings (must match graph_encoder.py)
ROOM_CATEGORIES = {
    "Unknown": 0,
    "LivingRoom": 1,
    "MasterRoom": 2,
    "Kitchen": 3,
    "Bathroom": 4,
    "DiningRoom": 5,
    "ChildRoom": 6,
    "StudyRoom": 7,
    "SecondRoom": 8,
    "GuestRoom": 9,
    "Balcony": 10,
    "Entrance": 11,
    "Storage": 12,
}

ROOM_LOCATIONS = [
    "north", "northwest", "west", "southwest",
    "south", "southeast", "east", "northeast",
    "center", "Unknown"
]

ROOM_SIZES = ["XS", "S", "M", "L", "XL", "Unknown"]

# Unit Type to Room Mapping
UNIT_ROOM_MAPPINGS = {
    "Studio": [
        {"type": "MasterRoom", "name": "Bedroom", "size": "M"},
        {"type": "Kitchen", "name": "Kitchen", "size": "S"},
        {"type": "Bathroom", "name": "Bathroom", "size": "XS"},
        {"type": "Entrance", "name": "Entry", "size": "XS"},
    ],
    "1BR": [
        {"type": "LivingRoom", "name": "Living", "size": "M"},
        {"type": "MasterRoom", "name": "Bedroom", "size": "M"},
        {"type": "Kitchen", "name": "Kitchen", "size": "S"},
        {"type": "Bathroom", "name": "Bathroom", "size": "S"},
        {"type": "Entrance", "name": "Entry", "size": "XS"},
    ],
    "2BR": [
        {"type": "LivingRoom", "name": "Living", "size": "L"},
        {"type": "MasterRoom", "name": "MasterBed", "size": "M"},
        {"type": "SecondRoom", "name": "SecondBed", "size": "M"},
        {"type": "Kitchen", "name": "Kitchen", "size": "S"},
        {"type": "Bathroom", "name": "MasterBath", "size": "S"},
        {"type": "Bathroom", "name": "SecondBath", "size": "XS"},
        {"type": "Entrance", "name": "Entry", "size": "XS"},
    ],
    "3BR": [
        {"type": "LivingRoom", "name": "Living", "size": "L"},
        {"type": "MasterRoom", "name": "MasterBed", "size": "M"},
        {"type": "SecondRoom", "name": "SecondBed", "size": "M"},
        {"type": "ChildRoom", "name": "ThirdBed", "size": "S"},
        {"type": "Kitchen", "name": "Kitchen", "size": "M"},
        {"type": "Bathroom", "name": "MasterBath", "size": "S"},
        {"type": "Bathroom", "name": "SecondBath", "size": "XS"},
        {"type": "Entrance", "name": "Entry", "size": "S"},
    ],
}

# Size Thresholds (SF)
SIZE_THRESHOLDS = {
    "XS": (0, 150),
    "S": (150, 300),
    "M": (300, 500),
    "L": (500, 800),
    "XL": (800, float('inf'))
}

# Model Constraints
MAX_ROOMS_PER_ZONE = 10  # From graph_encoder.py
MODEL_OUTPUT_SIZE = 64   # 64x64 pixels

# Zone Grid Configuration
DEFAULT_GRID_SIZE = 3    # 3x3 grid = 9 zones per floor

# Core Element Types
CORE_ELEMENTS = {
    "elevator_passenger": {"room_type": "Storage", "size": "S", "color": (128, 128, 128)},
    "elevator_freight": {"room_type": "Storage", "size": "M", "color": (100, 100, 100)},
    "stair": {"room_type": "Entrance", "size": "S", "color": (180, 180, 180)},
    "vestibule": {"room_type": "Entrance", "size": "XS", "color": (200, 200, 200)},
    "lobby": {"room_type": "LivingRoom", "size": "M", "color": (220, 220, 220)},
}

# Color Mapping for Visualization (matches image_process.py)
ROOM_COLORS = {
    0: (224, 255, 192),   # LivingRoom - light green
    1: (192, 255, 255),   # MasterRoom - cyan
    2: (255, 224, 128),   # Kitchen - yellow
    3: (192, 192, 224),   # Bathroom - lavender
    4: (255, 160, 96),    # DiningRoom - orange
    5: (255, 224, 224),   # ChildRoom - light pink
    6: (224, 224, 128),   # StudyRoom - khaki
    7: (224, 224, 255),   # SecondRoom - light blue
    8: (255, 192, 255),   # GuestRoom - pink
    9: (128, 255, 128),   # Balcony - green
    10: (255, 255, 255),  # Entrance - white
    11: (192, 192, 192),  # Storage - gray
    12: (255, 255, 128),  # Wall-in
    13: (255, 60, 128),   # External
    14: (0, 0, 0),        # ExteriorWall - black
    15: (96, 96, 96),     # FrontDoor
    16: (128, 128, 128),  # InteriorWall
    17: (160, 160, 160),  # InteriorDoor
}


@dataclass
class GenerationConfig:
    """Configuration for floor plan generation."""
    grid_size: int = DEFAULT_GRID_SIZE
    output_size: int = 512
    use_llm_enhancement: bool = True
    save_intermediate: bool = True
    output_dir: str = "./generated_floors"
