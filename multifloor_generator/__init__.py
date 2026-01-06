"""
Multi-Floor Apartment Massing Generator

Adapts ChatHouseDiffusion for multi-floor apartment layout generation
from CSV building program data.
"""

__version__ = "0.1.0"

from .csv_parser import parse_building_csv, BuildingSpec, FloorSpec, DwellingUnit
from .floor_tiler import FloorTiler, Zone
from .core_template import CoreTemplate
from .unit_zone_generator import UnitZoneGenerator
from .tile_assembler import TileAssembler
from .floor_stacker import FloorStacker
