"""
Coordinate mapping utilities.

Maps between real-world dimensions (feet) and pixel coordinates.
"""

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ScaleInfo:
    """Scale information for coordinate conversion."""
    floor_area_sf: float
    floor_side_ft: float
    pixels_total: int
    ft_per_pixel: float
    sf_per_pixel: float


class CoordinateMapper:
    """Maps between real-world coordinates and pixel coordinates."""

    def __init__(self, floor_area_sf: float, output_pixels: int = 192):
        """
        Initialize coordinate mapper.

        Args:
            floor_area_sf: Total floor area in square feet
            output_pixels: Output image size in pixels (default 192 for 3x64)
        """
        self.floor_area_sf = floor_area_sf
        self.output_pixels = output_pixels

        # Assume square floor plate
        self.floor_side_ft = math.sqrt(floor_area_sf)
        self.ft_per_pixel = self.floor_side_ft / output_pixels
        self.sf_per_pixel = self.ft_per_pixel ** 2

    def feet_to_pixels(self, feet: float) -> int:
        """Convert feet to pixels."""
        return int(round(feet / self.ft_per_pixel))

    def pixels_to_feet(self, pixels: int) -> float:
        """Convert pixels to feet."""
        return pixels * self.ft_per_pixel

    def sf_to_pixels(self, sf: float) -> int:
        """Convert square feet to pixel area (side length of square)."""
        side_ft = math.sqrt(sf)
        return self.feet_to_pixels(side_ft)

    def room_dimensions_to_pixels(self, width_ft: float, depth_ft: float) -> Tuple[int, int]:
        """Convert room dimensions to pixel dimensions."""
        return (
            self.feet_to_pixels(width_ft),
            self.feet_to_pixels(depth_ft)
        )

    def get_zone_size_ft(self, grid_size: int = 3) -> float:
        """Get the size of each zone in feet."""
        return self.floor_side_ft / grid_size

    def get_zone_size_pixels(self, grid_size: int = 3) -> int:
        """Get the size of each zone in pixels."""
        return self.output_pixels // grid_size

    def get_scale_info(self) -> ScaleInfo:
        """Get complete scale information."""
        return ScaleInfo(
            floor_area_sf=self.floor_area_sf,
            floor_side_ft=self.floor_side_ft,
            pixels_total=self.output_pixels,
            ft_per_pixel=self.ft_per_pixel,
            sf_per_pixel=self.sf_per_pixel
        )

    def zone_position_to_pixels(
        self,
        zone_row: int,
        zone_col: int,
        grid_size: int = 3
    ) -> Tuple[int, int, int, int]:
        """
        Get pixel bounds for a zone position.

        Args:
            zone_row: Row index (0 = top/north)
            zone_col: Column index (0 = left/west)
            grid_size: Grid dimension

        Returns:
            (x1, y1, x2, y2) pixel bounds
        """
        zone_size = self.get_zone_size_pixels(grid_size)
        x1 = zone_col * zone_size
        y1 = zone_row * zone_size
        x2 = x1 + zone_size
        y2 = y1 + zone_size
        return (x1, y1, x2, y2)

    def get_cardinal_direction(self, zone_row: int, zone_col: int, grid_size: int = 3) -> str:
        """
        Get cardinal direction for a zone position.

        For a 3x3 grid:
        - (0,0) = northwest, (0,1) = north, (0,2) = northeast
        - (1,0) = west, (1,1) = center, (1,2) = east
        - (2,0) = southwest, (2,1) = south, (2,2) = southeast
        """
        mid = grid_size // 2

        if zone_row < mid:
            ns = "north"
        elif zone_row > mid:
            ns = "south"
        else:
            ns = ""

        if zone_col < mid:
            ew = "west"
        elif zone_col > mid:
            ew = "east"
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
