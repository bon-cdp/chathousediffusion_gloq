"""
Tile Assembler.

Stitches zone images into complete floor plans.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont

from .config import MODEL_OUTPUT_SIZE, DEFAULT_GRID_SIZE, ROOM_COLORS
from .floor_tiler import Zone, ZoneType
from .core_template import CoreTemplate
from .unit_zone_generator import GenerationResult
from .utils.image_utils import ImageUtils
from .utils.coordinate_mapper import CoordinateMapper


class TileAssembler:
    """
    Assembles zone tiles into complete floor plans.

    Handles:
    - Placing zones in correct grid positions
    - Blending overlapping regions (corridors)
    - Adding corridor connections between zones
    - Scaling to real-world dimensions
    """

    def __init__(
        self,
        grid_size: int = DEFAULT_GRID_SIZE,
        zone_size: int = MODEL_OUTPUT_SIZE,
        output_size: int = 512
    ):
        """
        Initialize tile assembler.

        Args:
            grid_size: Grid dimension (e.g., 3 for 3x3)
            zone_size: Size of each zone in pixels
            output_size: Final output image size
        """
        self.grid_size = grid_size
        self.zone_size = zone_size
        self.assembly_size = grid_size * zone_size
        self.output_size = output_size

    def assemble_floor(
        self,
        core_image: Image.Image,
        wing_results: Dict[str, GenerationResult],
        zones: List[Zone],
        core_zone: Zone,
        add_corridors: bool = True
    ) -> Image.Image:
        """
        Assemble a complete floor plan from zone images.

        Args:
            core_image: Rendered core template image
            wing_results: Dictionary mapping zone_id to GenerationResult
            zones: List of all zones
            core_zone: The core zone specification
            add_corridors: Whether to draw corridor connections

        Returns:
            Assembled floor plan image
        """
        # Create base canvas
        canvas = Image.new("RGB", (self.assembly_size, self.assembly_size), (255, 255, 255))

        # Place each zone
        for zone in zones:
            row, col = zone.grid_position
            x = col * self.zone_size
            y = row * self.zone_size

            if zone.zone_type == ZoneType.CORE:
                # Place core image
                canvas.paste(core_image.resize((self.zone_size, self.zone_size)), (x, y))
            else:
                # Place wing zone
                if zone.zone_id in wing_results:
                    result = wing_results[zone.zone_id]
                    zone_img = result.image.resize((self.zone_size, self.zone_size))
                    canvas.paste(zone_img, (x, y))

        # Add corridor connections
        if add_corridors:
            canvas = self._add_corridor_connections(canvas, zones, core_zone)

        # Add grid lines for visualization (optional)
        # canvas = self._add_grid_lines(canvas)

        return canvas

    def _add_corridor_connections(
        self,
        image: Image.Image,
        zones: List[Zone],
        core_zone: Zone
    ) -> Image.Image:
        """Add corridor connections between zones."""
        draw = ImageDraw.Draw(image)
        corridor_color = (240, 240, 240)  # Light gray
        corridor_width = max(4, self.zone_size // 16)

        center = self.grid_size // 2
        center_pixel = center * self.zone_size + self.zone_size // 2

        for zone in zones:
            if zone.zone_type == ZoneType.CORE:
                continue

            row, col = zone.grid_position
            zone_center_x = col * self.zone_size + self.zone_size // 2
            zone_center_y = row * self.zone_size + self.zone_size // 2

            # Draw corridor from zone to core
            # Horizontal then vertical (L-shaped) or direct based on position
            if row == center:
                # Same row - horizontal connection
                x1 = min(zone_center_x, center_pixel)
                x2 = max(zone_center_x, center_pixel)
                draw.rectangle([
                    x1, center_pixel - corridor_width // 2,
                    x2, center_pixel + corridor_width // 2
                ], fill=corridor_color)
            elif col == center:
                # Same column - vertical connection
                y1 = min(zone_center_y, center_pixel)
                y2 = max(zone_center_y, center_pixel)
                draw.rectangle([
                    center_pixel - corridor_width // 2, y1,
                    center_pixel + corridor_width // 2, y2
                ], fill=corridor_color)
            else:
                # Diagonal - L-shaped connection
                # First horizontal to center column
                x1 = min(zone_center_x, center_pixel)
                x2 = max(zone_center_x, center_pixel)
                draw.rectangle([
                    x1, zone_center_y - corridor_width // 2,
                    x2, zone_center_y + corridor_width // 2
                ], fill=corridor_color)
                # Then vertical to center row
                y1 = min(zone_center_y, center_pixel)
                y2 = max(zone_center_y, center_pixel)
                draw.rectangle([
                    center_pixel - corridor_width // 2, y1,
                    center_pixel + corridor_width // 2, y2
                ], fill=corridor_color)

        return image

    def _add_grid_lines(self, image: Image.Image) -> Image.Image:
        """Add grid lines to visualize zone boundaries."""
        draw = ImageDraw.Draw(image)
        line_color = (200, 200, 200)

        for i in range(1, self.grid_size):
            # Vertical lines
            x = i * self.zone_size
            draw.line([(x, 0), (x, self.assembly_size)], fill=line_color, width=1)
            # Horizontal lines
            y = i * self.zone_size
            draw.line([(0, y), (self.assembly_size, y)], fill=line_color, width=1)

        return image

    def scale_to_output(self, image: Image.Image) -> Image.Image:
        """Scale floor plan to final output size."""
        return image.resize((self.output_size, self.output_size), Image.Resampling.NEAREST)

    def add_annotations(
        self,
        image: Image.Image,
        floor_side_ft: float,
        floor_number: int,
        scale_bar: bool = True
    ) -> Image.Image:
        """
        Add dimension annotations and labels to floor plan.

        Args:
            image: Floor plan image
            floor_side_ft: Floor side length in feet
            floor_number: Floor number for labeling
            scale_bar: Whether to add a scale bar

        Returns:
            Annotated floor plan image
        """
        # Add margin for annotations
        margin = 50
        new_size = (image.width + margin * 2, image.height + margin * 2)
        annotated = Image.new("RGB", new_size, (255, 255, 255))
        annotated.paste(image, (margin, margin))

        draw = ImageDraw.Draw(annotated)

        # Try to load font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()
            small_font = font

        # Add floor label
        label = f"Floor {floor_number}"
        draw.text((margin, 10), label, fill=(0, 0, 0), font=font)

        # Add dimension labels
        dim_label = f"{floor_side_ft:.0f} ft"

        # Bottom dimension
        bbox = draw.textbbox((0, 0), dim_label, font=small_font)
        text_w = bbox[2] - bbox[0]
        x_pos = margin + (image.width - text_w) // 2
        y_pos = margin + image.height + 5
        draw.text((x_pos, y_pos), dim_label, fill=(0, 0, 0), font=small_font)

        # Add dimension line
        draw.line([
            (margin, y_pos + 15),
            (margin + image.width, y_pos + 15)
        ], fill=(100, 100, 100), width=1)
        # End caps
        draw.line([(margin, y_pos + 10), (margin, y_pos + 20)], fill=(100, 100, 100))
        draw.line([(margin + image.width, y_pos + 10), (margin + image.width, y_pos + 20)], fill=(100, 100, 100))

        # Left dimension (rotated text is complex, just add label)
        draw.text((10, margin + image.height // 2), dim_label, fill=(0, 0, 0), font=small_font)

        # Add scale bar if requested
        if scale_bar:
            self._add_scale_bar(draw, annotated.width, annotated.height, floor_side_ft / image.width, small_font)

        return annotated

    def _add_scale_bar(
        self,
        draw: ImageDraw.Draw,
        img_width: int,
        img_height: int,
        ft_per_pixel: float,
        font
    ):
        """Add a scale bar to the image."""
        # Scale bar at 50 ft
        scale_ft = 50
        scale_px = int(scale_ft / ft_per_pixel)

        # Position in bottom right
        x1 = img_width - 60 - scale_px
        y = img_height - 30

        draw.rectangle([x1, y, x1 + scale_px, y + 5], fill=(50, 50, 50))
        draw.text((x1 + scale_px // 2 - 15, y + 8), f"{scale_ft} ft", fill=(50, 50, 50), font=font)

    def create_legend(self) -> Image.Image:
        """Create a color legend for room types."""
        legend_height = 20 * 13  # 13 room types
        legend_width = 150
        legend = Image.new("RGB", (legend_width, legend_height), (255, 255, 255))
        draw = ImageDraw.Draw(legend)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
        except:
            font = ImageFont.load_default()

        room_names = [
            "LivingRoom", "MasterRoom", "Kitchen", "Bathroom",
            "DiningRoom", "ChildRoom", "StudyRoom", "SecondRoom",
            "GuestRoom", "Balcony", "Entrance", "Storage", "Corridor"
        ]

        for i, name in enumerate(room_names):
            y = i * 20
            color = ROOM_COLORS.get(i, (200, 200, 200))

            # Color box
            draw.rectangle([5, y + 2, 25, y + 18], fill=color, outline=(100, 100, 100))
            # Label
            draw.text((30, y + 3), name, fill=(0, 0, 0), font=font)

        return legend


def assemble_floor_plan(
    core_template: CoreTemplate,
    zones: List[Zone],
    wing_results: Dict[str, GenerationResult],
    floor_side_ft: float,
    floor_number: int,
    output_size: int = 512,
    annotate: bool = True
) -> Image.Image:
    """
    Convenience function to assemble a complete floor plan.

    Args:
        core_template: Core zone template
        zones: List of all zones
        wing_results: Generated wing zone images
        floor_side_ft: Floor side in feet
        floor_number: Floor number for labeling
        output_size: Final image size
        annotate: Whether to add annotations

    Returns:
        Assembled and optionally annotated floor plan
    """
    grid_size = int(np.sqrt(len(zones) + 1))  # +1 for core if not in list
    assembler = TileAssembler(grid_size=grid_size, output_size=output_size)

    # Find core zone
    core_zone = next((z for z in zones if z.zone_type == ZoneType.CORE), None)
    if not core_zone:
        # Create default core zone
        center = grid_size // 2
        core_zone = Zone(
            zone_id="core",
            zone_type=ZoneType.CORE,
            grid_position=(center, center),
            direction="center",
            pixel_bounds=(0, 0, 64, 64)
        )

    # Render core template
    core_image = core_template.render()

    # Assemble floor
    floor_plan = assembler.assemble_floor(
        core_image=core_image,
        wing_results=wing_results,
        zones=zones,
        core_zone=core_zone
    )

    # Scale to output size
    floor_plan = assembler.scale_to_output(floor_plan)

    # Add annotations
    if annotate:
        floor_plan = assembler.add_annotations(
            floor_plan,
            floor_side_ft,
            floor_number
        )

    return floor_plan


if __name__ == "__main__":
    # Test assembly with mock data
    from .floor_tiler import Zone, ZoneType
    from .core_template import CoreTemplate

    # Create mock zones
    zones = []
    for row in range(3):
        for col in range(3):
            if row == 1 and col == 1:
                zone_type = ZoneType.CORE
                zone_id = "core"
            else:
                zone_type = ZoneType.WING
                directions = ["northwest", "north", "northeast",
                              "west", "center", "east",
                              "southwest", "south", "southeast"]
                zone_id = f"wing_{directions[row * 3 + col]}"

            zones.append(Zone(
                zone_id=zone_id,
                zone_type=zone_type,
                grid_position=(row, col),
                direction=directions[row * 3 + col] if zone_type == ZoneType.WING else "center",
                pixel_bounds=(col * 64, row * 64, (col + 1) * 64, (row + 1) * 64)
            ))

    # Create mock wing results
    wing_results = {}
    for zone in zones:
        if zone.zone_type == ZoneType.WING:
            mock_img = Image.new("RGB", (64, 64), (200, 220, 255))
            draw = ImageDraw.Draw(mock_img)
            draw.text((5, 25), zone.direction[:4], fill=(0, 0, 0))
            wing_results[zone.zone_id] = GenerationResult(
                zone_id=zone.zone_id,
                image=mock_img,
                grayscale=np.zeros((64, 64)),
                json_spec="{}",
                success=True
            )

    # Create core template
    core_template = CoreTemplate.create_default()

    # Assemble
    floor_plan = assemble_floor_plan(
        core_template=core_template,
        zones=zones,
        wing_results=wing_results,
        floor_side_ft=172.4,
        floor_number=3
    )

    floor_plan.save("assembled_floor_test.png")
    print("Saved assembled_floor_test.png")
