"""
Floor Stacker.

Creates multi-floor visualizations from individual floor plans.
"""

import os
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from .csv_parser import BuildingSpec


class FloorStacker:
    """
    Creates multi-floor visualizations from individual floor plans.

    Supports:
    - Vertical stacking (isometric-style view)
    - Side-by-side grid layout
    - Animated GIF showing floor-by-floor
    - Core alignment verification
    """

    def __init__(
        self,
        floor_height_ft: float = 10.5,
        spacing_px: int = 20
    ):
        """
        Initialize floor stacker.

        Args:
            floor_height_ft: Height of each floor in feet
            spacing_px: Pixel spacing between floors in visualizations
        """
        self.floor_height_ft = floor_height_ft
        self.spacing_px = spacing_px

    def create_stacked_view(
        self,
        floors: List[Tuple[int, Image.Image]],
        building_name: str = "Building"
    ) -> Image.Image:
        """
        Create a vertically stacked visualization.

        Args:
            floors: List of (floor_number, floor_image) tuples
            building_name: Name for title

        Returns:
            Stacked visualization image
        """
        if not floors:
            return Image.new("RGB", (400, 400), (255, 255, 255))

        # Sort floors by number (top floor first for display)
        floors = sorted(floors, key=lambda x: x[0], reverse=True)

        # Get dimensions
        sample_img = floors[0][1]
        floor_w, floor_h = sample_img.size
        num_floors = len(floors)

        # Calculate canvas size
        margin = 80
        total_height = num_floors * floor_h + (num_floors - 1) * self.spacing_px + margin * 2
        canvas_width = floor_w + margin * 2

        canvas = Image.new("RGB", (canvas_width, total_height), (245, 245, 250))
        draw = ImageDraw.Draw(canvas)

        # Load font
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
            small_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()
            small_font = font

        # Add title
        draw.text((margin, 15), building_name, fill=(0, 0, 0), font=font)

        # Place floors
        y_offset = margin
        for floor_num, floor_img in floors:
            # Place floor image
            canvas.paste(floor_img, (margin, y_offset))

            # Add floor label
            label = f"F{floor_num}"
            draw.text((10, y_offset + floor_h // 2 - 8), label, fill=(50, 50, 50), font=small_font)

            # Add height annotation
            height_label = f"{floor_num * self.floor_height_ft:.0f} ft"
            draw.text((canvas_width - 70, y_offset + floor_h // 2 - 8), height_label, fill=(100, 100, 100), font=small_font)

            y_offset += floor_h + self.spacing_px

        # Add building height at bottom
        total_height_ft = len(floors) * self.floor_height_ft
        draw.text(
            (canvas_width // 2 - 50, total_height - 30),
            f"Total: {total_height_ft:.0f} ft",
            fill=(0, 0, 0),
            font=font
        )

        return canvas

    def create_grid_view(
        self,
        floors: List[Tuple[int, Image.Image]],
        cols: int = 4
    ) -> Image.Image:
        """
        Create a grid layout of all floors.

        Args:
            floors: List of (floor_number, floor_image) tuples
            cols: Number of columns in grid

        Returns:
            Grid visualization image
        """
        if not floors:
            return Image.new("RGB", (400, 400), (255, 255, 255))

        # Sort floors by number
        floors = sorted(floors, key=lambda x: x[0])

        sample_img = floors[0][1]
        floor_w, floor_h = sample_img.size
        num_floors = len(floors)

        rows = (num_floors + cols - 1) // cols
        padding = 10
        label_height = 25

        canvas_width = cols * (floor_w + padding) + padding
        canvas_height = rows * (floor_h + padding + label_height) + padding

        canvas = Image.new("RGB", (canvas_width, canvas_height), (240, 240, 245))
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()

        for i, (floor_num, floor_img) in enumerate(floors):
            row = i // cols
            col = i % cols

            x = col * (floor_w + padding) + padding
            y = row * (floor_h + padding + label_height) + padding + label_height

            # Add label
            draw.text((x + floor_w // 2 - 20, y - label_height + 5), f"Floor {floor_num}", fill=(50, 50, 50), font=font)

            # Place image
            canvas.paste(floor_img, (x, y))

        return canvas

    def create_animated_gif(
        self,
        floors: List[Tuple[int, Image.Image]],
        output_path: str,
        duration_ms: int = 1000,
        loop: bool = True
    ) -> str:
        """
        Create an animated GIF showing floors in sequence.

        Args:
            floors: List of (floor_number, floor_image) tuples
            output_path: Path to save GIF
            duration_ms: Duration per frame in milliseconds
            loop: Whether to loop the animation

        Returns:
            Path to saved GIF
        """
        if not floors:
            return ""

        # Sort floors by number
        floors = sorted(floors, key=lambda x: x[0])

        # Add floor labels to each frame
        frames = []
        for floor_num, floor_img in floors:
            # Create frame with label
            frame = floor_img.copy()
            draw = ImageDraw.Draw(frame)

            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
            except:
                font = ImageFont.load_default()

            # Add floor number overlay
            label = f"Floor {floor_num}"
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            draw.rectangle([5, 5, text_w + 15, 30], fill=(255, 255, 255, 200))
            draw.text((10, 7), label, fill=(0, 0, 0), font=font)

            frames.append(frame)

        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0 if loop else 1
        )

        return output_path

    def verify_core_alignment(
        self,
        floors: List[Tuple[int, np.ndarray]],
        core_bounds: Tuple[int, int, int, int],
        tolerance_px: int = 2
    ) -> Dict[str, any]:
        """
        Verify that core elements align across all floors.

        Args:
            floors: List of (floor_number, grayscale_array) tuples
            core_bounds: (x1, y1, x2, y2) of core region
            tolerance_px: Allowed misalignment in pixels

        Returns:
            Dictionary with alignment analysis
        """
        if not floors or len(floors) < 2:
            return {"aligned": True, "message": "Not enough floors to verify"}

        x1, y1, x2, y2 = core_bounds
        reference = floors[0][1][y1:y2, x1:x2]
        misaligned_floors = []

        for floor_num, gray_array in floors[1:]:
            current = gray_array[y1:y2, x1:x2]

            # Check if core regions match
            diff = np.abs(reference.astype(float) - current.astype(float))
            max_diff = np.max(diff)

            if max_diff > tolerance_px:
                misaligned_floors.append({
                    "floor": floor_num,
                    "max_difference": max_diff
                })

        return {
            "aligned": len(misaligned_floors) == 0,
            "total_floors": len(floors),
            "misaligned_floors": misaligned_floors,
            "message": "All cores aligned" if not misaligned_floors else f"{len(misaligned_floors)} floors misaligned"
        }

    def create_section_view(
        self,
        floors: List[Tuple[int, Image.Image]],
        building_spec: BuildingSpec,
        section_line: str = "center"
    ) -> Image.Image:
        """
        Create a cross-section view of the building.

        Args:
            floors: List of (floor_number, floor_image) tuples
            building_spec: Building specification for dimensions
            section_line: Where to cut section ("center", "north", "south")

        Returns:
            Section view image
        """
        if not floors:
            return Image.new("RGB", (400, 400), (255, 255, 255))

        # Sort floors by number
        floors = sorted(floors, key=lambda x: x[0])

        sample_img = floors[0][1]
        floor_w, floor_h = sample_img.size
        num_floors = len(floors)

        # Section is a slice through the building
        section_y = floor_h // 2  # Center cut
        if section_line == "north":
            section_y = floor_h // 4
        elif section_line == "south":
            section_y = 3 * floor_h // 4

        # Create section canvas
        floor_height_px = 30  # Pixel height per floor in section
        margin = 60
        canvas_width = floor_w + margin * 2
        canvas_height = num_floors * floor_height_px + margin * 2

        canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
        except:
            font = ImageFont.load_default()

        # Draw each floor's section
        for i, (floor_num, floor_img) in enumerate(floors):
            y_pos = canvas_height - margin - (i + 1) * floor_height_px

            # Extract section line from floor image
            floor_array = np.array(floor_img)
            section_strip = floor_array[section_y:section_y + 1, :, :]

            # Scale to section height
            section_img = Image.fromarray(section_strip)
            section_img = section_img.resize((floor_w, floor_height_px - 2))

            canvas.paste(section_img, (margin, y_pos + 1))

            # Add floor line
            draw.line([(margin, y_pos), (margin + floor_w, y_pos)], fill=(100, 100, 100))

            # Add floor label
            draw.text((5, y_pos + floor_height_px // 2 - 5), f"F{floor_num}", fill=(50, 50, 50), font=font)

        # Add title
        draw.text((margin, 10), f"Section View ({section_line})", fill=(0, 0, 0), font=font)

        # Add ground line
        ground_y = canvas_height - margin
        draw.line([(margin - 20, ground_y), (margin + floor_w + 20, ground_y)], fill=(0, 0, 0), width=2)

        return canvas


def create_building_visualization(
    floors: List[Tuple[int, Image.Image]],
    building_spec: BuildingSpec,
    output_dir: str = "./output"
) -> Dict[str, str]:
    """
    Create all visualizations for a building.

    Args:
        floors: List of (floor_number, floor_image) tuples
        building_spec: Building specification
        output_dir: Directory to save outputs

    Returns:
        Dictionary mapping visualization type to file path
    """
    os.makedirs(output_dir, exist_ok=True)
    outputs = {}

    stacker = FloorStacker(floor_height_ft=building_spec.gba_sf / building_spec.gfa_sf * 10)

    # Stacked view
    stacked = stacker.create_stacked_view(floors, building_spec.name)
    stacked_path = os.path.join(output_dir, "building_stacked.png")
    stacked.save(stacked_path)
    outputs["stacked"] = stacked_path

    # Grid view
    grid = stacker.create_grid_view(floors)
    grid_path = os.path.join(output_dir, "building_grid.png")
    grid.save(grid_path)
    outputs["grid"] = grid_path

    # Animated GIF
    gif_path = os.path.join(output_dir, "building_animation.gif")
    stacker.create_animated_gif(floors, gif_path)
    outputs["animation"] = gif_path

    # Section view
    section = stacker.create_section_view(floors, building_spec)
    section_path = os.path.join(output_dir, "building_section.png")
    section.save(section_path)
    outputs["section"] = section_path

    return outputs


if __name__ == "__main__":
    # Test with mock floors
    mock_floors = []
    for i in range(1, 9):
        img = Image.new("RGB", (200, 200), (200, 220, 255))
        draw = ImageDraw.Draw(img)
        draw.text((80, 90), f"Floor {i}", fill=(0, 0, 0))
        # Add some variation
        draw.rectangle([50, 50, 150, 150], outline=(100, 100, 150), width=2)
        mock_floors.append((i, img))

    stacker = FloorStacker()

    # Test stacked view
    stacked = stacker.create_stacked_view(mock_floors, "Test Building")
    stacked.save("test_stacked.png")
    print("Saved test_stacked.png")

    # Test grid view
    grid = stacker.create_grid_view(mock_floors)
    grid.save("test_grid.png")
    print("Saved test_grid.png")

    # Test animated GIF
    stacker.create_animated_gif(mock_floors, "test_animation.gif")
    print("Saved test_animation.gif")
