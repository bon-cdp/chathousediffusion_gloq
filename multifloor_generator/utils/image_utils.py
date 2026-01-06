"""
Image manipulation utilities.

Handles image creation, blending, and visualization.
"""

import numpy as np
from typing import Tuple, List, Optional
from PIL import Image, ImageDraw, ImageFont

from ..config import ROOM_COLORS, MODEL_OUTPUT_SIZE


class ImageUtils:
    """Utility class for image operations."""

    @staticmethod
    def create_blank_image(
        size: int = 64,
        color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """Create a blank RGB image."""
        return Image.new("RGB", (size, size), color)

    @staticmethod
    def create_mask(
        size: int = 64,
        mask_regions: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> Image.Image:
        """
        Create a binary mask image.

        Args:
            size: Image size
            mask_regions: List of (x1, y1, x2, y2) regions to mask (black)

        Returns:
            Grayscale image (white = valid, black = masked)
        """
        mask = Image.new("L", (size, size), 255)
        if mask_regions:
            draw = ImageDraw.Draw(mask)
            for x1, y1, x2, y2 in mask_regions:
                draw.rectangle([x1, y1, x2, y2], fill=0)
        return mask

    @staticmethod
    def grayscale_to_rgb(gray_image: np.ndarray) -> Image.Image:
        """
        Convert grayscale room labels to RGB visualization.

        Matches image_process.py color mapping.
        """
        if isinstance(gray_image, Image.Image):
            gray_image = np.array(gray_image)

        h, w = gray_image.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for label, color in ROOM_COLORS.items():
            mask = gray_image == label
            rgb[mask] = color

        return Image.fromarray(rgb)

    @staticmethod
    def blend_zones(
        base: Image.Image,
        overlay: Image.Image,
        position: Tuple[int, int],
        blend_width: int = 4
    ) -> Image.Image:
        """
        Blend an overlay zone onto a base image with soft edges.

        Args:
            base: Base floor plan image
            overlay: Zone image to overlay
            position: (x, y) position for top-left of overlay
            blend_width: Width of blending zone in pixels
        """
        base = base.copy()
        x, y = position

        # Create alpha mask for blending
        ow, oh = overlay.size
        alpha = Image.new("L", (ow, oh), 255)
        draw = ImageDraw.Draw(alpha)

        # Fade edges
        for i in range(blend_width):
            alpha_val = int(255 * (i + 1) / blend_width)
            draw.rectangle([i, i, ow - 1 - i, oh - 1 - i], outline=alpha_val)

        # Composite
        overlay_rgba = overlay.convert("RGBA")
        overlay_rgba.putalpha(alpha)

        base_rgba = base.convert("RGBA")
        base_rgba.paste(overlay_rgba, position, overlay_rgba)

        return base_rgba.convert("RGB")

    @staticmethod
    def stitch_grid(
        zones: List[List[Image.Image]],
        grid_size: int = 3,
        zone_size: int = 64
    ) -> Image.Image:
        """
        Stitch a grid of zone images into a single floor plan.

        Args:
            zones: 2D list of zone images [row][col]
            grid_size: Grid dimension (e.g., 3 for 3x3)
            zone_size: Size of each zone image

        Returns:
            Combined floor plan image
        """
        output_size = grid_size * zone_size
        floor_plan = Image.new("RGB", (output_size, output_size), (255, 255, 255))

        for row in range(grid_size):
            for col in range(grid_size):
                if row < len(zones) and col < len(zones[row]):
                    zone_img = zones[row][col]
                    x = col * zone_size
                    y = row * zone_size
                    floor_plan.paste(zone_img, (x, y))

        return floor_plan

    @staticmethod
    def add_dimension_annotations(
        image: Image.Image,
        floor_side_ft: float,
        font_size: int = 12
    ) -> Image.Image:
        """Add real-world dimension annotations to floor plan."""
        img = image.copy()
        draw = ImageDraw.Draw(img)

        # Try to load a font, fall back to default
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            font = ImageFont.load_default()

        w, h = img.size

        # Add dimension labels
        label = f"{floor_side_ft:.0f} ft"

        # Bottom label
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text(((w - text_w) // 2, h - font_size - 5), label, fill=(0, 0, 0), font=font)

        # Left label (rotated)
        # For simplicity, just add at side
        draw.text((5, (h - font_size) // 2), label, fill=(0, 0, 0), font=font)

        return img

    @staticmethod
    def create_floor_stack_visualization(
        floors: List[Image.Image],
        floor_height_ft: float = 10.5,
        spacing: int = 10
    ) -> Image.Image:
        """
        Create a stacked visualization of multiple floors.

        Args:
            floors: List of floor plan images (bottom to top)
            floor_height_ft: Height of each floor in feet
            spacing: Pixel spacing between floors

        Returns:
            Vertically stacked visualization
        """
        if not floors:
            return Image.new("RGB", (100, 100), (255, 255, 255))

        w, h = floors[0].size
        num_floors = len(floors)
        total_height = num_floors * h + (num_floors - 1) * spacing + 100  # Extra for labels

        canvas = Image.new("RGB", (w + 80, total_height), (240, 240, 240))
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = ImageFont.load_default()

        # Place floors from top (highest) to bottom
        for i, floor_img in enumerate(reversed(floors)):
            y_pos = i * (h + spacing) + 50
            canvas.paste(floor_img, (70, y_pos))

            # Add floor label
            floor_num = num_floors - i
            label = f"F{floor_num}"
            draw.text((10, y_pos + h // 2 - 7), label, fill=(0, 0, 0), font=font)

        # Add title
        draw.text((w // 2, 10), "Floor Stack", fill=(0, 0, 0), font=font)

        return canvas

    @staticmethod
    def tensor_to_image(tensor: np.ndarray) -> Image.Image:
        """
        Convert model output tensor to image.

        Handles both grayscale (1, H, W) and one-hot (18, H, W) formats.
        """
        if tensor.ndim == 3:
            if tensor.shape[0] == 1:
                # Grayscale: (1, H, W)
                gray = (tensor[0] * 17).astype(np.uint8)
                return ImageUtils.grayscale_to_rgb(gray)
            elif tensor.shape[0] == 18:
                # One-hot: (18, H, W) -> argmax to get labels
                labels = tensor.argmax(axis=0).astype(np.uint8)
                return ImageUtils.grayscale_to_rgb(labels)
        elif tensor.ndim == 2:
            # Already grayscale
            gray = (tensor * 17).astype(np.uint8)
            return ImageUtils.grayscale_to_rgb(gray)

        # Fallback
        return Image.fromarray((tensor * 255).astype(np.uint8))

    @staticmethod
    def resize_floor_plan(image: Image.Image, target_size: int = 512) -> Image.Image:
        """Resize floor plan to target size with nearest-neighbor interpolation."""
        return image.resize((target_size, target_size), Image.Resampling.NEAREST)
