"""
Unit Zone Generator.

Generates dwelling unit layouts for wing zones using the diffusion model.
Handles the integration between zone specifications and the ChatHouseDiffusion model.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image

# Add ChatHouseDiffusion to path
CHATHOUSEDIFFUSION_PATH = Path(__file__).parent.parent / "chathousediffusion_gloq"
if str(CHATHOUSEDIFFUSION_PATH) not in sys.path:
    sys.path.insert(0, str(CHATHOUSEDIFFUSION_PATH))

from .config import MODEL_CONFIG, MODEL_OUTPUT_SIZE
from .floor_tiler import Zone, ZoneType
from .utils.image_utils import ImageUtils


@dataclass
class GenerationResult:
    """Result of a zone generation."""
    zone_id: str
    image: Image.Image
    grayscale: np.ndarray
    json_spec: str
    success: bool
    error: Optional[str] = None


class UnitZoneGenerator:
    """
    Generates floor plan layouts for wing zones using the diffusion model.

    Integrates with ChatHouseDiffusion's trainer.predict() method.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        milestone: int = MODEL_CONFIG["milestone"],
        inject_step: int = MODEL_CONFIG["inject_step"],
        cond_scale: float = MODEL_CONFIG["cond_scale"]
    ):
        """
        Initialize the zone generator.

        Args:
            model_path: Path to model checkpoint folder
            milestone: Model checkpoint milestone number
            inject_step: Step at which to inject conditioning
            cond_scale: Conditioning scale for guidance
        """
        self.model_path = model_path or MODEL_CONFIG["results_folder"]
        self.milestone = milestone
        self.inject_step = inject_step
        self.cond_scale = cond_scale

        self.trainer = None
        self._model_loaded = False

    def load_model(self) -> bool:
        """
        Load the diffusion model.

        Returns:
            True if model loaded successfully
        """
        if self._model_loaded:
            return True

        try:
            import pickle
            # Import model components from ChatHouseDiffusion
            from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

            print(f"Loading model from: {self.model_path}")

            # Load params
            params_path = os.path.join(self.model_path, "params.pkl")
            if not os.path.exists(params_path):
                raise FileNotFoundError(f"params.pkl not found at {params_path}")

            with open(params_path, "rb") as f:
                params = pickle.load(f)

            # Create model
            model = Unet(**params["unet_dict"])
            diffusion = GaussianDiffusion(model, **params["diffusion_dict"])

            # Create trainer in predict mode
            self.trainer = Trainer(
                diffusion,
                "",  # No training data needed
                "",
                "",
                **params["trainer_dict"],
                results_folder=self.model_path,
                train_num_workers=0,
                mode="predict",
                inject_step=self.inject_step
            )

            # Load model weights
            self.trainer.predict_load(self.milestone)

            # Configure trainer
            if hasattr(self.trainer, 'cond_scale'):
                self.trainer.cond_scale = self.cond_scale

            self._model_loaded = True
            print(f"Model loaded successfully (milestone {self.milestone})")
            return True

        except ImportError as e:
            print(f"Failed to import ChatHouseDiffusion: {e}")
            print("Make sure the model is available and dependencies are installed.")
            return False
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            return False
        except Exception as e:
            print(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_zone(
        self,
        zone: Zone,
        mask: Optional[Image.Image] = None,
        json_override: Optional[str] = None
    ) -> GenerationResult:
        """
        Generate a floor plan layout for a zone.

        Args:
            zone: Zone specification with room assignments
            mask: Optional constraint mask (white=valid, black=blocked)
            json_override: Override the zone's JSON spec

        Returns:
            GenerationResult with generated image
        """
        # Get JSON specification
        json_spec = json_override or zone.to_json()

        if not json_spec or json_spec == "{}":
            # Return empty zone
            return GenerationResult(
                zone_id=zone.zone_id,
                image=ImageUtils.create_blank_image(MODEL_OUTPUT_SIZE),
                grayscale=np.zeros((MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE)),
                json_spec=json_spec,
                success=True
            )

        # Create default mask if not provided
        if mask is None:
            mask = Image.new("L", (MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE), 255)

        # Convert mask to expected format
        mask_array = np.array(mask).astype(np.float32) / 255.0

        if not self._model_loaded:
            # Return mock result if model not loaded
            return self._generate_mock(zone, json_spec)

        try:
            # Call diffusion model
            # trainer.predict() returns a PIL Image directly (RGB)
            output = self.trainer.predict(
                feature=mask,  # Can be PIL Image, will be transformed internally
                text=json_spec,
                repredict=False
            )

            # output is already a PIL Image (RGB) from trainer.py:445
            if isinstance(output, Image.Image):
                image = output
                # Convert to grayscale for room segmentation analysis
                grayscale = np.array(output.convert('L'))
            else:
                # Fallback: handle tensor output (shouldn't happen with standard trainer)
                if hasattr(output, 'cpu'):
                    output_array = output.cpu().numpy()
                else:
                    output_array = np.array(output)

                # Handle different output shapes
                if output_array.ndim == 4:
                    output_array = output_array[0]  # Remove batch dim

                # Convert to image
                if output_array.shape[0] == 18:
                    grayscale = output_array.argmax(axis=0).astype(np.uint8)
                elif output_array.shape[0] == 1:
                    grayscale = (output_array[0] * 17).clip(0, 17).astype(np.uint8)
                else:
                    grayscale = output_array.astype(np.uint8)

                image = ImageUtils.grayscale_to_rgb(grayscale)

            return GenerationResult(
                zone_id=zone.zone_id,
                image=image,
                grayscale=grayscale,
                json_spec=json_spec,
                success=True
            )

        except Exception as e:
            return GenerationResult(
                zone_id=zone.zone_id,
                image=ImageUtils.create_blank_image(MODEL_OUTPUT_SIZE),
                grayscale=np.zeros((MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE)),
                json_spec=json_spec,
                success=False,
                error=str(e)
            )

    def _generate_mock(self, zone: Zone, json_spec: str) -> GenerationResult:
        """
        Generate a mock result when model is not available.

        Creates a placeholder visualization based on room specifications.
        """
        from PIL import ImageDraw
        from .config import ROOM_COLORS

        img = Image.new("RGB", (MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Parse room specs
        try:
            spec = json.loads(json_spec)
        except:
            spec = {}

        # Draw rooms in a grid layout
        room_list = []
        for room_type, data in spec.items():
            for room in data.get("rooms", []):
                room_list.append((room_type, room.get("name", "?"), room.get("size", "M")))

        if not room_list:
            return GenerationResult(
                zone_id=zone.zone_id,
                image=img,
                grayscale=np.ones((MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE), dtype=np.uint8) * 17,
                json_spec=json_spec,
                success=True
            )

        # Layout rooms
        num_rooms = len(room_list)
        cols = int(np.ceil(np.sqrt(num_rooms)))
        rows = int(np.ceil(num_rooms / cols))

        cell_w = MODEL_OUTPUT_SIZE // cols
        cell_h = MODEL_OUTPUT_SIZE // rows

        # Room type to color index
        type_to_idx = {
            "LivingRoom": 0, "MasterRoom": 1, "Kitchen": 2, "Bathroom": 3,
            "DiningRoom": 4, "ChildRoom": 5, "StudyRoom": 6, "SecondRoom": 7,
            "GuestRoom": 8, "Balcony": 9, "Entrance": 10, "Storage": 11
        }

        grayscale = np.ones((MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE), dtype=np.uint8) * 17

        for i, (room_type, name, size) in enumerate(room_list):
            row = i // cols
            col = i % cols

            x1 = col * cell_w + 2
            y1 = row * cell_h + 2
            x2 = (col + 1) * cell_w - 2
            y2 = (row + 1) * cell_h - 2

            # Get color
            idx = type_to_idx.get(room_type, 11)
            color = ROOM_COLORS.get(idx, (200, 200, 200))

            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(100, 100, 100))
            draw.text((x1 + 2, y1 + 2), name[:8], fill=(0, 0, 0))

            # Fill grayscale
            grayscale[y1:y2, x1:x2] = idx

        return GenerationResult(
            zone_id=zone.zone_id,
            image=img,
            grayscale=grayscale,
            json_spec=json_spec,
            success=True
        )

    def generate_multiple_zones(
        self,
        zones: List[Zone],
        masks: Optional[Dict[str, Image.Image]] = None
    ) -> List[GenerationResult]:
        """
        Generate layouts for multiple zones.

        Args:
            zones: List of zones to generate
            masks: Optional dictionary mapping zone_id to mask

        Returns:
            List of GenerationResult objects
        """
        results = []

        for zone in zones:
            mask = masks.get(zone.zone_id) if masks else None
            result = self.generate_zone(zone, mask)
            results.append(result)
            print(f"Generated {zone.zone_id}: {'success' if result.success else 'failed'}")

        return results

    def regenerate_zone(
        self,
        zone: Zone,
        previous_result: GenerationResult,
        mask: Optional[Image.Image] = None
    ) -> GenerationResult:
        """
        Regenerate a zone with a different random seed.

        Uses the repredict=True mode for variations.
        """
        if not self._model_loaded:
            return self._generate_mock(zone, zone.to_json())

        try:
            output = self.trainer.predict(
                feature=np.array(mask or Image.new("L", (MODEL_OUTPUT_SIZE, MODEL_OUTPUT_SIZE), 255)).astype(np.float32) / 255.0,
                text=zone.to_json(),
                repredict=True  # Enable variation mode
            )

            if hasattr(output, 'cpu'):
                output_array = output.cpu().numpy()
            else:
                output_array = np.array(output)

            if output_array.ndim == 4:
                output_array = output_array[0]

            if output_array.shape[0] == 18:
                grayscale = output_array.argmax(axis=0).astype(np.uint8)
            else:
                grayscale = (output_array[0] * 17).clip(0, 17).astype(np.uint8)

            image = ImageUtils.grayscale_to_rgb(grayscale)

            return GenerationResult(
                zone_id=zone.zone_id,
                image=image,
                grayscale=grayscale,
                json_spec=zone.to_json(),
                success=True
            )

        except Exception as e:
            return GenerationResult(
                zone_id=zone.zone_id,
                image=previous_result.image,
                grayscale=previous_result.grayscale,
                json_spec=zone.to_json(),
                success=False,
                error=str(e)
            )


class MockGenerator(UnitZoneGenerator):
    """
    Mock generator for testing without the actual diffusion model.

    Generates placeholder visualizations based on room specifications.
    """

    def __init__(self):
        super().__init__()
        self._model_loaded = False  # Always use mock generation

    def load_model(self) -> bool:
        """Mock model is always ready."""
        return True

    def generate_zone(
        self,
        zone: Zone,
        mask: Optional[Image.Image] = None,
        json_override: Optional[str] = None
    ) -> GenerationResult:
        """Generate mock layout."""
        json_spec = json_override or zone.to_json()
        return self._generate_mock(zone, json_spec)


def create_generator(use_mock: bool = False) -> UnitZoneGenerator:
    """
    Factory function to create appropriate generator.

    Args:
        use_mock: If True, create mock generator (for testing)

    Returns:
        UnitZoneGenerator instance
    """
    if use_mock:
        return MockGenerator()
    return UnitZoneGenerator()


if __name__ == "__main__":
    # Test with mock generator
    from .floor_tiler import Zone, ZoneType
    from .utils.room_mapper import RoomSpec

    # Create test zone
    test_zone = Zone(
        zone_id="wing_north",
        zone_type=ZoneType.WING,
        grid_position=(0, 1),
        direction="north",
        pixel_bounds=(64, 0, 128, 64),
        rooms=[
            RoomSpec("Living_1", "LivingRoom", "north", "M", ["Kitchen_1"]),
            RoomSpec("Kitchen_1", "Kitchen", "north", "S", ["Living_1"]),
            RoomSpec("Bedroom_1", "MasterRoom", "north", "M", ["Bath_1"]),
            RoomSpec("Bath_1", "Bathroom", "north", "XS", ["Bedroom_1"]),
        ]
    )

    generator = MockGenerator()
    result = generator.generate_zone(test_zone)

    print(f"Generated {result.zone_id}: success={result.success}")
    result.image.save("test_zone_generation.png")
    print("Saved test_zone_generation.png")
