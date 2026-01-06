#!/usr/bin/env python3
"""
Multi-Floor Apartment Massing Generator

Main entry point for generating multi-floor apartment layouts
from CSV building program data using the ChatHouseDiffusion model.

Usage:
    python -m multifloor_generator.main --csv data.csv --output ./generated
    python -m multifloor_generator.main --csv data.csv --mock  # Test without model
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from datetime import datetime

from PIL import Image

from .config import GenerationConfig, DASHSCOPE_CONFIG, MODEL_CONFIG
from .csv_parser import parse_building_csv, print_building_summary, BuildingSpec
from .floor_tiler import FloorTiler, Zone, create_floor_zones
from .core_template import CoreTemplate
from .unit_zone_generator import UnitZoneGenerator, MockGenerator, GenerationResult
from .llm_preprocessor import LLMPreprocessor, RuleBasedPreprocessor
from .tile_assembler import TileAssembler, assemble_floor_plan
from .floor_stacker import FloorStacker, create_building_visualization


class MultiFloorGenerator:
    """
    Main pipeline for generating multi-floor apartment layouts.
    """

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        use_mock: bool = False,
        use_llm: bool = True
    ):
        """
        Initialize the generator pipeline.

        Args:
            config: Generation configuration
            use_mock: Use mock generator (for testing without model)
            use_llm: Use LLM for room spec enhancement
        """
        self.config = config or GenerationConfig()

        # Initialize components
        if use_mock:
            self.generator = MockGenerator()
        else:
            self.generator = UnitZoneGenerator(
                model_path=MODEL_CONFIG["results_folder"],
                milestone=MODEL_CONFIG["milestone"]
            )

        if use_llm:
            self.preprocessor = LLMPreprocessor(
                api_key=DASHSCOPE_CONFIG["api_key"],
                base_url=DASHSCOPE_CONFIG["base_url"],
                model=DASHSCOPE_CONFIG["model"]
            )
        else:
            self.preprocessor = RuleBasedPreprocessor()

        self.assembler = TileAssembler(
            grid_size=self.config.grid_size,
            output_size=self.config.output_size
        )
        self.stacker = FloorStacker()

        self.building_spec: Optional[BuildingSpec] = None
        self.generated_floors: List[Tuple[int, Image.Image]] = []

    def load_building(self, csv_path: str) -> BuildingSpec:
        """
        Load building specification from CSV.

        Args:
            csv_path: Path to CSV file

        Returns:
            Parsed BuildingSpec
        """
        print(f"Loading building data from: {csv_path}")
        self.building_spec = parse_building_csv(csv_path)
        print_building_summary(self.building_spec)
        return self.building_spec

    def generate_floor(
        self,
        floor_number: int,
        save_intermediate: bool = True
    ) -> Tuple[Image.Image, Dict]:
        """
        Generate a single floor layout.

        Args:
            floor_number: Floor number to generate
            save_intermediate: Whether to save intermediate results

        Returns:
            Tuple of (floor_image, generation_info)
        """
        if not self.building_spec:
            raise ValueError("Building spec not loaded. Call load_building first.")

        # Get floor spec
        floor_spec = None
        for fs in self.building_spec.floor_specs:
            if fs.floor_number == floor_number:
                floor_spec = fs
                break

        if not floor_spec:
            raise ValueError(f"Floor {floor_number} not found in building spec")

        print(f"\n{'='*50}")
        print(f"Generating Floor {floor_number}")
        print(f"{'='*50}")

        # Create zones
        tiler = FloorTiler(floor_spec, self.config.grid_size)
        zones = tiler.create_zones()

        print(tiler.get_zone_summary())

        # Get core template
        core_template = tiler.get_core_template()

        # Enhance zone specs with LLM
        print("\nEnhancing room specifications...")
        wing_zones = [z for z in zones if z.zone_id != "core"]

        for zone in wing_zones:
            enhanced = self.preprocessor.enhance_zone_spec(
                zone,
                floor_context=f"Floor {floor_number} of {self.building_spec.name}"
            )
            if enhanced.success and enhanced.enhanced_json:
                # Update zone with enhanced spec (parsed back to rooms)
                pass  # Zone already has rooms, enhancement modifies JSON output

        # Generate wing zones
        print("\nGenerating wing zones...")
        wing_results: Dict[str, GenerationResult] = {}

        for zone in wing_zones:
            mask = tiler.get_wing_zone_mask(zone)
            result = self.generator.generate_zone(zone, mask)
            wing_results[zone.zone_id] = result
            print(f"  {zone.zone_id}: {'OK' if result.success else 'FAILED'} ({zone.room_count()} rooms)")

        # Assemble floor plan
        print("\nAssembling floor plan...")
        floor_plan = assemble_floor_plan(
            core_template=core_template,
            zones=zones,
            wing_results=wing_results,
            floor_side_ft=floor_spec.floor_side_ft,
            floor_number=floor_number,
            output_size=self.config.output_size,
            annotate=True
        )

        # Save intermediate results
        info = {
            "floor_number": floor_number,
            "zones_generated": len(wing_zones),
            "total_rooms": sum(z.room_count() for z in wing_zones),
            "success": all(r.success for r in wing_results.values())
        }

        if save_intermediate and self.config.save_intermediate:
            output_dir = Path(self.config.output_dir) / f"floor_{floor_number}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save floor plan
            floor_plan.save(output_dir / "floor_plan.png")

            # Save zone images
            for zone_id, result in wing_results.items():
                result.image.save(output_dir / f"zone_{zone_id}.png")

            # Save info
            with open(output_dir / "info.json", "w") as f:
                json.dump(info, f, indent=2)

            print(f"Saved intermediate results to: {output_dir}")

        return floor_plan, info

    def generate_all_floors(self) -> List[Tuple[int, Image.Image]]:
        """
        Generate all floors in the building.

        Returns:
            List of (floor_number, floor_image) tuples
        """
        if not self.building_spec:
            raise ValueError("Building spec not loaded")

        self.generated_floors = []

        for floor_spec in self.building_spec.floor_specs:
            floor_plan, info = self.generate_floor(floor_spec.floor_number)
            self.generated_floors.append((floor_spec.floor_number, floor_plan))

        return self.generated_floors

    def create_visualizations(self) -> Dict[str, str]:
        """
        Create all building visualizations.

        Returns:
            Dictionary mapping visualization type to file path
        """
        if not self.generated_floors:
            raise ValueError("No floors generated")

        return create_building_visualization(
            floors=self.generated_floors,
            building_spec=self.building_spec,
            output_dir=self.config.output_dir
        )

    def run(self, csv_path: str, floors: Optional[List[int]] = None) -> Dict[str, str]:
        """
        Run the complete generation pipeline.

        Args:
            csv_path: Path to building CSV
            floors: Specific floor numbers to generate (None = all)

        Returns:
            Dictionary of output file paths
        """
        # Load building
        self.load_building(csv_path)

        # Initialize model if needed
        if not isinstance(self.generator, MockGenerator):
            print("\nLoading diffusion model...")
            if not self.generator.load_model():
                print("Warning: Model not loaded, using mock generation")
                self.generator = MockGenerator()

        # Initialize LLM
        if isinstance(self.preprocessor, LLMPreprocessor):
            print("\nInitializing LLM preprocessor...")
            self.preprocessor.initialize()

        # Generate floors
        if floors:
            for floor_num in floors:
                floor_plan, info = self.generate_floor(floor_num)
                self.generated_floors.append((floor_num, floor_plan))
        else:
            self.generate_all_floors()

        # Create visualizations
        print("\nCreating visualizations...")
        outputs = self.create_visualizations()

        print(f"\n{'='*50}")
        print("Generation Complete!")
        print(f"{'='*50}")
        for viz_type, path in outputs.items():
            print(f"  {viz_type}: {path}")

        return outputs


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate multi-floor apartment layouts from CSV building data"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to building program CSV file"
    )

    parser.add_argument(
        "--output", "-o",
        default="./generated_floors",
        help="Output directory for generated images"
    )

    parser.add_argument(
        "--floors", "-f",
        type=int,
        nargs="+",
        help="Specific floor numbers to generate (default: all)"
    )

    parser.add_argument(
        "--grid-size", "-g",
        type=int,
        default=3,
        help="Grid size for zone tiling (default: 3)"
    )

    parser.add_argument(
        "--output-size", "-s",
        type=int,
        default=512,
        help="Output image size in pixels (default: 512)"
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock generator (for testing without model)"
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM enhancement (use rule-based)"
    )

    parser.add_argument(
        "--no-save-intermediate",
        action="store_true",
        help="Don't save intermediate zone images"
    )

    args = parser.parse_args()

    # Create configuration
    config = GenerationConfig(
        grid_size=args.grid_size,
        output_size=args.output_size,
        use_llm_enhancement=not args.no_llm,
        save_intermediate=not args.no_save_intermediate,
        output_dir=args.output
    )

    # Create generator
    generator = MultiFloorGenerator(
        config=config,
        use_mock=args.mock,
        use_llm=not args.no_llm
    )

    # Run pipeline
    try:
        outputs = generator.run(args.csv, args.floors)
        return 0
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
