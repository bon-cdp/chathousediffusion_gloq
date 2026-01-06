"""
Tests for CSV parser.

Uses expected values from 6464 Canoga Ave building data.
"""

import pytest
import os
from pathlib import Path

from multifloor_generator.csv_parser import (
    parse_building_csv,
    parse_sf_value,
    parse_dimension,
    parse_int_value,
    BuildingSpec,
    DwellingUnit,
    CoreElement,
)


# Expected values from 6464 Canoga Ave (first building in CSV)
EXPECTED_6464_CANOGA = {
    "name": "6464 Canoga Ave",
    "lot_size_sf": 77858,
    "far": 3.6,
    "gfa_sf": 279886,
    "gba_sf": 297432,
    "stories_total": 10,
    "stories_above_grade": 8,
    "stories_below_grade": 2,
    "typical_floor_plate_sf": 29743,
    "total_units": 377,
    "units_per_floor": 45,
    "dwelling_units": [
        {"type": "Studio", "count": 83, "sf": 447, "width": 17.2, "depth": 26.1, "baths": 1},
        {"type": "1BR", "count": 202, "sf": 552, "width": 22.3, "depth": 24.7, "baths": 1},
        {"type": "2BR", "count": 92, "sf": 867, "width": 41.6, "depth": 20.8, "baths": 2},
    ],
    "core_elements": {
        "elevator_passenger": {"count": 5, "sf_per_floor": 306},
        "elevator_freight": {"count": 2, "sf_per_floor": 252},
        "stair": {"count": 3, "sf_per_floor": 254},
        "vestibule": {"count": 3, "sf_per_floor": 120},
    },
    "circulation": {
        "corridor_width_in": 60,
        "corridor_sf": 24454,
        "elevator_lobby_sf": 3005,
    },
}


class TestParseHelpers:
    """Test helper parsing functions."""

    def test_parse_sf_value_with_commas(self):
        assert parse_sf_value("77,858") == 77858

    def test_parse_sf_value_with_sf_suffix(self):
        assert parse_sf_value("29,743 SF") == 29743

    def test_parse_sf_value_simple(self):
        assert parse_sf_value("447") == 447

    def test_parse_sf_value_empty(self):
        assert parse_sf_value("") == 0
        assert parse_sf_value("-") == 0
        assert parse_sf_value(None) == 0

    def test_parse_dimension_with_ft(self):
        assert parse_dimension("17.2 FT") == 17.2

    def test_parse_dimension_simple(self):
        assert parse_dimension("26.1") == 26.1

    def test_parse_dimension_inches(self):
        assert parse_dimension("60 IN") == 60

    def test_parse_int_value(self):
        assert parse_int_value("377") == 377
        assert parse_int_value("10") == 10
        assert parse_int_value("") == 0
        assert parse_int_value("-") == 0


class TestCSVParser:
    """Test CSV parsing for 6464 Canoga Ave."""

    @pytest.fixture
    def csv_path(self):
        """Get path to test CSV."""
        # Look for CSV in project root
        root = Path(__file__).parent.parent.parent
        csv_file = root / "gloq_example_floorplan_breakdowns - Sheet1.csv"
        if not csv_file.exists():
            pytest.skip(f"CSV file not found: {csv_file}")
        return str(csv_file)

    @pytest.fixture
    def building(self, csv_path):
        """Parse building from CSV."""
        return parse_building_csv(csv_path, building_index=0)

    def test_building_name(self, building):
        assert "canoga" in building.name.lower() or "6464" in building.name

    def test_lot_size(self, building):
        expected = EXPECTED_6464_CANOGA["lot_size_sf"]
        assert building.lot_size_sf == pytest.approx(expected, rel=0.01), \
            f"Expected lot_size {expected}, got {building.lot_size_sf}"

    def test_far(self, building):
        expected = EXPECTED_6464_CANOGA["far"]
        assert building.far == pytest.approx(expected, rel=0.01), \
            f"Expected FAR {expected}, got {building.far}"

    def test_gfa(self, building):
        expected = EXPECTED_6464_CANOGA["gfa_sf"]
        assert building.gfa_sf == pytest.approx(expected, rel=0.01), \
            f"Expected GFA {expected}, got {building.gfa_sf}"

    def test_gba(self, building):
        expected = EXPECTED_6464_CANOGA["gba_sf"]
        assert building.gba_sf == pytest.approx(expected, rel=0.01), \
            f"Expected GBA {expected}, got {building.gba_sf}"

    def test_stories_total(self, building):
        expected = EXPECTED_6464_CANOGA["stories_total"]
        assert building.stories_total == expected, \
            f"Expected {expected} stories, got {building.stories_total}"

    def test_stories_above_grade(self, building):
        expected = EXPECTED_6464_CANOGA["stories_above_grade"]
        assert building.stories_above_grade == expected, \
            f"Expected {expected} above grade, got {building.stories_above_grade}"

    def test_stories_below_grade(self, building):
        expected = EXPECTED_6464_CANOGA["stories_below_grade"]
        assert building.stories_below_grade == expected, \
            f"Expected {expected} below grade, got {building.stories_below_grade}"

    def test_floor_plate(self, building):
        expected = EXPECTED_6464_CANOGA["typical_floor_plate_sf"]
        assert building.typical_floor_plate_sf == pytest.approx(expected, rel=0.01), \
            f"Expected floor plate {expected}, got {building.typical_floor_plate_sf}"

    def test_floor_side_calculated(self, building):
        """Floor side should be sqrt of floor plate for square assumption."""
        import math
        expected_side = math.sqrt(EXPECTED_6464_CANOGA["typical_floor_plate_sf"])
        assert building.floor_side_ft == pytest.approx(expected_side, rel=0.01)

    def test_total_units(self, building):
        expected = EXPECTED_6464_CANOGA["total_units"]
        assert building.total_units == expected, \
            f"Expected {expected} units, got {building.total_units}"

    def test_units_per_floor(self, building):
        expected = EXPECTED_6464_CANOGA["units_per_floor"]
        assert building.units_per_floor == expected, \
            f"Expected {expected} units/floor, got {building.units_per_floor}"

    def test_dwelling_units_count(self, building):
        """Should have 3 unit types: Studio, 1BR, 2BR."""
        assert len(building.dwelling_units) == 3, \
            f"Expected 3 unit types, got {len(building.dwelling_units)}"

    def test_studio_units(self, building):
        expected = EXPECTED_6464_CANOGA["dwelling_units"][0]
        studio = next((u for u in building.dwelling_units if u.unit_type == "Studio"), None)
        assert studio is not None, "Studio units not found"
        assert studio.count == expected["count"], \
            f"Expected {expected['count']} studios, got {studio.count}"
        assert studio.sf == pytest.approx(expected["sf"], rel=0.05), \
            f"Expected {expected['sf']} SF, got {studio.sf}"
        assert studio.width_ft == pytest.approx(expected["width"], rel=0.05), \
            f"Expected width {expected['width']}, got {studio.width_ft}"
        assert studio.depth_ft == pytest.approx(expected["depth"], rel=0.05), \
            f"Expected depth {expected['depth']}, got {studio.depth_ft}"

    def test_1br_units(self, building):
        expected = EXPECTED_6464_CANOGA["dwelling_units"][1]
        unit = next((u for u in building.dwelling_units if u.unit_type == "1BR"), None)
        assert unit is not None, "1BR units not found"
        assert unit.count == expected["count"]
        assert unit.sf == pytest.approx(expected["sf"], rel=0.05)
        assert unit.width_ft == pytest.approx(expected["width"], rel=0.05)
        assert unit.depth_ft == pytest.approx(expected["depth"], rel=0.05)

    def test_2br_units(self, building):
        expected = EXPECTED_6464_CANOGA["dwelling_units"][2]
        unit = next((u for u in building.dwelling_units if u.unit_type == "2BR"), None)
        assert unit is not None, "2BR units not found"
        assert unit.count == expected["count"]
        assert unit.sf == pytest.approx(expected["sf"], rel=0.05)
        assert unit.width_ft == pytest.approx(expected["width"], rel=0.05)
        assert unit.depth_ft == pytest.approx(expected["depth"], rel=0.05)

    def test_core_elements_exist(self, building):
        """Should have 4 core element types."""
        assert len(building.core_elements) == 4, \
            f"Expected 4 core elements, got {len(building.core_elements)}"

    def test_passenger_elevators(self, building):
        expected = EXPECTED_6464_CANOGA["core_elements"]["elevator_passenger"]
        elem = next((e for e in building.core_elements if e.element_type == "elevator_passenger"), None)
        assert elem is not None, "Passenger elevators not found"
        assert elem.count == expected["count"]
        assert elem.sf_per_floor == pytest.approx(expected["sf_per_floor"], rel=0.05)

    def test_freight_elevators(self, building):
        expected = EXPECTED_6464_CANOGA["core_elements"]["elevator_freight"]
        elem = next((e for e in building.core_elements if e.element_type == "elevator_freight"), None)
        assert elem is not None, "Freight elevators not found"
        assert elem.count == expected["count"]
        assert elem.sf_per_floor == pytest.approx(expected["sf_per_floor"], rel=0.05)

    def test_stairs(self, building):
        expected = EXPECTED_6464_CANOGA["core_elements"]["stair"]
        elem = next((e for e in building.core_elements if e.element_type == "stair"), None)
        assert elem is not None, "Stairs not found"
        assert elem.count == expected["count"]
        assert elem.sf_per_floor == pytest.approx(expected["sf_per_floor"], rel=0.05)

    def test_vestibules(self, building):
        expected = EXPECTED_6464_CANOGA["core_elements"]["vestibule"]
        elem = next((e for e in building.core_elements if e.element_type == "vestibule"), None)
        assert elem is not None, "Vestibules not found"
        assert elem.count == expected["count"]
        assert elem.sf_per_floor == pytest.approx(expected["sf_per_floor"], rel=0.05)

    def test_corridor_width(self, building):
        expected = EXPECTED_6464_CANOGA["circulation"]["corridor_width_in"]
        assert building.circulation is not None, "Circulation spec not found"
        assert building.circulation.corridor_width_in == expected, \
            f"Expected corridor width {expected}, got {building.circulation.corridor_width_in}"

    def test_corridor_sf(self, building):
        expected = EXPECTED_6464_CANOGA["circulation"]["corridor_sf"]
        assert building.circulation is not None, "Circulation spec not found"
        assert building.circulation.corridor_sf == pytest.approx(expected, rel=0.05), \
            f"Expected corridor SF {expected}, got {building.circulation.corridor_sf}"

    def test_elevator_lobby_sf(self, building):
        expected = EXPECTED_6464_CANOGA["circulation"]["elevator_lobby_sf"]
        assert building.circulation is not None, "Circulation spec not found"
        assert building.circulation.elevator_lobby_sf == pytest.approx(expected, rel=0.05), \
            f"Expected elevator lobby SF {expected}, got {building.circulation.elevator_lobby_sf}"

    def test_floor_specs_generated(self, building):
        """Should generate floor specs for each above-grade floor."""
        expected_floors = EXPECTED_6464_CANOGA["stories_above_grade"]
        assert len(building.floor_specs) == expected_floors, \
            f"Expected {expected_floors} floor specs, got {len(building.floor_specs)}"

    def test_floor_specs_have_units(self, building):
        """Each floor spec should have dwelling units distributed."""
        for floor_spec in building.floor_specs:
            total_units = sum(u.count for u in floor_spec.dwelling_units)
            assert total_units > 0, f"Floor {floor_spec.floor_number} has no units"


class TestUnitSums:
    """Test that unit counts sum correctly."""

    @pytest.fixture
    def building(self):
        root = Path(__file__).parent.parent.parent
        csv_file = root / "gloq_example_floorplan_breakdowns - Sheet1.csv"
        if not csv_file.exists():
            pytest.skip(f"CSV file not found: {csv_file}")
        return parse_building_csv(str(csv_file), building_index=0)

    def test_unit_counts_sum_to_total(self, building):
        """Sum of all unit type counts should equal total units."""
        sum_units = sum(u.count for u in building.dwelling_units)
        assert sum_units == building.total_units, \
            f"Unit sum {sum_units} != total {building.total_units}"

    def test_floor_units_sum_to_total(self, building):
        """Sum of units across all floors should equal total."""
        floor_unit_sum = sum(
            sum(u.count for u in fs.dwelling_units)
            for fs in building.floor_specs
        )
        # Allow for rounding differences
        assert abs(floor_unit_sum - building.total_units) <= len(building.floor_specs), \
            f"Floor unit sum {floor_unit_sum} doesn't match total {building.total_units}"


# Expected values for 1723 Cloverfield (second building in CSV)
EXPECTED_1723_CLOVERFIELD = {
    "name": "1723 Cloverfield",
    "lot_size_sf": 56914,
    "far": 4.0,
    "gfa_sf": 226576,
    "gba_sf": 241127,
    "stories_total": 10,
    "stories_above_grade": 8,
    "stories_below_grade": 2,
    "typical_floor_plate_sf": 24113,
    "total_units": 258,
    "units_per_floor": 30,
    "dwelling_units": [
        {"type": "Studio", "count": 38, "sf": 501, "width": 17.3, "depth": 28.9, "baths": 1},
        {"type": "1BR", "count": 155, "sf": 656, "width": 23.8, "depth": 27.6, "baths": 1},
        {"type": "2BR", "count": 39, "sf": 966, "width": 42.8, "depth": 22.6, "baths": 2},
        {"type": "3BR", "count": 26, "sf": 1181, "width": 42.3, "depth": 27.9, "baths": 2},
    ],
    "core_elements": {
        "elevator_passenger": {"count": 3, "sf_per_floor": 186},
        "elevator_freight": {"count": 1, "sf_per_floor": 128},
        "stair": {"count": 3, "sf_per_floor": 257},
        "vestibule": {"count": 3, "sf_per_floor": 122},
    },
}


class TestCloverfield:
    """Test CSV parsing for 1723 Cloverfield (second building)."""

    @pytest.fixture
    def csv_path(self):
        root = Path(__file__).parent.parent.parent
        csv_file = root / "gloq_example_floorplan_breakdowns - Sheet1.csv"
        if not csv_file.exists():
            pytest.skip(f"CSV file not found: {csv_file}")
        return str(csv_file)

    @pytest.fixture
    def building(self, csv_path):
        return parse_building_csv(csv_path, building_index=1)

    def test_building_name(self, building):
        assert "cloverfield" in building.name.lower() or "1723" in building.name

    def test_lot_size(self, building):
        expected = EXPECTED_1723_CLOVERFIELD["lot_size_sf"]
        assert building.lot_size_sf == pytest.approx(expected, rel=0.01)

    def test_far(self, building):
        expected = EXPECTED_1723_CLOVERFIELD["far"]
        assert building.far == pytest.approx(expected, rel=0.01)

    def test_gfa(self, building):
        expected = EXPECTED_1723_CLOVERFIELD["gfa_sf"]
        assert building.gfa_sf == pytest.approx(expected, rel=0.01)

    def test_gba(self, building):
        expected = EXPECTED_1723_CLOVERFIELD["gba_sf"]
        assert building.gba_sf == pytest.approx(expected, rel=0.01)

    def test_stories_total(self, building):
        expected = EXPECTED_1723_CLOVERFIELD["stories_total"]
        assert building.stories_total == expected

    def test_stories_above_grade(self, building):
        expected = EXPECTED_1723_CLOVERFIELD["stories_above_grade"]
        assert building.stories_above_grade == expected

    def test_floor_plate(self, building):
        expected = EXPECTED_1723_CLOVERFIELD["typical_floor_plate_sf"]
        assert building.typical_floor_plate_sf == pytest.approx(expected, rel=0.01)

    def test_total_units(self, building):
        expected = EXPECTED_1723_CLOVERFIELD["total_units"]
        assert building.total_units == expected

    def test_units_per_floor(self, building):
        expected = EXPECTED_1723_CLOVERFIELD["units_per_floor"]
        assert building.units_per_floor == expected

    def test_dwelling_units_count(self, building):
        """Cloverfield has 4 unit types including 3BR."""
        assert len(building.dwelling_units) == 4

    def test_studio_units(self, building):
        expected = EXPECTED_1723_CLOVERFIELD["dwelling_units"][0]
        studio = next((u for u in building.dwelling_units if u.unit_type == "Studio"), None)
        assert studio is not None
        assert studio.count == expected["count"]
        assert studio.sf == pytest.approx(expected["sf"], rel=0.05)

    def test_1br_units(self, building):
        expected = EXPECTED_1723_CLOVERFIELD["dwelling_units"][1]
        unit = next((u for u in building.dwelling_units if u.unit_type == "1BR"), None)
        assert unit is not None
        assert unit.count == expected["count"]
        assert unit.sf == pytest.approx(expected["sf"], rel=0.05)

    def test_2br_units(self, building):
        expected = EXPECTED_1723_CLOVERFIELD["dwelling_units"][2]
        unit = next((u for u in building.dwelling_units if u.unit_type == "2BR"), None)
        assert unit is not None
        assert unit.count == expected["count"]
        assert unit.sf == pytest.approx(expected["sf"], rel=0.05)

    def test_3br_units(self, building):
        """Cloverfield has 3BR units unlike Canoga."""
        expected = EXPECTED_1723_CLOVERFIELD["dwelling_units"][3]
        unit = next((u for u in building.dwelling_units if u.unit_type == "3BR"), None)
        assert unit is not None
        assert unit.count == expected["count"]
        assert unit.sf == pytest.approx(expected["sf"], rel=0.05)

    def test_passenger_elevators(self, building):
        expected = EXPECTED_1723_CLOVERFIELD["core_elements"]["elevator_passenger"]
        elem = next((e for e in building.core_elements if e.element_type == "elevator_passenger"), None)
        assert elem is not None
        assert elem.count == expected["count"]
        assert elem.sf_per_floor == pytest.approx(expected["sf_per_floor"], rel=0.05)

    def test_freight_elevators(self, building):
        expected = EXPECTED_1723_CLOVERFIELD["core_elements"]["elevator_freight"]
        elem = next((e for e in building.core_elements if e.element_type == "elevator_freight"), None)
        assert elem is not None
        assert elem.count == expected["count"]
        assert elem.sf_per_floor == pytest.approx(expected["sf_per_floor"], rel=0.05)

    def test_stairs(self, building):
        expected = EXPECTED_1723_CLOVERFIELD["core_elements"]["stair"]
        elem = next((e for e in building.core_elements if e.element_type == "stair"), None)
        assert elem is not None
        assert elem.count == expected["count"]
        assert elem.sf_per_floor == pytest.approx(expected["sf_per_floor"], rel=0.05)

    def test_unit_counts_sum_to_total(self, building):
        sum_units = sum(u.count for u in building.dwelling_units)
        assert sum_units == building.total_units


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
