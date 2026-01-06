# CSV Parser Test Results

## Summary

**All 55 tests PASSED** (0.40s)

```
============================= test session starts ==============================
platform darwin -- Python 3.12.3, pytest-9.0.2
collected 55 items
multifloor_generator/tests/test_csv_parser.py ............................ [100%]
============================== 55 passed in 0.40s ==============================
```

## Test Coverage

| Category | Tests | Status |
|----------|-------|--------|
| Parse Helpers | 8 | PASSED |
| 6464 Canoga Ave | 28 | PASSED |
| 1723 Cloverfield | 19 | PASSED |

---

## Extracted Building Data

### 6464 Canoga Ave

| Property | Value |
|----------|-------|
| Floor Area | 29,743 SF |
| Floor Side | 172.5 ft (square plate) |
| Total Units | 377 |
| Units/Floor | 45 |
| Stories | 8 above grade, 2 below |
| FAR | 3.6 |
| GFA | 279,886 SF |
| GBA | 297,432 SF |

**Unit Mix:**
| Type | Count | Size (SF) | Dimensions |
|------|-------|-----------|------------|
| Studio | 83 | 447 | 17.2 ft × 26.1 ft |
| 1BR | 202 | 552 | 22.3 ft × 24.7 ft |
| 2BR | 92 | 867 | 41.6 ft × 20.8 ft |

**Core Elements:**
| Element | Count | SF/Floor |
|---------|-------|----------|
| Passenger Elevator | 5 | 306 |
| Freight Elevator | 2 | 252 |
| Stair | 3 | 254 |
| Vestibule | 3 | 120 |

**Circulation:**
- Corridor: 24,454 SF total
- Elevator Lobby: 3,005 SF

---

### 1723 Cloverfield Blvd

| Property | Value |
|----------|-------|
| Floor Area | 25,811 SF |
| Total Units | 258 |
| Units/Floor | 36 |
| Stories | 7 above grade, 2 below |
| FAR | 3.0 |

**Unit Mix:**
| Type | Count | Size (SF) |
|------|-------|-----------|
| Studio | 35 | 521 |
| 1BR | 124 | 645 |
| 2BR | 88 | 940 |
| 3BR | 11 | 1,162 |

**Core Elements:**
| Element | Count | SF/Floor |
|---------|-------|----------|
| Passenger Elevator | 3 | 270 |
| Freight Elevator | 1 | 200 |
| Stair | 2 | 220 |

---

## Test Categories

### Parse Helper Tests (8)
- `test_parse_sf_value_*` - Parse "1,234 SF" → 1234.0
- `test_parse_dimension_*` - Parse "17.2 ft" → 17.2
- `test_parse_int_value` - Parse "377" → 377

### Building Metric Tests
- Building name extraction
- Lot size, FAR, GFA, GBA
- Stories (total, above/below grade)
- Floor plate area
- Calculated floor side (sqrt of area)

### Unit Tests
- Total unit count
- Units per floor
- Individual unit types (Studio, 1BR, 2BR, 3BR)
- Unit dimensions (width × depth)
- Unit counts sum correctly

### Core Element Tests
- Elevator counts and sizes
- Stair counts and sizes
- Vestibule data
- Circulation areas

---

## Files

- **Parser**: `multifloor_generator/csv_parser.py`
- **Tests**: `multifloor_generator/tests/test_csv_parser.py`
- **Data**: `gloq_example_floorplan_breakdowns - Sheet1.csv`

## Run Tests

```bash
source venv/bin/activate
python3 -m pytest multifloor_generator/tests/test_csv_parser.py -v
```
