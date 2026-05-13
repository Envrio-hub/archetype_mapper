# archetype_mapper

A Python library for deriving Landscape Archetype and Climate-Land Unit rasters
as geospatial **exposure layers** in support of climate risk assessment.

---

## Conceptual framework

In climate risk assessment, **exposure** describes what is present in a location that
could be adversely affected — the landscape, its structural character, and the communities
and systems embedded within it. `LandArchetypes` operationalises this by classifying
the landscape into discrete, spatially explicit exposure units.

Two levels of exposure characterisation are available:

**Land Archetype** — a structural and morphological unit defined by land cover, habitat
type, elevation, imperviousness, population density, and proximity to coastlines and river
networks. It describes *what the land is*, which community systems are most likely present within it,
and which hazards pose the highest threat to it — serving as the primary exposure layer.

**Climate-Land Unit (CLU)** — an optional refinement that adds the climatic envelope
(mean annual precipitation and temperature) under which an archetype operates. CLUs are
derived by unsupervised clustering within each archetype class and carry labels of the
form `C4-1`, `C4-2`, encoding both structural identity and climate variant.

> **Terminology note**: a land archetype is a structural/morphological descriptor.
> A CLU is a distinct concept combining structural identity with climatic context.
> Both are exposure layers. Neither is a hazard map or a vulnerability indicator.

---

## Installation

```bash
pip install land-archetypes
```

Requires Python ≥ 3.12.

## Dependencies

- `xarray >= 2026.2.0`
- `rioxarray >= 0.22.0`
- `pandas >= 3.0.1`
- `geopandas >= 1.1.3`
- `scikit-learn >= 1.5.0`

---

## Stage 1 — Land Archetype map (mandatory)

The 15 built-in archetype classes span four groups:

| Group | Classes | Key discriminators |
|---|---|---|
| Coastal | A1–A4 | Coastal proximity, EUNIS marine/transitional codes |
| Urban | B1–B5 | Imperviousness, population density, elevation, river/coastal proximity |
| Rural | C1–C4 | CLC agricultural/natural codes, low imperviousness |
| Mountainous | D1–D2 | Elevation ≥ 300 m, alpine EUNIS codes |

### Indicative input data resources

| Spatial Evidence | Indicative Dataset | Spatial Resolution | Version | Last Updated |
|---|---|---|---|---|
| CORINE Land Cover | CORINE Land Cover | 100 × 100 m | 20.01 | 2020-05-13 |
| European Nature Information System | Ecosystem Types of Europe 2012 | 100 × 100 m | 3.1 | 2019-02-26 |
| Digital Elevation Model | Copernicus GLO-30 Digital Elevation Model | 30 × 30 m | — | 2015-01-07 |
| Surface Imperviousness Density | Imperviousness Density 2021, Europe (10 m and 100 m), 3-yearly | 10 × 10 m | 1.00 | 2025-08-01 |
| River Network | HydroRIVERS | — | 1.00 | — |
| Coast Line | OpenStreetMap Coastlines | — | — | 2026-02-20 |
| Population Density | WorldPOP Age and Sex Structures | 100 × 100 m | 1 | — |

Classification follows a first-match-wins precedence order over a configurable JSON rule set.
Climate constraints (mean annual precipitation and temperature) are optional — include the
corresponding rasters in `ras` to activate them; omit them and they are silently skipped.

```python
import json
from land_archetypes.archetype_classification import ArchetypeClassification

with open("archetype_classes/archetype_classes.json") as f:
    rules = json.load(f)

clf = ArchetypeClassification()
archetype_raster = clf.derive_archetype_raster_map(
    output_path="outputs/",
    archetype_map_name="archetypes.tif",
    ras={
        "clc":                clc_da,
        "eunis":              eunis_da,
        "coast_buffer":       coast_da,
        "river_buffer":       river_da,
        "imperviousness":     imp_da,
        "population_density": pop_da,
        "dem":                dem_da,
        # optional — activate mean_annual_precip_constraint / mean_annual_temp_constraint
        # fields defined in archetype_classes.json for selected archetypes
        "mean_precip":        precip_da,
        "mean_temp":          temp_da,
    },
    rules=rules,
    eunis_code_map=eunis_map,
    clc_code_map=clc_map,
)
```

### Classification precedence

Because a pixel can satisfy the rules of more than one archetype — for example, an urban
area on the coast satisfies both Coastal Urban and Inland Urban constraints — the
classifier uses a **first-match-wins** strategy: each pixel is assigned to the first
archetype in the precedence list whose rule it satisfies, and is then excluded from all
subsequent evaluations.

The default precedence is ordered from most spatially constrained to most general,
ensuring that specialised archetypes are not absorbed by broader ones:

| Priority | Code | Name |
|---|---|---|
| 1 | A2 | Beach-Dune System |
| 2 | A3 | Transitional Coastal Water System |
| 3 | A1 | Marine/Subtidal |
| 4 | A4 | Coastal Natural Plains & Forests |
| 5 | B3 | Coastal Urban |
| 6 | B2 | Riverine Urban |
| 7 | B1 | Inland Urban |
| 8 | B4 | Suburban |
| 9 | B5 | Mountainous Urban |
| 10 | D1 | Mountainous/Forested |
| 11 | D2 | High-Altitude Meadows & Scrub |
| 12 | C2 | Inland Waterbody Systems |
| 13 | C3 | Rural Settlements |
| 14 | C1 | Agricultural Land |
| 15 | C4 | Inland Natural Plains & Forests |

The order can be changed by passing a custom list to the `precedence` parameter:

```python
archetype_raster = clf.derive_archetype_raster_map(
    ...,
    precedence=["B5", "D2", "D1", "B3", "B2", "B1", "A2", "A3", "A1",
                "A4", "B4", "C2", "C3", "C1", "C4"],
)
```

### Overriding default rule thresholds

Per-archetype constraint values (elevation, imperviousness, population density, climate
ranges) can be overridden at call time without modifying the JSON file. Only the specified
fields are updated; all others retain their default values.

```python
archetype_raster = clf.derive_archetype_raster_map(
    ...,
    rule_overrides={
        "C4": {"elevation_constraint": [0, 400]},
        "B1": {"imperviousness_constraints": [40, 100]},
        "D2": {"mean_annual_temp_constraint": [-5, 4]},
    },
)
```

The coastline and riverline buffer distances are controlled at preprocessing time via
`GeospatialProcessingUtilities.create_line_buffer_raster(buffer_distance=...)`.

### Profiling a study area

Once you have an archetype raster, `ArchetypeProfiler` surfaces the domain knowledge
encoded in the rule set — which hazard layers are needed for your specific study area
and which community systems are at risk — without manual inspection of the JSON file.

```python
from land_archetypes.archetype_profiler import ArchetypeProfiler

profiler = ArchetypeProfiler()
report = profiler.profile(archetype_raster, rules)

# Complete set of hazard maps needed for this study area
print(report["required_hazard_layers"])
# ["coastal floods", "drought", "heatwaves", "slope instability/landslides", "wildfires"]

# Community systems at risk across the study area
print(report["community_systems_at_risk"])
# ["education", "environmental & ecosystem", "food", "health", "transportation", "water"]

# Per-archetype breakdown with pixel count and coverage
for key, info in report["archetypes_present"].items():
    print(f"{key} ({info['name']}): {info['coverage_pct']}% — hazards: {info['hazard_relevance']}")
```

The same profiler works on CLU rasters, adding climate centroids per variant:

```python
report = profiler.profile_clu(clu_raster, lookup, rules)

print(report["archetypes_present"]["C4"]["climate_variants"])
# {
#   "C4-1": {"pixel_count": 8200, "centroid": {"mean_precip": 320.4, "mean_temp": 17.8}},
#   "C4-2": {"pixel_count": 4250, "centroid": {"mean_precip": 680.1, "mean_temp": 11.2}},
# }
```

### Expanding community systems

`community_systems_at_risk` returns category names. Call `expand_community_systems`
to resolve those categories into a specific inventory of systems at risk, drawn from
the built-in KCS catalogue (32 systems across 8 categories).

```python
details = profiler.expand_community_systems(report["community_systems_at_risk"])
# {
#   "health": [
#       {"id": 19, "name": "hospitals",                  "description": "Medical institutions ..."},
#       {"id": 20, "name": "pharmacies",                 "description": "Facilities dispensing ..."},
#       {"id": 21, "name": "emergency medical services", "description": "Rapid-response ..."},
#   ],
#   "water": [
#       {"id": 1, "name": "drinking water distribution network", "description": "..."},
#       ...
#   ],
#   ...
# }
```

The KCS catalogue covers the following categories and systems:

| Category | Systems |
|---|---|
| Water | Drinking water distribution network; drinking water treatment plants; wastewater treatment plants; stormwater drainage system; irrigation water distribution system |
| Transportation | Ports/harbors; railways; airports; public transport systems; road networks |
| Energy | Power plants; transmission and distribution grid; renewable energy infrastructure; refineries |
| Food | Agricultural production; storages (e.g., silos); food processing facilities; local markets |
| Health | Hospitals; pharmacies; emergency medical services |
| Communication | Telecommunications (mobile, internet) |
| Education | Schools; universities; athletic centers |
| Environmental & Ecosystem | Wetlands, rivers, floodplains; soil; urban green spaces; dunes, reefs; forests; lagoons and freshwater lakes; groundwater resources |

---

## Stage 2 — Climate-Land Unit map (optional)

Requires the archetype raster from Stage 1. Within each archetype class, pixels are
clustered on mean annual precipitation and temperature to produce climate sub-types
(e.g. `C4-1`, `C4-2`). Features are z-score standardised before clustering so that
precipitation (mm) and temperature (°C) contribute equally.

```python
from land_archetypes.climate_land_unit_classification import ClimateLandUnitClassification

clf_clu = ClimateLandUnitClassification()
clu_raster, lookup = clf_clu.derive_climate_land_unit_map(
    output_path="outputs/",
    output_name="climate_land_units.tif",
    archetype_raster=archetype_raster,
    ras={
        "mean_precip": precip_da,   # mean annual precipitation (mm/year)
        "mean_temp":   temp_da,     # mean annual temperature (°C)
    },
    target_archetypes=["C4", "D1", "D2"],   # sub-type only these; others pass through
    n_clusters={"C4": 3},                    # fix k for C4; auto-select for the rest
    method="kmeans",                         # "kmeans" (silhouette) or "gmm" (BIC)
)
```

The returned `lookup` dict maps each CLU integer ID to its metadata:

```python
{
    1: {"archetype": "C4", "cluster": 1, "label": "C4-1",
        "centroid": {"mean_precip": 320.4, "mean_temp": 17.8}},
    2: {"archetype": "C4", "cluster": 2, "label": "C4-2",
        "centroid": {"mean_precip": 680.1, "mean_temp": 11.2}},
    ...
}
```

---

## Changelog

### 0.1.2

#### New: archetype area summary

- `ArchetypeProfiler.area_summary(archetype_raster, rules)` — returns a
  `pandas.DataFrame` with pixel count, area in hectares, and coverage
  percentage for each archetype present in the raster, sorted by coverage
  in descending order.
- Raises `ValueError` if the raster CRS is geographic (degrees); a projected
  CRS with metre units (e.g. EPSG:3035) is required for accurate area
  calculation.

```python
summary = profiler.area_summary(archetype_raster, rules)
print(summary)
#   code                          name  pixel_count   area_ha  coverage_pct
# 0   C4  Inland Natural Plains & ...        38200  38200.00         62.95
# 1   B1                  Inland Urban        22500  22500.00         37.07
```

---

### 0.1.1

#### New: Climate-Land Unit classification

- Added `climate_land_unit_classification.py` — `ClimateLandUnitClassification` class
  implementing the Stage 2 two-stage workflow:
  - Within-archetype unsupervised clustering on mean annual precipitation and temperature.
  - Supports `"kmeans"` (silhouette-based auto-k) and `"gmm"` (BIC-based auto-k).
  - Per-archetype fixed k via `n_clusters` dict; `target_archetypes` controls which
    archetypes are sub-typed vs. passed through.
  - Returns a UInt16 CLU raster (NoData = 65535) and a `lookup` dict with centroids.
  - `scikit-learn >= 1.5.0` added as a dependency.

#### New: archetype profiler

- Added `archetype_profiler.py` — `ArchetypeProfiler` class with two methods:
  - `profile(archetype_raster, rules)` — for archetype rasters: returns per-archetype
    pixel count, coverage %, `hazard_relevance`, and `kcs`; plus study-area-wide
    `required_hazard_layers` and `community_systems_at_risk` as sorted union sets.
  - `profile_clu(clu_raster, clu_lookup, rules)` — same as above for CLU rasters,
    with an additional `climate_variants` sub-dict per archetype showing per-CLU
    pixel counts and climate centroids.

#### New: per-archetype rule overrides

- `ArchetypeClassification.derive_archetype_raster_map` accepts
  `rule_overrides: Dict[str, Dict[str, Any]]` — a per-archetype dict that
  deep-merges with the loaded JSON rules at call time.
- Unknown archetype keys raise a `ValueError` with the list of valid keys.
- The original rules dict is never mutated (deep-copied before merging).

#### New: climate constraints in archetype rules

- `archetype_classes.json` includes `mean_annual_precip_constraint` and
  `mean_annual_temp_constraint` fields for all 15 archetypes. Indicative ranges
  set for B1, B3, C4, D1, D2; `null` for all others.
- `archetype_classification.py` supports optional `mean_precip` and `mean_temp`
  rasters. If absent from `ras`, climate constraints are silently skipped.

#### Fix: `GeospatialProcessingUtilities` (geoprocessing_tools.py)

- **Class rename**: `GeospacialProcessingUntilities` → `GeospatialProcessingUtilities`.
- **Bug — integer overflow**: `add_two_rasters` widens integer inputs to `int32` before
  addition to prevent silent `uint8` wraparound.
- **Bug — float transform comparison**: Replaced exact `Affine !=` equality with
  `np.allclose` on the six transform coefficients.
- **Bug — double processing of reference raster**: `reproject_rasters` collapsed two
  `reproject` calls into a single pass; reference key skipped inside the loop.
- **Pitfall — geometry repair**: `make_valid()` used consistently across all methods
  (previously `buffer(0)` in two methods, `make_valid()` in one).
- **Pitfall — deprecated `unary_union`**: Replaced with `union_all()` (Shapely 2.x).
- **Pitfall — unreliable CRS equality**: `!=` replaced with `.equals()`.
- **Pitfall — missing band-dim guard**: `_as_1band` promoted to class-level `@staticmethod`.
- **Performance — eager raster loading**: `chunks="auto"` added to all
  `rxr.open_rasterio` calls.
- **Performance — unnecessary `clip_box`**: Moved inside the `mask_outside_vector` branch.
- **Missing feature**: `compress` parameter added to `clip_raster_by_vector`.

### 0.1.0

- Initial release.
