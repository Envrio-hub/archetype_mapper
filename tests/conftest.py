import numpy as np
import pytest
import xarray as xr
import rioxarray  # noqa: F401 — activates .rio accessor


@pytest.fixture
def make_da():
    """Factory that creates minimal spatial DataArrays for testing."""
    def _inner(data, nodata=255, dtype=None, crs="EPSG:4326"):
        arr = np.array(data, dtype=dtype)
        h, w = arr.shape
        da = xr.DataArray(
            arr,
            dims=["y", "x"],
            coords={
                "y": np.linspace(48.0, 47.0, h),
                "x": np.linspace(10.0, 11.0, w),
            },
        )
        return da.rio.write_crs(crs).rio.write_nodata(nodata)
    return _inner


# Minimal two-archetype rule set used across multiple test modules.
# Uses only imperviousness to keep masks simple and deterministic.
MINIMAL_RULES = {
    "B1": {
        "archetype": "Urban",
        "name": "Inland Urban",
        "eunis_codes": [],
        "CLC_codes": [],
        "coastline_distance_constraint": None,
        "riverline_distance_constraint": None,
        "elevation_constraint": [0, 0],
        "imperviousness_constraint": [60, 100],
        "population_density_constraint": [0, 0],
        "mean_annual_precip_constraint": None,
        "mean_annual_temp_constraint": None,
        "hazard_relevance": ["heatwaves"],
        "kcs": ["health"],
    },
    "C4": {
        "archetype": "Rural",
        "name": "Inland Natural Plains & Forests",
        "eunis_codes": [],
        "CLC_codes": [],
        "coastline_distance_constraint": None,
        "riverline_distance_constraint": None,
        "elevation_constraint": [0, 0],
        "imperviousness_constraint": [0, 30],
        "population_density_constraint": [0, 0],
        "mean_annual_precip_constraint": None,
        "mean_annual_temp_constraint": None,
        "hazard_relevance": ["wildfires", "drought"],
        "kcs": ["environmental & ecosystem"],
    },
}


@pytest.fixture
def minimal_rules():
    import copy
    return copy.deepcopy(MINIMAL_RULES)


@pytest.fixture
def base_ras(make_da):
    """All 6 required rasters as a 2×3 zero-filled grid."""
    zeros = make_da(np.zeros((2, 3), dtype=np.float32))
    return {k: zeros for k in (
        "clc", "eunis", "coast_buffer", "river_buffer",
        "imperviousness", "population_density",
    )}


@pytest.fixture
def simple_archetype_raster(make_da):
    """2×3 UInt8 raster: row 0 = B1 (id=1), row 1 = C4 (id=2)."""
    data = np.array([[1, 1, 1], [2, 2, 2]], dtype=np.uint8)
    da = make_da(data, nodata=255, dtype=np.uint8)
    da.attrs["class_id_lookup"] = {"B1": 1, "C4": 2}
    return da
