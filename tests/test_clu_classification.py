import numpy as np
import pytest

from land_archetypes.climate_land_unit_classification import (
    ClimateLandUnitClassification,
    _make_entry,
)


class TestValidateInputs:
    def test_missing_class_id_lookup_raises(self, make_da):
        raster = make_da(np.zeros((2, 3), dtype=np.uint8), nodata=255)
        dummy = make_da(np.zeros((2, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="class_id_lookup"):
            ClimateLandUnitClassification._validate_inputs(
                raster, {"mean_precip": dummy, "mean_temp": dummy},
                "mean_precip", "mean_temp", "kmeans",
            )

    def test_missing_precip_raster_raises(self, simple_archetype_raster, make_da):
        dummy = make_da(np.zeros((2, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="mean_precip"):
            ClimateLandUnitClassification._validate_inputs(
                simple_archetype_raster, {"mean_temp": dummy},
                "mean_precip", "mean_temp", "kmeans",
            )

    def test_invalid_method_raises(self, simple_archetype_raster, make_da):
        dummy = make_da(np.zeros((2, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="method"):
            ClimateLandUnitClassification._validate_inputs(
                simple_archetype_raster,
                {"mean_precip": dummy, "mean_temp": dummy},
                "mean_precip", "mean_temp", "invalid_algo",
            )


class TestSelectK:
    @pytest.fixture
    def two_cluster_data(self):
        rng = np.random.default_rng(0)
        p = np.concatenate([rng.normal(100, 5, 60), rng.normal(600, 5, 60)])
        t = np.concatenate([rng.normal(5, 1, 60), rng.normal(25, 1, 60)])
        return p, t

    def test_fixed_k_returned_directly(self, two_cluster_data):
        p, t = two_cluster_data
        k = ClimateLandUnitClassification._select_k(
            p, t, fixed_k=3, max_k=4, method="kmeans", random_state=42
        )
        assert k == 3

    def test_too_few_pixels_returns_1(self):
        # 4 samples → max_k = min(4, 4//2) = 2, but also triggers k=1 path
        # when called via derive (len < min_pixels); test the k=1 ceiling directly
        p = np.array([100.0, 200.0])
        t = np.array([5.0, 10.0])
        # max_k = min(4, 2//2) = 1 → no valid range for silhouette → returns 1
        k = ClimateLandUnitClassification._select_k(
            p, t, fixed_k=None, max_k=4, method="kmeans", random_state=42
        )
        assert k == 1

    def test_kmeans_returns_positive_int(self, two_cluster_data):
        p, t = two_cluster_data
        k = ClimateLandUnitClassification._select_k(
            p, t, fixed_k=None, max_k=4, method="kmeans", random_state=42
        )
        assert isinstance(k, int)
        assert k >= 1

    def test_gmm_returns_positive_int(self, two_cluster_data):
        p, t = two_cluster_data
        k = ClimateLandUnitClassification._select_k(
            p, t, fixed_k=None, max_k=4, method="gmm", random_state=42
        )
        assert isinstance(k, int)
        assert k >= 1

    def test_gmm_allows_k1(self, make_da):
        # GMM includes k=1 as a candidate so it can select "no sub-typing"
        rng = np.random.default_rng(1)
        # Unimodal data — GMM should prefer k=1
        p = rng.normal(300, 3, 80)
        t = rng.normal(15, 0.5, 80)
        k = ClimateLandUnitClassification._select_k(
            p, t, fixed_k=None, max_k=4, method="gmm", random_state=42
        )
        assert k == 1


class TestFitPredict:
    @pytest.fixture
    def sample_data(self):
        rng = np.random.default_rng(1)
        return rng.normal(0, 1, 30), rng.normal(0, 1, 30)

    def test_kmeans_output_length(self, sample_data):
        p, t = sample_data
        labels = ClimateLandUnitClassification._fit_predict(
            p, t, k=2, method="kmeans", random_state=42
        )
        assert len(labels) == len(p)

    def test_kmeans_labels_are_binary(self, sample_data):
        p, t = sample_data
        labels = ClimateLandUnitClassification._fit_predict(
            p, t, k=2, method="kmeans", random_state=42
        )
        assert set(np.unique(labels)).issubset({0, 1})

    def test_gmm_output_length(self, sample_data):
        p, t = sample_data
        labels = ClimateLandUnitClassification._fit_predict(
            p, t, k=2, method="gmm", random_state=42
        )
        assert len(labels) == len(p)

    def test_gmm_labels_are_binary(self, sample_data):
        p, t = sample_data
        labels = ClimateLandUnitClassification._fit_predict(
            p, t, k=2, method="gmm", random_state=42
        )
        assert set(np.unique(labels)).issubset({0, 1})


class TestMakeEntry:
    def test_all_fields_present(self):
        entry = _make_entry("C4", 2, {"mean_precip": 300.0, "mean_temp": 15.0})
        assert entry["archetype"] == "C4"
        assert entry["cluster"] == 2
        assert entry["label"] == "C4-2"
        assert entry["centroid"]["mean_precip"] == 300.0

    def test_label_format(self):
        assert _make_entry("D1", 3, None)["label"] == "D1-3"

    def test_none_centroid_preserved(self):
        assert _make_entry("B1", 1, None)["centroid"] is None


class TestDeriveCLUMap:
    def test_shape_mismatch_raises(self, tmp_path, simple_archetype_raster, make_da):
        wrong_shape = make_da(np.zeros((3, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="Shape mismatch"):
            ClimateLandUnitClassification().derive_climate_land_unit_map(
                str(tmp_path), "clu.tif", simple_archetype_raster,
                {"mean_precip": wrong_shape, "mean_temp": wrong_shape},
            )

    def test_output_dtype_is_uint16(self, tmp_path, simple_archetype_raster, make_da):
        precip = make_da(np.full((2, 3), 300.0, dtype=np.float32))
        temp = make_da(np.full((2, 3), 15.0, dtype=np.float32))
        clu, _ = ClimateLandUnitClassification().derive_climate_land_unit_map(
            str(tmp_path), "clu.tif", simple_archetype_raster,
            {"mean_precip": precip, "mean_temp": temp},
            n_clusters={"B1": 1, "C4": 1},
        )
        assert clu.dtype == np.uint16

    def test_nodata_value_is_65535(self, tmp_path, simple_archetype_raster, make_da):
        precip = make_da(np.full((2, 3), 300.0, dtype=np.float32))
        temp = make_da(np.full((2, 3), 15.0, dtype=np.float32))
        clu, _ = ClimateLandUnitClassification().derive_climate_land_unit_map(
            str(tmp_path), "clu.tif", simple_archetype_raster,
            {"mean_precip": precip, "mean_temp": temp},
            n_clusters={"B1": 1, "C4": 1},
        )
        assert int(clu.rio.nodata) == 65535

    def test_lookup_contains_both_archetypes(self, tmp_path, simple_archetype_raster, make_da):
        precip = make_da(np.full((2, 3), 300.0, dtype=np.float32))
        temp = make_da(np.full((2, 3), 15.0, dtype=np.float32))
        _, lookup = ClimateLandUnitClassification().derive_climate_land_unit_map(
            str(tmp_path), "clu.tif", simple_archetype_raster,
            {"mean_precip": precip, "mean_temp": temp},
            n_clusters={"B1": 1, "C4": 1},
        )
        labels = {v["label"] for v in lookup.values()}
        assert labels == {"B1-1", "C4-1"}

    def test_lookup_entry_structure(self, tmp_path, simple_archetype_raster, make_da):
        precip = make_da(np.full((2, 3), 300.0, dtype=np.float32))
        temp = make_da(np.full((2, 3), 15.0, dtype=np.float32))
        _, lookup = ClimateLandUnitClassification().derive_climate_land_unit_map(
            str(tmp_path), "clu.tif", simple_archetype_raster,
            {"mean_precip": precip, "mean_temp": temp},
            n_clusters={"B1": 1, "C4": 1},
        )
        for entry in lookup.values():
            assert "archetype" in entry
            assert "cluster" in entry
            assert "label" in entry
            assert "centroid" in entry

    def test_all_valid_pixels_assigned(self, tmp_path, simple_archetype_raster, make_da):
        precip = make_da(np.full((2, 3), 300.0, dtype=np.float32))
        temp = make_da(np.full((2, 3), 15.0, dtype=np.float32))
        clu, _ = ClimateLandUnitClassification().derive_climate_land_unit_map(
            str(tmp_path), "clu.tif", simple_archetype_raster,
            {"mean_precip": precip, "mean_temp": temp},
            n_clusters={"B1": 1, "C4": 1},
        )
        assert 65535 not in np.unique(clu.values)

    def test_non_target_archetype_has_no_centroid(
        self, tmp_path, simple_archetype_raster, make_da
    ):
        # C4 is not in target_archetypes → passed through with centroid=None
        precip = make_da(np.full((2, 3), 300.0, dtype=np.float32))
        temp = make_da(np.full((2, 3), 15.0, dtype=np.float32))
        _, lookup = ClimateLandUnitClassification().derive_climate_land_unit_map(
            str(tmp_path), "clu.tif", simple_archetype_raster,
            {"mean_precip": precip, "mean_temp": temp},
            target_archetypes=["B1"],
            n_clusters={"B1": 1},
        )
        c4_entry = next(v for v in lookup.values() if v["archetype"] == "C4")
        assert c4_entry["centroid"] is None

    def test_centroid_keys_match_raster_keys(
        self, tmp_path, simple_archetype_raster, make_da
    ):
        precip = make_da(np.full((2, 3), 400.0, dtype=np.float32))
        temp = make_da(np.full((2, 3), 12.0, dtype=np.float32))
        _, lookup = ClimateLandUnitClassification().derive_climate_land_unit_map(
            str(tmp_path), "clu.tif", simple_archetype_raster,
            {"mean_precip": precip, "mean_temp": temp},
            n_clusters={"B1": 1, "C4": 1},
        )
        for entry in lookup.values():
            if entry["centroid"] is not None:
                assert "mean_precip" in entry["centroid"]
                assert "mean_temp" in entry["centroid"]
