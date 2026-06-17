import copy

import numpy as np
import pytest

from land_archetypes.archetype_classification import ArchetypeClassification


class TestGetRange:
    def test_returns_tuple_from_list(self):
        rule = {"elevation_constraint": [100, 500]}
        assert ArchetypeClassification._get_range(rule, ["elevation_constraint"]) == (100.0, 500.0)

    def test_returns_none_for_null_value(self):
        rule = {"mean_annual_precip_constraint": None}
        assert ArchetypeClassification._get_range(rule, ["mean_annual_precip_constraint"]) is None

    def test_returns_none_when_key_absent(self):
        assert ArchetypeClassification._get_range({}, ["elevation_constraint"]) is None

    def test_tries_fallback_key(self):
        rule = {"imperviousness_constraints": [40, 80]}
        result = ArchetypeClassification._get_range(
            rule, ["imperviousness_constraint", "imperviousness_constraints"]
        )
        assert result == (40.0, 80.0)

    def test_returns_floats(self):
        rule = {"elevation_constraint": [300, 2000]}
        lo, hi = ArchetypeClassification._get_range(rule, ["elevation_constraint"])
        assert isinstance(lo, float)
        assert isinstance(hi, float)


class TestAsInt01:
    def test_true_returns_1(self):
        assert ArchetypeClassification._as_int01(True) == 1

    def test_false_returns_0(self):
        assert ArchetypeClassification._as_int01(False) == 0

    def test_int_1_passthrough(self):
        assert ArchetypeClassification._as_int01(1) == 1

    def test_int_0_passthrough(self):
        assert ArchetypeClassification._as_int01(0) == 0


class TestApplyOverrides:
    def test_field_is_updated(self, minimal_rules):
        result = ArchetypeClassification._apply_overrides(
            minimal_rules, {"B1": {"imperviousness_constraint": [70, 100]}}
        )
        assert result["B1"]["imperviousness_constraint"] == [70, 100]

    def test_unrelated_fields_unchanged(self, minimal_rules):
        original_name = minimal_rules["B1"]["name"]
        result = ArchetypeClassification._apply_overrides(
            minimal_rules, {"B1": {"imperviousness_constraint": [70, 100]}}
        )
        assert result["B1"]["name"] == original_name

    def test_original_not_mutated(self, minimal_rules):
        snapshot = copy.deepcopy(minimal_rules)
        ArchetypeClassification._apply_overrides(
            minimal_rules, {"B1": {"imperviousness_constraint": [70, 100]}}
        )
        assert minimal_rules == snapshot

    def test_unknown_key_raises_value_error(self, minimal_rules):
        with pytest.raises(ValueError, match="unknown archetype keys"):
            ArchetypeClassification._apply_overrides(minimal_rules, {"INVALID": {}})

    def test_multiple_archetypes_overridden(self, minimal_rules):
        result = ArchetypeClassification._apply_overrides(
            minimal_rules,
            {
                "B1": {"imperviousness_constraint": [70, 100]},
                "C4": {"elevation_constraint": [0, 400]},
            },
        )
        assert result["B1"]["imperviousness_constraint"] == [70, 100]
        assert result["C4"]["elevation_constraint"] == [0, 400]


class TestDeriveArchetypeRasterMap:
    def test_missing_required_rasters_raises(self, tmp_path, minimal_rules, make_da):
        clf = ArchetypeClassification()
        incomplete = {"clc": make_da(np.zeros((2, 3)))}
        with pytest.raises(ValueError, match="Missing required rasters"):
            clf.derive_archetype_raster_map(
                str(tmp_path), "out.tif", incomplete, minimal_rules,
                eunis_code_map={}, clc_code_map={},
            )

    def test_basic_classification(self, tmp_path, minimal_rules, base_ras, make_da):
        # row 0: imp=80 → B1 (id=1),  row 1: imp=15 → C4 (id=2)
        base_ras["imperviousness"] = make_da(
            np.array([[80, 80, 80], [15, 15, 15]], dtype=np.float32)
        )
        result = ArchetypeClassification().derive_archetype_raster_map(
            str(tmp_path), "test.tif", base_ras, minimal_rules,
            eunis_code_map={}, clc_code_map={},
            precedence=["B1", "C4"],
        )
        assert np.all(result.values[0] == 1), "Row 0 should be classified as B1"
        assert np.all(result.values[1] == 2), "Row 1 should be classified as C4"

    def test_unmatched_pixel_gets_nodata(self, tmp_path, minimal_rules, base_ras, make_da):
        # imp=50 falls outside both B1 [60,100] and C4 [0,30]
        base_ras["imperviousness"] = make_da(np.full((2, 3), 50, dtype=np.float32))
        result = ArchetypeClassification().derive_archetype_raster_map(
            str(tmp_path), "test.tif", base_ras, minimal_rules,
            eunis_code_map={}, clc_code_map={},
            precedence=["B1", "C4"],
        )
        assert np.all(result.values == 255)

    def test_first_match_wins(self, tmp_path, make_da):
        # Both archetypes accept the full imperviousness range [0, 100].
        # The first archetype in the precedence list should claim all pixels.
        wide_rules = {
            "B1": {
                "archetype": "Urban", "name": "B1",
                "eunis_codes": [], "CLC_codes": [],
                "coastline_distance_constraint": None,
                "riverline_distance_constraint": None,
                "elevation_constraint": [0, 0],
                "imperviousness_constraint": [0, 100],
                "population_density_constraint": [0, 0],
                "mean_annual_precip_constraint": None,
                "mean_annual_temp_constraint": None,
                "hazard_relevance": [], "kcs": [],
            },
            "C4": {
                "archetype": "Rural", "name": "C4",
                "eunis_codes": [], "CLC_codes": [],
                "coastline_distance_constraint": None,
                "riverline_distance_constraint": None,
                "elevation_constraint": [0, 0],
                "imperviousness_constraint": [0, 100],
                "population_density_constraint": [0, 0],
                "mean_annual_precip_constraint": None,
                "mean_annual_temp_constraint": None,
                "hazard_relevance": [], "kcs": [],
            },
        }
        zeros = make_da(np.zeros((2, 3), dtype=np.float32))
        ras = {k: zeros for k in (
            "clc", "eunis", "coast_buffer", "river_buffer",
            "imperviousness", "population_density",
        )}
        result = ArchetypeClassification().derive_archetype_raster_map(
            str(tmp_path), "test.tif", ras, wide_rules,
            eunis_code_map={}, clc_code_map={},
            precedence=["B1", "C4"],
        )
        # B1 is first → every pixel assigned to B1 (id=1); C4 never reached
        assert np.all(result.values == 1)

    def test_class_id_lookup_stored_in_attrs(self, tmp_path, minimal_rules, base_ras):
        result = ArchetypeClassification().derive_archetype_raster_map(
            str(tmp_path), "test.tif", base_ras, minimal_rules,
            eunis_code_map={}, clc_code_map={},
            precedence=["B1", "C4"],
        )
        assert "class_id_lookup" in result.attrs
        assert result.attrs["class_id_lookup"] == {"B1": 1, "C4": 2}

    def test_climate_constraint_skipped_when_raster_absent(
        self, tmp_path, minimal_rules, base_ras, make_da
    ):
        # Adding a climate constraint to the rule but NOT supplying the raster
        # must produce the same result as having no climate constraint.
        rules = copy.deepcopy(minimal_rules)
        rules["C4"]["mean_annual_precip_constraint"] = [100, 800]

        base_ras["imperviousness"] = make_da(
            np.array([[80, 80, 80], [15, 15, 15]], dtype=np.float32)
        )
        result = ArchetypeClassification().derive_archetype_raster_map(
            str(tmp_path), "test.tif", base_ras, rules,
            eunis_code_map={}, clc_code_map={},
            precedence=["B1", "C4"],
        )
        assert np.all(result.values[0] == 1)
        assert np.all(result.values[1] == 2)

    def test_rule_overrides_applied(self, tmp_path, minimal_rules, base_ras, make_da):
        # Default B1 range is [60, 100]; override to [50, 100].
        # A pixel with imp=55 must match B1 after the override but not before.
        base_ras["imperviousness"] = make_da(np.full((2, 3), 55, dtype=np.float32))
        result = ArchetypeClassification().derive_archetype_raster_map(
            str(tmp_path), "test.tif", base_ras, minimal_rules,
            eunis_code_map={}, clc_code_map={},
            precedence=["B1", "C4"],
            rule_overrides={"B1": {"imperviousness_constraint": [50, 100]}},
        )
        assert np.all(result.values == 1)

    def test_custom_precedence_changes_ids(self, tmp_path, minimal_rules, base_ras, make_da):
        # With reversed precedence the id mapping flips: C4=1, B1=2
        base_ras["imperviousness"] = make_da(
            np.array([[80, 80, 80], [15, 15, 15]], dtype=np.float32)
        )
        result = ArchetypeClassification().derive_archetype_raster_map(
            str(tmp_path), "test.tif", base_ras, minimal_rules,
            eunis_code_map={}, clc_code_map={},
            precedence=["C4", "B1"],
        )
        assert result.attrs["class_id_lookup"] == {"C4": 1, "B1": 2}
        assert np.all(result.values[0] == 2)  # B1 pixels now carry id 2
        assert np.all(result.values[1] == 1)  # C4 pixels now carry id 1
