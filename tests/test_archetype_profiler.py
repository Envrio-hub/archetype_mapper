import numpy as np
import pytest

from land_archetypes.archetype_profiler import ArchetypeProfiler


@pytest.fixture
def simple_rules():
    return {
        "B1": {
            "archetype": "Urban",
            "name": "Inland Urban",
            "hazard_relevance": ["heatwaves"],
            "kcs": ["health"],
        },
        "C4": {
            "archetype": "Rural",
            "name": "Inland Natural Plains & Forests",
            "hazard_relevance": ["wildfires", "drought"],
            "kcs": ["environmental & ecosystem"],
        },
    }


class TestProfile:
    def test_missing_lookup_raises(self, make_da):
        raster = make_da(np.array([[1, 2], [1, 2]], dtype=np.uint8), nodata=255)
        with pytest.raises(ValueError, match="class_id_lookup"):
            ArchetypeProfiler.profile(raster, {})

    def test_returns_required_keys(self, simple_archetype_raster, simple_rules):
        report = ArchetypeProfiler.profile(simple_archetype_raster, simple_rules)
        assert "archetypes_present" in report
        assert "required_hazard_layers" in report
        assert "community_systems_at_risk" in report

    def test_both_archetypes_detected(self, simple_archetype_raster, simple_rules):
        report = ArchetypeProfiler.profile(simple_archetype_raster, simple_rules)
        assert "B1" in report["archetypes_present"]
        assert "C4" in report["archetypes_present"]

    def test_coverage_sums_to_100(self, simple_archetype_raster, simple_rules):
        report = ArchetypeProfiler.profile(simple_archetype_raster, simple_rules)
        total = sum(v["coverage_pct"] for v in report["archetypes_present"].values())
        assert abs(total - 100.0) < 0.01

    def test_hazard_layers_are_union(self, simple_archetype_raster, simple_rules):
        report = ArchetypeProfiler.profile(simple_archetype_raster, simple_rules)
        assert set(report["required_hazard_layers"]) == {"heatwaves", "wildfires", "drought"}

    def test_kcs_is_union(self, simple_archetype_raster, simple_rules):
        report = ArchetypeProfiler.profile(simple_archetype_raster, simple_rules)
        assert set(report["community_systems_at_risk"]) == {"health", "environmental & ecosystem"}

    def test_hazard_layers_sorted(self, simple_archetype_raster, simple_rules):
        report = ArchetypeProfiler.profile(simple_archetype_raster, simple_rules)
        hl = report["required_hazard_layers"]
        assert hl == sorted(hl)

    def test_per_archetype_pixel_count(self, simple_archetype_raster, simple_rules):
        report = ArchetypeProfiler.profile(simple_archetype_raster, simple_rules)
        # simple_archetype_raster: 2×3, row 0 = B1, row 1 = C4 → 3 pixels each
        assert report["archetypes_present"]["B1"]["pixel_count"] == 3
        assert report["archetypes_present"]["C4"]["pixel_count"] == 3

    def test_per_archetype_name_and_hazards(self, simple_archetype_raster, simple_rules):
        report = ArchetypeProfiler.profile(simple_archetype_raster, simple_rules)
        b1 = report["archetypes_present"]["B1"]
        assert b1["name"] == "Inland Urban"
        assert b1["hazard_relevance"] == ["heatwaves"]


class TestProfileCLU:
    @pytest.fixture
    def clu_lookup(self):
        return {
            1: {"archetype": "B1", "cluster": 1, "label": "B1-1",
                "centroid": {"mean_precip": 300.0, "mean_temp": 15.0}},
            2: {"archetype": "B1", "cluster": 2, "label": "B1-2",
                "centroid": {"mean_precip": 600.0, "mean_temp": 10.0}},
            3: {"archetype": "C4", "cluster": 1, "label": "C4-1",
                "centroid": {"mean_precip": 450.0, "mean_temp": 12.0}},
        }

    @pytest.fixture
    def clu_raster(self, make_da):
        # row 0: CLU 1 (2 pixels) + CLU 2 (1 pixel) — both B1
        # row 1: CLU 3 (3 pixels) — C4
        data = np.array([[1, 1, 2], [3, 3, 3]], dtype=np.uint16)
        return make_da(data, nodata=65535, dtype=np.uint16)

    def test_archetypes_grouped_correctly(self, clu_raster, clu_lookup, simple_rules):
        report = ArchetypeProfiler.profile_clu(clu_raster, clu_lookup, simple_rules)
        assert "B1" in report["archetypes_present"]
        assert "C4" in report["archetypes_present"]

    def test_climate_variants_present(self, clu_raster, clu_lookup, simple_rules):
        report = ArchetypeProfiler.profile_clu(clu_raster, clu_lookup, simple_rules)
        variants = report["archetypes_present"]["B1"]["climate_variants"]
        assert "B1-1" in variants
        assert "B1-2" in variants

    def test_pixel_count_aggregated_across_clu_variants(
        self, clu_raster, clu_lookup, simple_rules
    ):
        report = ArchetypeProfiler.profile_clu(clu_raster, clu_lookup, simple_rules)
        # B1-1 has 2 pixels, B1-2 has 1 pixel → B1 total = 3
        assert report["archetypes_present"]["B1"]["pixel_count"] == 3

    def test_coverage_sums_to_100(self, clu_raster, clu_lookup, simple_rules):
        report = ArchetypeProfiler.profile_clu(clu_raster, clu_lookup, simple_rules)
        total = sum(v["coverage_pct"] for v in report["archetypes_present"].values())
        assert abs(total - 100.0) < 0.01

    def test_hazard_union_carried_through(self, clu_raster, clu_lookup, simple_rules):
        report = ArchetypeProfiler.profile_clu(clu_raster, clu_lookup, simple_rules)
        assert set(report["required_hazard_layers"]) == {"heatwaves", "wildfires", "drought"}


class TestExpandCommunitySystems:
    def test_known_category_returned(self):
        result = ArchetypeProfiler.expand_community_systems(["water"])
        assert "water" in result
        names = [s["name"] for s in result["water"]]
        assert "drinking water distribution network" in names

    def test_unknown_category_silently_skipped(self):
        result = ArchetypeProfiler.expand_community_systems(["nonexistent_category"])
        assert result == {}

    def test_empty_input_returns_empty_dict(self):
        assert ArchetypeProfiler.expand_community_systems([]) == {}

    def test_multiple_categories(self):
        result = ArchetypeProfiler.expand_community_systems(["health", "education"])
        assert "health" in result
        assert "education" in result

    def test_health_has_three_systems(self):
        result = ArchetypeProfiler.expand_community_systems(["health"])
        assert len(result["health"]) == 3  # hospitals, pharmacies, EMS

    def test_each_entry_has_required_fields(self):
        result = ArchetypeProfiler.expand_community_systems(["water"])
        for system in result["water"]:
            assert "id" in system
            assert "name" in system
            assert "description" in system

    def test_mixed_known_and_unknown_categories(self):
        result = ArchetypeProfiler.expand_community_systems(["health", "unknown"])
        assert "health" in result
        assert "unknown" not in result
