import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import xarray as xr

_KCS_CATALOGUE_PATH = Path(__file__).parent / "kcs_catalogue.json"


class ArchetypeProfiler:
    """
    Derives analytical profiles from archetype and Climate-Land Unit rasters by
    surfacing the domain knowledge encoded in the archetype rule set — specifically
    the hazard relevance and key community systems (kcs) fields.

    Primary use: given an archetype raster for a study area, identify which hazard
    layers are needed for that area and which community systems are at risk, without
    manual inspection of the JSON rule file.
    """

    @staticmethod
    def profile(
        archetype_raster: xr.DataArray,
        rules: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Profile the archetypes present in an archetype raster.

        Parameters
        ----------
        archetype_raster:
            UInt8 raster from ArchetypeClassification.derive_archetype_raster_map.
            Must carry attrs["class_id_lookup"].
        rules:
            Archetype rule dict loaded from archetype_classes.json (or the enhanced
            variant). Used to retrieve hazard_relevance and kcs per archetype.

        Returns
        -------
        dict with keys:
            "archetypes_present": per-archetype dict with pixel_count,
                coverage_pct, hazard_relevance, kcs, archetype group, and name.
            "required_hazard_layers": sorted union of hazard_relevance across all
                archetypes present — the complete set of hazard maps needed for
                this study area.
            "community_systems_at_risk": sorted union of kcs across all archetypes
                present.

        Example
        -------
        profiler = ArchetypeProfiler()
        report = profiler.profile(archetype_raster, rules)

        print(report["required_hazard_layers"])
        # ["coastal floods", "drought", "heatwaves", "wildfires", ...]

        print(report["archetypes_present"]["C4"]["kcs"])
        # ["environmental & ecosystem", "water"]
        """
        if "class_id_lookup" not in archetype_raster.attrs:
            raise ValueError(
                "archetype_raster.attrs must contain 'class_id_lookup'. "
                "Use the raster returned by ArchetypeClassification.derive_archetype_raster_map."
            )

        id_to_key: Dict[int, str] = {
            v: k for k, v in archetype_raster.attrs["class_id_lookup"].items()
        }
        arch_nodata = int(archetype_raster.rio.nodata)
        arr = archetype_raster.squeeze().values if "band" in archetype_raster.dims else archetype_raster.values

        present_ids = np.unique(arr)
        present_ids = present_ids[present_ids != arch_nodata]
        total_valid = int(np.sum(arr != arch_nodata))

        archetypes_present: Dict[str, Any] = {}
        all_hazards: set = set()
        all_kcs: set = set()

        for arch_id in present_ids:
            arch_key = id_to_key.get(int(arch_id))
            if arch_key is None or arch_key not in rules:
                continue

            rule = rules[arch_key]
            pixel_count = int(np.sum(arr == arch_id))
            coverage_pct = round(pixel_count / total_valid * 100, 2) if total_valid > 0 else 0.0
            hazards = rule.get("hazard_relevance", [])
            kcs = rule.get("kcs", [])

            archetypes_present[arch_key] = {
                "archetype_group": rule.get("archetype", ""),
                "name":            rule.get("name", ""),
                "pixel_count":     pixel_count,
                "coverage_pct":    coverage_pct,
                "hazard_relevance": hazards,
                "kcs":             kcs,
            }
            all_hazards.update(hazards)
            all_kcs.update(kcs)

        return {
            "archetypes_present":       archetypes_present,
            "required_hazard_layers":   sorted(all_hazards),
            "community_systems_at_risk": sorted(all_kcs),
        }

    @staticmethod
    def profile_clu(
        clu_raster: xr.DataArray,
        clu_lookup: Dict[int, Dict[str, Any]],
        rules: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Profile the Climate-Land Units present in a CLU raster.

        Groups CLUs by their parent archetype so that hazard_relevance and kcs
        are reported at the archetype level, while climate centroids are reported
        at the CLU level.

        Parameters
        ----------
        clu_raster:
            UInt16 raster from ClimateLandUnitClassification.derive_climate_land_unit_map.
        clu_lookup:
            Lookup dict returned alongside clu_raster:
            {clu_id: {"archetype", "cluster", "label", "centroid"}}.
        rules:
            Archetype rule dict (same source used during classification).

        Returns
        -------
        dict with keys:
            "archetypes_present": per-archetype dict with total pixel count,
                coverage_pct, hazard_relevance, kcs, and a "climate_variants"
                sub-dict showing per-CLU pixel counts and climate centroids.
            "required_hazard_layers": sorted union of hazard_relevance.
            "community_systems_at_risk": sorted union of kcs.

        Example
        -------
        clu_raster, lookup = clf_clu.derive_climate_land_unit_map(...)

        profiler = ArchetypeProfiler()
        report = profiler.profile_clu(clu_raster, lookup, rules)

        print(report["archetypes_present"]["C4"]["climate_variants"])
        # {
        #   "C4-1": {"pixel_count": 8200, "centroid": {"mean_precip": 320.4, "mean_temp": 17.8}},
        #   "C4-2": {"pixel_count": 4250, "centroid": {"mean_precip": 680.1, "mean_temp": 11.2}},
        # }
        """
        clu_nodata = int(clu_raster.rio.nodata)
        arr = clu_raster.squeeze().values if "band" in clu_raster.dims else clu_raster.values

        present_ids = np.unique(arr)
        present_ids = present_ids[present_ids != clu_nodata]
        total_valid = int(np.sum(arr != clu_nodata))

        archetypes_present: Dict[str, Any] = {}
        all_hazards: set = set()
        all_kcs: set = set()

        for clu_id in present_ids:
            entry = clu_lookup.get(int(clu_id))
            if entry is None:
                continue

            arch_key = entry["archetype"]
            label = entry["label"]
            centroid = entry.get("centroid")
            pixel_count = int(np.sum(arr == clu_id))

            if arch_key not in archetypes_present:
                rule = rules.get(arch_key, {})
                hazards = rule.get("hazard_relevance", [])
                kcs = rule.get("kcs", [])
                all_hazards.update(hazards)
                all_kcs.update(kcs)

                archetypes_present[arch_key] = {
                    "archetype_group": rule.get("archetype", ""),
                    "name":            rule.get("name", ""),
                    "pixel_count":     0,
                    "coverage_pct":    0.0,
                    "hazard_relevance": hazards,
                    "kcs":             kcs,
                    "climate_variants": {},
                }

            archetypes_present[arch_key]["pixel_count"] += pixel_count
            archetypes_present[arch_key]["climate_variants"][label] = {
                "pixel_count": pixel_count,
                "centroid":    centroid,
            }

        # Compute coverage_pct once all pixels are accumulated
        if total_valid > 0:
            for entry in archetypes_present.values():
                entry["coverage_pct"] = round(entry["pixel_count"] / total_valid * 100, 2)

        return {
            "archetypes_present":        archetypes_present,
            "required_hazard_layers":    sorted(all_hazards),
            "community_systems_at_risk": sorted(all_kcs),
        }

    @staticmethod
    def expand_community_systems(
        categories: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Expand a list of KCS category names into their detailed constituent systems
        using the built-in KCS catalogue.

        Designed to be called with the "community_systems_at_risk" field returned
        by profile() or profile_clu(), turning category-level output into a
        specific, actionable inventory of systems at risk.

        Parameters
        ----------
        categories:
            List of KCS category names (e.g. ["water", "health", "transportation"]).
            Matches the category strings used in archetype_classes.json kcs fields.
            Unrecognised categories are silently skipped.

        Returns
        -------
        dict keyed by category name, each value a list of system dicts with keys
        "id", "name", and "description".

        Example
        -------
        report = profiler.profile(archetype_raster, rules)

        details = profiler.expand_community_systems(report["community_systems_at_risk"])
        # {
        #   "health": [
        #       {"id": 19, "name": "hospitals", "description": "..."},
        #       {"id": 20, "name": "pharmacies", "description": "..."},
        #       {"id": 21, "name": "emergency medical services", "description": "..."},
        #   ],
        #   "water": [...],
        # }
        """
        with _KCS_CATALOGUE_PATH.open(encoding="utf-8") as f:
            catalogue: Dict[str, List[Dict[str, Any]]] = json.load(f)

        return {
            cat: catalogue[cat]
            for cat in categories
            if cat in catalogue
        }
