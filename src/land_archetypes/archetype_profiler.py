import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
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
    def area_summary(
        archetype_raster: xr.DataArray,
        rules: Dict[str, Any],
    ) -> pd.DataFrame:
        """
        Return a DataFrame with pixel count, area in hectares, and coverage
        percentage for each archetype present in the raster.

        Parameters
        ----------
        archetype_raster:
            UInt8 raster from ArchetypeClassification.derive_archetype_raster_map.
            Must carry attrs["class_id_lookup"] and be in a projected CRS with
            metre units (e.g. EPSG:3035). Geographic CRS (degrees) raises a
            ValueError because pixel size in degrees cannot be converted to
            hectares without re-projection.
        rules:
            Archetype rule dict loaded from archetype_classes.json.

        Returns
        -------
        pd.DataFrame with columns:
            "code"          — archetype key (e.g. "B1", "C4")
            "name"          — archetype name from the rule set
            "pixel_count"   — number of pixels classified as this archetype
            "area_ha"       — total area in hectares
            "coverage_pct"  — percentage of the total valid (non-NoData) area

        Rows are sorted by coverage_pct descending (dominant archetypes first).

        Example
        -------
        summary = profiler.area_summary(archetype_raster, rules)
        print(summary)
        #   code                          name  pixel_count   area_ha  coverage_pct
        # 0   C4  Inland Natural Plains & ...        38200  38200.00         62.95
        # 1   B1                  Inland Urban        22500  22500.00         37.07
        """
        if "class_id_lookup" not in archetype_raster.attrs:
            raise ValueError(
                "archetype_raster.attrs must contain 'class_id_lookup'. "
                "Use the raster returned by ArchetypeClassification.derive_archetype_raster_map."
            )

        crs = archetype_raster.rio.crs
        if crs is None:
            raise ValueError(
                "archetype_raster has no CRS. Set a CRS before computing area."
            )
        if crs.is_geographic:
            raise ValueError(
                f"archetype_raster CRS ({crs}) is geographic (degrees). "
                "Reproject to a projected CRS with metre units (e.g. EPSG:3035) "
                "before computing area in hectares."
            )

        x_res, y_res = archetype_raster.rio.resolution()
        pixel_area_ha = abs(x_res * y_res) / 10_000

        id_to_key: Dict[int, str] = {
            v: k for k, v in archetype_raster.attrs["class_id_lookup"].items()
        }
        arch_nodata = int(archetype_raster.rio.nodata)
        arr = archetype_raster.squeeze().values if "band" in archetype_raster.dims else archetype_raster.values

        present_ids = np.unique(arr)
        present_ids = present_ids[present_ids != arch_nodata]
        total_valid = int(np.sum(arr != arch_nodata))

        rows = []
        for arch_id in present_ids:
            arch_key = id_to_key.get(int(arch_id))
            if arch_key is None or arch_key not in rules:
                continue

            pixel_count = int(np.sum(arr == arch_id))
            rows.append({
                "code":         arch_key,
                "name":         rules[arch_key].get("name", ""),
                "pixel_count":  pixel_count,
                "area_ha":      round(pixel_count * pixel_area_ha, 2),
                "coverage_pct": round(pixel_count / total_valid * 100, 2) if total_valid > 0 else 0.0,
            })

        df = pd.DataFrame(rows, columns=["code", "name", "pixel_count", "area_ha", "coverage_pct"])
        return df.sort_values("coverage_pct", ascending=False).reset_index(drop=True)

    @staticmethod
    def diagnose_unclassified(
        archetype_raster: xr.DataArray,
        ras: Dict[str, xr.DataArray],
        rules: Dict[str, Any],
        eunis_code_map: Dict[str, np.ndarray],
        clc_code_map: Dict[str, np.ndarray],
        precedence: Optional[List[str]] = None,
        sample_size: int = 50_000,
        skip_eunis: bool = False,
    ) -> Dict[str, Any]:
        """
        Diagnose why pixels are unclassified (value 255) in an archetype raster.

        For each unclassified pixel (up to sample_size), walks the classification
        precedence list, finds the first archetype whose CLC codes match, then
        identifies which constraint blocks classification. Returns a structured
        report with counts per failure reason and an actionable suggestion for
        each.

        Parameters
        ----------
        archetype_raster:
            UInt8 raster from ArchetypeClassification.derive_archetype_raster_map.
        ras:
            Dict of aligned input rasters used during classification. Expected
            keys: "clc", "eunis", "imperviousness", "dem", "population_density",
            "coast_buffer", "river_buffer". Missing keys are treated as all-NaN.
        rules:
            Archetype rule dict loaded from archetype_classes.json.
        eunis_code_map:
            Mapping from EUNIS L2 code string to raster integer value arrays,
            as returned by build_eunis_code_map().
        clc_code_map:
            Mapping from CLC 3-digit integer code to raster integer value arrays,
            as returned by build_clc_code_map().
        precedence:
            Classification precedence list. Defaults to the standard order.
        sample_size:
            Maximum number of 255 pixels to analyse (default 50 000). For large
            rasters a random sample is drawn; all counts refer to the sample.

        Returns
        -------
        dict with keys:
            "total_unclassified" — total count of 255 pixels in the raster.
            "unclassified_pct"   — percentage of the study area that is 255.
            "sample_size"        — actual number of pixels analysed.
            "failures"           — dict keyed by archetype code. Each entry:
                                   "name", "sampled_count", "reasons" (list of
                                   dicts with "description", "sampled_count",
                                   "suggestion").
            "no_clc_match"       — list of CLC codes present at 255 pixels but
                                   absent from all archetype rules, with counts
                                   and suggestions.
        """
        _DEFAULT_PREC = [
            "A2", "A3", "A1", "A4", "B3", "B2", "B1", "B4", "B5",
            "D1", "D2", "C2", "C3", "C1", "C4",
        ]
        _prec = precedence or _DEFAULT_PREC
        nodata = int(archetype_raster.rio.nodata or 255)
        arch = (
            archetype_raster.squeeze().values
            if "band" in archetype_raster.dims
            else archetype_raster.values
        )

        def _arr(key: str) -> np.ndarray:
            da = ras.get(key)
            if da is None:
                return np.full(arch.shape, np.nan)
            return da.squeeze().values.astype(float)

        clc_arr   = _arr("clc")
        eunis_arr = _arr("eunis")
        imp_arr   = _arr("imperviousness")
        dem_arr   = _arr("dem")
        pop_arr   = _arr("population_density")
        coast_arr = _arr("coast_buffer")
        river_arr = _arr("river_buffer")

        # Reverse maps: raster integer → code label
        clc_reverse: Dict[int, int] = {}
        for code, vals in clc_code_map.items():
            for v in np.asarray(vals).ravel():
                clc_reverse[int(v)] = int(code)

        eunis_reverse: Dict[int, str] = {}
        for code, vals in eunis_code_map.items():
            for v in np.asarray(vals).ravel():
                eunis_reverse[int(v)] = str(code)

        # EUNIS L2 code → human-readable label (e.g. "I1" → "Arable land and market gardens")
        _eunis_csv = Path(__file__).parent / "input_layers_mapping" / "eunis_l2_mapping.csv"
        eunis_labels: Dict[str, str] = {}
        if _eunis_csv.exists():
            _edf = pd.read_csv(_eunis_csv)
            for _, row in _edf.iterrows():
                l2_code = str(row["EUNIS"]).strip()
                raw_label = str(row["Label"]).strip()
                eunis_labels[l2_code] = (
                    raw_label.split(" - ", 1)[1] if " - " in raw_label else raw_label
                )

        mask_255  = arch == nodata
        total_255 = int(mask_255.sum())
        ys, xs    = np.where(mask_255)

        rng_gen = np.random.default_rng(42)
        if len(ys) > sample_size:
            idx    = rng_gen.choice(len(ys), sample_size, replace=False)
            ys, xs = ys[idx], xs[idx]
        actual_sample = len(ys)

        failure_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        no_clc_counts:  Dict[str, int]             = defaultdict(int)

        def _clc_match(rule, v):
            if np.isnan(v):
                return False, None
            code = clc_reverse.get(int(v))
            if code is None:
                return False, f"clc_raster_value_{int(v)}_unmapped"
            rule_codes = [int(c) for c in rule.get("CLC_codes", [])]
            return (not rule_codes or code in rule_codes), None

        def _eunis_match(rule, v):
            rule_codes = rule.get("eunis_codes", [])
            if not rule_codes:
                return True, None
            if np.isnan(v):
                return False, "EUNIS=NaN"
            l2 = eunis_reverse.get(int(v))
            if l2 is None:
                return False, f"EUNIS raster value {int(v)} not in mapping"
            label = eunis_labels.get(l2, "")
            desc = f"EUNIS {l2} ({label})" if label else f"EUNIS {l2}"
            return (l2 in rule_codes), f"{desc} not in rule"

        def _range_ok(val, rng_val, name):
            if rng_val is None or list(rng_val) == [0, 0]:
                return True, None
            if np.isnan(val):
                return False, f"{name}=NaN"
            lo, hi = float(rng_val[0]), float(rng_val[1])
            return (lo <= val <= hi), f"{name}={val:.1f} outside [{lo:.0f},{hi:.0f}]"

        def _buf_ok(val, constraint, name):
            if constraint is None:
                return True, None
            req = int(constraint)
            v   = int(val) if not np.isnan(val) else -1
            if req == 1 and v != 1:
                return False, f"{name} required but absent"
            if req == 0 and v != 0:
                return False, f"{name} present but excluded"
            return True, None

        for y, x in zip(ys, xs):
            pclc, peunis = clc_arr[y, x], eunis_arr[y, x]
            pimp,  pdem  = imp_arr[y, x],  dem_arr[y, x]
            ppop         = pop_arr[y, x]
            pcoast, priv = coast_arr[y, x], river_arr[y, x]

            matched = False
            for key in _prec:
                rule = rules.get(key, {})
                ok, _ = _clc_match(rule, pclc)
                if not ok:
                    continue
                matched = True
                _checks = [] if skip_eunis else [_eunis_match(rule, peunis)]
                _checks += [
                    _buf_ok(pcoast, rule.get("coastline_distance_constraint"), "coast"),
                    _buf_ok(priv,   rule.get("riverline_distance_constraint"),  "river"),
                    _range_ok(pdem,  rule.get("elevation_constraint"),          "elevation"),
                    _range_ok(pimp,  rule.get("imperviousness_constraint") or
                                     rule.get("imperviousness_constraints"),    "imperviousness"),
                    _range_ok(ppop,  rule.get("population_density_constraint"), "pop_density"),
                ]
                for ok, reason in _checks:
                    if not ok:
                        failure_counts[key][reason] += 1
                break

            if not matched and not np.isnan(pclc):
                clc_code = clc_reverse.get(int(pclc), int(pclc))
                no_clc_counts[str(clc_code)] += 1

        def _suggest(key: str, reason: str) -> str:
            if reason is None:
                return "Inspect pixel values manually."
            if "EUNIS" in reason and "not in rule" in reason:
                code = reason.split()[1]  # always the token after "EUNIS"
                return (
                    f"Add '{code}' to {key}'s eunis_codes in archetype_classes.json. "
                    f"Verify that {code} is ecologically consistent with {key}."
                )
            if "not in mapping" in reason:
                return (
                    "This EUNIS raster integer has no L2 mapping. "
                    "Check eunis_l2_mapping.csv for completeness."
                )
            if "EUNIS=NaN" in reason:
                return (
                    f"EUNIS raster has missing values at these pixels. "
                    f"Check EUNIS data coverage over the study area."
                )
            if "elevation=NaN" in reason:
                return (
                    f"DEM has missing values here. Consider setting "
                    f"elevation_constraint to [0, 0] for {key} to disable "
                    f"elevation filtering, or fill DEM gaps."
                )
            if "pop_density=NaN" in reason:
                return (
                    f"Population density raster has missing values. Consider "
                    f"setting population_density_constraint to [0, 0] for {key}."
                )
            if "imperviousness=NaN" in reason:
                return (
                    f"Imperviousness raster has missing values. Consider setting "
                    f"imperviousness_constraint to [0, 0] for {key}."
                )
            if "coast required but absent" in reason:
                return (
                    f"Pixels match {key}'s CLC/EUNIS but lie outside the coastal "
                    f"buffer. Consider increasing COAST_BUFFER_DIST."
                )
            if "river required but absent" in reason:
                return (
                    f"Pixels match {key}'s CLC/EUNIS but lie outside the river "
                    f"buffer. Consider increasing RIVER_BUFFER_DIST."
                )
            if "present but excluded" in reason:
                buf = "coast" if "coast" in reason else "river"
                return (
                    f"Pixels are within the {buf} buffer but {key} requires "
                    f"{buf} to be absent. Consider reducing the {buf} buffer distance."
                )
            if "outside" in reason:
                field = reason.split("=")[0]
                return (
                    f"Pixel {field} values fall outside {key}'s constraint range. "
                    f"Review {field}_constraint in archetype_classes.json or check "
                    f"input data units."
                )
            return f"Review {key}'s rule for: {reason}"

        failures: Dict[str, Any] = {}
        for key in _prec:
            if key not in failure_counts:
                continue
            rd = failure_counts[key]
            failures[key] = {
                "name": rules.get(key, {}).get("name", key),
                "sampled_count": sum(rd.values()),
                "reasons": [
                    {
                        "description":   r,
                        "sampled_count": c,
                        "suggestion":    _suggest(key, r),
                    }
                    for r, c in sorted(rd.items(), key=lambda x: -x[1])
                ],
            }

        no_clc_list = [
            {
                "clc_code":      int(code),
                "sampled_count": count,
                "suggestion": (
                    f"CLC code {code} is not covered by any archetype rule. "
                    f"Consider adding it to an appropriate archetype's CLC_codes "
                    f"in archetype_classes.json."
                ),
            }
            for code, count in sorted(no_clc_counts.items(), key=lambda x: -x[1])
        ]

        return {
            "total_unclassified": total_255,
            "unclassified_pct":   round(total_255 / arch.size * 100, 2),
            "sample_size":        actual_sample,
            "failures":           failures,
            "no_clc_match":       no_clc_list,
        }

    @staticmethod
    def export_diagnosis_rasters(
        archetype_raster: xr.DataArray,
        ras: Dict[str, xr.DataArray],
        rules: Dict[str, Any],
        eunis_code_map: Dict[str, np.ndarray],
        clc_code_map: Dict[str, np.ndarray],
        output_path: str,
        *,
        precedence: Optional[List[str]] = None,
        skip_eunis: bool = False,
        compress: str = "LZW",
    ) -> Dict[str, Any]:
        """
        Run the full unclassified-pixel diagnosis on ALL unclassified pixels
        (no sampling) and export GeoTIFF rasters to *output_path*.

        For every (archetype, constraint) failure category a binary TIF is
        written (1 = failure, 255 = nodata/not applicable).  A combined
        ``diagnosis_map.tif`` encodes the primary failure per pixel as an
        integer value, and a ``diagnosis_map_legend.json`` maps each integer
        to its archetype, slug, and human-readable description.

        Parameters mirror ``diagnose_unclassified``; the return dict has the
        same structure but with exact (not sampled) pixel counts.
        """
        import re
        import rasterio

        _DEFAULT_PREC = [
            "A2", "A3", "A1", "A4", "B3", "B2", "B1", "B4", "B5",
            "D1", "D2", "C2", "C3", "C1", "C4",
        ]
        _prec = precedence or _DEFAULT_PREC
        nodata_val = int(archetype_raster.rio.nodata or 255)

        arch = (
            archetype_raster.squeeze().values
            if "band" in archetype_raster.dims
            else archetype_raster.values
        )

        def _arr(key: str) -> np.ndarray:
            da = ras.get(key)
            if da is None:
                return np.full(arch.shape, np.nan)
            v = da.squeeze().values if "band" in da.dims else da.values
            return v.astype(float)

        clc_arr   = _arr("clc")
        eunis_arr = _arr("eunis")
        imp_arr   = _arr("imperviousness")
        dem_arr   = _arr("dem")
        pop_arr   = _arr("population_density")
        coast_arr = _arr("coast_buffer")
        river_arr = _arr("river_buffer")

        shape = arch.shape

        # Reverse-lookup tables: raster integer value → code string
        clc_rval_to_code: Dict[int, int] = {}
        for code_str, vals in clc_code_map.items():
            for v in np.asarray(vals).ravel():
                clc_rval_to_code[int(v)] = int(code_str)

        eunis_rval_to_l2: Dict[int, str] = {}
        for l2, vals in eunis_code_map.items():
            for v in np.asarray(vals).ravel():
                eunis_rval_to_l2[int(v)] = str(l2)

        # Fast CLC lookup via integer table
        clc_int = np.where(np.isnan(clc_arr), -1, clc_arr.astype(int))
        if clc_rval_to_code:
            _max_rv = max(clc_rval_to_code.keys())
            _clc_lut = np.full(_max_rv + 2, -1, dtype=np.int32)
            for rv, code in clc_rval_to_code.items():
                _clc_lut[rv] = code
            clc_code_2d = np.where(
                (clc_int >= 0) & (clc_int <= _max_rv),
                _clc_lut[np.clip(clc_int, 0, _max_rv)],
                -1,
            )
        else:
            clc_code_2d = clc_int.copy()

        eunis_int = np.where(np.isnan(eunis_arr), -1, eunis_arr.astype(int))

        # Pixel sets — handle both masked (NaN) and unmasked (255) archetype rasters
        unclassified = (arch == nodata_val) | np.isnan(arch)
        inside_sa    = ~np.isnan(clc_arr)
        unclass_mask = unclassified & inside_sa
        total_unclass = int(unclassified.sum())
        total_valid   = int(inside_sa.sum())

        # Storage
        failure_arrs: Dict[tuple, np.ndarray] = {}
        assigned        = np.zeros(shape, dtype=bool)
        arch_pixel_cnt: Dict[str, int] = {}
        primary_code    = np.zeros(shape, dtype=np.uint16)
        code_registry: Dict[tuple, int] = {}
        _next = [1]

        def _get_code(ak: str, sg: str) -> int:
            k = (ak, sg)
            if k not in code_registry:
                code_registry[k] = _next[0]
                _next[0] += 1
            return code_registry[k]

        def _fail_arr(ak: str, sg: str) -> np.ndarray:
            k = (ak, sg)
            if k not in failure_arrs:
                failure_arrs[k] = np.zeros(shape, dtype=bool)
            return failure_arrs[k]

        def _record(ak: str, sg: str, pixels: np.ndarray) -> None:
            if not pixels.any():
                return
            _fail_arr(ak, sg)[...] |= pixels
            code = _get_code(ak, sg)
            no_primary = pixels & (primary_code == 0)
            if no_primary.any():
                primary_code[no_primary] = code

        # Main pass
        for key in _prec:
            rule = rules.get(key, {})

            clc_rule = [int(c) for c in rule.get("CLC_codes", [])]
            clc_match = np.isin(clc_code_2d, clc_rule) if clc_rule else inside_sa.copy()
            candidates = unclass_mask & clc_match & ~assigned
            if not candidates.any():
                continue

            assigned[candidates] = True
            arch_pixel_cnt[key] = int(candidates.sum())

            if not skip_eunis:
                eunis_rule = rule.get("eunis_codes", [])
                if eunis_rule:
                    valid_rv = [rv for rv, l2 in eunis_rval_to_l2.items() if l2 in eunis_rule]
                    _record(key, "eunis_mismatch", candidates & ~np.isin(eunis_int, valid_rv))

            coast_c = rule.get("coastline_distance_constraint")
            if coast_c is not None:
                req_c = int(coast_c)
                c_int = np.where(np.isnan(coast_arr), -1, coast_arr.astype(int))
                slug  = "coast_required_but_absent" if req_c == 1 else "coast_present_but_excluded"
                _record(key, slug, candidates & (c_int != req_c))

            river_c = rule.get("riverline_distance_constraint")
            if river_c is not None:
                req_r = int(river_c)
                r_int = np.where(np.isnan(river_arr), -1, river_arr.astype(int))
                slug  = "river_required_but_absent" if req_r == 1 else "river_present_but_excluded"
                _record(key, slug, candidates & (r_int != req_r))

            for rng_keys, sg, val in [
                (["elevation_constraint"],
                 "elevation_out_of_range", dem_arr),
                (["imperviousness_constraint", "imperviousness_constraints"],
                 "imperviousness_out_of_range", imp_arr),
                (["population_density_constraint"],
                 "pop_density_out_of_range", pop_arr),
            ]:
                rng = next(
                    (rule.get(k) for k in rng_keys if k in rule and rule[k] is not None),
                    None,
                )
                if rng is None or list(rng) == [0, 0]:
                    continue
                lo, hi = float(rng[0]), float(rng[1])
                _record(key, sg, candidates & (np.isnan(val) | (val < lo) | (val > hi)))

        # Pixels with no CLC-matching archetype
        no_match = unclass_mask & ~assigned
        if no_match.any():
            _fail_arr("no_clc_match", "no_clc_match")[...] |= no_match
            code_nm = _get_code("no_clc_match", "no_clc_match")
            primary_code[no_match & (primary_code == 0)] = code_nm

        # --- Write GeoTIFFs ---
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        raster_profile = {
            "driver":    "GTiff",
            "dtype":     "uint8",
            "width":     shape[1],
            "height":    shape[0],
            "count":     1,
            "crs":       archetype_raster.rio.crs,
            "transform": archetype_raster.rio.transform(),
            "nodata":    255,
            "compress":  compress,
        }

        def _write_binary(fname: str, mask: np.ndarray) -> None:
            out = np.full(shape, 255, dtype=np.uint8)
            out[mask] = 1
            with rasterio.open(str(out_dir / fname), "w", **raster_profile) as dst:
                dst.write(out, 1)

        for (ak, sg), barr in failure_arrs.items():
            fname = "no_clc_match.tif" if ak == "no_clc_match" else f"{ak}_{sg}.tif"
            _write_binary(fname, barr)

        # Combined map: 0=classified, 1-253=primary failure code, 255=nodata
        combined = np.full(shape, 255, dtype=np.uint8)
        combined[~unclassified & inside_sa] = 0
        has_code = unclass_mask & (primary_code > 0)
        if has_code.any():
            combined[has_code] = np.clip(primary_code[has_code], 1, 254).astype(np.uint8)
        with rasterio.open(str(out_dir / "diagnosis_map.tif"), "w", **raster_profile) as dst:
            dst.write(combined, 1)

        # Legend JSON
        legend: Dict[str, Any] = {
            "0":   {"archetype": None, "slug": None, "tif": None,
                    "description": "Classified pixel"},
            "255": {"archetype": None, "slug": None, "tif": None,
                    "description": "Outside study area (nodata)"},
        }
        for (ak, sg), code in sorted(code_registry.items(), key=lambda x: x[1]):
            if ak == "no_clc_match":
                desc = "No archetype has CLC codes matching this pixel"
                tif  = "no_clc_match.tif"
            else:
                arch_name = rules.get(ak, {}).get("name", ak)
                desc = f"{ak} ({arch_name}): {sg.replace('_', ' ')}"
                tif  = f"{ak}_{sg}.tif"
            legend[str(code)] = {
                "archetype": ak, "slug": sg, "tif": tif, "description": desc,
            }
        with open(str(out_dir / "diagnosis_map_legend.json"), "w", encoding="utf-8") as fj:
            json.dump(legend, fj, indent=2, ensure_ascii=False)

        # --- QGIS style file (.qml) for diagnosis_map.tif ---
        _COLORS = [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
            "#a65628", "#f781bf", "#17becf", "#bcbd22", "#e377c2",
            "#8c564b", "#1f77b4", "#2ca02c", "#d62728", "#9467bd",
            "#7f7f7f", "#e6b800", "#6baed6", "#74c476", "#fd8d3c",
            "#c5b0d5", "#ffbb78", "#aec7e8", "#98df8a",
        ]

        def _xe(s: str) -> str:
            return (s.replace("&", "&amp;").replace("<", "&lt;")
                     .replace(">", "&gt;").replace('"', "&quot;"))

        entries = [
            '      <paletteEntry alpha="0"   color="#d3d3d3" value="0"   label="Classified pixel (transparent)"/>',
        ]
        for (ak, sg), code in sorted(code_registry.items(), key=lambda x: x[1]):
            color = _COLORS[(code - 1) % len(_COLORS)]
            if ak == "no_clc_match":
                label = _xe("No CLC match for any archetype rule")
            else:
                arch_name = rules.get(ak, {}).get("name", ak)
                label = _xe(f"{ak} – {sg.replace('_', ' ')} ({arch_name})")
            entries.append(
                f'      <paletteEntry alpha="230" color="{color}" value="{code}" label="{label}"/>'
            )
        entries.append(
            '      <paletteEntry alpha="0"   color="#000000" value="255" label="Outside study area (nodata)"/>',
        )

        qml = (
            "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
            '<qgis version="3.0" styleCategories="AllStyleCategories">\n'
            " <pipe>\n"
            '  <rasterrenderer alphaBand="-1" opacity="1" type="paletted" band="1" nodataColor="">\n'
            "   <rasterTransparency/>\n"
            "   <colorPalette>\n"
            + "\n".join(entries) + "\n"
            "   </colorPalette>\n"
            "  </rasterrenderer>\n"
            '  <brightnesscontrast brightness="0" contrast="0" gamma="1"/>\n'
            '  <huesaturation colorizeBlue="128" colorizeGreen="128" colorizeOn="0"'
            ' colorizeRed="255" colorizeStrength="100" grayscaleMode="0" saturation="0"/>\n'
            '  <rasterresampler maxOversampling="2"/>\n'
            " </pipe>\n"
            " <blendMode>0</blendMode>\n"
            "</qgis>\n"
        )
        with open(str(out_dir / "diagnosis_map.qml"), "w", encoding="utf-8") as fq:
            fq.write(qml)

        # --- Build return dict (same structure as diagnose_unclassified) ---
        failures_out: Dict[str, Any] = {}
        for ak in [k for k in _prec if k in arch_pixel_cnt]:
            reasons = [
                {
                    "description":   sg.replace("_", " "),
                    "sampled_count": int(barr.sum()),
                    "suggestion":    f"See {ak}_{sg}.tif in {output_path}",
                }
                for (a, sg), barr in failure_arrs.items()
                if a == ak
            ]
            if reasons:
                failures_out[ak] = {
                    "name":          rules.get(ak, {}).get("name", ak),
                    "sampled_count": arch_pixel_cnt[ak],
                    "reasons":       sorted(reasons, key=lambda r: -r["sampled_count"]),
                }

        no_match_arr = failure_arrs.get(("no_clc_match", "no_clc_match"),
                                        np.zeros(shape, dtype=bool))
        no_clc_out: List[Dict[str, Any]] = []
        if no_match_arr.any():
            clc_counts: Dict[str, int] = defaultdict(int)
            for y, x in zip(*np.where(no_match_arr)):
                v = clc_int[y, x]
                if v >= 0:
                    clc_counts[str(clc_rval_to_code.get(int(v), int(v)))] += 1
            for cc, cnt in sorted(clc_counts.items(), key=lambda i: -i[1]):
                no_clc_out.append({
                    "clc_code":      cc,
                    "sampled_count": cnt,
                    "suggestion":    f"See no_clc_match.tif in {output_path}",
                })

        return {
            "total_unclassified": total_unclass,
            "unclassified_pct":   round(total_unclass / max(total_valid, 1) * 100, 2),
            "total_diagnosed":    int(unclass_mask.sum()),
            "failures":           failures_out,
            "no_clc_match":       no_clc_out,
        }

    @staticmethod
    def print_diagnosis(
        diag: Dict[str, Any],
        *,
        output_path: Optional[str] = None,
    ) -> None:
        """
        Pretty-print the dict returned by ``diagnose_unclassified`` or
        ``export_diagnosis_rasters``.

        Works with both methods: ``diagnose_unclassified`` uses the key
        ``"sample_size"`` (sampled count); ``export_diagnosis_rasters``
        uses ``"total_diagnosed"`` (full count).  Pass ``output_path``
        to append a line pointing to the written rasters.

        Example
        -------
        diag = ArchetypeProfiler.export_diagnosis_rasters(
            archetype_raster, ras, rules, eunis_map, clc_map,
            output_path=str(DIAG_DIR),
        )
        ArchetypeProfiler.print_diagnosis(diag, output_path=str(DIAG_DIR))
        """
        total    = diag["total_unclassified"]
        pct      = diag["unclassified_pct"]
        analysed = diag.get("total_diagnosed") or diag.get("sample_size", total)
        label    = "Pixels diagnosed" if "total_diagnosed" in diag else "Pixels analysed (sample)"

        print(f"Unclassified pixels : {total:,}  ({pct:.1f}% of study area)")
        print(f"{label:<26}: {analysed:,}\n")

        for arch_key, info in diag["failures"].items():
            print(f"{arch_key}  {info['name']}  —  {info['sampled_count']:,} pixels")
            for r in info["reasons"]:
                print(f"    {r['sampled_count']:>6,}  {r['description']}")
                print(f"           → {r['suggestion']}")
            print()

        if diag["no_clc_match"]:
            print("CLC codes not covered by any archetype rule:")
            for entry in diag["no_clc_match"]:
                print(f"    CLC {entry['clc_code']:>4}  ({entry['sampled_count']:,} pixels)")
                print(f"         → {entry['suggestion']}")
            print()

        if output_path is not None:
            print(f"Diagnosis rasters written to: {output_path}")

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
