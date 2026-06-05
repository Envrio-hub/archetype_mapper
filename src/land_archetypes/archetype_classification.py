import json
from pathlib import Path
from typing import Dict, Any, Iterable, Optional
import copy
import xarray as xr
import numpy as np

class ArchetypeClassification():

    def derive_archetype_raster_map(
            self,
            output_path: str,
            archetype_map_name: str,
            ras: Dict[str, xr.DataArray],
            rules: Dict[str, Any],
            *,
            eunis_code_map: Dict[str, np.ndarray],
            clc_code_map: Dict[str, np.ndarray],
            output_nodata: int = 255,
            precedence: Optional[list[str]] = None,
            rule_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
            clc_fallback: bool = False,
            dem_key: str = "dem",
            precip_key: str = "mean_precip",
            temp_key: str = "mean_temp",
            compress: Optional[str] = "LZW"
            ) -> xr.DataArray:
        """
        Produces a UInt8 archetype raster with class IDs (1..N) and NoData=255.
        Optionally uses mean annual precipitation (mm) and mean annual temperature (°C)
        rasters for climate-informed refinement. Pass them in `ras` under the keys
        defined by `precip_key` and `temp_key`. If absent, climate constraints are skipped
        regardless of what is defined in the rules.

        rule_overrides:
            Per-archetype constraint overrides applied on top of the loaded rules.
            Only the specified fields are updated; all other rule fields remain unchanged.

            Example — adjust elevation for D2, set a precipitation window for C4:

                rule_overrides = {
                    "D2": {"elevation_constraint": [500, 2000]},
                    "C4": {"mean_annual_precip_constraint": [100, 600]},
                }

        clc_fallback:
            When True, runs a second classification pass on pixels that remain
            unclassified (255) after the first pass. The second pass ignores the
            EUNIS constraint, relying on CLC and all other spatial/thematic
            constraints only. This handles pixels where CLC and EUNIS disagree due
            to input data inconsistencies (e.g. agricultural CLC with forest EUNIS).

            The first-pass result is saved under `archetype_map_name`. The
            second-pass result (which supersedes it for 255 pixels) is saved as
            `<stem>_clc_fallback<ext>` (e.g. archetypes_clc_fallback.tif).
            Outside-study-area pixels (CLC = NaN) are never affected.
        """
        required = {"clc", "eunis", "coast_buffer", "river_buffer", "imperviousness", "population_density"}
        missing = required - set(ras.keys())
        if missing:
            raise ValueError(f"Missing required rasters: {missing}")

        if rule_overrides:
            rules = self._apply_overrides(rules, rule_overrides)

        if precedence is None:
            precedence = ["A2", "A3", "A1", "A4", "B3", "B2", "B1", "B4", "B5", "B6", "C2", "C3", "C1", "C4", "D1", "D2"]

        clc     = ras["clc"]
        eunis   = ras["eunis"]
        coast_buf = ras["coast_buffer"]
        river_buf = ras["river_buffer"]
        imp     = ras["imperviousness"]
        pop     = ras["population_density"]
        dem     = ras.get(dem_key, None)
        precip  = ras.get(precip_key, None)
        temp    = ras.get(temp_key, None)

        key_to_id = {k: i + 1 for i, k in enumerate(precedence)}
        out = xr.full_like(clc, fill_value=output_nodata, dtype=np.uint8).rio.write_nodata(output_nodata)

        # --- Pass 1: full classification (CLC + EUNIS + all constraints) ---
        out = self._run_pass(
            out, rules, precedence,
            clc, eunis, coast_buf, river_buf, imp, pop, dem, precip, temp,
            eunis_code_map, clc_code_map, key_to_id, output_nodata,
            skip_eunis=False,
        )
        out = out.compute()  # materialise before rio.to_raster alters the lazy graph
        out = out.rio.set_crs("EPSG:3035")
        out.attrs["class_id_lookup"] = {k: int(v) for k, v in key_to_id.items()}
        out.attrs["_FillValue"] = 255
        _tif_path = f"{output_path}/{archetype_map_name}"
        out.rio.to_raster(_tif_path, compress=compress)

        # Embed key_to_id as a TIFF tag so the fallback can verify it later.
        import rasterio as _rio
        with _rio.open(_tif_path, "r+") as _ds:
            _ds.update_tags(class_id_lookup=json.dumps(
                {k: int(v) for k, v in key_to_id.items()}
            ))

        # --- Pass 2 (optional): CLC fallback for remaining 255 pixels ---
        if clc_fallback:
            import rioxarray as _rxr

            # Guard: verify the saved TIF was built with the same key_to_id.
            # A mismatch means Pass 1 was run with a different precedence and
            # must be re-run before the fallback to avoid corrupted class IDs.
            with _rio.open(_tif_path) as _ds:
                _stored_tag = _ds.tags().get("class_id_lookup")
            if _stored_tag is not None:
                _stored = {k: int(v) for k, v in json.loads(_stored_tag).items()}
                _current = {k: int(v) for k, v in key_to_id.items()}
                if _stored != _current:
                    _conflicts = {
                        k: {"pass_1": _stored.get(k), "fallback": _current.get(k)}
                        for k in set(_stored) | set(_current)
                        if _stored.get(k) != _current.get(k)
                    }
                    raise ValueError(
                        f"Fallback aborted: '{archetype_map_name}' was built with "
                        f"a different class ID mapping than this fallback call. "
                        f"Re-run the first-pass classification (Step 5) with the "
                        f"current precedence before running the fallback.\n"
                        f"Conflicting IDs: {_conflicts}"
                    )

            # Reload first-pass result from disk and load all inputs to numpy so
            # that _run_pass operates on concrete arrays with no dask involvement.
            out = _rxr.open_rasterio(_tif_path, masked=False).load()
            out = self._run_pass(
                out, rules, precedence,
                clc.load(), eunis.load(), coast_buf.load(), river_buf.load(),
                imp.load(), pop.load(),
                dem.load() if dem is not None else None,
                precip.load() if precip is not None else None,
                temp.load() if temp is not None else None,
                eunis_code_map, clc_code_map, key_to_id, output_nodata,
                skip_eunis=True,
            )
            out.attrs["class_id_lookup"] = {k: int(v) for k, v in key_to_id.items()}
            p = Path(archetype_map_name)
            fallback_name = f"{p.stem}_clc_fallback{p.suffix}"
            out.rio.to_raster(f"{output_path}/{fallback_name}", compress=compress)

        return out

    def _run_pass(
            self,
            out: xr.DataArray,
            rules: Dict[str, Any],
            precedence: list,
            clc: xr.DataArray,
            eunis: xr.DataArray,
            coast_buf: xr.DataArray,
            river_buf: xr.DataArray,
            imp: xr.DataArray,
            pop: xr.DataArray,
            dem: Optional[xr.DataArray],
            precip: Optional[xr.DataArray],
            temp: Optional[xr.DataArray],
            eunis_code_map: Dict[str, np.ndarray],
            clc_code_map: Dict[str, np.ndarray],
            key_to_id: Dict[str, int],
            output_nodata: int,
            skip_eunis: bool = False,
    ) -> xr.DataArray:
        """Single classification pass. When skip_eunis=True the EUNIS mask is
        set to all-True so only CLC and spatial/thematic constraints apply."""

        for key in precedence:
            rule = rules[key]

            # --- EUNIS membership ---
            if skip_eunis:
                m_eunis = xr.ones_like(eunis, dtype=bool)
            else:
                eunis_codes = rule.get("eunis_codes", [])
                eunis_ids = []
                for code in eunis_codes:
                    if code in eunis_code_map:
                        eunis_ids.append(eunis_code_map[code])
                if eunis_ids:
                    eunis_ids = np.unique(np.concatenate(eunis_ids)).astype(np.int32)
                    m_eunis = xr.apply_ufunc(np.isin, eunis, eunis_ids)
                else:
                    m_eunis = xr.ones_like(eunis, dtype=bool)

            # --- CLC membership ---
            clc_codes = rule.get("CLC_codes", [])
            clc_ids = []
            for code in clc_codes:
                if code in clc_code_map:
                    clc_ids.append(clc_code_map[code])
            if clc_ids:
                clc_ids = np.unique(np.concatenate(clc_ids)).astype(np.int32)
                m_clc = xr.apply_ufunc(np.isin, clc, clc_ids)
            else:
                m_clc = xr.ones_like(clc, dtype=bool)

            # --- Coast/river constraints ---
            coast_raw = rule.get("coastline_distance_constraint", None)
            river_raw = rule.get("riverline_distance_constraint", None)

            if coast_raw is None:
                m_coast = xr.ones_like(coast_buf, dtype=bool)
            else:
                coast_req = self._as_int01(coast_raw)
                m_coast = (coast_buf == 1) if coast_req == 1 else (coast_buf == 0)

            if river_raw is None:
                m_river = xr.ones_like(river_buf, dtype=bool)
            else:
                river_req = self._as_int01(river_raw)
                m_river = (river_buf == 1) if river_req == 1 else (river_buf == 0)

            # --- Range constraints ---
            elev_rng   = self._get_range(rule, ["elevation_constraint"])
            imp_rng    = self._get_range(rule, ["imperviousness_constraint", "imperviousness_constraints"])
            pop_rng    = self._get_range(rule, ["population_density_constraint"])
            precip_rng = self._get_range(rule, ["mean_annual_precip_constraint"])
            temp_rng   = self._get_range(rule, ["mean_annual_temp_constraint"])

            m_elev = xr.ones_like(clc, dtype=bool)
            if dem is not None and elev_rng is not None and not (elev_rng == (0.0, 0.0)):
                m_elev = (dem >= elev_rng[0]) & (dem <= elev_rng[1])

            m_imp = xr.ones_like(clc, dtype=bool)
            if imp_rng is not None and not (imp_rng == (0.0, 0.0)):
                m_imp = (imp >= imp_rng[0]) & (imp <= imp_rng[1])

            m_pop = xr.ones_like(clc, dtype=bool)
            if pop_rng is not None and not (pop_rng == (0.0, 0.0)):
                m_pop = (pop >= pop_rng[0]) & (pop <= pop_rng[1])

            m_precip = xr.ones_like(clc, dtype=bool)
            if precip is not None and precip_rng is not None:
                m_precip = (precip >= precip_rng[0]) & (precip <= precip_rng[1])

            m_temp = xr.ones_like(clc, dtype=bool)
            if temp is not None and temp_rng is not None:
                m_temp = (temp >= temp_rng[0]) & (temp <= temp_rng[1])

            mask = m_eunis & m_clc & m_coast & m_river & m_elev & m_imp & m_pop & m_precip & m_temp

            # first-match-wins; pass 2 only updates pixels still at nodata
            out = xr.where((out == output_nodata) & mask, np.uint8(key_to_id[key]), out)

        return out

    @staticmethod
    def _apply_overrides(
            rules: Dict[str, Any],
            overrides: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        unknown = set(overrides) - set(rules)
        if unknown:
            raise ValueError(
                f"rule_overrides contains unknown archetype keys: {unknown}. "
                f"Valid keys are: {sorted(rules.keys())}"
            )
        rules = copy.deepcopy(rules)
        for key, fields in overrides.items():
            rules[key].update(fields)
        return rules

    @staticmethod
    def _get_range(rule: Dict[str, Any], keys: Iterable[str]) -> Optional[tuple[float, float]]:
        for k in keys:
            if k in rule:
                v = rule[k]
                if v is None:
                    return None
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    return float(v[0]), float(v[1])
        return None

    @staticmethod
    def _as_int01(v) -> int:
        # allows 0/1 or True/False in JSON
        if isinstance(v, bool):
            return 1 if v else 0
        return int(v)

    # ------------------------------------------------------------------
    # Rule-management helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_rule(key: str, rule: Dict[str, Any]) -> None:
        """
        Raise ValueError if any constraint in *rule* is structurally invalid.

        Checked invariants
        ------------------
        - Buffer constraints (coast, river) must be 0, 1, or null.
        - Range constraints must be null or [lo, hi] with lo ≤ hi.
        - List fields (CLC_codes, eunis_codes, …) must be lists.
        """
        _BUFFER_KEYS = [
            "coastline_distance_constraint",
            "riverline_distance_constraint",
        ]
        _RANGE_KEYS = [
            "elevation_constraint",
            "imperviousness_constraint",
            "imperviousness_constraints",
            "population_density_constraint",
            "mean_annual_precip_constraint",
            "mean_annual_temp_constraint",
        ]
        _LIST_KEYS = ["CLC_codes", "eunis_codes", "hazard_relevance", "kcs"]

        for field in _BUFFER_KEYS:
            v = rule.get(field)
            if v is None:
                continue
            try:
                iv = int(v)
            except (TypeError, ValueError):
                raise ValueError(
                    f"[{key}] {field} must be 0, 1, or null — got {v!r}"
                )
            if iv not in (0, 1):
                raise ValueError(
                    f"[{key}] {field} must be 0, 1, or null — got {iv!r}"
                )

        for field in _RANGE_KEYS:
            v = rule.get(field)
            if v is None:
                continue
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                raise ValueError(
                    f"[{key}] {field} must be [lo, hi] or null — got {v!r}"
                )
            try:
                lo, hi = float(v[0]), float(v[1])
            except (TypeError, ValueError):
                raise ValueError(
                    f"[{key}] {field} values must be numeric — got {v!r}"
                )
            if lo > hi:
                raise ValueError(
                    f"[{key}] {field}: lo ({lo}) must be ≤ hi ({hi})"
                )

        for field in _LIST_KEYS:
            v = rule.get(field)
            if v is not None and not isinstance(v, list):
                raise ValueError(
                    f"[{key}] {field} must be a list — got {type(v).__name__!r}"
                )

    @staticmethod
    def rules_table(rules: Dict[str, Any]) -> Any:
        """
        Print a summary table of all archetype rules and return it as a
        pandas DataFrame.

        Columns: code, name, group, CLC_codes, eunis_codes, coast,
        river, elevation, imperviousness, pop_density.

        Example
        -------
        clf.rules_table(rules)
        """
        import pandas as pd

        def _buf(v) -> str:
            if v is None:
                return "—"
            return "required" if int(v) == 1 else "excluded"

        def _rng(v) -> str:
            if v is None:
                return "—"
            if isinstance(v, (list, tuple)) and list(v) == [0, 0]:
                return "—"
            if isinstance(v, (list, tuple)):
                return f"[{v[0]}, {v[1]}]"
            return str(v)

        rows = []
        for key, rule in rules.items():
            imp = rule.get("imperviousness_constraint") or rule.get("imperviousness_constraints")
            rows.append({
                "code":           key,
                "name":           rule.get("name", ""),
                "group":          rule.get("archetype", ""),
                "CLC_codes":      ", ".join(str(c) for c in rule.get("CLC_codes", [])),
                "eunis_codes":    ", ".join(str(c) for c in rule.get("eunis_codes", [])),
                "coast":          _buf(rule.get("coastline_distance_constraint")),
                "river":          _buf(rule.get("riverline_distance_constraint")),
                "elevation":      _rng(rule.get("elevation_constraint")),
                "imperviousness": _rng(imp),
                "pop_density":    _rng(rule.get("population_density_constraint")),
            })

        df = pd.DataFrame(rows)
        print(df.to_string(index=False))
        return df

    def update_class(
        self,
        rules: Dict[str, Any],
        key: str,
        *,
        save_path: Optional[str] = None,
        **fields: Any,
    ) -> Dict[str, Any]:
        """
        Update one or more fields on an existing archetype class and
        validate the result.

        Returns a new rules dict.  The original is never mutated.
        Pass ``save_path`` to also write the result to a JSON file
        (a warning is raised if the path is inside site-packages).

        Example
        -------
        rules = clf.update_class(
            rules, "A2",
            riverline_distance_constraint=None,
            elevation_constraint=[0, 100],
        )
        """
        if key not in rules:
            raise KeyError(
                f"Archetype '{key}' not found. Valid keys: {sorted(rules.keys())}"
            )
        rules = copy.deepcopy(rules)
        rules[key].update(fields)
        self._validate_rule(key, rules[key])
        if save_path is not None:
            self._save_rules(rules, save_path)
        return rules

    def add_class(
        self,
        rules: Dict[str, Any],
        key: str,
        *,
        name: str,
        archetype: str = "",
        operational_rule: str = "",
        spatial_evidence: str = "",
        hazard_relevance: Optional[list] = None,
        kcs: Optional[list] = None,
        eunis_codes: Optional[list] = None,
        CLC_codes: Optional[list] = None,
        coastline_distance_constraint: Optional[int] = None,
        riverline_distance_constraint: Optional[int] = None,
        elevation_constraint: Optional[list] = None,
        imperviousness_constraint: Optional[list] = None,
        population_density_constraint: Optional[list] = None,
        save_path: Optional[str] = None,
        **extra_fields: Any,
    ) -> Dict[str, Any]:
        """
        Add a new archetype class to the rules dict, validate it, and
        warn about precedence.

        Returns a new rules dict.  Raises ValueError if *key* already
        exists — use ``update_class`` to modify an existing entry.

        The new class is NOT in the default precedence list and will be
        silently ignored by ``derive_archetype_raster_map`` unless you
        insert it explicitly with ``insert_into_precedence``.

        Example
        -------
        rules = clf.add_class(
            rules, "E1",
            name="Coastal Wetland",
            archetype="Coastal",
            CLC_codes=["411", "412"],
            eunis_codes=["D5", "D6"],
            coastline_distance_constraint=1,
            elevation_constraint=[-3, 10],
        )
        precedence = clf.insert_into_precedence(
            ["A2","A3","A1","A4","B3","B2","B1","B4","B5",
             "D1","D2","C2","C3","C1","C4"],
            "E1", after="A4",
        )
        """
        import warnings

        if key in rules:
            raise ValueError(
                f"Archetype '{key}' already exists. "
                f"Use update_class() to modify it."
            )
        rules = copy.deepcopy(rules)
        rules[key] = {
            "archetype":                      archetype,
            "name":                           name,
            "operational_rule":               operational_rule,
            "spatial_evidence":               spatial_evidence,
            "hazard_relevance":               hazard_relevance or [],
            "kcs":                            kcs or [],
            "eunis_codes":                    eunis_codes or [],
            "CLC_codes":                      CLC_codes or [],
            "coastline_distance_constraint":  coastline_distance_constraint,
            "riverline_distance_constraint":  riverline_distance_constraint,
            "elevation_constraint":           elevation_constraint,
            "imperviousness_constraint":      imperviousness_constraint,
            "population_density_constraint":  population_density_constraint,
            "mean_annual_precip_constraint":  None,
            "mean_annual_temp_constraint":    None,
            **extra_fields,
        }
        self._validate_rule(key, rules[key])
        warnings.warn(
            f"'{key}' was added to the rules dict but is NOT in the default "
            f"precedence list — it will be ignored by derive_archetype_raster_map "
            f"unless you pass a custom precedence containing it. "
            f"Use clf.insert_into_precedence(precedence, '{key}', after='...') "
            f"to position it before classifying.",
            UserWarning,
            stacklevel=2,
        )
        if save_path is not None:
            self._save_rules(rules, save_path)
        return rules

    def delete_class(
        self,
        rules: Dict[str, Any],
        key: str,
        *,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Remove an archetype class from the rules dict.

        Returns a new rules dict.  The original is never mutated.
        Pass ``save_path`` to also write the result to a JSON file.

        Example
        -------
        rules = clf.delete_class(rules, "D2")
        """
        if key not in rules:
            raise KeyError(
                f"Archetype '{key}' not found. Valid keys: {sorted(rules.keys())}"
            )
        rules = copy.deepcopy(rules)
        del rules[key]
        if save_path is not None:
            self._save_rules(rules, save_path)
        return rules

    @staticmethod
    def insert_into_precedence(
        precedence: list,
        key: str,
        *,
        after: Optional[str] = None,
        before: Optional[str] = None,
        position: Optional[int] = None,
    ) -> list:
        """
        Return a new precedence list with *key* inserted at the
        specified position.  Exactly one of ``after``, ``before``, or
        ``position`` must be supplied.

        Example
        -------
        precedence = clf.insert_into_precedence(precedence, "E1", after="A4")
        precedence = clf.insert_into_precedence(precedence, "E1", before="B3")
        precedence = clf.insert_into_precedence(precedence, "E1", position=0)
        """
        if sum(x is not None for x in [after, before, position]) != 1:
            raise ValueError(
                "Exactly one of 'after', 'before', or 'position' must be supplied."
            )
        prec = list(precedence)
        if key in prec:
            raise ValueError(f"'{key}' is already in the precedence list.")
        if position is not None:
            prec.insert(position, key)
        elif after is not None:
            if after not in prec:
                raise KeyError(f"'{after}' not found in precedence list.")
            prec.insert(prec.index(after) + 1, key)
        else:
            if before not in prec:
                raise KeyError(f"'{before}' not found in precedence list.")
            prec.insert(prec.index(before), key)
        return prec

    def preview_rule_change(
        self,
        rules_before: Dict[str, Any],
        rules_after: Dict[str, Any],
        ras: Dict[str, Any],
        eunis_code_map: Dict[str, np.ndarray],
        clc_code_map: Dict[str, np.ndarray],
        *,
        precedence: Optional[list] = None,
    ) -> Any:
        """
        Run both rule sets in memory and show how pixel counts change.

        No files are written.  Returns a pandas DataFrame with columns
        code, name, before_px, after_px, Δ_px; only rows where
        something changed are printed.

        New keys in *rules_after* that are absent from the default
        precedence are appended at the end of the 'after' run.  Pass
        ``precedence`` to control ordering explicitly.

        Example
        -------
        rules_new = clf.update_class(rules, "A2",
                                     riverline_distance_constraint=None)
        clf.preview_rule_change(rules, rules_new, ras, eunis_map, clc_map)
        """
        import pandas as pd

        _DEFAULT_PREC = [
            "A2", "A3", "A1", "A4", "B3", "B2", "B1", "B4", "B5",
            "D1", "D2", "C2", "C3", "C1", "C4",
        ]
        output_nodata = 255
        base_prec     = precedence or _DEFAULT_PREC

        prec_before = [k for k in base_prec if k in rules_before]
        new_keys    = [k for k in rules_after  if k not in rules_before and k not in base_prec]
        prec_after  = [k for k in base_prec if k in rules_after] + new_keys

        clc       = ras["clc"]
        eunis     = ras["eunis"]
        coast_buf = ras["coast_buffer"]
        river_buf = ras["river_buffer"]
        imp       = ras["imperviousness"]
        pop       = ras["population_density"]
        dem       = ras.get("dem")
        precip    = ras.get("mean_precip")
        temp      = ras.get("mean_temp")

        def _classify(rules, prec):
            kid = {k: i + 1 for i, k in enumerate(prec)}
            out = xr.full_like(clc, fill_value=output_nodata,
                               dtype=np.uint8).rio.write_nodata(output_nodata)
            out = self._run_pass(
                out, rules, prec,
                clc, eunis, coast_buf, river_buf, imp, pop,
                dem, precip, temp,
                eunis_code_map, clc_code_map, kid, output_nodata,
            )
            return out.compute().squeeze().values, kid

        print("Classifying (before) …")
        arr_b, kid_b = _classify(rules_before, prec_before)
        print("Classifying (after)  …")
        arr_a, kid_a = _classify(rules_after,  prec_after)

        def _counts(arr, kid):
            c = {k: int(np.sum(arr == v)) for k, v in kid.items()}
            c["(unclassified)"] = int(np.sum(arr == output_nodata))
            return c

        cnt_b = _counts(arr_b, kid_b)
        cnt_a = _counts(arr_a, kid_a)

        all_keys = list(dict.fromkeys(
            list(kid_b) + list(kid_a) + ["(unclassified)"]
        ))
        rows = []
        for k in all_keys:
            b = cnt_b.get(k, 0)
            a = cnt_a.get(k, 0)
            rule = rules_after.get(k) or rules_before.get(k, {})
            rows.append({
                "code":      k,
                "name":      "(unclassified)" if k == "(unclassified)"
                             else rule.get("name", k),
                "before_px": b,
                "after_px":  a,
                "Δ_px":      a - b,
            })

        df = pd.DataFrame(rows)
        changed = df[df["Δ_px"] != 0]
        if changed.empty:
            print("No pixel counts changed.")
        else:
            print(changed.to_string(index=False))
        return df

    @staticmethod
    def _save_rules(rules: Dict[str, Any], path: str) -> None:
        """
        Write *rules* to a JSON file, creating parent directories as
        needed.  Warns if the target path is inside site-packages (it
        would be overwritten on the next ``pip install``).
        """
        import warnings, sysconfig

        p = Path(path).resolve()
        site_lib = Path(sysconfig.get_path("purelib")).resolve()
        if str(p).startswith(str(site_lib)):
            warnings.warn(
                f"Saving to {p} which is inside the Python environment's "
                f"site-packages directory. This file will be overwritten "
                f"the next time the package is reinstalled. "
                f"Consider saving to your project directory instead.",
                UserWarning,
                stacklevel=3,
            )
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rules, f, indent=4, ensure_ascii=False)
        print(f"Rules saved → {p}")
