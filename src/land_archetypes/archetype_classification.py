from typing import Dict, Any, Iterable, Optional
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
            dem_key: str = "dem",
            compress: Optional[str] = "LZW"
            ) -> xr.DataArray:
        """
        Produces a UInt8 archetype raster with class IDs (1..N) and NoData=255.
        """
        required = {"clc", "eunis", "coast_buffer", "river_buffer", "imperviousness", "population_density"}
        missing = required - set(ras.keys())
        if missing:
            raise ValueError(f"Missing required rasters: {missing}")

        if precedence is None:
            precedence = ["A2", "A3", "A1", "A4", "B3", "B2", "B1", "B4", "B5", "D1", "D2", "C2", "C3", "C1", "C4"]

        clc = ras["clc"]
        eunis = ras["eunis"]
        coast_buf = ras["coast_buffer"]
        river_buf = ras["river_buffer"]
        imp = ras["imperviousness"]
        pop = ras["population_density"]
        dem = ras.get(dem_key, None)

        key_to_id = {k: i + 1 for i, k in enumerate(precedence)}
        out = xr.full_like(clc, fill_value=output_nodata, dtype=np.uint8).rio.write_nodata(output_nodata)

        for key in precedence:
            rule = rules[key]

            # --- EUNIS membership ---
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

            # --- Coast/river constraints (binary masks). None => no constraint ---
            coast_raw = rule.get("coastline_distance_constraint", None)
            river_raw = rule.get("riverline_distance_constraint", None)

            if coast_raw is None:
                m_coast = xr.ones_like(coast_buf, dtype=bool)
            else:
                coast_req = self._as_int01(coast_raw)  # expects 0/1-like
                m_coast = (coast_buf == 1) if coast_req == 1 else (coast_buf == 0)

            if river_raw is None:
                m_river = xr.ones_like(river_buf, dtype=bool)
            else:
                river_req = self._as_int01(river_raw)
                m_river = (river_buf == 1) if river_req == 1 else (river_buf == 0)

            # --- Ranges ---
            elev_rng = self._get_range(rule, ["elevation_constraint"])
            imp_rng = self._get_range(rule, ["imperviousness_constraint", "imperviousness_constraints"])
            pop_rng = self._get_range(rule, ["population_density_constraint"])

            m_elev = xr.ones_like(clc, dtype=bool)
            if dem is not None and elev_rng is not None and not (elev_rng == (0.0, 0.0)):
                m_elev = (dem >= elev_rng[0]) & (dem <= elev_rng[1])

            m_imp = xr.ones_like(clc, dtype=bool)
            if imp_rng is not None and not (imp_rng == (0.0, 0.0)):
                m_imp = (imp >= imp_rng[0]) & (imp <= imp_rng[1])

            m_pop = xr.ones_like(clc, dtype=bool)
            if pop_rng is not None and not (pop_rng == (0.0, 0.0)):
                m_pop = (pop >= pop_rng[0]) & (pop <= pop_rng[1])

            mask = m_eunis & m_clc & m_coast & m_river & m_elev & m_imp & m_pop

            # first-match-wins
            out = xr.where((out == output_nodata) & mask, np.uint8(key_to_id[key]), out)

        out.attrs["class_id_lookup"] = {k: int(v) for k, v in key_to_id.items()}
        out.rio.to_raster(f"{output_path}/{archetype_map_name}", compress=compress)
        return out
    
    @staticmethod
    def _get_range(rule: Dict[str, Any], keys: Iterable[str]) -> Optional[tuple[float, float]]:
        for k in keys:
            if k in rule:
                v = rule[k]
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    return float(v[0]), float(v[1])
        return None

    @staticmethod
    def _as_int01(v) -> int:
        # allows 0/1 or True/False in JSON
        if isinstance(v, bool):
            return 1 if v else 0
        return int(v)