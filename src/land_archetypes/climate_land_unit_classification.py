from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class ClimateLandUnitClassification:
    """
    Derives Climate-Land Units (CLUs) by sub-typing landscape archetypes using
    unsupervised clustering on mean annual precipitation and temperature.

    CLUs are a distinct concept from land archetypes: an archetype describes
    what the land *is* (structural/morphological character); a CLU expresses
    which climatic envelope that archetype operates under. The sub-type label
    encodes both — e.g. "C4-1" is the first climate variant of archetype C4.

    This class is designed as Stage 2 of a two-stage workflow:
        Stage 1 — ArchetypeClassification.derive_archetype_raster_map (mandatory)
        Stage 2 — ClimateLandUnitClassification.derive_climate_land_unit_map (optional)
    """

    def derive_climate_land_unit_map(
        self,
        output_path: str,
        output_name: str,
        archetype_raster: xr.DataArray,
        ras: Dict[str, xr.DataArray],
        *,
        precip_key: str = "mean_precip",
        temp_key: str = "mean_temp",
        target_archetypes: Optional[List[str]] = None,
        n_clusters: Optional[Dict[str, int]] = None,
        max_clusters: int = 4,
        method: str = "kmeans",
        min_pixels: int = 100,
        random_state: int = 42,
        compress: str = "LZW",
    ) -> Tuple[xr.DataArray, Dict[int, Dict[str, Any]]]:
        """
        Parameters
        ----------
        output_path:
            Directory for the output GeoTIFF.
        output_name:
            Output filename (e.g. "climate_land_units.tif").
        archetype_raster:
            UInt8 raster produced by ArchetypeClassification.derive_archetype_raster_map.
            Must carry attrs["class_id_lookup"] = {"A1": 1, "B1": 5, ...}.
        ras:
            Dict of rasters containing at minimum precip_key and temp_key, already
            reprojected and grid-matched to archetype_raster (use
            GeoprocessingUtilities.reproject_rasters to align them beforehand).
        precip_key:
            Key for mean annual precipitation raster in ras (mm/year).
        temp_key:
            Key for mean annual temperature raster in ras (°C).
        target_archetypes:
            Archetypes to sub-type into climate variants (e.g. ["C4", "D1", "D2"]).
            Archetypes not listed are passed through as a single CLU (label "X-1")
            with no clustering applied. If None, all archetypes present in the
            raster are sub-typed.
        n_clusters:
            Fixed cluster count per archetype, e.g. {"C4": 3, "D1": 2}.
            Archetypes not listed here use automatic k-selection up to max_clusters.
        max_clusters:
            Upper bound on k during automatic k-selection.
        method:
            Clustering algorithm — "kmeans" (default) or "gmm".
            k-means uses silhouette score for automatic k selection.
            GMM uses BIC; k=1 (no sub-typing) is included as a valid candidate.
        min_pixels:
            Archetypes with fewer valid (non-NaN) pixels than this threshold are
            not sub-typed and are assigned a single cluster (label "X-1").
        random_state:
            Random seed passed to the clustering algorithm for reproducibility.
        compress:
            GeoTIFF compression algorithm (default "LZW").

        Returns
        -------
        clu_raster : xr.DataArray
            UInt16 raster; NoData = 65535. Each unique integer ID maps to one
            Climate-Land Unit. Pixels with NaN climate values within a valid
            archetype area retain the NoData value.
        lookup : dict
            {clu_id: {"archetype", "cluster", "label", "centroid"}} where
            centroid is {precip_key: float, temp_key: float} or None for
            single-cluster pass-throughs.

        Notes
        -----
        Features are z-score standardised before clustering so that precipitation
        (mm) and temperature (°C) contribute equally regardless of their differing
        numerical ranges.

        Example
        -------
        clf_arch = ArchetypeClassification()
        arch_raster = clf_arch.derive_archetype_raster_map(...)

        clf_clu = ClimateLandUnitClassification()
        clu_raster, lookup = clf_clu.derive_climate_land_unit_map(
            output_path="outputs/",
            output_name="climate_land_units.tif",
            archetype_raster=arch_raster,
            ras={"mean_precip": precip_da, "mean_temp": temp_da},
            target_archetypes=["C4", "D1", "D2"],
            n_clusters={"C4": 3},
            method="kmeans",
        )
        """
        self._validate_inputs(archetype_raster, ras, precip_key, temp_key, method)

        id_to_key: Dict[int, str] = {
            v: k for k, v in archetype_raster.attrs["class_id_lookup"].items()
        }
        nodata_out = np.uint16(65535)
        arch_nodata = int(archetype_raster.rio.nodata)

        arch_da = archetype_raster.squeeze() if "band" in archetype_raster.dims else archetype_raster
        precip_da = ras[precip_key].squeeze() if "band" in ras[precip_key].dims else ras[precip_key]
        temp_da = ras[temp_key].squeeze() if "band" in ras[temp_key].dims else ras[temp_key]

        arch_arr = arch_da.values
        precip_arr = precip_da.values
        temp_arr = temp_da.values

        if precip_arr.shape != arch_arr.shape or temp_arr.shape != arch_arr.shape:
            raise ValueError(
                "Shape mismatch: precip, temp, and archetype_raster must share the same grid. "
                "Use GeoprocessingUtilities.reproject_rasters to align them first."
            )

        out_arr = np.full(arch_arr.shape, nodata_out, dtype=np.uint16)
        lookup: Dict[int, Dict[str, Any]] = {}
        clu_counter = 1

        present_ids = np.unique(arch_arr)
        present_ids = present_ids[present_ids != arch_nodata]

        for arch_id in present_ids:
            arch_key = id_to_key.get(int(arch_id))
            if arch_key is None:
                continue

            pixel_mask = arch_arr == arch_id
            do_subtype = (target_archetypes is None) or (arch_key in target_archetypes)

            if not do_subtype:
                lookup[clu_counter] = _make_entry(arch_key, 1, None)
                out_arr[pixel_mask] = clu_counter
                clu_counter += 1
                continue

            p_flat = precip_arr[pixel_mask].ravel()
            t_flat = temp_arr[pixel_mask].ravel()
            valid = np.isfinite(p_flat) & np.isfinite(t_flat)
            p_valid, t_valid = p_flat[valid], t_flat[valid]
            valid_indices = np.flatnonzero(pixel_mask)[valid]

            fixed_k = n_clusters.get(arch_key) if n_clusters else None

            k = (
                1 if len(p_valid) < min_pixels
                else self._select_k(
                    p_valid, t_valid,
                    fixed_k=fixed_k,
                    max_k=max_clusters,
                    method=method,
                    random_state=random_state,
                )
            )

            cluster_labels = (
                np.zeros(len(p_valid), dtype=int)
                if k == 1
                else self._fit_predict(p_valid, t_valid, k=k, method=method, random_state=random_state)
            )

            flat_out = out_arr.ravel()
            for cl in np.unique(cluster_labels):
                cl_mask = cluster_labels == cl
                centroid = {
                    precip_key: float(np.mean(p_valid[cl_mask])),
                    temp_key: float(np.mean(t_valid[cl_mask])),
                }
                lookup[clu_counter] = _make_entry(arch_key, int(cl) + 1, centroid)
                flat_out[valid_indices[cl_mask]] = clu_counter
                clu_counter += 1

        clu_da = (
            xr.DataArray(out_arr, dims=arch_da.dims, coords=arch_da.coords)
            .rio.write_crs(arch_da.rio.crs)
            .rio.write_nodata(int(nodata_out))
        )
        clu_da.attrs["clu_lookup"] = lookup
        clu_da.rio.to_raster(f"{output_path}/{output_name}", compress=compress)
        return clu_da, lookup

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        archetype_raster: xr.DataArray,
        ras: Dict[str, xr.DataArray],
        precip_key: str,
        temp_key: str,
        method: str,
    ) -> None:
        if "class_id_lookup" not in archetype_raster.attrs:
            raise ValueError(
                "archetype_raster.attrs must contain 'class_id_lookup'. "
                "Use the raster returned by ArchetypeClassification.derive_archetype_raster_map."
            )
        for key in (precip_key, temp_key):
            if key not in ras:
                raise ValueError(f"Required raster '{key}' not found in ras.")
        if method not in ("kmeans", "gmm"):
            raise ValueError(f"method must be 'kmeans' or 'gmm', got '{method}'.")

    @staticmethod
    def _select_k(
        p_vals: np.ndarray,
        t_vals: np.ndarray,
        *,
        fixed_k: Optional[int],
        max_k: int,
        method: str,
        random_state: int,
    ) -> int:
        if fixed_k is not None:
            return max(1, int(fixed_k))

        # Require at least 2 samples per cluster
        max_k = min(max_k, len(p_vals) // 2)
        if max_k < 2:
            return 1

        X = StandardScaler().fit_transform(np.column_stack([p_vals, t_vals]))

        if method == "kmeans":
            scores: Dict[int, float] = {}
            for k in range(2, max_k + 1):
                labels = KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit_predict(X)
                if len(np.unique(labels)) < 2:
                    continue
                scores[k] = silhouette_score(X, labels)
            return max(scores, key=scores.get) if scores else 1

        # GMM: include k=1 so BIC can select "no sub-typing"
        bic: Dict[int, float] = {}
        for k in range(1, max_k + 1):
            gm = GaussianMixture(n_components=k, random_state=random_state)
            gm.fit(X)
            bic[k] = gm.bic(X)
        return min(bic, key=bic.get)

    @staticmethod
    def _fit_predict(
        p_vals: np.ndarray,
        t_vals: np.ndarray,
        *,
        k: int,
        method: str,
        random_state: int,
    ) -> np.ndarray:
        X = StandardScaler().fit_transform(np.column_stack([p_vals, t_vals]))
        if method == "kmeans":
            return KMeans(n_clusters=k, random_state=random_state, n_init="auto").fit_predict(X)
        return GaussianMixture(n_components=k, random_state=random_state).fit_predict(X)


def _make_entry(
    arch_key: str,
    cluster_number: int,
    centroid: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    return {
        "archetype": arch_key,
        "cluster": cluster_number,
        "label": f"{arch_key}-{cluster_number}",
        "centroid": centroid,
    }
