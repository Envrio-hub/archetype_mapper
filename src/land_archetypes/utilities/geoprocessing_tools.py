from dataclasses import dataclass
from typing import Optional, Union, Dict
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
import numpy as np

from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterio.crs import CRS

@dataclass
class GeospacialProcessingUntilities:
    output_path: str
    study_area_path: str
    study_area_layer: Optional[str] = None 

    def clip_raster_by_vector(
        self,
        output_tif_name: str,
        raster_path: str,
        vector_path: str = None,
        *,
        output_crs: Optional[Union[str, int]] = None,
        crop_to_vector_bounds: bool = True,
        mask_outside_vector: bool = True,
        all_touched: bool = True,
        vector_layer: Optional[str] = None,
        vector_crs_override: Optional[Union[str, int]] = None,
        raster_crs_override: Optional[Union[str, int]] = None,
        output_nodata: Optional[float] = None,
        resampling: Resampling = Resampling.nearest,
    ) -> None:
        """ 
        Parameters
        ----------
        output_crs:
            CRS for the saved GeoTIFF (e.g., "EPSG:4326" or 4326). If None, keep raster CRS.
        crop_to_vector_bounds:
            If True, output is cropped to polygon bounds (smaller raster). If False, keeps original raster extent.
        mask_outside_vector:
            If True, pixels outside polygon are set to nodata (masked) even if within bbox/extent.
        all_touched:
            If True, mask includes all pixels touched by geometry edges.
        vector_layer:
            Optional layer name for multi-layer vector sources (e.g., GeoPackage).
        vector_crs_override / raster_crs_override:
            If provided, assigns CRS when missing/incorrect in source metadata.
        output_nodata:
            If provided, forces nodata value in output; otherwise uses raster nodata (or keeps existing).
        resampling:
            Resampling used if output_crs reprojection is requested.
        """

        # ---- 1) Load raster ----
        raster = rxr.open_rasterio(raster_path, masked=True)

        # Ensure spatial dims are set (usually "x","y" already)
        raster = raster.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

        # Optionally override/assign raster CRS if missing/wrong
        if raster_crs_override is not None:
            raster = raster.rio.write_crs(raster_crs_override, inplace=False)

        if raster.rio.crs is None:
            raise ValueError(
                "Raster has no CRS. Provide raster_crs_override (e.g., 'EPSG:32634') "
                "or fix the raster metadata."
            )

        # ---- 2) Load vector properly with geopandas ----
        vector_path = vector_path or self.study_area_path
        vector_layer = vector_layer or self.study_area_layer
        gdf = gpd.read_file(vector_path, layer=vector_layer) if vector_layer else gpd.read_file(vector_path)

        if gdf.empty:
            raise ValueError("Vector file contains no features.")

        # Fix invalid geometries (common cause of clip failures)
        # buffer(0) can fix many self-intersections; if it fails, you can switch to shapely.make_valid
        gdf["geometry"] = gdf.geometry.buffer(0)

        # Optionally override/assign vector CRS if missing/wrong
        if vector_crs_override is not None:
            gdf = gdf.set_crs(vector_crs_override, allow_override=True)

        if gdf.crs is None:
            raise ValueError(
                "Vector has no CRS. Provide vector_crs_override (e.g., 'EPSG:4326') "
                "or fix the vector metadata."
            )

        # ---- 3) Reproject vector to raster CRS if needed ----
        if gdf.crs != raster.rio.crs:
            gdf = gdf.to_crs(raster.rio.crs)

        minx, miny, maxx, maxy = gdf.total_bounds
        subset = raster.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)

        # Dissolve to a single geometry to avoid slivers / per-feature masking issues
        geom = gdf.unary_union

        # ---- 4) Clip + mask outside polygon ----
        # rioxarray.clip accepts GeoJSON-like mappings or shapely geometries
        # drop=True -> crops to bounds; drop=False -> keeps extent but masks outside geometry
        drop = crop_to_vector_bounds

        if mask_outside_vector:
            clipped = subset.rio.clip(
                [geom],
                crs=raster.rio.crs,
                drop=drop,
                all_touched=all_touched,
            )
        else:
            # If you don't want masking, you essentially just crop to bounds (if enabled)
            # Keeping extent without masking doesn't make much sense; we implement crop-only behavior.
            if not crop_to_vector_bounds:
                clipped = raster  # no-op
            else:
                clipped = raster.rio.clip_box(*gdf.total_bounds)

        # ---- 5) Enforce/choose nodata behavior ----
        # If raster already has nodata, masked=True will preserve it as NaN/mask.
        # You can write an explicit nodata if you want consistent output.
        if output_nodata is not None:
            clipped = clipped.rio.write_nodata(output_nodata, inplace=False)

        # ---- 6) Optional: reproject output to user-defined CRS ----
        if output_crs is not None:
            clipped = clipped.rio.reproject(
                output_crs,
                resampling=resampling
            )

        # ---- 7) Write GeoTIFF ----
        clipped.rio.to_raster(f'{self.output_path}/{output_tif_name}')

    def clip_vector_by_vector(
        self,
        output_vector_name: str,
        vector_path: str,
        clipper_path: Optional[str] = None,
        *,
        output_crs: Optional[Union[str, int]] = None,
        keep_geom_type: bool = True,
        vector_layer: Optional[str] = None,
        clipper_layer: Optional[str] = None,
        vector_crs_override: Optional[Union[str, int]] = None,
        clipper_crs_override: Optional[Union[str, int]] = None,
        driver: str = "GPKG",
    ) -> None:
        """
        Clip a vector dataset by another vector dataset (mask).

        Parameters
        ----------
        output_vector_name:
            Output filename (e.g., "rivers_clipped.gpkg" or "rivers_clipped.shp").
        vector_path:
            Input vector to be clipped.
        clipper_path:
            Clipping/mask vector. If None, defaults to self.study_area_path.
        output_crs:
            CRS for output. If None, keep input vector CRS.
        keep_geom_type:
            If True, keeps only geometries of the same type as input (prevents mixed types after clip).
        vector_layer / clipper_layer:
            Optional layer names for multi-layer vector sources (GeoPackage).
        vector_crs_override / clipper_crs_override:
            Assign CRS if missing/incorrect in metadata.
        driver:
            Output driver. Use "GPKG" (recommended) or "ESRI Shapefile".
        """

        # ---- 1) Load input vector ----
        gdf = gpd.read_file(vector_path, layer=vector_layer) if vector_layer else gpd.read_file(vector_path)
        if gdf.empty:
            raise ValueError("Input vector file contains no features.")

        # Fix invalid geometries (common cause of overlay failures)
        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf["geometry"] = gdf.geometry.buffer(0)

        if vector_crs_override is not None:
            gdf = gdf.set_crs(vector_crs_override, allow_override=True)

        if gdf.crs is None:
            raise ValueError(
                "Input vector has no CRS. Provide vector_crs_override (e.g., 'EPSG:4326') "
                "or fix the vector metadata."
            )

        # ---- 2) Load clipper vector ----
        clipper_path = clipper_path or self.study_area_path
        clipper_layer = clipper_layer or getattr(self, "study_area_layer", None)

        clip_gdf = (
            gpd.read_file(clipper_path, layer=clipper_layer)
            if clipper_layer
            else gpd.read_file(clipper_path)
        )

        if clip_gdf.empty:
            raise ValueError("Clipper vector file contains no features.")

        clip_gdf = clip_gdf[clip_gdf.geometry.notnull()].copy()
        clip_gdf["geometry"] = clip_gdf.geometry.buffer(0)

        if clipper_crs_override is not None:
            clip_gdf = clip_gdf.set_crs(clipper_crs_override, allow_override=True)

        if clip_gdf.crs is None:
            raise ValueError(
                "Clipper vector has no CRS. Provide clipper_crs_override (e.g., 'EPSG:4326') "
                "or fix the vector metadata."
            )

        # ---- 3) Reproject clipper to input CRS if needed ----
        if clip_gdf.crs != gdf.crs:
            clip_gdf = clip_gdf.to_crs(gdf.crs)

        # Dissolve clipper to a single geometry to avoid slivers and speed up clip
        clip_geom = clip_gdf.unary_union

        # ---- 4) Clip ----
        clipped = gpd.clip(gdf, clip_geom, keep_geom_type=keep_geom_type)

        # Optional: drop empties that can appear after clip
        clipped = clipped[clipped.geometry.notnull() & ~clipped.geometry.is_empty].copy()

        # ---- 5) Optional: reproject output ----
        if output_crs is not None:
            clipped = clipped.to_crs(output_crs)

        # ---- 6) Write output ----
        out_path = f"{self.output_path}/{output_vector_name}"
        clipped.to_file(out_path, driver=driver)
    
    def create_line_buffer_raster(
        self,
        buffer_name: str,
        line_path: str,
        buffer_distance: float,
        *,
        line_layer: Optional[str] = None,
        reference_raster_path: Optional[str] = None,
        pixel_size: Optional[float] = None,
        out_crs: Optional[Union[str, int]] = None,
        all_touched: bool = True,
        compress: str = "LZW",
    ) -> None:
        """
        Create a binary raster: 1 within buffer_distance of coastline, 0 elsewhere,
        and NoData outside the study area polygon. Processing is done in EPSG:3035.
        If out_crs is None, output is saved in EPSG:3035; otherwise reprojected.
        """
        work_crs = CRS.from_epsg(3035)
        nodata = np.uint8(255)

        # --- Load & reproject study area to EPSG:3035 ---
        study_gdf = gpd.read_file(self.study_area_path, layer=self.study_area_layer) if self.study_area_layer else gpd.read_file(self.study_area_path)
        if study_gdf.empty:
            raise ValueError("Study area layer is empty.")
        if study_gdf.crs is None:
            raise ValueError("Study area CRS is missing.")

        study_gdf = study_gdf[study_gdf.geometry.notnull()].copy()
        study_gdf["geometry"] = study_gdf.geometry.make_valid()
        study_gdf = study_gdf.to_crs(work_crs)
        study_union = study_gdf.unary_union

        # --- Load & reproject coastline to EPSG:3035 ---
        coast_gdf = gpd.read_file(line_path, layer=line_layer) if line_layer else gpd.read_file(line_path)
        if coast_gdf.empty:
            raise ValueError("Coastline layer is empty.")
        if coast_gdf.crs is None:
            raise ValueError("Coastline CRS is missing.")

        coast_gdf = coast_gdf[coast_gdf.geometry.notnull()].copy()
        coast_gdf["geometry"] = coast_gdf.geometry.make_valid()
        coast_gdf = coast_gdf.to_crs(work_crs)

        # --- Clip coastline to study area polygon ---
        coast_clipped = gpd.clip(coast_gdf, study_union)

        # --- Buffer coastline (meters in EPSG:3035) ---
        buffer_geom = None
        if not coast_clipped.empty:
            buffer_geom = coast_clipped.geometry.buffer(buffer_distance).unary_union

        # --- Build target grid (in EPSG:3035) as an xarray DataArray ---
        if reference_raster_path is not None:
            ref = rxr.open_rasterio(reference_raster_path, masked=True)

            if ref.rio.crs is None:
                raise ValueError("Reference raster CRS is missing.")

            # Reproject reference raster to EPSG:3035 to obtain a working grid
            ref_3035 = ref.rio.reproject(
                work_crs,
                resampling=Resampling.nearest,
            )

            # Use first band as template
            template = ref_3035.isel(band=0)
            transform = template.rio.transform()
            height, width = template.shape
        else:
            if pixel_size is None:
                raise ValueError("Provide either reference_raster_path or pixel_size (meters).")

            minx, miny, maxx, maxy = study_union.bounds
            width = int(np.ceil((maxx - minx) / pixel_size))
            height = int(np.ceil((maxy - miny) / pixel_size))
            transform = from_bounds(minx, miny, maxx, maxy, width, height)

            # Create coordinates aligned with transform
            # x = centers from left to right, y = centers from top to bottom
            x0 = transform.c + transform.a / 2.0
            y0 = transform.f + transform.e / 2.0
            xs = x0 + np.arange(width) * transform.a
            ys = y0 + np.arange(height) * transform.e  # transform.e is negative for north-up

            template = xr.DataArray(
                np.zeros((height, width), dtype=np.uint8),
                dims=("y", "x"),
                coords={"y": ys, "x": xs},
                name="coast_buffer",
            ).rio.write_crs(work_crs).rio.write_transform(transform)

        # --- Rasterize buffer (1 inside buffer, 0 elsewhere) ---
        if buffer_geom is None:
            buf_arr = np.zeros((height, width), dtype=np.uint8)
        else:
            buf_arr = rasterize(
                [(buffer_geom, 1)],
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype="uint8",
                all_touched=all_touched,
            )

        # --- Rasterize study area mask (1 inside, 0 outside) ---
        mask_arr = rasterize(
            [(study_union, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        )

        # --- Apply NoData outside study area ---
        out_arr = buf_arr.copy()
        out_arr[mask_arr == 0] = nodata

        out_da = xr.DataArray(
            out_arr,
            dims=("y", "x"),
            coords=template.coords,
            name="coast_buffer",
        ).rio.write_crs(work_crs).rio.write_transform(transform).rio.write_nodata(int(nodata))

        # --- Reproject if requested ---
        if out_crs is not None:
            dst_crs = CRS.from_user_input(out_crs)
            out_da = out_da.rio.reproject(dst_crs, resampling=Resampling.nearest)

        # --- Write GeoTIFF ---
        # Ensure a single-band GeoTIFF (DataArray)
        out_da.rio.to_raster(f'{self.output_path}/{buffer_name}', compress=compress)

    def add_two_rasters(
        self,
        output_name: str,
        raster1_path: str,
        raster2_path: str,
        target_crs: Union[str, int],
        *,
        output_path: Optional[str] = None,
        min_value: Optional[float] = None,
        integer_nodata: Optional[int] = None,
        resampling: Resampling = Resampling.nearest,
        compress: str = "LZW",
    ) -> None:
        """
        Read 2 rasters, reproject both to target_crs, verify identical grid
        (extent, pixel size, transform), add them, then optionally mask values
        below min_value as NoData/NaN so they won't display.

        Parameters
        ----------
        min_value:
            If provided, any output cell with value < min_value is set to NoData/NaN.
        integer_nodata:
            Used only if output dtype is integer. Must be a value not used by your data.
            If None and integer output, defaults to 255 for uint8, else raises.
        """

        # Ensure extension/driver
        if not output_name.lower().endswith(".tif"):
            output_name += ".tif"

        target_crs = CRS.from_user_input(target_crs)

        # --- Load rasters (first band) ---
        r1 = rxr.open_rasterio(raster1_path, masked=True).isel(band=0)
        r2 = rxr.open_rasterio(raster2_path, masked=True).isel(band=0)

        if r1.rio.crs is None or r2.rio.crs is None:
            raise ValueError("Both rasters must have a defined CRS.")

        # --- Reproject both to target CRS ---
        r1 = r1.rio.reproject(target_crs, resampling=resampling)
        r2 = r2.rio.reproject(target_crs, resampling=resampling)

        # --- Strict grid checks ---
        if r1.rio.crs != r2.rio.crs:
            raise ValueError("CRS mismatch after reprojection.")

        if r1.shape != r2.shape:
            raise ValueError("Raster shapes differ (rows/columns mismatch).")

        if not np.allclose(r1.rio.resolution(), r2.rio.resolution()):
            raise ValueError("Raster resolutions differ (pixel size mismatch).")

        if r1.rio.transform() != r2.rio.transform():
            raise ValueError("Raster transforms differ (pixel alignment mismatch).")

        if not np.allclose(r1.rio.bounds(), r2.rio.bounds()):
            raise ValueError("Raster extents differ.")

        # --- Add ---
        result = r1 + r2

        # --- Apply threshold mask (< min_value => NoData/NaN) ---
        if min_value is not None:
            is_integer = np.issubdtype(result.dtype, np.integer)

            if not is_integer:
                # float output -> NaN mask
                result = result.where(result >= min_value)
            else:
                # integer output -> need a nodata sentinel
                if integer_nodata is None:
                    # Safe default only for uint8; otherwise user must decide
                    if result.dtype == np.uint8:
                        integer_nodata = 255
                    else:
                        raise ValueError(
                            "Result is integer dtype; provide integer_nodata "
                            "(a value not used by your data) to mask values."
                        )

                result = result.rio.write_nodata(int(integer_nodata), inplace=False)
                result = result.where(result >= min_value, other=int(integer_nodata))

        # --- Write output ---
        output_path = self.output_path if output_path is None else output_path
        out_path = f"{output_path}/{output_name}"
        result.rio.to_raster(
            out_path,
            driver="GTiff",
            compress=compress,
        )

    def reproject_rasters(
        self,
        rasters: Dict[str, xr.DataArray],
        target_crs: Union[str, int],
        *,
        reference_key: str,
        pixel_size: float = 30.0,
        categorical_keys: Optional[set] = None,
    ) -> Dict[str, xr.DataArray]:
        """
        Reproject all rasters to EPSG:3035 and match them to a common 30m grid.
        """
        def _as_1band(da: xr.DataArray) -> xr.DataArray:
            return da.isel(band=0) if "band" in da.dims else da
        
        if categorical_keys is None:
            categorical_keys = {"clc", "eunis", "coast_buffer", "river_buffer"}

        ref = _as_1band(rasters[reference_key])
        if ref.rio.crs is None:
            raise ValueError(f"Reference raster '{reference_key}' has no CRS.")

        # Create a 30m reference grid in EPSG:3035
        ref_3035 = ref.rio.reproject(target_crs, resampling=Resampling.nearest)
        ref_30 = ref_3035.rio.reproject(
            target_crs,
            resolution=pixel_size,
            resampling=Resampling.nearest,
        )

        out = {reference_key: ref_30}

        for k, da in rasters.items():
            da1 = _as_1band(da)
            if da1.rio.crs is None:
                raise ValueError(f"Raster '{k}' has no CRS.")

            resamp = Resampling.nearest if k in categorical_keys else Resampling.bilinear

            da_3035 = da1.rio.reproject(target_crs, resampling=resamp)
            da_match = da_3035.rio.reproject_match(ref_30, resampling=resamp)

            out[k] = da_match

        return out