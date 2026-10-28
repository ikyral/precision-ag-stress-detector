"""Raster zones -> topology-preserving polygons -> GeoJSON/Shapefile + economics."""
from __future__ import annotations

from datetime import datetime
import heapq
import json
import math
import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
import rasterio
from rasterio.features import shapes
from shapely import make_valid, segmentize
from shapely.geometry import Polygon, MultiPolygon, shape, mapping
from shapely.ops import transform as transform_geometry, unary_union
from shapely.strtree import STRtree

from .economics import ZONE_NAMES, _nonnegative, calculate_economics, validate_parameters
from .coverage import coverage_report


def normalize_date(date_str: str) -> str:
    """Accept calendar dates YYYY_MM_DD or YYYY-MM-DD; return YYYY_MM_DD."""
    if not isinstance(date_str, str) or not re.fullmatch(
        r"\d{4}([_-])\d{2}\1\d{2}", date_str
    ):
        raise ValueError("date_str must have format YYYY_MM_DD or YYYY-MM-DD")
    normalized = date_str.replace("-", "_")
    try:
        datetime.strptime(normalized, "%Y_%m_%d")
    except ValueError as exc:
        raise ValueError(f"Invalid calendar date: {date_str}") from exc
    return normalized


def _polygon_parts(geometry):
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return [geometry]
    if isinstance(geometry, MultiPolygon):
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        return [part for item in geometry.geoms for part in _polygon_parts(item)]
    return []


def _metric_crs(source_crs: CRS, source_geometries) -> CRS:
    # Keep source UTM grids unchanged: their pixels and the module-2 hectare
    # statistics then have exactly the same projected-area interpretation.
    if source_crs.utm_zone and all(
        math.isclose(axis.unit_conversion_factor, 1.0) for axis in source_crs.axis_info
    ):
        return source_crs
    minx = min(g.bounds[0] for g in source_geometries)
    miny = min(g.bounds[1] for g in source_geometries)
    maxx = max(g.bounds[2] for g in source_geometries)
    maxy = max(g.bounds[3] for g in source_geometries)
    lon, lat = Transformer.from_crs(source_crs, 4326, always_xy=True).transform(
        (minx + maxx) / 2, (miny + maxy) / 2
    )
    if not all(math.isfinite(v) for v in (lon, lat)) or not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError("Raster coordinates cannot be transformed to WGS84")
    if source_crs.is_geographic and maxx - minx > 180:
        raise ValueError("A raster spanning the antimeridian must be split into local fields")
    if -80 <= lat <= 84:
        zone = min(60, max(1, int((lon + 180) // 6) + 1))
        return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)
    return CRS.from_proj4(
        f"+proj=laea +lat_0={lat:.10f} +lon_0={lon:.10f} +datum=WGS84 +units=m +no_defs"
    )


def _extract(zones_tif_path):
    with rasterio.open(zones_tif_path) as src:
        if src.count != 1:
            raise ValueError("Zone raster must have exactly one band")
        if src.crs is None:
            raise ValueError("Zone raster has no CRS")
        crs = CRS.from_user_input(src.crs)
        if not (crs.is_projected or crs.is_geographic):
            raise ValueError("Zone raster CRS must be projected or geographic")
        affine = src.transform
        if not all(math.isfinite(v) for v in affine) or abs(affine.determinant) == 0:
            raise ValueError("Zone raster has an invalid or degenerate geotransform")
        band = src.read(1, masked=True)
        values = np.asarray(band.data)
        valid = ~np.ma.getmaskarray(band) & (values != 0)
        inside = values[valid]
        if inside.size == 0:
            raise ValueError("Zone raster contains no valid field pixels")
        if not np.all(np.isfinite(inside)):
            raise ValueError("Zone raster contains unmasked non-finite values")
        if not np.all(np.isin(inside, (1, 2, 3))):
            invalid = np.unique(inside[~np.isin(inside, (1, 2, 3))])[:10]
            raise ValueError(f"Zone raster must contain classes 0, 1, 2, 3; found {invalid.tolist()}")
        # Only cast after validation, preventing silent truncation of float labels.
        labels = np.zeros(values.shape, dtype=np.uint8)
        labels[valid] = inside.astype(np.uint8)
        records = []
        for geometry, label in shapes(labels, mask=valid, transform=affine, connectivity=4):
            polygon = shape(geometry)
            if not polygon.is_valid:
                polygon = make_valid(polygon)
            for part in _polygon_parts(polygon):
                if not part.is_empty and part.area > 0:
                    records.append({"zone_class": int(label), "geometry": part})
        metadata = {
            "valid_pixel_count": int(valid.sum()),
            "excluded_pixel_count": int(valid.size - valid.sum()),
            "source_crs": crs.to_string(),
            "raster_width": src.width,
            "raster_height": src.height,
            "pixel_size_source_units": [math.hypot(affine.a, affine.d), math.hypot(affine.b, affine.e)],
        }
    if not records:
        raise ValueError("No polygonal geometry could be extracted from the raster")
    return records, crs, metadata


def _merge_small_regions(records, project, threshold):
    """Merge connected tiny regions along the longest shared boundary.

    Topology operations remain in the original grid CRS, avoiding gaps caused
    by separately reprojecting collinear raster boundaries. Edge distances and
    area thresholds are measured in the selected metric CRS. Isolated regions
    are retained; field coverage and nodata holes are never discarded.
    """
    geometries = [r["geometry"] for r in records]
    areas = [float(project(g).area / 10000) for g in geometries]
    if any(not math.isfinite(a) or a <= 0 for a in areas):
        raise ValueError("The projected polygon areas must be positive and finite")
    regions = {
        i: {"geometry": g, "zone_class": records[i]["zone_class"],
            "area": areas[i], "origins": [i], "version": 0}
        for i, g in enumerate(geometries)
    }
    if threshold == 0 or not any(a < threshold for a in areas):
        return list(regions.values()), areas, 0
    tree = STRtree(geometries)
    adjacency = {i: {} for i in regions}
    for i, geometry in enumerate(geometries):
        for raw_j in tree.query(geometry, predicate="intersects"):
            j = int(raw_j)
            if j <= i:
                continue
            shared = geometry.boundary.intersection(geometries[j].boundary)
            if shared.is_empty or shared.length == 0:
                continue
            length = float(project(shared).length)
            if length > 0 and math.isfinite(length):
                adjacency[i][j] = length
                adjacency[j][i] = length
    queue = [(r["area"], i, 0) for i, r in regions.items() if r["area"] < threshold]
    heapq.heapify(queue)
    merges = 0
    while queue:
        _, i, version = heapq.heappop(queue)
        if i not in regions or regions[i]["version"] != version:
            continue
        region = regions[i]
        if region["area"] >= threshold:
            continue
        neighbors = adjacency[i]
        if not neighbors:
            continue
        # Deterministic ties: larger neighbor, lower class, lower original id.
        j = max(neighbors, key=lambda n: (
            neighbors[n], regions[n]["area"], -regions[n]["zone_class"], -n
        ))
        target = regions[j]
        target["geometry"] = target["geometry"].union(region["geometry"])
        if not target["geometry"].is_valid:
            raise ValueError("Merging small regions created invalid geometry")
        target["area"] = float(project(target["geometry"]).area / 10000)
        target["origins"].extend(region["origins"])
        target["version"] += 1
        adjacency[j].pop(i, None)
        for k, length in list(neighbors.items()):
            adjacency[k].pop(i, None)
            if k != j:
                adjacency[j][k] = adjacency[j].get(k, 0.0) + length
                adjacency[k][j] = adjacency[j][k]
        del adjacency[i]
        del regions[i]
        merges += 1
        if target["area"] < threshold:
            heapq.heappush(queue, (target["area"], j, target["version"]))
    return list(regions.values()), areas, merges


def generate_prescription(
    zones_tif_path: str,
    output_dir: str,
    date_str: str,
    base_rate_kg_ha: float = 150.0,
    rates_by_zone: dict | None = None,
    price_per_kg: float = 45.0,
    *,
    min_polygon_area_ha: float = 0.05,
    unit: str = "kg",
    reference_mask_path: str | None = None,
) -> tuple[str, str, str]:
    """Generate GeoJSON, zipped ESRI Shapefile, and a JSON economic report.

    Norms are illustrative project defaults, not agronomic recommendations.
    With unit='l', legacy *_kg arguments are interpreted as liters/ha and
    RUB/liter. Missing/background pixels in the input zones are excluded.
    Classes filled upstream by M2 are included; classified does not mean observed.
    Returns paths to (GeoJSON, Shapefile ZIP, metrics JSON).
    """
    date = normalize_date(date_str)
    base, rates, price = validate_parameters(base_rate_kg_ha, rates_by_zone, price_per_kg, unit)
    threshold = _nonnegative(min_polygon_area_ha, "min_polygon_area_ha")
    records, source_crs, metadata = _extract(zones_tif_path)
    area_crs = _metric_crs(source_crs, [r["geometry"] for r in records])
    transformer = Transformer.from_crs(source_crs, area_crs, always_xy=True)
    # Densify projected curves at half the source pixel spacing. Otherwise
    # transforming a long simplified edge and its several shorter neighboring
    # edges can produce different chords and tiny artificial gaps/area drift.
    max_segment_length = min(metadata["pixel_size_source_units"]) / 2
    if source_crs == area_crs:
        project = lambda geometry: geometry
    else:
        project = lambda geometry: transform_geometry(
            transformer.transform, segmentize(geometry, max_segment_length)
        )
    regions, original_areas, merge_count = _merge_small_regions(records, project, threshold)
    source_areas = {
        z: math.fsum(a for r, a in zip(records, original_areas) if r["zone_class"] == z)
        for z in ZONE_NAMES
    }
    # Dissolve by final class, then split into connected polygon features.
    # This removes artificial internal boundaries without bridging gaps.
    output_records = []
    for zone in ZONE_NAMES:
        group = [r["geometry"] for r in regions if r["zone_class"] == zone]
        if not group:
            continue
        merged_geometry = unary_union(group)
        for geometry in sorted(_polygon_parts(merged_geometry), key=lambda g: g.bounds):
            metric_geometry = project(geometry)
            if not metric_geometry.is_valid or metric_geometry.is_empty:
                raise ValueError("Projection created invalid prescription geometry")
            area = float(metric_geometry.area / 10000)
            output_records.append({
                "zone_class": zone,
                "zone_name": ZONE_NAMES[zone],
                "area_ha": area,
                "fertilizer_rate": rates[zone],
                f"rate_{unit}_ha": rates[zone],
                "rate_unit": f"{unit}/ha",
                f"total_{unit}": area * rates[zone],
                "total_quantity": area * rates[zone],
                "quantity_unit": unit,
                "geometry": metric_geometry,
            })
    gdf = gpd.GeoDataFrame(output_records, geometry="geometry", crs=area_crs)
    final_areas = {
        z: math.fsum(r["area_ha"] for r in output_records if r["zone_class"] == z)
        for z in ZONE_NAMES
    }
    source_total = math.fsum(source_areas.values())
    final_total = math.fsum(final_areas.values())
    # Projection can introduce sub-mm numerical noise; never accept material loss.
    if not math.isclose(source_total, final_total, rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError("Polygon processing changed the total observed field area")
    report = calculate_economics(final_areas, base, rates, price, unit=unit)
    report.update({"schema_version": "1.0", "date": date,
                   "input_raster": str(Path(zones_tif_path).resolve())})
    reassigned = math.fsum(
        original_areas[i] for region in regions for i in region["origins"]
        if records[i]["zone_class"] != region["zone_class"]
    )
    retained_small = sum(r["area_ha"] < threshold for r in output_records)
    report["processing"] = {
        **metadata,
        "area_crs": area_crs.to_string(),
        "area_method": "projected_area_m2_divided_by_10000",
        "min_polygon_area_ha": threshold,
        "input_polygon_count": len(records),
        "output_polygon_count": len(output_records),
        "merged_polygon_count": merge_count,
        "retained_small_polygon_count": retained_small,
        "reassigned_area_ha": reassigned,
        "source_total_field_area_ha": source_total,
        "source_area_by_zone": [{"zone_class": z, "area_ha": source_areas[z]} for z in ZONE_NAMES],
        "source_stressed_area_ha": source_areas[1] + source_areas[2],
        "source_stressed_area_percent": (source_areas[1] + source_areas[2]) / source_total * 100,
    }
    report["warnings"] = []
    if reference_mask_path is not None:
        report["coverage"] = coverage_report(zones_tif_path, reference_mask_path, project)
        if not report["coverage"]["is_complete"]:
            report["warnings"].append(
                "Incomplete reference-field coverage: no rate is assigned to unclassified pixels; "
                "economics applies only to classified area."
            )
    if retained_small:
        report["warnings"].append(
            f"{retained_small} isolated polygon(s) below the area threshold were retained to preserve coverage."
        )
    if reassigned > 0:
        report["warnings"].append(
            "Micro-polygon merging changed zone assignments; source zone areas are recorded in processing."
        )
    report["warnings"].append(
        "Areas cover valid classified pixels only. Background and nodata are excluded; "
        "upstream-filled classes are included and are not direct observations. "
        "See M2 index_stats for observed_area_ha and filled_area_ha."
    )
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    geojson_name = f"prescription_{date}.geojson"
    metrics_name = f"metrics_{date}.json"
    shapefile_dir_name = f"shapefile_{date}"
    zip_name = f"{shapefile_dir_name}.zip"
    shp_name = f"prescription_{date}.shp"
    report["files"] = {
        "geojson": geojson_name,
        "shapefile_zip": zip_name,
        "shapefile": f"{shapefile_dir_name}/{shp_name}",
        "metrics": metrics_name,
    }
    # Stage all artifacts before replacing the named outputs. ZIP input is a
    # fresh directory, so reruns cannot include unrelated or stale sidecar files.
    with TemporaryDirectory(prefix=".vra-", dir=out) as temporary:
        stage = Path(temporary)
        shp_stage = stage / shapefile_dir_name
        shp_stage.mkdir()
        short_names = {"fertilizer_rate": "fert_rate", "total_quantity": "total_qty",
                       "quantity_unit": "qty_unit"}
        shp_frame = gdf.rename(columns=short_names)
        if any(len(name) > 10 for name in shp_frame.columns if name != "geometry"):
            raise ValueError("Shapefile field names exceed the DBF 10-character limit")
        shp_frame.to_file(shp_stage / shp_name, driver="ESRI Shapefile", encoding="UTF-8", index=False)
        for suffix in (".shp", ".shx", ".dbf", ".prj"):
            if not (shp_stage / Path(shp_name).with_suffix(suffix)).is_file():
                raise RuntimeError(f"Missing required Shapefile component: {suffix}")
        # GDAL normally creates .cpg; write an explicit encoding file if needed.
        cpg = shp_stage / Path(shp_name).with_suffix(".cpg")
        if not cpg.exists():
            cpg.write_text("UTF-8", encoding="ascii")
        shutil.make_archive(str(stage / shapefile_dir_name), "zip", root_dir=shp_stage)
        web = gdf.to_crs(epsg=4326)
        features = []
        for record in web.to_dict("records"):
            geometry = record.pop("geometry")
            if not geometry.is_valid or geometry.is_empty or not all(math.isfinite(v) for v in geometry.bounds):
                raise ValueError("Invalid WGS84 geometry in GeoJSON output")
            features.append({"type": "Feature", "properties": record, "geometry": mapping(geometry)})
        collection = {"type": "FeatureCollection", "features": features}
        (stage / geojson_name).write_text(
            json.dumps(collection, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        (stage / metrics_name).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
        )
        shp_destination = out / shapefile_dir_name
        shp_destination.mkdir(exist_ok=True)
        for sidecar in shp_stage.iterdir():
            os.replace(sidecar, shp_destination / sidecar.name)
        for name in (geojson_name, zip_name, metrics_name):
            os.replace(stage / name, out / name)
    return str(out / geojson_name), str(out / zip_name), str(out / metrics_name)
