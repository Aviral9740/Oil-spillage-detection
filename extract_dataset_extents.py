"""
extract_dataset_extents.py
---------------------------------
Extracts the real-world bounding box + centroid of every georeferenced
Zenodo image (Part I oil-positive, Part II no-oil/lookalike), and clusters
the centroids into distinct regions -- so the backend/AIS work can target
actual regions instead of one meaningless global bounding box.

Fast: rasterio only reads the file's header/metadata here, not the pixel
data, so this runs quickly even across thousands of scenes.

Usage:
    python extract_dataset_extents.py /path/to/one/or/more/image/folders
"""

import sys
import glob
import os
import csv
import rasterio
from rasterio.warp import transform_bounds


def scan_folder(folder: str):
    """Returns list of dicts: filename, lon_min, lat_min, lon_max, lat_max, center_lon, center_lat"""
    paths = sorted(glob.glob(os.path.join(folder, "*.tif")) + glob.glob(os.path.join(folder, "*.tiff")))
    results = []
    for p in paths:
        try:
            with rasterio.open(p) as src:
                if src.crs is None:
                    continue  # masks aren't georeferenced -- skip anything without a CRS
                bounds = src.bounds
                lon_min, lat_min, lon_max, lat_max = transform_bounds(src.crs, "EPSG:4326", *bounds)
        except Exception as e:
            print(f"  skipped {p}: {e}")
            continue
        results.append({
            "file": os.path.basename(p),
            "folder": os.path.basename(folder.rstrip("/")),
            "lon_min": lon_min, "lat_min": lat_min,
            "lon_max": lon_max, "lat_max": lat_max,
            "center_lon": (lon_min + lon_max) / 2,
            "center_lat": (lat_min + lat_max) / 2,
        })
    return results


def cluster_centroids(rows, grid_deg=5.0):
    """Cheap clustering: snap each centroid to a grid_deg x grid_deg grid cell.
    Good enough to separate 'Gulf of Mexico' from 'North Sea' from 'Persian Gulf'
    without pulling in a real clustering library. Tighten grid_deg for finer
    separation if scenes turn out to be close together within one sea."""
    clusters = {}
    for r in rows:
        key = (round(r["center_lat"] / grid_deg) * grid_deg,
               round(r["center_lon"] / grid_deg) * grid_deg)
        clusters.setdefault(key, []).append(r)
    return clusters


def main(folders):
    all_rows = []
    for folder in folders:
        print(f"Scanning {folder} ...")
        rows = scan_folder(folder)
        print(f"  {len(rows)} georeferenced scenes found")
        all_rows.extend(rows)

    if not all_rows:
        print("No georeferenced scenes found -- check the folder paths point at the *_images "
              "directories (masks aren't georeferenced per the dataset notes).")
        return

    # Save the full per-scene table -- hand this CSV to the backend dev directly,
    # it's more useful than any summary.
    out_csv = "zenodo_scene_extents.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote per-scene extents -> {out_csv} ({len(all_rows)} rows)")

    # Overall envelope (for reference only -- see caveat below)
    lon_min = min(r["lon_min"] for r in all_rows)
    lon_max = max(r["lon_max"] for r in all_rows)
    lat_min = min(r["lat_min"] for r in all_rows)
    lat_max = max(r["lat_max"] for r in all_rows)
    print(f"\nOverall bounding envelope (all scenes): "
          f"lon [{lon_min:.2f}, {lon_max:.2f}], lat [{lat_min:.2f}, {lat_max:.2f}]")
    print("CAVEAT: this envelope almost certainly includes huge amounts of land/irrelevant "
          "ocean between clusters if scenes are scattered globally -- see the cluster "
          "breakdown below for the actual regions to hand to the AIS work.")

    # Region clusters
    clusters = cluster_centroids(all_rows, grid_deg=5.0)
    print(f"\n{len(clusters)} distinct region cluster(s) (~5-degree grid):")
    for (lat, lon), rows in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        print(f"  ~({lat:.1f}, {lon:.1f})  -- {len(rows)} scenes")


if __name__ == "__main__":
    folders = sys.argv[1:]
    if not folders:
        print("Usage: python extract_dataset_extents.py <images_folder> [<images_folder2> ...]")
        print("Example:")
        print("  python extract_dataset_extents.py "
              "E:/01_Train_Val_Oil_Spill_images/Oil")
        sys.exit(1)
    main(folders)