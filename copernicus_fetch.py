"""
copernicus_sigma0_fetch.py
---------------------------------
Fetches Sentinel-1 GRD Sigma0 (dB) imagery from the Copernicus Data Space
Ecosystem (CDSE) using the same kind of processing chain the Zenodo dataset
used (calibration -> noise removal -> terrain/ellipsoid correction -> dB),
and gives you tools to check that a freshly-fetched scene actually looks
statistically consistent with a Zenodo training scene.

SETUP (one-time, manual):
1. Create a free account at https://dataspace.copernicus.eu
2. Go to your Dashboard -> "Settings" -> "OAuth clients" -> create a new
   OAuth client. Copy the client_id and client_secret it gives you.
3. pip install sentinelhub rasterio pyproj --break-system-packages
4. Set the two env vars below (or hardcode them locally, never commit them):
   export CDSE_CLIENT_ID=...
   export CDSE_CLIENT_SECRET=...

NOTE: Sentinel Hub / CDSE's Process API evolves — if any call here 404s or
rejects a param, check the current docs at
https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Process.html
before assuming the preprocessing itself is wrong.
"""

import os
import math
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from sentinelhub import (
    SHConfig, DataCollection, SentinelHubRequest, BBox, CRS, MimeType,
    bbox_to_dimensions,
)

# ---------------------------------------------------------------------------
# 1. CDSE config
# ---------------------------------------------------------------------------
def build_cdse_config() -> SHConfig:
    config = SHConfig()
    config.sh_client_id = os.environ.get("CDSE_CLIENT_ID", "")
    config.sh_client_secret = os.environ.get("CDSE_CLIENT_SECRET", "")
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.sh_token_url = (
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE"
        "/protocol/openid-connect/token"
    )
    if not config.sh_client_id or not config.sh_client_secret:
        raise RuntimeError(
            "Set CDSE_CLIENT_ID / CDSE_CLIENT_SECRET env vars first "
            "(see module docstring for how to create them)."
        )
    return config


# Sentinel-1 IW GRD collection, pointed at the CDSE service instead of the
# default AWS-hosted Sentinel Hub deployment.
S1_CDSE = DataCollection.SENTINEL1_IW.define_from(
    "S1_CDSE", service_url="https://sh.dataspace.copernicus.eu"
)

# ---------------------------------------------------------------------------
# 2. Evalscript: replicate "Sigma0 in dB" the way the Zenodo dataset states
#    it was produced (calibration + terrain correction baked into backCoeff,
#    then a manual linear -> dB conversion here).
# ---------------------------------------------------------------------------
EVALSCRIPT_SIGMA0_DB = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["VV", "VH"], units: "LINEAR_POWER" }],
    output: { bands: 2, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  function toDb(x) { return x > 0 ? 10 * Math.log10(x) : -9999; }
  return [toDb(sample.VV), toDb(sample.VH)];
}
"""


def bbox_from_center(lat: float, lon: float, size_px: int = 2048, resolution_m: float = 10.0) -> BBox:
    """Build a bbox of size_px x size_px pixels at resolution_m ground
    resolution, centered at (lat, lon). Matches the Zenodo scenes' native
    2048x2048 @ ~10m Sentinel-1 IW resolution."""
    half_extent_m = (size_px * resolution_m) / 2
    # rough meters-per-degree at this latitude
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat))
    dlat = half_extent_m / m_per_deg_lat
    dlon = half_extent_m / m_per_deg_lon
    return BBox(bbox=[lon - dlon, lat - dlat, lon + dlon, lat + dlat], crs=CRS.WGS84)


def fetch_sigma0_db(lat: float, lon: float, date_from: str, date_to: str,
                     out_path: str, size_px: int = 2048,
                     back_coeff: str = "SIGMA0_ELLIPSOID") -> str:
    """Fetch a Sigma0-dB, VV+VH GeoTIFF for the given center point and date
    range. back_coeff can be 'SIGMA0_ELLIPSOID' or 'SIGMA0_TERRAIN' -- try
    both and see which one's statistics line up better with the Zenodo
    scenes over flat ocean (ellipsoid is usually fine for open water)."""
    config = build_cdse_config()
    bbox = bbox_from_center(lat, lon, size_px=size_px)
    size = bbox_to_dimensions(bbox, resolution=10)

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_SIGMA0_DB,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=S1_CDSE,
                time_interval=(date_from, date_to),
                other_args={
                    "processing": {
                        "backCoeff": back_coeff,
                        "orthorectify": True,
                    }
                },
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
        # NOTE: we deliberately do NOT rely on save_data=True here --
        # sentinelhub writes into its own hashed subfolder under data_folder,
        # not to out_path directly, which silently produces a missing-file
        # error later. We fetch the array in memory and write the GeoTIFF
        # ourselves instead, so out_path is guaranteed to be where the file
        # actually is, with a correct CRS/transform attached.
    )
    data = request.get_data(save_data=False)
    array = data[0]  # (H, W, 2) -- VV, VH
    print(f"Fetched array shape: {array.shape}")

    width, height = size
    transform = from_bounds(bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y, width, height)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=height, width=width,
        count=2, dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(array[:, :, 0], 1)  # VV
        dst.write(array[:, :, 1], 2)  # VH

    print(f"Saved georeferenced GeoTIFF -> {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# 3. Pull the exact bbox out of a Zenodo training GeoTIFF (they ARE
#    georeferenced per the dataset notes), so you can re-fetch the *same*
#    location from Copernicus and compare apples to apples -- isolating
#    preprocessing differences from geography differences.
# ---------------------------------------------------------------------------
def extract_bbox_from_geotiff(path: str):
    """Returns (lon_min, lat_min, lon_max, lat_max) in WGS84 for a
    georeferenced Zenodo image TIFF."""
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"{path} has no CRS -- is this really one of the "
                              f"georeferenced Sigma0 images, not a mask?")
        bounds = src.bounds
        lon_min, lat_min, lon_max, lat_max = transform_bounds(
            src.crs, "EPSG:4326", *bounds
        )
    return lon_min, lat_min, lon_max, lat_max


# ---------------------------------------------------------------------------
# 4. Compare statistics between a Zenodo training scene and a freshly
#    fetched Copernicus scene (ideally over the same bbox, per step 3).
# ---------------------------------------------------------------------------
def compare_stats(zenodo_path: str, fetched_path: str, zenodo_band_order=("VH", "VV")):
    """zenodo_band_order: confirmed empirically for this Zenodo dataset --
    band 0 is VH and band 1 is VV (opposite of what you'd naively assume).
    If you point this at a different dataset later, re-verify this with the
    multi-scene mean-comparison check before trusting the labels below."""
    with rasterio.open(zenodo_path) as src:
        zen = src.read().astype(np.float32)  # (2, H, W) -- order per zenodo_band_order
    with rasterio.open(fetched_path) as src:
        fetched = src.read().astype(np.float32)

    # -60 dB rather than -100: real open-water Sigma0 rarely goes below
    # ~-35 to -40 dB, so anything past -60 is almost always a border/
    # noise-floor artifact, not signal worth including in the comparison.
    OUTLIER_FLOOR = -60.0

    for name, arr, band_order in [
        ("Zenodo", zen, zenodo_band_order),
        ("Copernicus fetch", fetched, ("VV", "VH")),
    ]:
        print(f"\n{name}: shape={arr.shape} (band order: {band_order})")
        for i, pol in enumerate(band_order):
            band = arr[i]
            band = band[np.isfinite(band) & (band > OUTLIER_FLOOR)]
            if band.size == 0:
                print(f"  {pol}: no valid pixels")
                continue
            print(f"  {pol}: mean={band.mean():.2f} dB, std={band.std():.2f}, "
                  f"min={band.min():.2f}, max={band.max():.2f}")

    print(
        "\nWhat to look for: means/stds within a couple dB of each other and "
        "similar min/max range suggest the preprocessing chains are "
        "consistent. A large, systematic offset (e.g. Copernicus fetch is "
        "~5-10 dB brighter/darker across the board) usually means a "
        "backCoeff/orthorectify mismatch, not a real scene difference -- "
        "try the other backCoeff option and re-run."
    )


if __name__ == "__main__":
    # --- Example usage ---
    # 1) Point this at one of your downloaded Zenodo training images:
    zenodo_sample = "D:/MINE/Projects/00000.tif"  # adjust path

    # 2) Pull its real-world bbox, then re-fetch that same area from
    #    Copernicus for a *different* date (any recent date works fine --
    #    we're checking preprocessing consistency, not re-detecting the
    #    same spill).
    lon_min, lat_min, lon_max, lat_max = extract_bbox_from_geotiff(zenodo_sample)
    center_lat = (lat_min + lat_max) / 2
    center_lon = (lon_min + lon_max) / 2
    print(f"Zenodo scene center: {center_lat:.4f}, {center_lon:.4f}")

    fetched_path = fetch_sigma0_db(
        lat=center_lat, lon=center_lon,
        date_from="2026-08-01", date_to="2026-08-15",
        out_path="./cdse_fetch/sigma0.tif",
    )

    compare_stats(zenodo_sample, fetched_path)