"""Synthetic test for terrain_diffusion.data.grid.canopy. Run from the terrain-diffusion repo root."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

sys.path.insert(0, ".")
from terrain_diffusion.data.grid.canopy import quadkey_bounds  # noqa: E402
from terrain_diffusion.data.grid.tiles import child_tiles, parent_cells, parents_intersecting  # noqa: E402

root = Path(sys.argv[1])
chm = root / "chm"
out = root / "tiles"
shutil.rmtree(root, ignore_errors=True)
chm.mkdir(parents=True)

# quadkey check against a known zoom-1 tile: "0" is the NW quadrant
assert quadkey_bounds("0") == (-180.0, 0.0, 0.0, 85.0511287798066), quadkey_bounds("0")

# fake Meta tile: zoom-9 quadkey in the US, 60 m pixels in EPSG:3857, float32, nodata -1
qk = "023013213000"  # zoom 12, about 10 km, so a 2 m source stays small
lon0, lat0, lon1, lat1 = quadkey_bounds(qk)
l, b, r, t = transform_bounds("EPSG:4326", "EPSG:3857", lon0, lat0, lon1, lat1)
res = 2.0
w = int(round((r - l) / res)); h = int(round((t - b) / res))
tr = from_bounds(l, b, r, t, w, h)
rows = np.arange(h)[:, None]; cols = np.arange(w)[None, :]
height = np.where(rows < h // 2, 10.0, 0.0) + np.zeros((h, w))      # north half 10 m, south half ground
height = height.astype(np.float32)
height[(rows % 7 == 0) & (cols >= 0)] = 3.0                         # every 7th row = 3 m (between the thresholds)
height[:, : w // 10] = -1.0                                         # nodata strip on the west edge
with rasterio.open(chm / f"{qk}.tif", "w", driver="GTiff", dtype="float32", count=1, width=w, height=h,
                   crs="EPSG:3857", transform=tr, nodata=-1.0, tiled=True, compress="deflate") as ds:
    ds.write(height, 1)
print(f"source {w}x{h} px, bounds {lon0:.3f},{lat0:.3f},{lon1:.3f},{lat1:.3f}")

# pick the parent whose centre is nearest the quadkey centre
clon, clat = (lon0 + lon1) / 2, (lat0 + lat1) / 2
cands = parents_intersecting([(lon0, lat0, lon1, lat1)])
best = min(cands, key=lambda i: (sum(parent_cells()[i][0::2]) / 2 - clon) ** 2 + (sum(parent_cells()[i][1::2]) / 2 - clat) ** 2)
print("parent", best, parent_cells()[best])

cmd = [sys.executable, "-m", "terrain_diffusion.data.grid.canopy", "--chm-dir", str(chm), "--out", str(out),
       "--workers", "1", "--parents", str(best)]
subprocess.run(cmd, check=True)

# ---- checks
kids = child_tiles(best)
for k in kids:
    assert (out / k.path / "canopy.tif").exists(), k.id
    assert (out / k.path / "canopy.json").exists()
with rasterio.open(out / f"30m/{k.parent}/canopy.tif") as ds:
    p = ds.read(); ptr = ds.transform; descs = ds.descriptions
assert descs == ("height", "cover2", "cover5", "valid"), descs
print("parent valid mean", float(np.nanmean(p[3])), "height range", np.nanmin(p[0]), np.nanmax(p[0]))

# geometry: sample the parent at a point in the north half and south half, inside footprint
def sample(arr, transform, lon, lat, k=10):
    """Mean over a (2k)^2 neighbourhood so the every-7th-row pattern averages out."""
    c, r = ~transform * (lon, lat)
    r, c = int(r), int(c)
    return np.nanmean(arr[:, r - k:r + k, c - k:c + k], axis=(1, 2))

pmin_lon, pmin_lat, pmax_lon, pmax_lat = parent_cells()[best]
assert pmin_lon < lon0 and pmax_lon > lon1 and pmin_lat < lat0 and pmax_lat > lat1, "quadkey must sit inside the parent"
n_pt = (clon + 0.02, lat1 - 0.01)
s_pt = (clon + 0.02, lat0 + 0.01)
vn = sample(p, ptr, *n_pt); vs = sample(p, ptr, *s_pt)
print("north sample", vn, "south sample", vs)
# north: 10 m with every 7th 2 m-row at 3 m -> height 9.0, cover2 1, cover5 6/7, valid 1
assert abs(vn[0] - 9.0) < 0.15 and vn[1] > 0.99 and abs(vn[2] - 6 / 7) < 0.03 and vn[3] > 0.99, vn
# south: ground with every 7th row 3 m -> height 3/7, cover2 1/7, cover5 0, valid 1
assert abs(vs[0] - 3 / 7) < 0.05 and abs(vs[1] - 1 / 7) < 0.03 and vs[2] < 0.01 and vs[3] > 0.99, vs

# nodata strip: west tenth of the quadkey is invalid -> valid 0 and NaN height
v = sample(p, ptr, lon0 + (lon1 - lon0) * 0.05, clat, k=1)
assert v[3] == 0 and np.isnan(v[0]), v
print("nodata strip ok", v)
# outside the quadkey entirely: valid 0, height NaN
v = sample(p, ptr, lon1 + 0.05, clat, k=1)
assert v[3] == 0 and np.isnan(v[0]), v
# total valid area in the parent should equal 90 percent of the quadkey area, in pixels
px_area = (ptr.a * 111320 * np.cos(np.radians(clat))) * (-ptr.e * 111320)
qk_area = (lon1 - lon0) * 111320 * np.cos(np.radians(clat)) * (lat1 - lat0) * 111320
print("valid px area", np.nansum(p[3]) * px_area / 1e6, "km2, expected", 0.9 * qk_area / 1e6)
assert abs(np.nansum(p[3]) * px_area / (0.9 * qk_area) - 1) < 0.02

# child/parent consistency: parent == 3x3 block mean of child mosaic
n = kids[0].size
mos = np.full((4, 3 * n, 3 * n), np.nan, dtype=np.float32)
for i, k in enumerate(kids):
    r, c = divmod(i, 3)
    with rasterio.open(out / k.path / "canopy.tif") as ds:
        mos[:, r * n:(r + 1) * n, c * n:(c + 1) * n] = ds.read()
w4 = np.nan_to_num(mos[3]).reshape(2048, 3, 2048, 3)
num = np.nan_to_num(mos[0] * mos[3]).reshape(2048, 3, 2048, 3).sum(axis=(1, 3))
den = w4.sum(axis=(1, 3))
ref = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
diff = np.nanmax(np.abs(ref - p[0]))
print("parent vs child-mosaic max diff", diff)
assert diff < 1e-3
# orientation of the child mosaic: the rows holding valid data must be the rows the quadkey's latitude band maps to
# under the parent transform refined 3x (child row 0 = north)
from rasterio.transform import Affine  # noqa: E402
mtr = ptr * Affine.scale(1 / 3)
rows_valid = np.where(np.nan_to_num(mos[3]).sum(axis=1) > 0)[0]
exp_top = (~mtr * (clon, lat1))[1]; exp_bot = (~mtr * (clon, lat0))[1]
print("valid rows", rows_valid.min(), rows_valid.max(), "expected", exp_top, exp_bot)
assert abs(rows_valid.min() - exp_top) < 2 and abs(rows_valid.max() - exp_bot) < 2
print(json.loads((out / kids[0].path / "canopy.json").read_text()))
print("ALL CHECKS PASSED")
