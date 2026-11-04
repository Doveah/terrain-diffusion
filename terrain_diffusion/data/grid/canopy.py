"""P1 reducer: Meta/WRI 1 m canopy height -> 10 m children and 30 m parents (DESIGN.md D4, PIPELINE.md P1).

Input: a directory of Meta canopy tiles named ``<quadkey>.tif`` (EPSG:3857, ~1 m). Nothing about their resolution or
nodata value is assumed; footprints come from the quadkey, everything else from each file's own metadata.

Output per 10 m child tile (``<out>/10m/<parent>/<child>/canopy.tif``), float32, four bands on the child grid:

    1 height   mean canopy height in metres over all valid 1 m pixels (ground counts as 0)
    2 cover2   fraction of valid 1 m pixels with height > 2 m
    3 cover5   fraction of valid 1 m pixels with height > 5 m
    4 valid    fraction of the 10 m pixel covered by valid 1 m pixels (0 outside the Meta footprint)

Height and cover are NaN where ``valid`` is 0. The 30 m parent product (``<out>/30m/<parent>/canopy.tif``) is the exact
3 x 3 block mean of the nine children, weighted by ``valid``.

How it works: each child is processed in blocks. For every block the intersecting source tiles are read with a halo
(boundless, so area outside a tile's footprint is explicitly invalid). Height, the two indicators and validity are
carried as densities (each multiplied by validity), block-meaned from 1 m to ``DECIMATE`` m in the source projection,
and reprojected onto the block with GDAL's average resampling; dividing by the reprojected validity gives the exact
validity-weighted means. Sources meeting inside one block add their densities, so tile seams are exact. The decimation
cuts the reproject work by DECIMATE^2 (16x at the default) and only quantises where a 10 m pixel edge falls in the source
to 4 m, below the source's own positional accuracy. ``--decimate 1`` gives the undecimated reference.

Run (inside the data-prep container):

    /opt/p1/bin/python -m terrain_diffusion.data.grid.canopy --chm-dir /data/a/meta_chm/chm --out /data/b/tiles --workers 24

``--bbox`` limits the run, ``--limit`` takes the first N parents, existing outputs are skipped so a run can be resumed.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

if __package__ in (None, ""):  # allow `python canopy.py` without an installed package
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from terrain_diffusion.data.grid.tiles import (  # noqa: E402
    CRS,
    CHILD_SPLIT,
    Bounds,
    Tile,
    block_windows,
    child_tiles,
    intersects,
    parent_tile,
    parents_intersecting,
)

VERSION = "canopy-reducer-2"  # 2: densities decimated before the reproject (exact up to 4 m edge quantisation)
BANDS = ("height", "cover2", "cover5", "valid")
THRESHOLDS = (2.0, 5.0)
HALO_PX = 4
DECIMATE = 4  # block-mean the ~1 m source to this many source pixels per cell before the reproject (1 = off)


# --------------------------------------------------------------------------------------------- source index

def quadkey_bounds(qk: str) -> Bounds:
    """Lon/lat bounds of a Bing/Web-Mercator quadkey tile."""
    x = y = 0
    z = len(qk)
    for i, ch in enumerate(qk):
        mask = 1 << (z - 1 - i)
        d = int(ch)
        if d & 1:
            x |= mask
        if d & 2:
            y |= mask
    n = 2 ** z

    def lon(tx: int) -> float:
        return tx / n * 360.0 - 180.0

    def lat(ty: int) -> float:
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))

    return (lon(x), lat(y + 1), lon(x + 1), lat(y))


def index_sources(chm_dir: Path) -> Dict[str, Bounds]:
    out = {}
    for p in sorted(chm_dir.glob("*.tif")):
        qk = p.stem
        if not qk.isdigit():
            continue
        out[str(p)] = quadkey_bounds(qk)
    if not out:
        raise SystemExit(f"no <quadkey>.tif files in {chm_dir}")
    return out


# --------------------------------------------------------------------------------------------- per-worker state

_SOURCES: Dict[str, Bounds] = {}
_OPEN: Dict[str, rasterio.DatasetReader] = {}
_MAX_OPEN = 16
_DECIMATE = DECIMATE


def _init_worker(sources: Dict[str, Bounds], decimate: int = DECIMATE) -> None:
    global _SOURCES, _DECIMATE
    _SOURCES = sources
    _DECIMATE = decimate
    os.environ.setdefault("GDAL_CACHEMAX", "512")
    os.environ.setdefault("GDAL_NUM_THREADS", "1")


def _open(path: str) -> rasterio.DatasetReader:
    ds = _OPEN.get(path)
    if ds is None:
        if len(_OPEN) >= _MAX_OPEN:
            _OPEN.pop(next(iter(_OPEN))).close()
        ds = rasterio.open(path)
        _OPEN[path] = ds
    return ds


# --------------------------------------------------------------------------------------------- reduction

def _read_source_block(ds: rasterio.DatasetReader, bounds: Bounds, d: int) -> Tuple[np.ndarray, np.ndarray, object]:
    """Read the part of ``ds`` covering lon/lat ``bounds`` plus a halo, with a window whose size is a multiple of ``d``.
    Returns (height, valid_mask, transform)."""
    left, bottom, right, top = transform_bounds(CRS, ds.crs, *bounds, densify_pts=21)
    win = rasterio.windows.from_bounds(left, bottom, right, top, ds.transform)
    col0 = math.floor(win.col_off) - HALO_PX
    row0 = math.floor(win.row_off) - HALO_PX
    col1 = math.ceil(win.col_off + win.width) + HALO_PX
    row1 = math.ceil(win.row_off + win.height) + HALO_PX
    col1 = col0 + -(-(col1 - col0) // d) * d
    row1 = row0 + -(-(row1 - row0) // d) * d
    w = Window(col0, row0, col1 - col0, row1 - row0)
    # Read only the part inside the file and place it in a zero/invalid canvas: several times faster than a boundless
    # masked read, and the file extent is the footprint, so area outside it is explicitly invalid even when the file
    # declares no nodata (the canvas's 0 = ground must never count as valid canopy at tile seams).
    data = np.zeros((row1 - row0, col1 - col0), dtype=np.float32)
    valid = np.zeros(data.shape, dtype=bool)
    ic0, ir0 = max(col0, 0), max(row0, 0)
    ic1, ir1 = min(col1, ds.width), min(row1, ds.height)
    if ic1 > ic0 and ir1 > ir0:
        a = ds.read(1, window=Window(ic0, ir0, ic1 - ic0, ir1 - ir0))
        v = np.isfinite(a) if np.issubdtype(a.dtype, np.floating) else np.ones(a.shape, dtype=bool)
        if ds.nodata is not None:
            v &= a != ds.nodata
        sl = (slice(ir0 - row0, ir1 - row0), slice(ic0 - col0, ic1 - col0))
        data[sl] = np.where(v, a, 0)
        valid[sl] = v
    return data, valid, window_transform(w, ds.transform)


def _decimate(a: np.ndarray, d: int) -> np.ndarray:
    """Mean over d x d cells; ``a`` has dimensions that are multiples of ``d``."""
    if d == 1:
        return a.astype(np.float32, copy=False)
    # Strided slice adds, accumulated in place: about five times faster than reshape().mean(axis=(1, 3)) and it
    # takes bool input (the indicator and validity layers) without a full-size float conversion.
    r = a[0::d].astype(np.float32)
    for i in range(1, d):
        r += a[i::d]
    c = r[:, 0::d].copy()
    for i in range(1, d):
        c += r[:, i::d]
    c *= np.float32(1.0 / (d * d))
    return c


def _average_onto(src: np.ndarray, src_transform, src_crs, dst_shape, dst_transform) -> np.ndarray:
    """Area-weighted mean of ``src`` on the destination grid; 0 where the source does not reach."""
    dst = np.zeros(dst_shape, dtype=np.float32)
    reproject(
        source=np.ascontiguousarray(src, dtype=np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=None,
        dst_transform=dst_transform,
        dst_crs=CRS,
        dst_nodata=0.0,
        resampling=Resampling.average,
        num_threads=1,
    )
    return dst


def reduce_block(bounds: Bounds, shape: Tuple[int, int], dst_transform, sources: Sequence[str],
                 decimate: int = DECIMATE) -> np.ndarray:
    """Four-band float32 block (height, cover2, cover5, valid) for one destination block.

    Every quantity is carried as a per-area density (height * valid, indicator * valid, valid), so decimating the
    1 m source to ``decimate`` m by block mean before the reproject is exact: the destination pixel ends up with
    sum(height * valid) / sum(valid) over the 1 m pixels it covers either way. Only the placement of the destination
    pixel edges is quantised to the decimated grid, which at 4 m against 10 m output is below the source's own
    positional accuracy. Several sources meeting in one block add their densities, so seams are exact too.
    """
    acc = np.zeros((len(BANDS), *shape), dtype=np.float64)  # densities summed over sources; band 3 is the weight
    for path in sources:
        ds = _open(path)
        height, valid, tr = _read_source_block(ds, bounds, decimate)
        if not valid.any():
            continue
        layers = [height * valid] + [(height > t) & valid for t in THRESHOLDS] + [valid]
        tr_d = tr * Affine.scale(decimate)
        for b, layer in enumerate(layers):
            acc[b] += _average_onto(_decimate(layer, decimate), tr_d, ds.crs, shape, dst_transform)
    out = np.full((len(BANDS), *shape), np.nan, dtype=np.float32)
    wsum = acc[3]
    covered = wsum > 0
    for b in range(3):
        out[b][covered] = (acc[b][covered] / wsum[covered]).astype(np.float32)
    out[1:3] = np.clip(out[1:3], 0.0, 1.0)
    out[3] = np.clip(wsum, 0.0, 1.0).astype(np.float32)
    return out


def _profile(tile: Tile) -> dict:
    return dict(
        driver="GTiff", dtype="float32", count=len(BANDS), width=tile.size, height=tile.size, crs=CRS,
        transform=tile.transform, nodata=float("nan"), tiled=True, blockxsize=256, blockysize=256,
        compress="deflate", predictor=3, zlevel=6, BIGTIFF="NO",
    )


def _write(tile: Tile, data: np.ndarray, out_dir: Path, meta: dict) -> Path:
    d = out_dir / tile.path
    d.mkdir(parents=True, exist_ok=True)
    final = d / "canopy.tif"
    tmp = d / "canopy.tif.part"
    with rasterio.open(tmp, "w", **_profile(tile)) as dst:
        dst.write(data)
        for i, name in enumerate(BANDS, start=1):
            dst.set_band_description(i, name)
        dst.update_tags(VERSION=VERSION, THRESHOLDS=",".join(str(t) for t in THRESHOLDS), **{k: str(v) for k, v in meta.items()})
    os.replace(tmp, final)
    valid = data[3]
    stats = dict(
        tile=tile.id, level=tile.level, bounds=tile.bounds, version=VERSION,
        valid_mean=float(np.nanmean(valid)) if valid.size else 0.0,
        height_mean=float(np.nanmean(data[0])) if np.isfinite(data[0]).any() else None,
        cover2_mean=float(np.nanmean(data[1])) if np.isfinite(data[1]).any() else None,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"), **meta,
    )
    (d / "canopy.json").write_text(json.dumps(stats))
    return final


def reduce_child(tile: Tile, out_dir: Path, block: int) -> Optional[Path]:
    final = out_dir / tile.path / "canopy.tif"
    if final.exists():
        return None
    cand = [p for p, b in _SOURCES.items() if intersects(b, tile.bounds)]
    data = np.full((len(BANDS), tile.size, tile.size), np.nan, dtype=np.float32)
    data[3] = 0.0
    if cand:
        tr = tile.transform
        for r0, c0, h, w in block_windows(tile.size, block):
            win = Window(c0, r0, w, h)
            btr = window_transform(win, tr)
            bb = rasterio.windows.bounds(win, tr)  # (left, bottom, right, top) in lon/lat
            srcs = [p for p in cand if intersects(_SOURCES[p], bb)]
            if not srcs:
                continue
            data[:, r0:r0 + h, c0:c0 + w] = reduce_block(bb, (h, w), btr, srcs, _DECIMATE)
    return _write(tile, data, out_dir, dict(n_sources=len(cand), decimate=_DECIMATE))


def aggregate_parent(index: int, out_dir: Path) -> Optional[Path]:
    """30 m parent = valid-weighted 3 x 3 block mean of the nine 10 m children, read back from disk."""
    parent = parent_tile(index)
    final = out_dir / parent.path / "canopy.tif"
    if final.exists():
        return None
    kids = child_tiles(index)
    n = kids[0].size
    big = n * CHILD_SPLIT
    sums = np.zeros((3, big, big), dtype=np.float64)
    wsum = np.zeros((big, big), dtype=np.float64)
    for k, kid in enumerate(kids):
        r, c = divmod(k, CHILD_SPLIT)
        with rasterio.open(out_dir / kid.path / "canopy.tif") as ds:
            arr = ds.read()
        w = np.nan_to_num(arr[3], nan=0.0)
        sl = (slice(r * n, (r + 1) * n), slice(c * n, (c + 1) * n))
        for b in range(3):
            sums[b][sl] = np.nan_to_num(arr[b], nan=0.0) * w
        wsum[sl] = w
    s = CHILD_SPLIT
    m = parent.size
    sums = sums.reshape(3, m, s, m, s).sum(axis=(2, 4))
    wsum = wsum.reshape(m, s, m, s).sum(axis=(1, 3))
    data = np.full((len(BANDS), m, m), np.nan, dtype=np.float32)
    covered = wsum > 0
    for b in range(3):
        data[b][covered] = (sums[b][covered] / wsum[covered]).astype(np.float32)
    data[3] = (wsum / (s * s)).astype(np.float32)
    n_src = sum(1 for b in _SOURCES.values() if intersects(b, parent.bounds))
    return _write(parent, data, out_dir, dict(n_sources=n_src, children=len(kids)))


def process_parent(args: Tuple[int, str, int]) -> Tuple[int, float, int]:
    index, out, block = args
    out_dir = Path(out)
    t0 = time.time()
    written = 0
    for kid in child_tiles(index):
        if reduce_child(kid, out_dir, block) is not None:
            written += 1
    if aggregate_parent(index, out_dir) is not None:
        written += 1
    return index, time.time() - t0, written


# --------------------------------------------------------------------------------------------- CLI

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chm-dir", required=True, type=Path, help="directory of <quadkey>.tif Meta canopy tiles")
    ap.add_argument("--out", required=True, type=Path, help="tile root; writes <out>/10m and <out>/30m")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--block", type=int, default=256, help="destination block size in pixels")
    ap.add_argument("--bbox", type=float, nargs=4, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    ap.add_argument("--limit", type=int, help="process only the first N parents")
    ap.add_argument("--parents", type=int, nargs="*", help="explicit parent indices")
    ap.add_argument("--decimate", type=int, default=DECIMATE, help="source pixels per side folded into one cell before reprojecting; 1 disables")
    a = ap.parse_args(argv)

    sources = index_sources(a.chm_dir)
    bbox = tuple(a.bbox) if a.bbox else None
    parents = a.parents if a.parents else parents_intersecting(sources.values(), bbox)
    if a.limit:
        parents = parents[: a.limit]
    print(f"{len(sources)} source tiles, {len(parents)} parent tiles, {a.workers} workers, decimate {a.decimate} -> {a.out}", flush=True)

    jobs = [(i, str(a.out), a.block) for i in parents]
    done = 0
    t_start = time.time()
    if a.workers <= 1:
        _init_worker(sources, a.decimate)
        results = map(process_parent, jobs)
    else:
        pool = mp.Pool(a.workers, initializer=_init_worker, initargs=(sources, a.decimate))
        results = pool.imap_unordered(process_parent, jobs)
    for index, dt, written in results:
        done += 1
        elapsed = time.time() - t_start
        eta = elapsed / done * (len(jobs) - done)
        print(f"[{done}/{len(jobs)}] parent {index:06d} {written} files {dt:6.1f}s  elapsed {elapsed/3600:.2f}h  eta {eta/3600:.2f}h",
              flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
