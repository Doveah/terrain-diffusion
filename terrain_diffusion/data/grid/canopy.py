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
(boundless, so area outside a tile's footprint is explicitly invalid), and GDAL's average resampling reprojects height,
the two indicator rasters and a validity raster onto the block. Several sources meeting inside one block are combined by
validity-weighted average, so tile seams are exact. No source mosaic or VRT is needed.

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

VERSION = "canopy-reducer-1"
BANDS = ("height", "cover2", "cover5", "valid")
THRESHOLDS = (2.0, 5.0)
SENTINEL = -9999.0
HALO_PX = 4


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


def _init_worker(sources: Dict[str, Bounds]) -> None:
    global _SOURCES
    _SOURCES = sources
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

def _read_source_block(ds: rasterio.DatasetReader, bounds: Bounds) -> Tuple[np.ndarray, np.ndarray, object]:
    """Read the part of ``ds`` covering lon/lat ``bounds`` plus a halo. Returns (height, valid_mask, transform)."""
    left, bottom, right, top = transform_bounds(CRS, ds.crs, *bounds, densify_pts=21)
    win = rasterio.windows.from_bounds(left, bottom, right, top, ds.transform)
    col0 = math.floor(win.col_off) - HALO_PX
    row0 = math.floor(win.row_off) - HALO_PX
    col1 = math.ceil(win.col_off + win.width) + HALO_PX
    row1 = math.ceil(win.row_off + win.height) + HALO_PX
    w = Window(col0, row0, col1 - col0, row1 - row0)
    arr = ds.read(1, window=w, boundless=True, masked=True, fill_value=ds.nodata)
    valid = ~np.ma.getmaskarray(arr)
    data = np.asarray(arr.filled(0), dtype=np.float32)
    valid &= np.isfinite(data)
    # Explicit footprint: anything outside the file's own extent is invalid even when the file declares no nodata,
    # otherwise the boundless fill (0 = ground) would count as valid canopy at tile seams.
    rows = np.arange(row0, row1)[:, None]
    cols = np.arange(col0, col1)[None, :]
    valid &= (rows >= 0) & (rows < ds.height) & (cols >= 0) & (cols < ds.width)
    return data, valid, window_transform(w, ds.transform)


def _average_onto(src: np.ndarray, src_transform, src_crs, dst_shape, dst_transform,
                  nodata: Optional[float]) -> np.ndarray:
    dst = np.full(dst_shape, SENTINEL if nodata is not None else 0.0, dtype=np.float32)
    reproject(
        source=src,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=nodata,
        dst_transform=dst_transform,
        dst_crs=CRS,
        dst_nodata=SENTINEL if nodata is not None else 0.0,
        resampling=Resampling.average,
        num_threads=1,
    )
    return dst


def reduce_block(bounds: Bounds, shape: Tuple[int, int], dst_transform, sources: Sequence[str]) -> np.ndarray:
    """Four-band float32 block (height, cover2, cover5, valid) for one destination block."""
    acc = np.zeros((len(BANDS), *shape), dtype=np.float64)  # weighted sums; last band is the weight sum
    for path in sources:
        ds = _open(path)
        height, valid, tr = _read_source_block(ds, bounds)
        if not valid.any():
            continue
        h = np.where(valid, height, SENTINEL).astype(np.float32)
        ind = [np.where(valid, (height > t).astype(np.float32), SENTINEL).astype(np.float32) for t in THRESHOLDS]
        w = _average_onto(valid.astype(np.float32), tr, ds.crs, shape, dst_transform, nodata=None)
        hm = _average_onto(h, tr, ds.crs, shape, dst_transform, nodata=SENTINEL)
        cm = [_average_onto(a, tr, ds.crs, shape, dst_transform, nodata=SENTINEL) for a in ind]
        w = np.clip(w, 0.0, 1.0)
        ok = (w > 0) & (hm != SENTINEL)
        acc[0][ok] += hm[ok] * w[ok]
        for k, c in enumerate(cm):
            acc[1 + k][ok] += c[ok] * w[ok]
        acc[3][ok] += w[ok]
    out = np.full((len(BANDS), *shape), np.nan, dtype=np.float32)
    wsum = acc[3]
    covered = wsum > 0
    for b in range(3):
        out[b][covered] = (acc[b][covered] / wsum[covered]).astype(np.float32)
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
            data[:, r0:r0 + h, c0:c0 + w] = reduce_block(bb, (h, w), btr, srcs)
    return _write(tile, data, out_dir, dict(n_sources=len(cand)))


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
    return _write(parent, data, out_dir, dict(n_sources=0, children=len(kids)))


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
    a = ap.parse_args(argv)

    sources = index_sources(a.chm_dir)
    bbox = tuple(a.bbox) if a.bbox else None
    parents = a.parents if a.parents else parents_intersecting(sources.values(), bbox)
    if a.limit:
        parents = parents[: a.limit]
    print(f"{len(sources)} source tiles, {len(parents)} parent tiles, {a.workers} workers -> {a.out}", flush=True)

    jobs = [(i, str(a.out), a.block) for i in parents]
    done = 0
    t_start = time.time()
    if a.workers <= 1:
        _init_worker(sources)
        results = map(process_parent, jobs)
    else:
        pool = mp.Pool(a.workers, initializer=_init_worker, initargs=(sources,))
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
