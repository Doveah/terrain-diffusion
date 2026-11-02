"""Nested tile grid shared by every P1 product (see PIPELINE.md, P1).

Parent grid: the author's equal-area cells at 2048 px x 30 m (about 61 km). The cell maths is copied verbatim from
``data/downloading/world_grid.py`` so that this module imports no plotting libraries and can run in the data-prep
container. Parent ids are the zero-based index into that list, formatted as six digits.

Child grid: each parent is split 3 x 3 into 10 m children of 2048 px (about 20 km). Child ``r`` counts rows from the
north, ``c`` columns from the west, so child (0, 0) is the north-west corner. Upsampling a parent's 2048 px field by 3
gives a 6144 px grid, and child (r, c) is exactly rows ``2048 r .. 2048 (r + 1)`` and columns ``2048 c .. 2048 (c + 1)``
of that grid. The 30 m product of a parent is the 3 x 3 block mean of the mosaic of its nine children.

All tiles are in EPSG:4326 with a north-up affine transform. Pixels are not square in degrees; they are close to square
in metres because the parent longitude span is stretched by 1 / cos(latitude).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rasterio.transform import Affine, from_bounds

Bounds = Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)

PARENT_PX = 2048
PARENT_RES_M = 30
CHILD_SPLIT = 3
CHILD_PX = 2048
CHILD_RES_M = PARENT_RES_M // CHILD_SPLIT
CRS = "EPSG:4326"


def equal_area_cells(target_size: Tuple[float, float]) -> List[Bounds]:
    """Verbatim copy of ``world_grid.create_equal_area_grid`` (same floats, same cells)."""
    EARTH_RADIUS = 6378000
    MIN_LAT, MAX_LAT = -60, 60

    base_lon_spacing = np.degrees(target_size[0] / (EARTH_RADIUS * np.cos(np.radians(0))))
    base_lat_spacing = np.degrees(target_size[1] / EARTH_RADIUS)

    grid_cells = []
    current_lat = MIN_LAT
    while current_lat + base_lat_spacing < MAX_LAT:
        next_lat = current_lat + base_lat_spacing
        cos_lat = np.cos(np.radians(current_lat + (next_lat - current_lat) / 2))
        adjusted_lon_spacing = base_lon_spacing / cos_lat
        current_lon = -180
        while current_lon < 180:
            next_lon = current_lon + adjusted_lon_spacing
            if next_lon > 180:
                break
            grid_cells.append((current_lon, current_lat, next_lon, next_lat))
            current_lon += adjusted_lon_spacing
        current_lat = next_lat
    return grid_cells


@lru_cache(maxsize=1)
def parent_cells() -> List[Bounds]:
    return equal_area_cells((PARENT_PX * PARENT_RES_M, PARENT_PX * PARENT_RES_M))


def parent_id(index: int) -> str:
    return f"{index:06d}"


def child_id(index: int, r: int, c: int) -> str:
    return f"{parent_id(index)}_{r}{c}"


def intersects(a: Bounds, b: Bounds) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def child_bounds(parent: Bounds, r: int, c: int) -> Bounds:
    min_lon, min_lat, max_lon, max_lat = parent
    dlon = (max_lon - min_lon) / CHILD_SPLIT
    dlat = (max_lat - min_lat) / CHILD_SPLIT
    # row 0 is the northern strip
    top = max_lat - r * dlat
    return (min_lon + c * dlon, top - dlat, min_lon + (c + 1) * dlon, top)


@dataclass(frozen=True)
class Tile:
    id: str
    level: str  # "30m" or "10m"
    bounds: Bounds
    size: int
    parent: str

    @property
    def transform(self) -> Affine:
        return from_bounds(*self.bounds, self.size, self.size)

    @property
    def path(self) -> str:
        """Relative output directory: ``30m/<parent>`` or ``10m/<parent>/<child>``."""
        if self.level == "30m":
            return f"30m/{self.id}"
        return f"10m/{self.parent}/{self.id}"


def parent_tile(index: int) -> Tile:
    return Tile(parent_id(index), "30m", parent_cells()[index], PARENT_PX, parent_id(index))


def child_tiles(index: int) -> List[Tile]:
    cell = parent_cells()[index]
    return [
        Tile(child_id(index, r, c), "10m", child_bounds(cell, r, c), CHILD_PX, parent_id(index))
        for r in range(CHILD_SPLIT)
        for c in range(CHILD_SPLIT)
    ]


def parents_intersecting(bounds_list: Iterable[Bounds], bbox: Optional[Bounds] = None) -> List[int]:
    """Indices of parent cells that touch any of ``bounds_list`` (and ``bbox`` when given)."""
    cells = parent_cells()
    boxes = list(bounds_list)
    out = []
    for i, cell in enumerate(cells):
        if bbox is not None and not intersects(cell, bbox):
            continue
        if any(intersects(cell, b) for b in boxes):
            out.append(i)
    return out


def block_windows(size: int, block: int) -> Sequence[Tuple[int, int, int, int]]:
    """(row_off, col_off, height, width) blocks covering a size x size tile."""
    out = []
    for r in range(0, size, block):
        for c in range(0, size, block):
            out.append((r, c, min(block, size - r), min(block, size - c)))
    return out
