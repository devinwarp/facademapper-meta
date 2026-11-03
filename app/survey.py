"""Fixture survey (UV map): where every real light fixture sits, in canvas space.

A survey holds several elevations side by side. We read each fixture, split the canvas into elevations at the
gaps, and map the elevation that matches the tower in the photo onto the tower's corners (a homography)."""
import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # surveys are ~128 MP masks


@dataclass
class Survey:
    cx: np.ndarray  # fixture centres, canvas px
    cy: np.ndarray
    w: float        # typical fixture size, canvas px
    h: float
    canvas: tuple[int, int]  # (width, height)


def read_uv_mask_bool(im: Image.Image) -> np.ndarray:
    """Decode any standard media-server UV mask format into a boolean fixture mask:
    1. White/coloured fixture pixels on black (`max(R,G,B) > 127`)
    2. Transparent stencil cutouts punched in a black mask (`alpha < 128` when >70% is opaque black)
    3. Opaque fixture pixels on a transparent sheet (`alpha > 127`).
    """
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        rgba = np.asarray(im.convert("RGBA"))
        rgb_hi = rgba[:, :, :3].max(axis=2) > 127
        if rgb_hi.mean() > 0.0004:
            return rgb_hi
        alpha = rgba[:, :, 3]
        if (alpha < 128).mean() > 0.0004 and (alpha > 200).mean() > 0.70:
            return alpha < 128
        return alpha > 127
    return np.asarray(im.convert("L")) > 127


def load_survey(mask_path: str, cache: str | None = None) -> Survey:
    """Connected fixture blobs in the UV mask = fixtures. Cached as .npz (parsing 128 MP masks is slow)."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return Survey(z["cx"], z["cy"], float(z["w"]), float(z["h"]), tuple(z["canvas"]))
    from scipy import ndimage as ndi
    a = read_uv_mask_bool(Image.open(mask_path))
    lab, n = ndi.label(a)
    sl = ndi.find_objects(lab)
    cy = np.array([(s[0].start + s[0].stop) / 2 for s in sl]); cx = np.array([(s[1].start + s[1].stop) / 2 for s in sl])
    hs = np.array([s[0].stop - s[0].start for s in sl]); ws = np.array([s[1].stop - s[1].start for s in sl])
    sv = Survey(cx, cy, float(np.median(ws)), float(np.median(hs)), (a.shape[1], a.shape[0]))
    if cache:
        np.savez(cache, cx=cx, cy=cy, w=sv.w, h=sv.h, canvas=np.array(sv.canvas))
    return sv


def elevations(sv: Survey, gap: float | None = None, min_height: float = 0.4) -> list[tuple[float, float, float, float]]:
    """Split fixtures into elevations at x-gaps wider than `gap`; keep the tall ones (towers, not podium bands).
    Returns canvas rects (x0, y0, x1, y1), left to right."""
    use_gap = float(gap) if gap is not None else max(16.0, 0.013 * float(sv.canvas[0]))
    order = np.argsort(sv.cx)
    x = sv.cx[order]
    cut = np.where(np.diff(x) > use_gap)[0]
    groups = np.split(order, cut + 1)
    out = []
    for g in groups:
        x0, x1, y0, y1 = sv.cx[g].min(), sv.cx[g].max(), sv.cy[g].min(), sv.cy[g].max()
        if (y1 - y0) >= min_height * sv.canvas[1]:
            out.append((x0 - sv.w, y0 - sv.h, x1 + sv.w, y1 + sv.h))
    return out


def pick_elevation(rects, aspect: float) -> int:
    """Index of the elevation whose width/height best matches the tower's width/height in the photo.
    ponytail: aspect heuristic; a user-pinned elevation (the old app's `visible` flag) should override it."""
    return int(np.argmin([abs(np.log(((r[2] - r[0]) / (r[3] - r[1])) / aspect)) for r in rects]))


def quad_from_mask(mask: np.ndarray, band: float = 0.04) -> np.ndarray:
    """Corners (TL, TR, BR, BL) of a tower silhouette: its left/right extremes in the top and bottom bands."""
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max()
    b = max(2, int((y1 - y0) * band))
    top, bot = ys <= y0 + b, ys >= y1 - b
    return np.array([[xs[top].min(), y0], [xs[top].max(), y0], [xs[bot].max(), y1], [xs[bot].min(), y1]], float)


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 matrix mapping 4 src points to 4 dst points (direct linear transform)."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y, -u], [0, 0, 0, x, y, 1, -v * x, -v * y, -v]]
    return np.linalg.svd(np.array(A))[2][-1].reshape(3, 3)


def apply(H: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = H @ np.stack([x, y, np.ones_like(x)])
    return p[0] / p[2], p[1] / p[2]


def map_elevation(sv: Survey, rect, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Photo positions of every fixture inside the canvas rect, by mapping the rect onto the photo quad."""
    x0, y0, x1, y1 = rect
    inside = (sv.cx >= x0) & (sv.cx <= x1) & (sv.cy >= y0) & (sv.cy <= y1)
    H = homography(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float), quad)
    return apply(H, sv.cx[inside], sv.cy[inside])
