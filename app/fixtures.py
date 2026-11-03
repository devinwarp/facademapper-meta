"""Architectural fixture placement & discrete LED optical rendering.

Extracts real structural lines (continuous vertical/helical mullions & fins via ridge-snapped
streamlines, floor-slab spandrels, crown parapets, setback ledges, and existing lit night
fixtures) inside SAM 3.1 element masks so light is emitted strictly from discrete physical
linear fixtures — never as a plain surface wash over the building.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

from app.mapper import Element


@dataclass
class FixtureRigOption:
    id: str
    name: str
    subtitle: str
    recommended: bool
    fixture_type: str
    fixture_count: int
    pitch_desc: str
    est_power_kw: float
    zones_used: list[str]
    xs: np.ndarray          # float32 (N,) in render px
    ys: np.ndarray          # float32 (N,) in render px
    lengths: np.ndarray     # float32 (N,) in render px
    widths: np.ndarray      # float32 (N,) in render px
    angles: np.ndarray      # float32 (N,) radians (direction of linear bar)
    elem_ids: np.ndarray    # int32 (N,) primary element id
    elem_membership: dict[int, np.ndarray] = field(default_factory=dict)
    render_size: tuple[int, int] = (0, 0)  # (W, H)
    u_local: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))   # (N,) 0..1 across visible side(s) w/ 3D wrap
    v_elev: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))    # (N,) 0..1 top-to-base elevation
    view_cos: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))  # (N,) 3D grazing falloff (0.58..1.0)
    default_wrap: float = 0.45      # fraction of 360-deg building perimeter visible in photo (1.0 for stitched 4-sides)
    default_u_offset: float = 0.25  # azimuth offset on 360-deg canvas (0.0 for stitched 4-sides)
    spine_u: float = 0.50           # normalized [0..1] azimuth of densest structural spine
    crop_box: tuple[int, int, int, int] | None = None                 # tight architectural fixture crop (x0, y0, x1, y1)
    panel_spans: list[tuple[int, int]] = field(default_factory=list)  # per-side (x0, x1) spans when 4 elevations are stitched
    elev_y_span: tuple[int, int] | None = None                        # aligned (y_crown, y_base) when stitched

    def summary_dict(self, preview_b64: str | None = None) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "subtitle": self.subtitle,
            "recommended": self.recommended,
            "fixture_type": self.fixture_type,
            "fixture_count": int(self.fixture_count),
            "pitch_desc": self.pitch_desc,
            "est_power_kw": round(float(self.est_power_kw), 1),
            "zones_used": self.zones_used,
        }
        if preview_b64 is not None:
            d["preview_jpg_b64"] = preview_b64
        return d


def compute_render_size(img: Image.Image, els: list[Element], target_h: float = 1680.0) -> tuple[int, int, float]:
    """Choose an even (W, H) render canvas so fine vertical fins (38-44 columns) resolve with crisp dark gaps."""
    solids = [e for e in els if e.mask is not None]
    if solids:
        ys_list = [np.where(np.asarray(e.mask) > 127)[0] for e in solids]
        ys_all = np.concatenate([y for y in ys_list if len(y)])
        span = float(np.ptp(ys_all)) if len(ys_all) > 1 else float(img.height)
    else:
        span = float(img.height)
    max_src = max(img.width, img.height)
    eff_target = 1440.0 if max_src <= 700 else target_h
    max_canvas = 1800.0 if max_src <= 700 else 2240.0
    scale = float(np.clip(eff_target / max(span, 160.0), 1.0, 2.5 if max_src <= 700 else 2.6))
    if max(img.width * scale, img.height * scale) > max_canvas:
        scale = max_canvas / max(img.width, img.height)
    W = max(2, int(round(img.width * scale)) // 2 * 2)
    H = max(2, int(round(img.height * scale)) // 2 * 2)
    return W, H, W / img.width


def prepare_night_base(rgb: np.ndarray, building_mask: np.ndarray) -> tuple[np.ndarray, bool]:
    """Photorealistic architectural night/blue-hour grading that preserves real dusk skies, warm golden
    interior office windows, and foreground streetlamps while cleanly suppressing pre-existing exterior LEDs."""
    H, W = rgb.shape[:2]
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]

    sky_probe = (~building_mask) & (yy < 0.42)
    sky_med = float(np.median(luma[sky_probe])) if sky_probe.sum() > 100 else float(np.median(luma))

    def _clean_existing_leds(src: np.ndarray) -> np.ndarray:
        cleaned = src.copy()
        local_bg = ndi.grey_opening(luma, size=(7, 9))
        narrow_spike = luma - local_bg
        is_warm_interior = (src[:, :, 0] > src[:, :, 2] * 1.28) & (src[:, :, 1] > src[:, :, 2] * 1.08) & (local_bg > 85.0)
        led_dot = building_mask & (narrow_spike > 18.0) & (~is_warm_interior)
        if led_dot.any():
            dil = ndi.binary_dilation(led_dot, structure=np.ones((3, 5), bool)) & building_mask
            med_rgb = np.stack([ndi.grey_opening(src[:, :, c], size=(7, 9)) for c in range(3)], axis=2)
            cleaned[dil] = med_rgb[dil] * 0.76
        return cleaned

    # 1. True Night Photo (dark night sky < 78 luma)
    if sky_med < 78.0:
        out = _clean_existing_leds(rgb)
        out[building_mask] *= 0.80
        return np.clip(out, 0, 255).astype(np.float32), True

    # 2. Dusk / Blue-Hour / Twilight Photo (78 <= sky_med < 142, e.g. sunset/twilight skyline with warm lamps & windows)
    if sky_med < 142.0:
        cleaned = _clean_existing_leds(rgb)
        norm = np.clip(cleaned / 255.0, 0.0, 1.0)
        # Deep nocturnal indigo-obsidian grade so subtle pearlescent & indigo light-art reads with Museum-of-Light contrast
        graded = (norm ** 1.62) * 255.0 * np.array([0.34, 0.39, 0.56], dtype=np.float32)
        warm_glow = np.clip((cleaned[:, :, 0] - cleaned[:, :, 2] * 0.95) / 105.0, 0.0, 1.0) * np.clip((luma - 140.0) / 110.0, 0.0, 1.0)
        out = graded * (1.0 - 0.42 * warm_glow[:, :, None]) + cleaned * (0.42 * warm_glow[:, :, None])
        bld_soft = ndi.gaussian_filter(building_mask.astype(np.float32), sigma=1.2)[:, :, None]
        out = out * (1.0 - 0.32 * bld_soft)
        return np.clip(out, 0, 255).astype(np.float32), True

    # 3. Bright Daytime Photo -> Cinematic Blue-Hour Architectural Twilight
    blue_ratio = rgb[:, :, 2] / (rgb[:, :, 0] + 12.0)
    raw_sky = (~building_mask) & (luma > 95.0) & (blue_ratio > 0.84) & (yy < 0.90)
    sky_alpha = ndi.gaussian_filter(raw_sky.astype(np.float32), sigma=2.5)
    sky_alpha[building_mask] = 0.0

    sky_top = np.array([11.0, 18.0, 38.0], dtype=np.float32)
    sky_hor = np.array([28.0, 39.0, 66.0], dtype=np.float32)
    cloud_tex = np.clip((luma - 145.0) / 115.0, -0.15, 0.28)[:, :, None] * np.array([5.0, 7.0, 13.0], dtype=np.float32)
    dusk_sky = sky_top[None, None, :] * (1.0 - yy[:, :, None]) + sky_hor[None, None, :] * yy[:, :, None] + cloud_tex

    norm = np.clip(rgb / 255.0, 0.0, 1.0)
    context_night = (norm ** 1.55) * 255.0 * np.array([0.24, 0.28, 0.38], dtype=np.float32)
    bg = dusk_sky * sky_alpha[:, :, None] + context_night * (1.0 - sky_alpha[:, :, None])

    low = ndi.gaussian_filter(luma, sigma=12.0)
    mullion_spec = np.clip((luma - low) / 38.0, -0.25, 0.85)[:, :, None]
    facade_twilight = (
        (norm ** 1.72) * 255.0 * np.array([0.24, 0.29, 0.42], dtype=np.float32)
        + mullion_spec * np.array([14.0, 18.0, 26.0], dtype=np.float32)
    )
    bld_alpha = ndi.gaussian_filter(building_mask.astype(np.float32), sigma=0.9)[:, :, None]
    out = bg * (1.0 - bld_alpha) + np.clip(facade_twilight, 4.0, 85.0) * bld_alpha
    return np.clip(out, 0, 255).astype(np.float32), False


def _estimate_pitch_1d(profile: np.ndarray, lo: int, hi: int) -> int:
    """Estimate fundamental spatial period in `[lo, hi]`, locking onto the true first harmonic `T`
    rather than `2*T` or `3*T` on dense multi-fin skyscrapers."""
    lo, hi = max(4, int(lo)), max(int(lo) + 2, int(hi))
    if len(profile) <= hi + 4:
        return (lo + hi) // 2
    p = profile.astype(np.float64) - ndi.gaussian_filter1d(profile.astype(np.float64), sigma=max(2.5, hi * 0.45))
    n = len(p)
    ac = np.correlate(p, p, mode="full")[n - 1:]
    end = min(hi + 1, len(ac))
    window = ac[lo:end]
    if len(window) == 0 or np.all(window <= 0) or ac[0] <= 1e-9:
        return (lo + hi) // 2

    nac = ac[:end] / float(ac[0])
    k_max = int(lo + np.argmax(window))
    best_val = float(nac[k_max])
    if best_val <= 0.04:
        return k_max

    # Check if an earlier local autocorrelation peak `k` is the fundamental period `T` (with `k_max ~ m * T` or strong prominence)
    for k in range(lo, k_max):
        if k <= 1 or k + 1 >= len(nac):
            continue
        if nac[k] >= nac[k - 1] and nac[k] >= nac[k + 1] and nac[k] >= max(0.08, 0.58 * best_val):
            ratio = k_max / float(k)
            if abs(ratio - round(ratio)) <= 0.20 or nac[k] >= 0.78 * best_val:
                return int(k)
    return k_max


def _nms_1d(signal: np.ndarray, min_dist: int, thresh: float) -> np.ndarray:
    min_dist = max(2, int(min_dist))
    mx = ndi.maximum_filter1d(signal, size=2 * min_dist + 1)
    cand = np.where((signal >= mx - 1e-6) & (signal >= thresh))[0]
    if len(cand) <= 1:
        return cand.astype(np.int32)
    order = cand[np.argsort(signal[cand])[::-1]]
    picked: list[int] = []
    for idx in order:
        if all(abs(int(idx) - p) >= min_dist for p in picked):
            picked.append(int(idx))
    return np.array(sorted(picked), dtype=np.int32)


def _assign_membership(
    xs: np.ndarray,
    ys: np.ndarray,
    masks: dict[int, np.ndarray],
    solids_order: list[int],
    H: int,
    W: int,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    yi = np.clip(np.round(ys).astype(int), 0, H - 1)
    xi = np.clip(np.round(xs).astype(int), 0, W - 1)
    elem_ids = np.zeros(len(xs), dtype=np.int32)
    membership: dict[int, np.ndarray] = {}
    areas = sorted(masks.keys(), key=lambda k: masks[k].sum(), reverse=True)
    for eid in areas:
        hit = masks[eid][yi, xi]
        idx = np.where(hit)[0].astype(np.int32)
        if len(idx):
            membership[eid] = idx
            elem_ids[hit] = eid
    unassigned = np.where(elem_ids == 0)[0]
    if len(unassigned) and solids_order:
        # Assign to element whose vertical box contains y
        for u in unassigned:
            yu, xu = yi[u], xi[u]
            best_eid = solids_order[0]
            best_d = 1e9
            for eid in solids_order:
                eys, exs = np.where(masks[eid])
                if len(eys):
                    cy_m, cx_m = eys.mean(), exs.mean()
                    d = abs(yu - cy_m) + 0.5 * abs(xu - cx_m)
                    if d < best_d:
                        best_d, best_eid = d, eid
            elem_ids[u] = best_eid
        for eid in solids_order:
            extra = np.where(elem_ids == eid)[0].astype(np.int32)
            if len(extra):
                prev = membership.get(eid, np.empty(0, np.int32))
                membership[eid] = np.unique(np.concatenate([prev, extra]))
    return elem_ids, membership


def _taper_vertical_bars(
    xs: np.ndarray,
    ys: np.ndarray,
    active_surface: np.ndarray,
    nominal_len: float,
    taper_zone: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Taper vertical LED bar lengths smoothly over ~1.4 floor tiers as fins approach slanted crown roofs
    or diagonal SAM 3.1 void canyons (matching CAD UV maps where edge modules step down organically)."""
    if len(xs) == 0:
        e = np.empty(0, np.float32)
        return e, e, e, np.empty(0, bool)
    H, W = active_surface.shape
    zone_px = max(4, int(np.ceil(taper_zone if taper_zone is not None else nominal_len * 1.8)))
    min_half_hard = max(2, int(np.ceil(nominal_len * 0.55)))
    out_x = xs.copy()
    out_y = ys.copy()
    out_l = np.full(len(xs), nominal_len, dtype=np.float32)
    keep = np.ones(len(xs), dtype=bool)

    for i in range(len(xs)):
        ix = int(np.clip(round(float(xs[i])), 0, W - 1))
        iy = int(np.clip(round(float(ys[i])), 0, H - 1))
        if not active_surface[iy, ix]:
            keep[i] = False
            continue
        up = 0
        while up < zone_px and iy - up - 1 >= 0 and active_surface[iy - up - 1, ix]:
            up += 1
        dn = 0
        while dn < zone_px and iy + dn + 1 < H and active_surface[iy + dn + 1, ix]:
            dn += 1
        span = float(up + dn + 1)
        if span < max(3.0, nominal_len * 0.36):
            keep[i] = False
            continue
        min_clear = min(up, dn)
        if min_clear < zone_px:
            if up < min_half_hard or dn < min_half_hard:
                shift = np.clip(0.32 * float(dn - up), -1.8, 1.8)
                out_y[i] = float(ys[i]) + shift
            u_t = float(min_clear) / max(float(zone_px), 1.0)
            taper = 0.42 + 0.58 * (u_t * u_t * (3.0 - 2.0 * u_t))
            hard_cap = float(2 * min_clear + 1) * 0.90 if min_clear < min_half_hard else nominal_len
            out_l[i] = float(np.clip(min(nominal_len * taper, hard_cap), nominal_len * 0.38, nominal_len))
    return out_x[keep], out_y[keep], out_l[keep], keep


def _trace_mullion_tracks(
    mullion_score: np.ndarray,
    theta_vert: np.ndarray,
    inner_mask: np.ndarray,
    active_surface: np.ndarray,
    tier_ys: list[int],
    min_dx: int,
    dense_spine_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace vertical / helical mullion streamlines along the structure tensor field,
    respecting SAM 3.1 negative-space cutouts while preserving foreshortened perimeter fins,
    and seat crisp upright linear LED modules along horizontal floor tiers."""
    H, W = mullion_score.shape
    cot_theta = np.cos(theta_vert) / np.maximum(np.sin(theta_vert), 0.35)  # dx per +1 dy
    tier_set = np.zeros(H, bool)
    for y in tier_ys:
        if 0 <= y < H:
            tier_set[y] = True

    occupied = np.zeros((H, W), bool)
    half_excl_std = max(3, int(round(min_dx * 0.66)))
    half_excl_dense = max(3, int(round(min_dx * 0.54)))

    ys_valid = np.where(inner_mask.any(axis=1))[0]
    if len(ys_valid) < 10:
        return np.empty(0, np.float32), np.empty(0, np.float32), np.empty(0, np.float32)
    y0_all, y1_all = int(ys_valid.min()), int(ys_valid.max())
    seed_step = max(56, min(90, (y1_all - y0_all) // 10))
    seed_rows = list(range(y0_all + 16, y1_all - 12, seed_step))
    if not seed_rows:
        seed_rows = [(y0_all + y1_all) // 2]

    out_x, out_y, out_ang = [], [], []
    min_track_len = max(12, min(18, (y1_all - y0_all) // 18))

    for sy in seed_rows:
        row_valid = inner_mask[sy] & active_surface[sy]
        if not row_valid.any():
            continue
        sig = mullion_score[sy].copy()
        sig[~row_valid | occupied[sy]] = 0.0
        step_dx = max(4, int(round(min_dx * 0.76))) if (dense_spine_mask is not None and dense_spine_mask[sy].any()) else max(4, int(round(min_dx * 0.86)))
        seeds = list(_nms_1d(sig, min_dist=step_dx, thresh=0.34))
        # Fill gaps only inside active fin/mullion intervals (never across SAM 3.1 recessed glass canyons!)
        runs_lab, n_runs = ndi.label(row_valid & (~occupied[sy]))
        interp_seeds: list[int] = []
        for r_id in range(1, n_runs + 1):
            r_xs = np.where(runs_lab == r_id)[0]
            if len(r_xs) < max(4, int(min_dx * 0.8)):
                continue
            in_run_seeds = [int(s) for s in seeds if r_xs[0] <= s <= r_xs[-1]]
            anchors_x = sorted(set([int(r_xs[0]) + 2] + in_run_seeds + [int(r_xs[-1]) - 2]))
            for ax, bx in zip(anchors_x, anchors_x[1:]):
                gap_x = bx - ax
                loc_dx = int(round(min_dx * 0.80)) if (dense_spine_mask is not None and dense_spine_mask[sy, (ax + bx) // 2]) else min_dx
                if gap_x >= int(loc_dx * 1.65):
                    n_fill = max(2, int(round(gap_x / max(loc_dx, 4))))
                    for kf in range(1, n_fill):
                        cand = int(round(ax + kf * gap_x / n_fill))
                        if row_valid[cand] and not occupied[sy, cand]:
                            interp_seeds.append(cand)
        seeds = sorted(set(int(s) for s in seeds) | set(interp_seeds), key=lambda c: float(sig[c]), reverse=True)

        for sx in seeds:
            if occupied[sy, sx]:
                continue
            pts_y = [sy]
            pts_x = [float(sx)]

            for step_y in (-1, 1):
                cx = float(sx)
                cy = sy + step_y
                gap_run = 0
                while 3 <= cy < H - 3:
                    xi_est = int(round(cx + step_y * cot_theta[cy, int(np.clip(round(cx), 0, W - 1))]))
                    if xi_est < 3 or xi_est >= W - 3 or not inner_mask[cy, xi_est]:
                        gap_run += 1
                        if gap_run > 6:
                            break
                        cx = float(np.clip(xi_est, 3, W - 4))
                        cy += step_y
                        continue
                    gap_run = 0
                    w0, w1 = max(3, xi_est - 1), min(W - 3, xi_est + 2)
                    local_r = mullion_score[cy, w0:w1]
                    best_x = w0 + int(np.argmax(local_r))
                    cx = 0.70 * float(xi_est) + 0.30 * (float(best_x) if local_r.max() > 0.68 else float(xi_est))
                    ix = int(round(cx))
                    if occupied[cy, ix]:
                        break
                    if step_y == -1:
                        pts_y.insert(0, cy)
                        pts_x.insert(0, cx)
                    else:
                        pts_y.append(cy)
                        pts_x.append(cx)
                    cy += step_y

            if len(pts_y) < min_track_len:
                continue

            arr_y = np.array(pts_y, dtype=np.int32)
            arr_x = ndi.gaussian_filter1d(np.array(pts_x, dtype=np.float32), sigma=3.2)
            for py_i, px_f in zip(arr_y, arr_x):
                ix = int(np.clip(round(px_f), 0, W - 1))
                he = half_excl_dense if (dense_spine_mask is not None and dense_spine_mask[py_i, ix]) else half_excl_std
                occupied[py_i, max(0, ix - he):min(W, ix + he + 1)] = True

            # Seat crisp upright vertical linear LED bars along each floor tier strictly inside active fin/mullion surface
            for py_i, px_f in zip(arr_y, arr_x):
                ix = int(np.clip(round(px_f), 0, W - 1))
                if tier_set[py_i] and inner_mask[py_i, ix] and active_surface[py_i, ix]:
                    out_x.append(float(px_f))
                    out_y.append(float(py_i))
                    out_ang.append(float(np.pi / 2.0))

    return np.array(out_x, np.float32), np.array(out_y, np.float32), np.array(out_ang, np.float32)


def _extract_exterior_led_blobs(
    rgb: np.ndarray,
    active_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Detect crisp glowing exterior linear LED dashes on dusk/night facades while rejecting
    wide warm-yellow interior office windows (`R > B * 1.25` with broad ambient fill)."""
    H, W = rgb.shape[:2]
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    sky_probe = (~active_mask) & (np.arange(H)[:, None] < 0.42 * H)
    sky_med = float(np.median(luma[sky_probe])) if sky_probe.sum() > 100 else float(np.median(luma))
    empty = np.empty(0, np.float32)
    if sky_med >= 142.0 or not active_mask.any():
        return empty, empty, empty, empty, empty, sky_med

    open_h = max(7, int(round(H / 160.0)) | 1)
    open_w = max(7, int(round(W / 110.0)) | 1)
    local_bg = ndi.grey_opening(luma, size=(open_h, open_w))
    tophat = luma - local_bg
    max_c = rgb.max(axis=2)
    min_c = rgb.min(axis=2)
    sat = (max_c - min_c) / (max_c + 1e-3)
    broad = ndi.uniform_filter(luma, size=15)
    # Reject warm golden interior office windows (broad amber glow behind smooth glass)
    warm_office = (rgb[:, :, 0] > rgb[:, :, 2] * 1.24) & (rgb[:, :, 1] > rgb[:, :, 2] * 1.05) & (local_bg > 78.0)
    thresh_th = 24.0 if sky_med >= 90.0 else 30.0
    emit = active_mask & (~warm_office) & (tophat > thresh_th) & (luma > 92.0) & ((tophat > 42.0) | (sat > 0.20) | (broad < 120.0))
    lab, n = ndi.label(emit)
    if n < 30:
        return empty, empty, empty, empty, empty, sky_med

    max_bh = max(28, int(round(H / 52.0)))
    max_bw = max(18, int(round(W / 65.0)))
    objs = ndi.find_objects(lab)
    xs, ys, ls, ws, angs = [], [], [], [], []
    for k, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        sy, sx = sl
        bh, bw = sy.stop - sy.start, sx.stop - sx.start
        if bh > max_bh or bw > max_bw or (bh * bw) < 2 or (bh > 16 and bw > 12):
            continue
        sub = lab[sy, sx] == k
        area = int(sub.sum())
        if area < 2 or area > 130:
            continue
        py, px = np.where(sub)
        cy = float(sy.start + py.mean())
        cx = float(sx.start + px.mean())
        if len(px) >= 3:
            cov_xx = float(np.var(px)) + 0.25
            cov_yy = float(np.var(py)) + 0.25
            cov_xy = float(np.mean((px - px.mean()) * (py - py.mean())))
            ang = 0.5 * np.arctan2(2 * cov_xy, cov_xx - cov_yy)
        else:
            ang = np.pi / 2 if bh >= bw else 0.0
        length = float(np.clip(max(bh, bw) * 1.0, 5.0, 14.0))
        width = float(np.clip(min(bh, bw) * 0.45, 1.4, 2.1))
        xs.append(cx); ys.append(cy); ls.append(length); ws.append(width); angs.append(float(ang))

    return (
        np.array(xs, np.float32),
        np.array(ys, np.float32),
        np.array(ls, np.float32),
        np.array(ws, np.float32),
        np.array(angs, np.float32),
        sky_med,
    )


def _measure_empirical_led_pitch(
    xs: np.ndarray,
    ys: np.ndarray,
    med_w: float,
    b_height: float,
) -> tuple[float | None, float | None]:
    """Infer exact hardware fin pitch `dx` and floor spacing `dy` when >=45 exterior LEDs are glowing at dusk/night."""
    if len(xs) < 45:
        return None, None
    order_x = np.argsort(xs)
    sx, sy = xs[order_x], ys[order_x]
    dy_tol = max(4.5, b_height / 180.0)
    dx_lo, dx_hi = max(4.5, med_w / 54.0), max(12.0, med_w / 14.0)
    dxs: list[float] = []
    for i in range(len(sx)):
        best = 1e9
        for j in range(i + 1, len(sx)):
            dx = float(sx[j] - sx[i])
            if dx > dx_hi:
                break
            if dx >= dx_lo and abs(float(sy[j] - sy[i])) <= dy_tol:
                best = dx
                break
        if best < 1e8:
            dxs.append(best)

    order_y = np.argsort(ys)
    ux, uy = xs[order_y], ys[order_y]
    # Allow helical diagonal shift (`med_w / 58`) between adjacent floors so twisting towers match floor-to-floor instead of skipping 2 floors!
    dx_tol = max(6.5, med_w / 58.0)
    dy_lo, dy_hi = max(8.0, b_height / 85.0), max(18.0, b_height / 32.0)
    dys: list[float] = []
    for i in range(len(uy)):
        best = 1e9
        for j in range(i + 1, len(uy)):
            dy = float(uy[j] - uy[i])
            if dy > dy_hi:
                break
            if dy >= dy_lo and abs(float(ux[j] - ux[i])) <= dx_tol:
                best = dy
                break
        if best < 1e8:
            dys.append(best)

    emp_dx = float(np.percentile(dxs, 38)) if len(dxs) >= 24 else None
    emp_dy = float(np.percentile(dys, 35)) if len(dys) >= 24 else None
    return emp_dx, emp_dy


def _detect_existing_night_fixtures(
    rgb: np.ndarray,
    active_mask: np.ndarray,
    els: list[Element],
    masks: dict[int, np.ndarray],
) -> FixtureRigOption | None:
    """When a user uploads a night photo where the full tower rig (>60% height) is already glowing,
    offer the direct as-photographed LED coordinates as an option."""
    H, W = rgb.shape[:2]
    xs_a, ys_a, ls_a, ws_a, angs_a, sky_med = _extract_exterior_led_blobs(rgb, active_mask)
    if sky_med >= 105.0 or len(xs_a) < 220:
        return None
    ys_b = np.where(active_mask.any(axis=1))[0]
    b_span = float(ys_b.max() - ys_b.min()) if len(ys_b) > 2 else float(H)
    if float(ys_a.max() - ys_a.min()) < 0.62 * b_span:
        return None

    solids_ids = [e.id for e in els if e.id in masks]
    eids, memb = _assign_membership(xs_a, ys_a, masks, solids_ids, H, W)
    zones = [e.label for e in els if e.id in memb and len(memb[e.id]) > 8]
    return FixtureRigOption(
        id="detected_night_rig",
        name="Detected Existing Facade Rig",
        subtitle="Exact glowing LED fixture positions extracted directly from this night photograph",
        recommended=True,
        fixture_type="As-Photographed Direct-View Linear LED Pixel Bars",
        fixture_count=len(xs_a),
        pitch_desc="Exact native mullion & floor fixture coordinates",
        est_power_kw=len(xs_a) * 0.016,
        zones_used=zones[:6] or ["Full Facade"],
        xs=xs_a,
        ys=ys_a,
        lengths=ls_a,
        widths=ws_a,
        angles=angs_a,
        elem_ids=eids,
        elem_membership=memb,
        render_size=(W, H),
    )


def _survey_rig_option(
    survey,
    els: list[Element],
    masks: dict[int, np.ndarray],
    building_mask: np.ndarray,
    W: int,
    H: int,
    recommended: bool = True,
) -> FixtureRigOption | None:
    """Map an as-built UV survey cleanly onto curved or straight tower silhouettes without crushing columns."""
    from app.survey import elevations

    tower_kinds = {"crown", "tower", "setback"}
    tower_union = np.zeros((H, W), bool)
    for e in els:
        if e.kind in tower_kinds and e.id in masks:
            tower_union |= masks[e.id]
    if not tower_union.any():
        tower_union = building_mask

    # Bridge any 1-8px seams between stacked element boxes before selecting main tower component
    tower_union = ndi.binary_closing(tower_union, structure=np.ones((15, 5), bool))
    lab, n = ndi.label(tower_union)
    if n > 1:
        sizes = ndi.sum(tower_union, lab, range(1, n + 1))
        tower_union = lab == (int(np.argmax(sizes)) + 1)

    ys_t = np.where(tower_union.any(axis=1))[0]
    if len(ys_t) < 20:
        return None
    y_top, y_bot = int(ys_t.min()), int(ys_t.max())
    tower_h = max(1, y_bot - y_top)

    row_l = np.zeros(H, np.float32)
    row_r = np.zeros(H, np.float32)
    for y in range(y_top, y_bot + 1):
        xs_row = np.where(tower_union[y])[0]
        if len(xs_row):
            row_l[y], row_r[y] = float(xs_row.min()), float(xs_row.max())
        elif y > y_top:
            row_l[y], row_r[y] = row_l[y - 1], row_r[y - 1]
    row_l[y_top:y_bot + 1] = ndi.gaussian_filter1d(row_l[y_top:y_bot + 1], sigma=6.0)
    row_r[y_top:y_bot + 1] = ndi.gaussian_filter1d(row_r[y_top:y_bot + 1], sigma=6.0)
    med_w = float(np.median(row_r[y_top:y_bot + 1] - row_l[y_top:y_bot + 1]))
    aspect = med_w / max(tower_h, 1.0)

    tall_rects = elevations(survey, min_height=0.35)
    if not tall_rects:
        return None

    # Pick the single elevation whose aspect best matches the visible tower face so columns stay crisp & uncrushed
    best_span = min(
        tall_rects,
        key=lambda r: abs(np.log(max((r[2] - r[0]) / max(r[3] - r[1], 1.0), 1e-3) / max(aspect, 1e-3))),
    )
    u0, v0, u1, v1 = best_span
    inside = (survey.cx >= u0) & (survey.cx <= u1) & (survey.cy >= v0) & (survey.cy <= v1)
    u_norm = (survey.cx[inside] - u0) / max(u1 - u0, 1.0)
    v_norm = (survey.cy[inside] - v0) / max(v1 - v0, 1.0)

    py = np.clip(y_top + v_norm * tower_h, y_top, y_bot)
    yi = np.clip(np.round(py).astype(int), y_top, y_bot)
    margin = np.maximum(3.0, 0.03 * (row_r[yi] - row_l[yi]))
    px = (row_l[yi] + margin) + u_norm * np.maximum(4.0, (row_r[yi] - row_l[yi]) - 2.0 * margin)

    keep = tower_union[np.clip(np.round(py).astype(int), 0, H - 1), np.clip(np.round(px).astype(int), 0, W - 1)]
    px, py = px[keep].astype(np.float32), py[keep].astype(np.float32)

    # Upright architectural linear bar per surveyed node
    angles = np.full(len(px), np.pi / 2, dtype=np.float32)
    sy = tower_h / max(v1 - v0, 1.0)
    bar_len = float(np.clip(survey.h * sy * 0.98, 5.8, 9.8))
    bar_w = float(np.clip(med_w / 145.0, 1.85, 2.35))
    lengths = np.full(len(px), bar_len, dtype=np.float32)
    widths = np.full(len(px), bar_w, dtype=np.float32)

    solids_ids = [e.id for e in els if e.id in masks]
    eids, memb = _assign_membership(px, py, masks, solids_ids, H, W)
    zones = [e.label for e in els if e.id in memb and len(memb[e.id]) > 8]
    return FixtureRigOption(
        id="as_built_survey",
        name="As-Built Surveyed Fixture Rig (UV Map)",
        subtitle="Exact CAD/UV survey elevation warped onto the full SAM tower silhouette",
        recommended=recommended,
        fixture_type="As-Built Dynamic Linear Facade Nodes (DYN Zone)",
        fixture_count=len(px),
        pitch_desc=f"Surveyed {int(survey.canvas[0])}×{int(survey.canvas[1])} UV canvas ({len(px):,} active nodes)",
        est_power_kw=len(px) * 0.018,
        zones_used=zones[:6] or ["Full Tower"],
        xs=px,
        ys=py,
        lengths=lengths,
        widths=widths,
        angles=angles,
        elem_ids=eids,
        elem_membership=memb,
        render_size=(W, H),
    )


def propose_fixture_rigs(
    img: Image.Image,
    els: list[Element],
    survey=None,
    exact_size: bool = False,
) -> tuple[list[FixtureRigOption], np.ndarray, np.ndarray]:
    """Analyze the building's real mullions, fins, slab edges, crown and setbacks inside SAM 3.1 masks
    using full-resolution Building-ROI extraction + SAM 3.1 negative-space cutouts, returning 3+ realistic
    fixture placement options plus (night_base, building_mask) at render size."""
    raw_rgb_pil = img.convert("RGB")
    raw_w, raw_h = raw_rgb_pil.size
    if exact_size:
        W = max(2, raw_w // 2 * 2)
        H = max(2, raw_h // 2 * 2)
    else:
        W, H, _ = compute_render_size(raw_rgb_pil, els)
    rgb = np.asarray(raw_rgb_pil.resize((W, H), Image.LANCZOS), dtype=np.float32)

    solids = [e for e in els if e.mask is not None]
    masks = {e.id: np.asarray(e.mask.resize((W, H), Image.NEAREST)) > 127 for e in solids}
    masks = {k: m for k, m in masks.items() if m.any()}
    building_mask = np.zeros((H, W), bool)
    for m in masks.values():
        building_mask |= m
    building_mask = ndi.binary_closing(building_mask, structure=np.ones((9, 5), bool))
    building_mask = ndi.binary_fill_holes(building_mask)

    # Exclude only small ground-level lobby doors (top of element > 82% down the image), NEVER glazed tower shafts!
    lobby_mask = np.zeros((H, W), bool)
    for e in solids:
        if e.kind == "glass" and e.id in masks:
            ys_g = np.where(masks[e.id])[0]
            if len(ys_g) and ys_g.min() > 0.82 * H:
                lobby_mask |= masks[e.id]

    # Collect SAM 3.1 negative-space void mask (recessed glazed canyons, interior office reveals, louver bands)
    void_audited = any(bool(getattr(e, "void_audited", False)) for e in els)
    sam_void_render = np.zeros((H, W), bool)
    sam_void_raw = np.zeros((raw_h, raw_w), bool)
    has_sam_void = False
    for e in els:
        vm = getattr(e, "void_mask", None)
        if vm is not None:
            has_sam_void = True
            sam_void_render |= np.asarray(vm.resize((W, H), Image.NEAREST)) > 127
            sam_void_raw |= np.asarray(vm.resize((raw_w, raw_h), Image.NEAREST)) > 127

    night_base, _ = prepare_night_base(rgb, building_mask)

    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gx = ndi.sobel(luma, axis=1)
    gy = ndi.sobel(luma, axis=0)

    # Filter out detached small dark street billboards at very bottom of frame (never main wide building bases!)
    local_detail = ndi.gaussian_filter(np.abs(gx) + np.abs(gy), sigma=6.0)
    yy_idx = np.arange(H)[:, None]
    raw_bb = building_mask & (yy_idx > 0.84 * H) & (local_detail < np.percentile(local_detail[building_mask], 14)) & (luma < 82.0)
    lab_bb, n_bb = ndi.label(raw_bb)
    billboard = np.zeros((H, W), bool)
    for k_bb in range(1, n_bb + 1):
        comp = lab_bb == k_bb
        xs_c = np.where(comp)[1]
        if len(xs_c) and (xs_c.max() - xs_c.min()) < 0.32 * W:
            billboard |= ndi.binary_dilation(comp, iterations=3) & (yy_idx > 0.83 * H)

    usable_mask = building_mask & (~lobby_mask) & (~billboard)
    if usable_mask.sum() < 0.5 * building_mask.sum():
        usable_mask = building_mask

    ys_b, xs_b = np.where(usable_mask)
    y_min, y_max = int(ys_b.min()), int(ys_b.max())
    x_min, x_max = int(xs_b.min()), int(xs_b.max())
    b_height = max(40, y_max - y_min)
    row_widths = usable_mask[y_min:y_max + 1].sum(axis=1)
    med_width = float(np.median(row_widths[row_widths > 20])) if (row_widths > 20).any() else float(x_max - x_min)

    # -------------------------------------------------------------------------
    # UPGRADE #1: Native Full-Resolution Building-ROI Mullion & Floor Extraction
    # Crop tightly around the building directly from `raw_rgb_pil` BEFORE any whole-image downscale
    # so high-MP camera uploads keep 100% native fin sharpness and narrow towers upscale cleanly.
    # -------------------------------------------------------------------------
    pad_rx = max(10, int(round(0.04 * (x_max - x_min) * raw_w / max(W, 1))))
    pad_ry = max(10, int(round(0.025 * (y_max - y_min) * raw_h / max(H, 1))))
    rx0 = max(0, int(np.floor(x_min * raw_w / max(W, 1))) - pad_rx)
    ry0 = max(0, int(np.floor(y_min * raw_h / max(H, 1))) - pad_ry)
    rx1 = min(raw_w, int(np.ceil((x_max + 1) * raw_w / max(W, 1))) + pad_rx)
    ry1 = min(raw_h, int(np.ceil((y_max + 1) * raw_h / max(H, 1))) + pad_ry)
    cw, ch = max(16, rx1 - rx0), max(16, ry1 - ry0)

    raw_med_w = med_width * (raw_w / max(W, 1))
    raw_bh = b_height * (raw_h / max(H, 1))
    roi_scale = max(1.0, 560.0 / max(raw_med_w, 24.0), 1560.0 / max(raw_bh, 40.0))
    roi_scale = min(roi_scale, 3.2, 2800.0 / max(cw, ch))
    W_roi = max(32, int(round(cw * roi_scale)))
    H_roi = max(32, int(round(ch * roi_scale)))

    roi_crop_pil = raw_rgb_pil.crop((rx0, ry0, rx1, ry1))
    if (W_roi, H_roi) != (cw, ch):
        roi_crop_pil = roi_crop_pil.resize((W_roi, H_roi), Image.LANCZOS)
    roi_rgb = np.asarray(roi_crop_pil, dtype=np.float32)

    # Build native-precision `usable_roi` & `sam_void_roi` on `(H_roi, W_roi)`
    raw_bmask = np.zeros((raw_h, raw_w), bool)
    for e in solids:
        if e.mask is not None:
            m_raw = np.asarray(e.mask.resize((raw_w, raw_h), Image.NEAREST)) > 127
            if e.kind == "glass":
                ys_gr = np.where(m_raw)[0]
                if len(ys_gr) and ys_gr.min() > 0.82 * raw_h:
                    continue
            raw_bmask |= m_raw
    raw_bmask = ndi.binary_closing(raw_bmask, structure=np.ones((7, 5), bool))
    raw_bmask = ndi.binary_fill_holes(raw_bmask)
    if billboard.any():
        bb_raw = np.asarray(Image.fromarray(billboard.astype(np.uint8) * 255).resize((raw_w, raw_h), Image.NEAREST)) > 127
        raw_bmask &= ~bb_raw
    usable_crop = raw_bmask[ry0:ry1, rx0:rx1]
    if not usable_crop.any():
        usable_crop = np.ones((ch, cw), bool)
    usable_roi = (
        np.asarray(Image.fromarray(usable_crop.astype(np.uint8) * 255).resize((W_roi, H_roi), Image.NEAREST)) > 127
        if (W_roi, H_roi) != (cw, ch)
        else usable_crop.copy()
    )

    roi_luma = 0.299 * roi_rgb[:, :, 0] + 0.587 * roi_rgb[:, :, 1] + 0.114 * roi_rgb[:, :, 2]
    roi_gx = ndi.sobel(roi_luma, axis=1)
    roi_gy = ndi.sobel(roi_luma, axis=0)

    # Structure tensor on high-res ROI for smooth local mullion / helical fin tangent orientation
    sig_st = float(np.clip(W_roi / 56.0, 7.0, 14.0))
    Jxx = ndi.gaussian_filter(roi_gx * roi_gx, sigma=sig_st)
    Jyy = ndi.gaussian_filter(roi_gy * roi_gy, sigma=sig_st)
    Jxy = ndi.gaussian_filter(roi_gx * roi_gy, sigma=sig_st)
    grad_angle = 0.5 * np.arctan2(2.0 * Jxy, Jxx - Jyy)
    raw_vert_ang = grad_angle + np.pi / 2.0
    delta_v = np.arctan2(np.sin(raw_vert_ang - np.pi / 2.0), np.cos(raw_vert_ang - np.pi / 2.0))
    theta_vert_roi = (np.pi / 2.0 + np.clip(delta_v, -0.45, 0.45)).astype(np.float32)

    ys_r, xs_r = np.where(usable_roi)
    ry_min, ry_max = int(ys_r.min()), int(ys_r.max())
    rx_min, rx_max = int(xs_r.min()), int(xs_r.max())
    roi_bh = max(40, ry_max - ry_min)
    roi_rwidths = usable_roi[ry_min:ry_max + 1].sum(axis=1)
    roi_med_w = float(np.median(roi_rwidths[roi_rwidths > 20])) if (roi_rwidths > 20).any() else float(rx_max - rx_min)

    # Adaptive vertical fin ridge enhancement scaled to ROI tower width so fine 40+ fin facades pop crisply even on 500px thumbnails
    sig_fin = float(np.clip(roi_med_w / 195.0, 1.5, 3.6))
    hp_x = roi_luma - ndi.gaussian_filter1d(roi_luma, sigma=sig_fin, axis=1)
    fin_raw = np.maximum(hp_x, 0.0) + 0.25 * np.abs(roi_gx)
    fin_smooth = ndi.gaussian_filter1d(fin_raw, sigma=3.0, axis=0)
    local_scale = ndi.gaussian_filter(fin_smooth, sigma=14.0) + 1.6
    mullion_score_roi = fin_smooth / local_scale

    # Illumination-invariant vertical fin energy (immune to cool blue cast shadows on sunlit ceramic/stone fins!)
    low_luma_roi = ndi.gaussian_filter(roi_luma, sigma=9.0) + 18.0
    hp_norm_roi = np.abs(roi_luma - ndi.gaussian_filter1d(roi_luma, sigma=max(1.4, sig_fin * 0.65), axis=1)) / low_luma_roi
    fin_energy_roi = ndi.gaussian_filter(hp_norm_roi, sigma=(7.0, 4.5))

    inner_roi = ndi.binary_erosion(usable_roi, iterations=max(2, int(round(W_roi / 220.0))))
    dense_spine_roi = np.zeros((H_roi, W_roi), bool)
    if usable_roi.sum() > 400:
        fe_p68 = float(np.percentile(fin_energy_roi[usable_roi], 68))
        dense_spine_roi = usable_roi & (fin_energy_roi >= fe_p68)

    # Extract any glowing exterior LED dashes on dusk/night uploads BEFORE finalizing voids so lit exterior fins are never masked out!
    led_xs_roi, led_ys_roi, _, _, _, _ = _extract_exterior_led_blobs(roi_rgb, usable_roi)
    lit_fin_guard = np.zeros((H_roi, W_roi), bool)
    if len(led_xs_roi) >= 35:
        ly_i = np.clip(np.round(led_ys_roi).astype(int), 0, H_roi - 1)
        lx_i = np.clip(np.round(led_xs_roi).astype(int), 0, W_roi - 1)
        seed_pts = np.zeros((H_roi, W_roi), np.int32)
        np.add.at(seed_pts, (ly_i, lx_i), 1)
        bridged = ndi.binary_dilation(seed_pts > 0, structure=np.ones((23, 19), bool))
        lab_lg, n_lg = ndi.label(bridged)
        for lg_id in range(1, n_lg + 1):
            comp = lab_lg == lg_id
            if int(seed_pts[comp].sum()) >= 10:
                lit_fin_guard |= comp

    # -------------------------------------------------------------------------
    # UPGRADE #2: SAM 3.1 Negative-Space Cutouts (Zero Shadow/Colour False Positives!)
    # -------------------------------------------------------------------------
    active_roi = usable_roi.copy()
    active_render = usable_mask.copy()
    if has_sam_void:
        void_crop = sam_void_raw[ry0:ry1, rx0:rx1]
        void_roi = (
            np.asarray(Image.fromarray(void_crop.astype(np.uint8) * 255).resize((W_roi, H_roi), Image.NEAREST)) > 127
            if (W_roi, H_roi) != (cw, ch)
            else void_crop.copy()
        )
        void_roi &= ~lit_fin_guard
        # Smoothly preserve at least 14% outer fin-spine margin on any row where a wide glass canyon spans most of the face
        # (NEVER zero out an entire row, which would draw horizontal dotted lines across the void!)
        for ry_i in range(H_roi):
            u_xs = np.where(usable_roi[ry_i])[0]
            if len(u_xs) > 16 and int((void_roi[ry_i] & usable_roi[ry_i]).sum()) > int(0.84 * len(u_xs)):
                keep_margin = max(3, int(round(0.14 * len(u_xs))))
                left_e = float(fin_energy_roi[ry_i, u_xs[:keep_margin]].mean())
                right_e = float(fin_energy_roi[ry_i, u_xs[-keep_margin:]].mean())
                if right_e >= left_e:
                    void_roi[ry_i, u_xs[-keep_margin:]] = False
                else:
                    void_roi[ry_i, u_xs[:keep_margin]] = False
        active_roi = usable_roi & (~void_roi)
        # Project cleaned `void_roi` back to render space so Option 2 & 3 also match cleanly
        void_clean_raw = np.zeros((raw_h, raw_w), bool)
        void_clean_raw[ry0:ry1, rx0:rx1] = (
            np.asarray(Image.fromarray(void_roi.astype(np.uint8) * 255).resize((cw, ch), Image.NEAREST)) > 127
            if (W_roi, H_roi) != (cw, ch)
            else void_roi
        )
        sam_void_render = np.asarray(Image.fromarray(void_clean_raw.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)) > 127
        active_render = usable_mask & (~sam_void_render)
    elif not void_audited and usable_roi.sum() > 600:
        # Offline fallback (only when `map_facade`'s SAM 3.1 void audit wasn't invoked):
        # Strictly illumination-normalized (`hp_norm_roi`) — NEVER uses `R - B` colour warmth so cast shadows stay 100% populated!
        fe_vals = fin_energy_roi[usable_roi]
        fe_p12, fe_med, fe_p88 = np.percentile(fe_vals, [12, 50, 88])
        if fe_p88 / max(fe_p12, 1e-4) > 2.85:
            raw_void = usable_roi & (fin_energy_roi < 0.34 * fe_med) & (~lit_fin_guard)
            lab_v, n_v = ndi.label(raw_void)
            clean_v = np.zeros((H_roi, W_roi), bool)
            for vid in range(1, n_v + 1):
                comp = lab_v == vid
                if int(comp.sum()) >= max(420, int(0.012 * usable_roi.sum())):
                    clean_v |= comp
            active_roi = usable_roi & (~clean_v)

    # Slenderness-aware pitch bounds: tall high-rise towers (`slenderness >= 2.25`) ALWAYS carry 44-68 individual stories
    # and 24-46 vertical fin columns (never locking onto 16-floor macro refuge bands on compressed web thumbnails!)
    slenderness = float(roi_bh) / max(roi_med_w, 1.0)
    dx_lo = max(6, int(round(roi_med_w / 46.0)))
    dx_hi = max(11, int(round(roi_med_w / (24.0 if slenderness >= 2.25 else 11.5))))
    dy_lo = max(12, int(round(roi_bh / 68.0)))
    dy_hi = max(18, int(round(roi_bh / (44.0 if slenderness >= 2.25 else 16.0))))

    mid_ry0, mid_ry1 = ry_min + int(0.22 * roi_bh), ry_min + int(0.78 * roi_bh)
    horiz_prof = fin_smooth[mid_ry0:mid_ry1].mean(axis=0) if mid_ry1 > mid_ry0 else fin_smooth.mean(axis=0)
    mullion_dx_roi = _estimate_pitch_1d(horiz_prof[rx_min:rx_max], lo=dx_lo, hi=dx_hi)

    hp_y_roi = np.abs(roi_luma - ndi.gaussian_filter1d(roi_luma, sigma=max(2.0, roi_bh / 520.0), axis=0))
    slab_map_roi = ndi.gaussian_filter1d(hp_y_roi, sigma=max(5.0, roi_med_w / 70.0), axis=1)
    row_prof_roi = np.array([
        slab_map_roi[y, active_roi[y]].mean() if active_roi[y].any() else (slab_map_roi[y, usable_roi[y]].mean() if usable_roi[y].any() else 0.0)
        for y in range(ry_min, ry_max + 1)
    ], dtype=np.float32)
    floor_dy_roi = _estimate_pitch_1d(row_prof_roi, lo=dy_lo, hi=dy_hi)

    # Dusk / Night Empirical LED Hardware Calibration: if >=45 real exterior LEDs are glowing in ANY zone (e.g. upper tower),
    # lock onto their exact physical column & floor spacing!
    emp_dx, emp_dy = _measure_empirical_led_pitch(led_xs_roi, led_ys_roi, roi_med_w, float(roi_bh))
    if emp_dx is not None:
        mullion_dx_roi = int(np.clip(round(emp_dx), dx_lo, dx_hi))
    if emp_dy is not None:
        floor_dy_roi = int(np.clip(round(emp_dy), dy_lo, dy_hi))

    # Floor-slab y lines on ROI: snap to true horizontal spandrel peaks, interpolating steady floor tiers in smooth zones
    slab_peaks_roi = _nms_1d(row_prof_roi, min_dist=max(6, int(floor_dy_roi * 0.76)), thresh=float(np.percentile(row_prof_roi, 28))) + ry_min
    if len(slab_peaks_roi) >= 4:
        tier_ys_roi = sorted(int(v) for v in slab_peaks_roi)
        filled_r: list[int] = []
        anchors_r = [ry_min + floor_dy_roi // 2] + tier_ys_roi + [ry_max - floor_dy_roi // 2]
        for a, b in zip(anchors_r, anchors_r[1:]):
            gap = b - a
            if gap >= int(floor_dy_roi * 1.62):
                steps = max(2, int(round(gap / floor_dy_roi)))
                for k in range(1, steps):
                    filled_r.append(int(round(a + k * gap / steps)))
        tier_ys_roi = sorted(set(tier_ys_roi + filled_r))
    else:
        tier_ys_roi = list(range(ry_min + floor_dy_roi // 2, ry_max - floor_dy_roi // 3, floor_dy_roi))

    # Coordinate conversion helpers: ROI `(W_roi, H_roi)` -> Render `(W, H)` with exact pixel-center alignment
    sx_r2w = (cw / float(W_roi)) * (W / float(raw_w))
    sy_r2h = (ch / float(H_roi)) * (H / float(raw_h))

    def _roi_to_render_xy(xr: np.ndarray, yr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xw = (rx0 + (xr + 0.5) * (cw / float(W_roi))) * (W / float(raw_w)) - 0.5
        yh = (ry0 + (yr + 0.5) * (ch / float(H_roi))) * (H / float(raw_h)) - 0.5
        return np.clip(xw, 1.0, W - 2.0).astype(np.float32), np.clip(yh, 1.0, H - 2.0).astype(np.float32)

    mullion_dx = max(5, int(round(mullion_dx_roi * sx_r2w)))
    floor_dy = max(8, int(round(floor_dy_roi * sy_r2h)))
    tier_ys = sorted(set(int(np.clip(round((ry0 + (ty + 0.5) * (ch / float(H_roi))) * (H / float(raw_h)) - 0.5), 4, H - 5)) for ty in tier_ys_roi))

    # Estimate clean perspective roof/slab horizontal slope per row
    top_xs = np.where(usable_mask[y_min:min(H, y_min + 25)])[1]
    horiz_slope = 0.0
    if len(top_xs) > 10:
        c_cols = np.unique(top_xs)
        if len(c_cols) > 12:
            top_y_per_x = [y_min + int(np.where(usable_mask[y_min:min(H, y_min + 35), c])[0].min()) for c in c_cols]
            slope_fit = float(np.polyfit(c_cols, top_y_per_x, 1)[0])
            horiz_slope = float(np.clip(np.arctan(slope_fit), -0.10, 0.10))

    solids_ids = [e.id for e in solids if e.id in masks]
    options: list[FixtureRigOption] = []

    detected_opt = _detect_existing_night_fixtures(rgb, active_render, solids, masks)
    if detected_opt is not None:
        options.append(detected_opt)

    if survey is not None:
        sv_opt = _survey_rig_option(
            survey, solids, masks, building_mask, W, H,
            recommended=(detected_opt is None),
        )
        if sv_opt is not None:
            options.append(sv_opt)

    primary_recommended = (len(options) == 0)
    nom_len_roi = float(np.clip(floor_dy_roi * 0.48, 5.5, 22.0))
    bar_len_v = float(np.clip(floor_dy * 0.46, 5.2, 10.6))
    bar_w_v = float(np.clip(mullion_dx * 0.165, 1.38, 2.25))

    # -------------------------------------------------------------------------
    # OPTION 1: Full Mullion & Fin Linear Pixel Matrix (Native High-Res Streamlines)
    # -------------------------------------------------------------------------
    xs1_r, ys1_r, angs1 = _trace_mullion_tracks(
        mullion_score_roi, theta_vert_roi, inner_roi, active_roi, tier_ys_roi,
        min_dx=mullion_dx_roi, dense_spine_mask=dense_spine_roi,
    )
    xs1_r, ys1_r, ls1_r, keep1 = _taper_vertical_bars(
        xs1_r, ys1_r, active_roi, nom_len_roi, taper_zone=floor_dy_roi * 1.45,
    )
    angs1 = angs1[keep1]
    xs1, ys1 = _roi_to_render_xy(xs1_r, ys1_r)
    taper_ratio = np.clip(ls1_r / max(nom_len_roi, 1e-3), 0.38, 1.0)
    ls1 = np.clip(bar_len_v * taper_ratio, 2.6, 10.8).astype(np.float32)

    if len(xs1) >= 20:
        eids1, memb1 = _assign_membership(xs1, ys1, masks, solids_ids, H, W)
        zones1 = [e.label for e in solids if e.id in memb1 and len(memb1[e.id]) > 8]
        options.append(FixtureRigOption(
            id="mullion_matrix",
            name="Full Mullion & Fin Pixel Matrix",
            subtitle="Direct-view linear RGBW bars tracked at native ROI resolution along mullions & helical fins",
            recommended=primary_recommended,
            fixture_type="1200mm Direct-View Mullion Linear Pixel Bar",
            fixture_count=len(xs1),
            pitch_desc=f"~{mullion_dx}px fin pitch × {floor_dy}px floor tier spacing",
            est_power_kw=len(xs1) * 0.018,
            zones_used=zones1[:6] or ["Full Facade"],
            xs=xs1,
            ys=ys1,
            lengths=ls1,
            widths=np.full(len(xs1), bar_w_v, np.float32),
            angles=angs1,
            elem_ids=eids1,
            elem_membership=memb1,
            render_size=(W, H),
        ))

    # -------------------------------------------------------------------------
    # OPTION 2: Architectural Crown, Setback & Structural Spine Rig
    # -------------------------------------------------------------------------
    inner_mask = ndi.binary_erosion(usable_mask, iterations=3)
    a_xs, a_ys, a_ls, a_ws, a_angs = [], [], [], [], []
    crown_mask = np.zeros((H, W), bool)
    for e in solids:
        if e.kind == "crown" and e.id in masks:
            crown_mask |= masks[e.id]
    if crown_mask.sum() < 200:
        crown_cut = y_min + int(0.11 * b_height)
        crown_mask = usable_mask & (np.arange(H)[:, None] <= crown_cut)

    # (a) Dense crown mullions from tracked streamlines inside crown_mask
    if len(xs1):
        in_cr = crown_mask[np.clip(ys1.astype(int), 0, H - 1), np.clip(xs1.astype(int), 0, W - 1)]
        for cx, cy, cl, ca in zip(xs1[in_cr], ys1[in_cr], ls1[in_cr], angs1[in_cr]):
            a_xs.append(float(cx)); a_ys.append(float(cy))
            a_ls.append(float(cl) * 1.05); a_ws.append(bar_w_v * 1.15); a_angs.append(float(ca))

    # (b) Clean horizontal architectural ledges on crown parapet, setbacks, transfer bands, and podium
    seg_step = max(13, int(mullion_dx * 1.35))
    for e in solids:
        if e.id not in masks or e.kind == "glass":
            continue
        em = masks[e.id] & active_render
        eys = np.where(em.any(axis=1))[0]
        if len(eys) < 8:
            continue
        ledge_rows = [int(eys.min()) + 4]
        if e.kind in {"band", "podium"} and (eys.max() - eys.min()) > 16:
            ledge_rows.append(int(eys.max()) - 4)
        for ly in ledge_rows:
            if ly < 4 or ly >= H - 4:
                continue
            xs_row = np.where(em[ly] & inner_mask[ly])[0]
            if len(xs_row) < 14:
                continue
            for x in range(int(xs_row.min()) + seg_step // 2, int(xs_row.max()) - seg_step // 3, seg_step):
                if inner_mask[ly, x] and active_render[ly, x]:
                    slope_here = horiz_slope * max(0.0, 1.0 - (ly - y_min) / max(b_height, 1.0))
                    a_xs.append(float(x)); a_ys.append(float(ly))
                    a_ls.append(float(seg_step * 0.76)); a_ws.append(bar_w_v * 1.2)
                    a_angs.append(float(slope_here))

    # (c) Smooth outer perimeter spines + top 2 primary structural megacolumns
    xs_sp_r, ys_sp_r, angs_sp = _trace_mullion_tracks(
        mullion_score_roi, theta_vert_roi, inner_roi, active_roi, tier_ys_roi, min_dx=max(28, int(roi_med_w / 4.2)),
    )
    xs_sp, ys_sp = _roi_to_render_xy(xs_sp_r, ys_sp_r)
    for cx, cy, ca in zip(xs_sp, ys_sp, angs_sp):
        a_xs.append(float(cx)); a_ys.append(float(cy))
        a_ls.append(bar_len_v * 1.1); a_ws.append(bar_w_v * 1.2); a_angs.append(float(ca))

    # Also trace smooth left & right silhouette edge contours (including sculpted SAM 3.1 twist edges!)
    contour_mask = inner_mask & active_render
    for y in tier_ys:
        if y < 4 or y >= H - 4 or not contour_mask[y].any():
            continue
        row_bool = contour_mask[y]
        edges_x = np.where(row_bool[1:] != row_bool[:-1])[0]
        for ex in edges_x:
            sx = int(ex + 3 if row_bool[min(W - 1, ex + 2)] else ex - 3)
            if 4 <= sx < W - 4 and contour_mask[y, sx]:
                y_up, y_dn = max(0, y - 8), min(H - 1, y + 8)
                xs_up = np.where(inner_mask[y_up])[0]
                xs_dn = np.where(inner_mask[y_dn])[0]
                if len(xs_up) and len(xs_dn):
                    is_left = abs(sx - xs_dn.min()) < abs(sx - xs_dn.max())
                    dx_e = (xs_dn.min() - xs_up.min()) if is_left else (xs_dn.max() - xs_up.max())
                    ang_e = float(np.arctan2(y_dn - y_up, dx_e))
                else:
                    ang_e = float(np.pi / 2.0)
                a_xs.append(float(sx)); a_ys.append(float(y))
                a_ls.append(bar_len_v * 1.08); a_ws.append(bar_w_v * 1.18); a_angs.append(ang_e)

    if len(a_xs) >= 16:
        pts = np.stack([a_xs, a_ys], axis=1)
        keep_idx = [0]
        for i in range(1, len(pts)):
            if np.sum((pts[keep_idx] - pts[i]) ** 2, axis=1).min() >= 36.0:
                keep_idx.append(i)
        xs2 = np.array([a_xs[i] for i in keep_idx], np.float32)
        ys2 = np.array([a_ys[i] for i in keep_idx], np.float32)
        eids2, memb2 = _assign_membership(xs2, ys2, masks, solids_ids, H, W)
        zones2 = [e.label for e in solids if e.id in memb2 and len(memb2[e.id]) > 4]
        options.append(FixtureRigOption(
            id="architectural_accent",
            name="Crown, Setback & Structural Spine Rig",
            subtitle="Perimeter linear grazing on the crown, setback ledges, and primary structural spines (low CapEx)",
            recommended=False,
            fixture_type="1000mm Architectural Contour & Spine Linear Bar",
            fixture_count=len(xs2),
            pitch_desc="Crown matrix + setback parapets + structural edge spines",
            est_power_kw=len(xs2) * 0.020,
            zones_used=zones2[:6] or ["Crown & Spines"],
            xs=xs2,
            ys=ys2,
            lengths=np.array([a_ls[i] for i in keep_idx], np.float32),
            widths=np.array([a_ws[i] for i in keep_idx], np.float32),
            angles=np.array([a_angs[i] for i in keep_idx], np.float32),
            elem_ids=eids2,
            elem_membership=memb2,
            render_size=(W, H),
        ))

    # -------------------------------------------------------------------------
    # OPTION 3: Horizontal Floor-Slab & Spandrel Ribbon Rig
    # -------------------------------------------------------------------------
    h_xs, h_ys, h_ls, h_ws, h_angs = [], [], [], [], []
    horiz_step = max(13, int(mullion_dx * 1.45))
    horiz_len = float(horiz_step * 0.72)
    use_every = 2 if len(tier_ys) > 42 else 1
    for idx_t, y in enumerate(tier_ys):
        if y < 4 or y >= H - 4 or not (inner_mask[y] & active_render[y]).any():
            continue
        in_crown = bool(crown_mask[y].any())
        if (idx_t % use_every != 0) and not in_crown:
            continue
        xs_row = np.where(inner_mask[y] & active_render[y])[0]
        if len(xs_row) < 10:
            continue
        slope_here = horiz_slope * max(0.0, 1.0 - (y - y_min) / max(b_height, 1.0))
        x = int(xs_row.min()) + horiz_step // 2
        x_end = int(xs_row.max()) - horiz_step // 3
        while x <= x_end:
            span_ok = inner_mask[y, max(0, x - 4):min(W, x + 5)] & active_render[y, max(0, x - 4):min(W, x + 5)]
            if span_ok.all():
                h_xs.append(float(x)); h_ys.append(float(y))
                h_ls.append(horiz_len); h_ws.append(bar_w_v * 1.05)
                h_angs.append(float(slope_here))
            x += horiz_step

    if len(h_xs) >= 16:
        xs3, ys3 = np.array(h_xs, np.float32), np.array(h_ys, np.float32)
        eids3, memb3 = _assign_membership(xs3, ys3, masks, solids_ids, H, W)
        zones3 = [e.label for e in solids if e.id in memb3 and len(memb3[e.id]) > 6]
        options.append(FixtureRigOption(
            id="horizontal_tiers",
            name="Horizontal Floor-Slab & Spandrel Ribbons",
            subtitle="Segmented horizontal LED bars mounted along detected floor-slab edges and louver bands",
            recommended=False,
            fixture_type="1200mm Slab-Edge Spandrel Linear Ribbon",
            fixture_count=len(xs3),
            pitch_desc=f"~{floor_dy * use_every}px vertical slab rhythm × {horiz_step}px bay modules",
            est_power_kw=len(xs3) * 0.018,
            zones_used=zones3[:6] or ["Floor Slab Tiers"],
            xs=xs3,
            ys=ys3,
            lengths=np.array(h_ls, np.float32),
            widths=np.array(h_ws, np.float32),
            angles=np.array(h_angs, np.float32),
            elem_ids=eids3,
            elem_membership=memb3,
            render_size=(W, H),
        ))

    # Populate 3D wrap-around surface coordinates (u_local, v_elev, view_cos) for every rig option
    # so a 360-degree full-building show maps realistically onto the visible side(s) in the photo!
    row_l = np.zeros(H, dtype=np.float32)
    row_r = np.full(H, float(W - 1), dtype=np.float32)
    for ry in range(y_min, y_max + 1):
        rxs = np.where(building_mask[ry])[0]
        if len(rxs) >= 2:
            row_l[ry] = float(rxs.min())
            row_r[ry] = float(rxs.max())
    valid_span = slice(y_min, y_max + 1)
    row_l[valid_span] = ndi.gaussian_filter1d(row_l[valid_span], sigma=6.0)
    row_r[valid_span] = ndi.gaussian_filter1d(row_r[valid_span], sigma=6.0)

    # Wide front elevations (>55% of image width) default to 1-side/front wrap (0.32); slender towers default to 0.45 (~162 deg)
    auto_wrap = 0.32 if (med_width / max(W, 1)) > 0.55 else 0.45

    for opt in options:
        if opt.fixture_count == 0:
            continue
        yi_c = np.clip(np.round(opt.ys).astype(np.int32), y_min, y_max)
        rl = row_l[yi_c]
        rr = np.maximum(row_r[yi_c], rl + 8.0)
        s = np.clip((opt.xs - rl) / (rr - rl), 0.015, 0.985).astype(np.float32)
        # Blend 62% cylindrical arccos foreshortening + 38% planar so patterns curve naturally around grazing edges
        phi_cyl = (np.arccos(np.clip(1.0 - 2.0 * s, -0.99, 0.99)) / np.pi).astype(np.float32)
        opt.u_local = (0.62 * phi_cyl + 0.38 * s).astype(np.float32)
        opt.v_elev = np.clip((opt.ys - y_min) / max(float(y_max - y_min), 1.0), 0.0, 1.0).astype(np.float32)
        opt.view_cos = (0.62 + 0.38 * np.sin(np.pi * s)).astype(np.float32)
        opt.default_wrap = auto_wrap

        # Structural spine azimuth (`spine_u`) = peak fixture density in central third (just like 3S Wasl viz.py SPINE_X!)
        hist, edges = np.histogram(opt.u_local, bins=36, range=(0.0, 1.0))
        smooth_h = ndi.gaussian_filter1d(hist.astype(np.float32), sigma=1.5)
        mid_lo, mid_hi = 10, 26
        best_bin = mid_lo + int(np.argmax(smooth_h[mid_lo:mid_hi]))
        opt.spine_u = float(0.5 * (edges[best_bin] + edges[best_bin + 1]))

        # Tight architectural crop box (using smoothed mass > 5% of peak like 3S viz.py so sparse specks don't pad empty margins)
        col_hist, col_edges = np.histogram(opt.xs, bins=max(16, W // 16), range=(0.0, float(W)))
        col_sm = ndi.gaussian_filter1d(col_hist.astype(np.float32), sigma=1.5)
        act_cols = np.where(col_sm > col_sm.max() * 0.05)[0]
        fx_x0 = float(col_edges[act_cols.min()]) if len(act_cols) else float(opt.xs.min())
        fx_x1 = float(col_edges[act_cols.max() + 1]) if len(act_cols) else float(opt.xs.max())
        fx_y0, fx_y1 = float(opt.ys.min()), float(opt.ys.max())
        pad_x = max(24.0, (fx_x1 - fx_x0) * 0.10)
        pad_y = max(28.0, (fx_y1 - fx_y0) * 0.05)
        cx0 = max(0, int(round(fx_x0 - pad_x)) // 2 * 2)
        cy0 = max(0, int(round(fx_y0 - pad_y)) // 2 * 2)
        cx1 = min(W, int(round(fx_x1 + pad_x + 1)) // 2 * 2)
        cy1 = min(H, int(round(fx_y1 + pad_y + 1)) // 2 * 2)
        if cx1 - cx0 >= 64 and cy1 - cy0 >= 64:
            opt.crop_box = (cx0, cy0, cx1, cy1)

    return options, night_base, building_mask


def rasterize_oriented_splats(
    xs: np.ndarray,
    ys: np.ndarray,
    lengths: np.ndarray,
    widths: np.ndarray,
    angles: np.ndarray,
    H: int,
    W: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute anti-aliased pixel coordinates (py, px, owner, weight) sorted by ascending weight
    for fast O(1) per-frame array assignment, plus a tight 2D mullion grazing mask."""
    n = len(xs)
    if n == 0:
        empty_i = np.empty(0, np.int32)
        return empty_i, empty_i, empty_i, np.empty(0, np.float32), np.zeros((H, W), np.float32)

    max_L = int(np.ceil(lengths.max())) if len(lengths) else 8
    steps_along = max(3, max_L * 2 + 1)
    t_vals = np.linspace(-0.5, 0.5, steps_along, dtype=np.float32)

    cos_a = np.cos(angles)[:, None]
    sin_a = np.sin(angles)[:, None]
    L_col = lengths[:, None]

    sx = xs[:, None] + (t_vals[None, :] * L_col) * cos_a
    sy = ys[:, None] + (t_vals[None, :] * L_col) * sin_a
    end_taper = np.clip(1.16 - 0.52 * np.abs(t_vals * 2.0) ** 3, 0.52, 1.0)[None, :]

    nx = -sin_a[:, 0]
    ny = cos_a[:, 0]
    half_w = np.clip((widths - 0.68) * 0.52, 0.30, 0.82)
    edge_wt = np.clip((widths - 1.10) * 0.55, 0.34, 0.62)[:, None]

    py_parts, px_parts, own_parts, wt_parts = [], [], [], []
    owners_grid = np.broadcast_to(np.arange(n, dtype=np.int32)[:, None], sx.shape)

    for offset_scale, is_edge in ((-1.0, True), (1.0, True), (0.0, False)):
        ox = sx + (offset_scale * half_w)[:, None] * nx[:, None]
        oy = sy + (offset_scale * half_w)[:, None] * ny[:, None]
        iy = np.clip(np.round(oy).astype(np.int32), 0, H - 1).ravel()
        ix = np.clip(np.round(ox).astype(np.int32), 0, W - 1).ravel()
        ow = owners_grid.ravel()
        c_wt = edge_wt if is_edge else 1.0
        wt = np.broadcast_to(end_taper * c_wt, sx.shape).ravel().astype(np.float32)
        py_parts.append(iy); px_parts.append(ix); own_parts.append(ow); wt_parts.append(wt)

    py_all = np.concatenate(py_parts)
    px_all = np.concatenate(px_parts)
    own_all = np.concatenate(own_parts)
    wt_all = np.concatenate(wt_parts)

    lin = own_all.astype(np.int64) * (H * W) + py_all.astype(np.int64) * W + px_all.astype(np.int64)
    order2 = np.lexsort((-wt_all, lin))
    lin2 = lin[order2]
    uniq_mask = np.r_[True, lin2[1:] != lin2[:-1]]
    keep = order2[uniq_mask]

    # Sort ascending by weight so fast assignment `emitter[py, px] = ...` always lets the bright 1.0 core win!
    w_sort = np.argsort(wt_all[keep], kind="mergesort")
    keep = keep[w_sort]

    py_u = py_all[keep]
    px_u = px_all[keep]
    own_u = own_all[keep]
    wt_u = wt_all[keep]

    emitter_map = np.zeros((H, W), np.float32)
    emitter_map[py_u, px_u] = wt_u
    dilated = ndi.grey_dilation(emitter_map, size=(5, 5))
    halo_mask = ndi.gaussian_filter(np.clip(dilated, 0.0, 1.0), sigma=1.0).astype(np.float32)
    return py_u, px_u, own_u, wt_u, halo_mask


def composite_fixture_frame(
    night_base: np.ndarray,
    py: np.ndarray,
    px: np.ndarray,
    owner: np.ndarray,
    weight: np.ndarray,
    halo_mask: np.ndarray,
    fixture_rgb: np.ndarray,
) -> np.ndarray:
    """Render one frame on the nocturnal camera base where ONLY discrete physical LED bars emit light,
    with HDR white-core saturation and tight mullion grazing — zero surface wash on glass bays."""
    H, W = night_base.shape[:2]
    if len(py) == 0 or len(fixture_rgb) == 0:
        return night_base.clip(0, 255).astype(np.uint8)

    # HDR roll-off: values > 255 bleed into incandescent white-gold core while preserving rich hue below 255
    hdr_excess = np.maximum(0.0, fixture_rgb - 255.0).max(axis=1, keepdims=True)
    chroma_rgb = np.clip(fixture_rgb, 0.0, 255.0)

    sat_layer = np.zeros((H, W, 3), np.float32)
    sat_layer[py, px] = chroma_rgb[owner] * weight[:, None]

    # Tight optical grazing confined strictly to immediate mullion proximity (never washes across glass!)
    grazing = np.asarray(
        Image.fromarray(sat_layer.clip(0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=2.2)),
        np.float32,
    ) * (halo_mask[:, :, None] * 1.65)

    # Crisp direct-view diode core preserving full chromatic saturation + incandescent HDR star pop
    core_col = np.clip(chroma_rgb * 1.12 + hdr_excess * 0.92, 0.0, 255.0)
    emitter = np.zeros((H, W, 3), np.float32)
    emitter[py, px] = core_col[owner] * weight[:, None]

    light = np.clip(emitter + grazing, 0.0, 255.0) / 255.0
    base_n = np.clip(night_base, 0.0, 255.0) / 255.0
    screened = (1.0 - (1.0 - base_n) * (1.0 - light)) * 255.0
    out = np.maximum(screened, emitter * 0.96)
    return np.clip(out, 0, 255).astype(np.uint8)


def composite_pure_fixture_map(
    py: np.ndarray,
    px: np.ndarray,
    owner: np.ndarray,
    weight: np.ndarray,
    fixture_rgb: np.ndarray,
    H: int,
    W: int,
    crop_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Render the Museum-of-Light Pure Architectural Fixture-Map Preview (exact 3S submission aesthetic):
    discrete fixture mask emission -> density-calibrated two-pass optical lens bloom (tight + wide)
    -> 0.85 gamma lift on deep #050512 obsidian midnight, cropped cleanly around the active facade."""
    obsidian = np.array([0x05, 0x05, 0x12], dtype=np.float32) / 255.0
    if len(py) == 0 or len(fixture_rgb) == 0:
        blank = np.broadcast_to((obsidian * 255.0).astype(np.uint8), (H, W, 3)).copy()
        if crop_box is not None:
            x0, y0, x1, y1 = crop_box
            blank = blank[y0:y1, x0:x1]
        ch, cw = blank.shape[:2]
        return blank[:max(2, ch // 2 * 2), :max(2, cw // 2 * 2)]

    # HDR field in [0 .. 2.2]: excess > 1.0 creates incandescent white-gold star/bloom cores
    hdr = np.clip(fixture_rgb / 255.0, 0.0, 2.2)
    excess = np.maximum(0.0, hdr.max(axis=1, keepdims=True) - 1.0)
    core_hdr = np.clip(hdr + excess * 0.65, 0.0, 1.0)

    lit = np.zeros((H, W, 3), dtype=np.float32)
    lit[py, px] = core_hdr[owner] * weight[:, None]

    # Adaptive density calibration: keep optical lens bloom rich on sparse rigs without clouding dense 42-fin matrices!
    bbox_area = max(float((px.max() - px.min() + 1) * (py.max() - py.min() + 1)), 1000.0)
    fill_density = float(weight.sum()) / bbox_area
    bloom_gain = float(np.clip(0.095 / max(fill_density, 0.032), 0.85, 2.35))

    bloom_raw = np.zeros((H, W, 3), dtype=np.float32)
    bloom_raw[py, px] = np.clip(hdr[owner], 0.0, 1.35) * weight[:, None]
    im_b = Image.fromarray(np.clip(bloom_raw * 215.0, 0.0, 255.0).astype(np.uint8))
    r_tight = float(np.clip(H / 235.0, 5.2, 8.0))
    r_wide = float(np.clip(H / 54.0, 22.0, 34.0))
    tight = np.asarray(im_b.filter(ImageFilter.GaussianBlur(radius=r_tight)), dtype=np.float32) / 255.0
    wide = np.asarray(im_b.filter(ImageFilter.GaussianBlur(radius=r_wide)), dtype=np.float32) / 255.0

    out = np.clip(lit * 1.12 + tight * (1.05 * bloom_gain) + wide * (0.52 * bloom_gain), 0.0, 1.0) ** 0.85
    out = np.clip(out + obsidian[None, None, :] * (1.0 - out), 0.0, 1.0)
    rgb_u8 = (out * 255.0).astype(np.uint8)
    if crop_box is not None:
        x0, y0, x1, y1 = crop_box
        rgb_u8 = rgb_u8[y0:y1, x0:x1]
    ch, cw = rgb_u8.shape[:2]
    return rgb_u8[:max(2, ch // 2 * 2), :max(2, cw // 2 * 2)]


def render_rig_blueprint(
    night_base: np.ndarray,
    opt: FixtureRigOption,
    els: list[Element],
) -> Image.Image:
    """Render a crisp architectural fixture-layout preview colour-coded by architectural zone."""
    H, W = night_base.shape[:2]
    py, px, own, wt, halo = rasterize_oriented_splats(opt.xs, opt.ys, opt.lengths, opt.widths, opt.angles, H, W)
    palette = {
        "crown": np.array([255.0, 196.0, 55.0], np.float32),
        "tower": np.array([35.0, 195.0, 255.0], np.float32),
        "setback": np.array([80.0, 165.0, 255.0], np.float32),
        "band": np.array([255.0, 155.0, 45.0], np.float32),
        "fins": np.array([50.0, 225.0, 210.0], np.float32),
        "podium": np.array([255.0, 145.0, 65.0], np.float32),
        "other": np.array([95.0, 180.0, 255.0], np.float32),
    }
    kind_by_id = {e.id: e.kind for e in els}
    cols = np.zeros((opt.fixture_count, 3), np.float32)
    for k in range(opt.fixture_count):
        kind = kind_by_id.get(int(opt.elem_ids[k]), "tower")
        cols[k] = palette.get(kind, palette["tower"])
    frame = composite_fixture_frame(night_base, py, px, own, wt, halo, cols)
    return Image.fromarray(frame)
