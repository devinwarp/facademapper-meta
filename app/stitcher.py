"""Architectural 4-Side Facade Map Stitcher (360° North · East · South · West).

When a user wants to author and export a show for the ENTIRE video facade map, they upload
all 4 elevations of the building (North 0°–90°, East 90°–180°, South 180°–270°, West 270°–360°).

Unlike consumer panorama stitchers (which fail on 4 non-overlapping opposite building faces),
this module performs true architectural elevation stitching:
1. Segments all 4 sides concurrently via SAM 3.1 + Muse Spark (`map_facade`).
2. Normalizes crown-to-podium scale (`y_crown` .. `y_base`) across all 4 photos while preserving
   each elevation's true aspect ratio.
3. Extracts discrete mullion, fin, crown, and slab-edge LED fixtures natively on each side.
4. Harmonizes twilight sky & facade exposure across seam joints and maps each side's fixtures into
   its exact 90° quadrant (`u_360 in [k*0.25, (k+1)*0.25)`) of one unified 360° Facade Rig.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
from scipy import ndimage as ndi

from app.fixtures import (
    FixtureRigOption,
    _assign_membership,
    propose_fixture_rigs,
)
from app.mapper import Element, map_facade

SIDE_HEADERS = [
    "SIDE 1 · NORTH (0°–90°)",
    "SIDE 2 · EAST (90°–180°)",
    "SIDE 3 · SOUTH (180°–270°)",
    "SIDE 4 · WEST (270°–360°)",
]


@dataclass
class StitchedFacadeResult:
    stitched_original: Image.Image
    night_base: np.ndarray
    building_mask: np.ndarray
    elements: list[Element]
    rig_options: list[FixtureRigOption]
    panel_spans: list[tuple[int, int]]       # [(bx0, bx1), ...] building x-spans per side
    panel_dividers: list[int] | None = None
    elev_y_span: tuple[int, int] = (74, 846)
    side_counts: list[int] | None = None     # recommended rig fixture count per side


def _font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"):
        try:
            return ImageFont.truetype(p, px)
        except OSError:
            pass
    return ImageFont.load_default()


def _place_with_edge_pad(src: np.ndarray, dst_h: int, dst_w: int, paste_x: int, paste_y: int) -> np.ndarray:
    """Place `src` onto a `(dst_h, dst_w)` canvas at `(paste_x, paste_y)`, filling any out-of-bounds
    margins via smooth edge clamping instead of black bars."""
    sh, sw = src.shape[:2]
    yy = np.clip(np.arange(dst_h) - paste_y, 0, sh - 1)
    xx = np.clip(np.arange(dst_w) - paste_x, 0, sw - 1)
    canvas = src[np.ix_(yy, xx)].copy()
    # Gently blur only any clamped border strips so padded sky/street edges look clean
    in_y = (np.arange(dst_h) >= paste_y) & (np.arange(dst_h) < paste_y + sh)
    in_x = (np.arange(dst_w) >= paste_x) & (np.arange(dst_w) < paste_x + sw)
    inside = np.outer(in_y, in_x)
    if not inside.all():
        if src.ndim == 3:
            blurred = np.stack([ndi.gaussian_filter(canvas[:, :, c].astype(np.float32), sigma=9.0) for c in range(3)], axis=2)
            canvas[~inside] = np.clip(blurred[~inside], 0, 255).astype(src.dtype)
        else:
            canvas[~inside] = 0
    return canvas


def _align_elevation_panel(
    img: Image.Image,
    els: list[Element],
    target_tower_h: int,
    y_crown: int,
    panel_h: int,
    max_panel_w: int = 580,
) -> tuple[Image.Image, list[Element], np.ndarray, tuple[int, int]]:
    """Scale & align one building side so its crown sits at `y_crown` and its base sits at
    `y_crown + target_tower_h`, centered horizontally in an even-width panel."""
    rgb = np.asarray(img.convert("RGB"))
    ih, iw = rgb.shape[:2]

    bmask = np.zeros((ih, iw), bool)
    for e in els:
        if e.mask is not None:
            bmask |= np.asarray(e.mask.resize((iw, ih), Image.NEAREST)) > 127
    if bmask.any():
        bmask = ndi.binary_fill_holes(bmask)
        ys, xs = np.where(bmask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
    else:
        y0, y1 = int(0.10 * ih), int(0.90 * ih)
        x0, x1 = int(0.20 * iw), int(0.80 * iw)
        bmask[y0:y1, x0:x1] = True

    bh = max(24, y1 - y0)
    bw = max(24, x1 - x0)

    scale = float(target_tower_h) / float(bh)
    # Prevent ultra-wide podiums/wings from overflowing max_panel_w
    if bw * scale > max_panel_w - 36:
        scale = float(max_panel_w - 36) / float(bw)

    new_w = max(32, int(round(iw * scale)))
    new_h = max(32, int(round(ih * scale)))
    scaled_rgb = np.asarray(img.convert("RGB").resize((new_w, new_h), Image.LANCZOS))

    bw_s = bw * scale
    panel_w = int(np.clip(round(bw_s * 1.40), 340, max_panel_w)) // 2 * 2

    cx_s = 0.5 * (x0 + x1) * scale
    y0_s = y0 * scale
    paste_x = int(round(panel_w * 0.5 - cx_s))
    paste_y = int(round(y_crown - y0_s))

    panel_rgb = _place_with_edge_pad(scaled_rgb, panel_h, panel_w, paste_x, paste_y)
    panel_img = Image.fromarray(panel_rgb)

    aligned_els: list[Element] = []
    panel_bmask = np.zeros((panel_h, panel_w), bool)
    panel_void = np.zeros((panel_h, panel_w), bool)
    void_audited = any(bool(getattr(e, "void_audited", False)) for e in els)
    has_void = False
    for e in els:
        vm = getattr(e, "void_mask", None)
        if vm is not None:
            vm_s = np.asarray(vm.resize((new_w, new_h), Image.NEAREST)) > 127
            vm_p = _place_with_edge_pad(vm_s.astype(np.uint8) * 255, panel_h, panel_w, paste_x, paste_y) > 127
            if vm_p.any():
                panel_void |= vm_p
                has_void = True
        if e.mask is None:
            continue
        m_s = np.asarray(e.mask.resize((new_w, new_h), Image.NEAREST)) > 127
        m_p = _place_with_edge_pad(m_s.astype(np.uint8) * 255, panel_h, panel_w, paste_x, paste_y) > 127
        if not m_p.any():
            continue
        panel_bmask |= m_p
        eys, exs = np.where(m_p)
        bx = (int(exs.min()), int(eys.min()), int(exs.max() + 1), int(eys.max() + 1))
        aligned_els.append(Element(e.id, e.label, e.kind, bx, Image.fromarray(m_p.astype(np.uint8) * 255)))

    if not aligned_els:
        panel_bmask[y_crown:y_crown + target_tower_h, panel_w // 5:4 * panel_w // 5] = True
        aligned_els.append(Element(
            1, "facade elevation", "tower",
            (panel_w // 5, y_crown, 4 * panel_w // 5, y_crown + target_tower_h),
            Image.fromarray(panel_bmask.astype(np.uint8) * 255),
        ))

    aligned_els[0].void_audited = void_audited
    if has_void:
        aligned_els[0].void_mask = Image.fromarray((panel_void & panel_bmask).astype(np.uint8) * 255)

    bx_xs = np.where(panel_bmask)[1]
    bx_span = (int(bx_xs.min()), int(bx_xs.max())) if len(bx_xs) else (int(panel_w * 0.2), int(panel_w * 0.8))
    return panel_img, aligned_els, panel_bmask, bx_span


def _harmonize_stitched_night_base(
    night_panels: list[np.ndarray],
    bmask_panels: list[np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    """Join 4 night-relit elevation panels into one cohesive 360° architectural sheet, matching
    background sky tone across sides and gently feathering non-building seam joints."""
    H = night_panels[0].shape[0]
    yy = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None]

    # Compute target median twilight sky gradient across all 4 panels so day + night mixes harmonize
    sky_medians = []
    for nb, bm in zip(night_panels, bmask_panels):
        sky = (~bm) & (yy < 0.45)
        if sky.sum() > 80:
            sky_medians.append(np.median(nb[sky], axis=0))
    target_sky = np.median(np.stack(sky_medians, axis=0), axis=0) if sky_medians else np.array([18.0, 26.0, 48.0], np.float32)
    target_sky = np.clip(target_sky, [12.0, 18.0, 34.0], [34.0, 46.0, 74.0])

    norm_panels: list[np.ndarray] = []
    for nb, bm in zip(night_panels, bmask_panels):
        p = nb.copy()
        sky = (~bm) & (yy < 0.45)
        if sky.sum() > 80:
            cur_sky = np.maximum(np.median(p[sky], axis=0), 6.0)
            gain = np.clip(target_sky / cur_sky, 0.55, 1.65)
            bg_weight = (1.0 - ndi.gaussian_filter(bm.astype(np.float32), sigma=2.0))[:, :, None]
            p = p * (1.0 - 0.72 * bg_weight) + (p * gain[None, None, :]) * (0.72 * bg_weight)
        norm_panels.append(np.clip(p, 0.0, 255.0))

    stitched = np.concatenate(norm_panels, axis=1)
    full_bm = np.concatenate(bmask_panels, axis=1)

    # Feather 14px around each internal panel seam strictly in the background sky/ground (never blur building!)
    x_offsets: list[int] = []
    acc_w = 0
    for idx, p in enumerate(norm_panels):
        x_offsets.append(acc_w)
        acc_w += p.shape[1]
        if idx < len(norm_panels) - 1:
            sx = acc_w
            w0, w1 = max(0, sx - 16), min(stitched.shape[1], sx + 16)
            band = stitched[:, w0:w1].copy()
            smooth_band = np.stack([ndi.gaussian_filter1d(band[:, :, c], sigma=7.0, axis=1) for c in range(3)], axis=2)
            bg_mask = ~ndi.binary_dilation(full_bm[:, w0:w1], iterations=2)
            stitched[:, w0:w1][bg_mask] = smooth_band[bg_mask]
            # Subtle vertical architectural elevation hairline in top header band
            stitched[:48, max(0, sx - 1):min(stitched.shape[1], sx + 1)] = [36.0, 54.0, 88.0]

    # Stamp crisp compass elevation pills along the top margin (above crown at y=74)
    pil_nb = Image.fromarray(stitched.clip(0, 255).astype(np.uint8)).convert("RGBA")
    ov = Image.new("RGBA", pil_nb.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(ov)
    font = _font(13)
    for k, x0 in enumerate(x_offsets):
        pw = norm_panels[k].shape[1]
        label = SIDE_HEADERS[k] if k < len(SIDE_HEADERS) else f"SIDE {k + 1}"
        cx = x0 + pw // 2
        bw = len(label) * 7 + 20
        draw.rounded_rectangle((cx - bw // 2, 12, cx + bw // 2, 36), radius=6, fill=(8, 13, 26, 195), outline=(56, 189, 248, 120), width=1)
        draw.text((cx, 24), label, fill=(205, 224, 252, 235), font=font, anchor="mm")

    final_nb = np.asarray(Image.alpha_composite(pil_nb, ov).convert("RGB"), dtype=np.float32)
    return final_nb, x_offsets


def _build_unified_360_elements(
    panel_els_list: list[list[Element]],
    bmask_panels: list[np.ndarray],
    x_offsets: list[int],
    W_total: int,
    H_total: int,
    y_crown: int,
    y_base: int,
) -> list[Element]:
    """Group per-side elements into synchronized 360° architectural tiers across all 4 elevations
    so choreographies sweep, climb, and shimmer seamlessly around the entire building."""
    crown_m = np.zeros((H_total, W_total), bool)
    upper_m = np.zeros((H_total, W_total), bool)
    mid_m = np.zeros((H_total, W_total), bool)
    base_m = np.zeros((H_total, W_total), bool)

    span_h = max(40, y_base - y_crown)
    y_cut_crown = y_crown + int(0.14 * span_h)
    y_cut_upper = y_crown + int(0.48 * span_h)
    y_cut_mid = y_crown + int(0.78 * span_h)
    yy = np.arange(H_total)[:, None]

    for k, (els_k, bm_k, x_off) in enumerate(zip(panel_els_list, bmask_panels, x_offsets)):
        pw = bm_k.shape[1]
        has_crown = False
        has_band = False
        for e in els_k:
            if e.mask is None:
                continue
            em = np.asarray(e.mask) > 127
            if e.kind in {"crown", "spire"}:
                crown_m[:, x_off:x_off + pw] |= em
                has_crown = True
            elif e.kind in {"setback", "band"}:
                mid_m[:, x_off:x_off + pw] |= em
                has_band = True
            elif e.kind in {"podium", "arch"}:
                base_m[:, x_off:x_off + pw] |= em

        if not has_crown:
            crown_m[:, x_off:x_off + pw] |= bm_k & (yy <= y_cut_crown)
        upper_m[:, x_off:x_off + pw] |= bm_k & (yy > y_cut_crown) & (yy <= y_cut_upper)
        if not has_band:
            mid_m[:, x_off:x_off + pw] |= bm_k & (yy > y_cut_upper) & (yy <= y_cut_mid)
        else:
            mid_m[:, x_off:x_off + pw] |= bm_k & (yy > y_cut_upper) & (yy <= y_cut_mid)
        base_m[:, x_off:x_off + pw] |= bm_k & (yy > y_cut_mid)

    tiers = [
        (1, "crown & parapet (all 4 sides)", "crown", crown_m),
        (2, "upper facade shaft (all 4 sides)", "tower", upper_m),
        (3, "mid facade & setbacks (all 4 sides)", "setback", mid_m),
        (4, "lower shaft & podium (all 4 sides)", "podium", base_m),
    ]
    out: list[Element] = []
    for eid, label, kind, m in tiers:
        if not m.any():
            continue
        ys, xs = np.where(m)
        box = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
        out.append(Element(eid, label, kind, box, Image.fromarray(m.astype(np.uint8) * 255)))
    return out


async def stitch_four_elevations(
    side_images: list[Image.Image],
    map_fn: Callable[[Image.Image], Awaitable[list[Element]]] = map_facade,
) -> StitchedFacadeResult:
    """Segment, align, relight, and stitch 4 building elevation photographs (N · E · S · W) into
    one unified 360° Full-Building Facade Map & discrete LED Fixture Rig suite."""
    if len(side_images) != 4:
        raise ValueError(f"Expected 4 building side photos (North, East, South, West), got {len(side_images)}.")

    rgb_imgs = [ImageOps.exif_transpose(im).convert("RGB") for im in side_images]

    sem = asyncio.Semaphore(2)

    async def _map_one(im: Image.Image) -> list[Element]:
        async with sem:
            return await map_fn(im)

    raw_els_list = list(await asyncio.gather(*(_map_one(im) for im in rgb_imgs)))

    # Measure median building aspect ratio so both slender skyscrapers and wide low-rises fit cleanly
    ars: list[float] = []
    for im, els in zip(rgb_imgs, raw_els_list):
        bm = np.zeros((im.height, im.width), bool)
        for e in els:
            if e.mask is not None:
                bm |= np.asarray(e.mask) > 127
        if bm.any():
            ys, xs = np.where(bm)
            ars.append(float(xs.max() - xs.min() + 1) / max(float(ys.max() - ys.min() + 1), 1.0))
        else:
            ars.append(float(im.width) / max(float(im.height), 1.0))

    ref_ar = float(np.percentile(ars, 75))
    target_tower_h = int(np.clip(round(440.0 / max(ref_ar, 0.42)), 460, 1024)) // 2 * 2
    y_crown = 74
    y_base = y_crown + target_tower_h
    panel_h = (y_base + 74) // 2 * 2

    panel_imgs: list[Image.Image] = []
    panel_els_list: list[list[Element]] = []
    bmask_panels: list[np.ndarray] = []
    local_bx_spans: list[tuple[int, int]] = []

    for im, els in zip(rgb_imgs, raw_els_list):
        p_img, p_els, p_bm, bx_s = _align_elevation_panel(
            im, els, target_tower_h=target_tower_h, y_crown=y_crown, panel_h=panel_h,
        )
        panel_imgs.append(p_img)
        panel_els_list.append(p_els)
        bmask_panels.append(p_bm)
        local_bx_spans.append(bx_s)

    # Extract native mullion/fin/ledge fixtures independently on each aligned elevation
    per_side_rigs: list[dict[str, FixtureRigOption]] = []
    night_panels: list[np.ndarray] = []
    for p_img, p_els in zip(panel_imgs, panel_els_list):
        opts_k, nb_k, _ = propose_fixture_rigs(p_img, p_els, exact_size=True)
        per_side_rigs.append({o.id: o for o in opts_k})
        night_panels.append(nb_k)

    night_base, x_offsets = _harmonize_stitched_night_base(night_panels, bmask_panels)
    H_total, W_total = night_base.shape[:2]
    full_bmask = np.concatenate(bmask_panels, axis=1)

    # Stitched daytime/original panorama with subtle compass headers & seam lines
    orig_concat = np.concatenate([np.asarray(p) for p in panel_imgs], axis=1).copy()
    for sx in x_offsets[1:]:
        orig_concat[:, max(0, sx - 1):min(W_total, sx + 1)] = [24, 38, 64]
    pil_orig = Image.fromarray(orig_concat).convert("RGBA")
    ov_o = Image.new("RGBA", pil_orig.size, (0, 0, 0, 0))
    dr_o = ImageDraw.Draw(ov_o)
    font = _font(13)
    for k, x0 in enumerate(x_offsets):
        pw = panel_imgs[k].width
        lbl = SIDE_HEADERS[k]
        cx = x0 + pw // 2
        bw = len(lbl) * 7 + 20
        dr_o.rounded_rectangle((cx - bw // 2, 12, cx + bw // 2, 36), radius=6, fill=(8, 13, 26, 205), outline=(56, 189, 248, 180), width=1)
        dr_o.text((cx, 24), lbl, fill=(230, 240, 255, 255), font=font, anchor="mm")
    stitched_orig = Image.alpha_composite(pil_orig, ov_o).convert("RGB")

    unified_els = _build_unified_360_elements(
        panel_els_list, bmask_panels, x_offsets, W_total, H_total, y_crown, y_base,
    )
    unified_masks = {e.id: np.asarray(e.mask) > 127 for e in unified_els if e.mask is not None}
    solids_order = [e.id for e in unified_els]

    panel_spans = [(x_offsets[k] + local_bx_spans[k][0], x_offsets[k] + local_bx_spans[k][1]) for k in range(4)]

    # Merge each rig family across all 4 elevations into a seamless 360° [0, 1) rig!
    rig_specs = [
        (
            "mullion_matrix",
            "360° Full-Building Mullion & Fin Matrix (All 4 Sides)",
            "Continuous linear RGBW bars tracked across North · East · South · West mullions & fins",
            True,
        ),
        (
            "architectural_accent",
            "360° Crown, Setback & Structural Spine Rig (All 4 Sides)",
            "Synchronized 4-side crown beacons, horizontal setback ledges & corner structural spines",
            False,
        ),
        (
            "horizontal_tiers",
            "360° Floor-Slab Spandrel Ribbons (All 4 Sides)",
            "Aligned horizontal floor-slab LED grazing bands wrapping all 4 elevations",
            False,
        ),
    ]
    if sum(1 for d in per_side_rigs if "detected_night_rig" in d) >= 2:
        rig_specs.insert(0, (
            "detected_night_rig",
            "360° Detected Existing Facade Rig (All 4 Sides)",
            "As-photographed glowing LED nodes stitched across all 4 building elevations",
            True,
        ))
        rig_specs[1] = (rig_specs[1][0], rig_specs[1][1], rig_specs[1][2], False)

    merged_options: list[FixtureRigOption] = []
    rec_side_counts: list[int] = [0, 0, 0, 0]

    for fam_id, fam_name, fam_sub, is_rec in rig_specs:
        xs_parts, ys_parts, ls_parts, ws_parts, ang_parts = [], [], [], [], []
        uloc_parts, velev_parts, vcos_parts = [], [], []
        side_cnts = []
        sample_ft = "1200mm Direct-View Mullion Linear Pixel Bar"

        for k in range(4):
            opt_k = per_side_rigs[k].get(fam_id) or per_side_rigs[k].get("mullion_matrix") or next(iter(per_side_rigs[k].values()), None)
            if opt_k is None or opt_k.fixture_count == 0:
                side_cnts.append(0)
                continue
            sample_ft = opt_k.fixture_type
            x_off = float(x_offsets[k])
            xs_k = opt_k.xs + x_off
            ys_k = opt_k.ys
            xs_parts.append(xs_k)
            ys_parts.append(ys_k)
            ls_parts.append(opt_k.lengths)
            ws_parts.append(opt_k.widths)
            ang_parts.append(opt_k.angles)

            # Map side k's [0..1] azimuth cleanly into quadrant [k*0.25 .. (k+1)*0.25) of 360° wrap!
            uk = opt_k.u_local if len(opt_k.u_local) == opt_k.fixture_count else np.linspace(0.0, 1.0, opt_k.fixture_count, dtype=np.float32)
            uloc_parts.append((k * 0.25 + 0.25 * np.clip(uk, 0.0, 0.999)).astype(np.float32))

            vk = np.clip((ys_k - y_crown) / max(float(y_base - y_crown), 1.0), 0.0, 1.0).astype(np.float32)
            velev_parts.append(vk)

            ck = opt_k.view_cos if len(opt_k.view_cos) == opt_k.fixture_count else np.ones(opt_k.fixture_count, dtype=np.float32)
            vcos_parts.append(ck)
            side_cnts.append(int(opt_k.fixture_count))

        if not xs_parts:
            continue

        xs_all = np.concatenate(xs_parts).astype(np.float32)
        ys_all = np.concatenate(ys_parts).astype(np.float32)
        eids_all, memb_all = _assign_membership(xs_all, ys_all, unified_masks, solids_order, H_total, W_total)
        zones_used = [e.label for e in unified_els if e.id in memb_all and len(memb_all[e.id]) > 8]
        per_side_str = " · ".join(f"S{k+1}:{c}" for k, c in enumerate(side_cnts))

        if is_rec:
            rec_side_counts = side_cnts

        merged_options.append(FixtureRigOption(
            id=fam_id,
            name=fam_name,
            subtitle=fam_sub,
            recommended=is_rec,
            fixture_type=sample_ft,
            fixture_count=len(xs_all),
            pitch_desc=f"4 Stitched Sides ({per_side_str})",
            est_power_kw=len(xs_all) * 0.018,
            zones_used=zones_used or ["All 4 Elevations"],
            xs=xs_all,
            ys=ys_all,
            lengths=np.concatenate(ls_parts).astype(np.float32),
            widths=np.concatenate(ws_parts).astype(np.float32),
            angles=np.concatenate(ang_parts).astype(np.float32),
            elem_ids=eids_all,
            elem_membership=memb_all,
            render_size=(W_total, H_total),
            u_local=np.concatenate(uloc_parts).astype(np.float32),
            v_elev=np.concatenate(velev_parts).astype(np.float32),
            view_cos=np.concatenate(vcos_parts).astype(np.float32),
            default_wrap=1.0,
            default_u_offset=0.0,
            panel_spans=panel_spans,
            elev_y_span=(y_crown, y_base),
            spine_u=0.375,
            crop_box=(0, max(0, y_crown - 36), W_total, min(H_total, y_base + 36)),
        ))

    return StitchedFacadeResult(
        stitched_original=stitched_orig,
        night_base=night_base,
        building_mask=full_bmask,
        elements=unified_els,
        rig_options=merged_options,
        panel_spans=panel_spans,
        elev_y_span=(y_crown, y_base),
        side_counts=rec_side_counts,
    )


def make_demo_four_sides(base_img: Image.Image) -> list[Image.Image]:
    """Generate 4 realistic N·E·S·W architectural elevation views from a single sample tower photo
    (varying perspective crop, helical orientation, and sun/dusk lighting) for 1-click demo testing."""
    im = base_img.convert("RGB")
    w, h = im.size
    # Side 1 (North): Direct reference elevation
    s1 = im.copy()
    # Side 2 (East): Mirrored helical twist + slightly warmer late-afternoon sun
    s2 = ImageEnhance.Color(ImageOps.mirror(im).crop((int(0.03 * w), 0, int(0.97 * w), h)).resize((w, h), Image.LANCZOS)).enhance(1.08)
    # Side 3 (South): Slightly tighter telephoto framing + cooler shadow elevation
    s3 = ImageEnhance.Brightness(im.crop((int(0.05 * w), int(0.02 * h), int(0.95 * w), int(0.99 * h))).resize((w, h), Image.LANCZOS)).enhance(0.88)
    # Side 4 (West): Opposite helical profile + golden-hour contrast
    s4 = ImageEnhance.Contrast(ImageOps.mirror(im).crop((int(0.02 * w), int(0.01 * h), int(0.98 * w), h)).resize((w, h), Image.LANCZOS)).enhance(1.06)
    return [s1, s2, s3, s4]
