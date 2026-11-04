"""F4: 360° Full-Building Choreography Projected onto Discrete Architectural Fixtures.

Models landmark-tower media-facade production (3S Lighting's standard):
- Show content runs seamlessly across the ENTIRE 360° building perimeter (`u_360 in [0, 1)`) and
  full elevation height (`v in [0, 1]`), blending curl-advected `muse-image-1.0` silk plates with
  procedural 3D helical ribbons, rising comet streams, geometric diamond lattices, and champagne shimmer.
- The photograph captures one viewpoint (1 flat elevation or 2 curved/angled faces spanning
  `wrap_fraction` of the 360° perimeter); only those visible discrete linear LED fixtures are lit,
  with realistic 3D grazing cosine shading and zero surface wash over curtain-wall glass.
- Media-server deliverables (`render_unmasked_canvas`) export the full unmasked 360° 4-elevation
  colour field edge-to-edge for Disguise / Resolume ingestion.
"""
from __future__ import annotations

import subprocess, sys
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.fixtures import (
    FixtureRigOption,
    composite_fixture_frame,
    composite_pure_fixture_map,
    propose_fixture_rigs,
    rasterize_oriented_splats,
)
from app.mapper import Element
from app.planner import Plan, Step
from app.visuals import (
    _hex_rgb,
    evaluate_360_shader,
    procedural_texture,
)

FIXTURES = 1200
CROSSFADE_S = 0.65


@dataclass
class Fx:
    """Fixtures of one element or unified element group, in render pixels."""
    ys: np.ndarray
    xs: np.ndarray
    y0: float
    y1: float
    x0: float
    x1: float
    vertical: bool = True
    u_360: np.ndarray | None = None
    v_norm: np.ndarray | None = None


@dataclass
class Rig:
    base: np.ndarray                       # dusk/night relighted photo, float32 HxWx3
    xs: np.ndarray                         # float32 (N,)
    ys: np.ndarray                         # float32 (N,)
    fx: dict[int, Fx]                      # element id -> Fx (kept for test compatibility)
    elem_membership: dict[int, np.ndarray] # element id -> int32 fixture indices
    py: np.ndarray
    px: np.ndarray
    owner: np.ndarray
    weight: np.ndarray
    halo_mask: np.ndarray
    n: int
    texture: np.ndarray | None = None
    option: FixtureRigOption | None = field(default=None)
    u_local: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    v_elev: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    view_cos: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    wrap_fraction: float = 0.45   # visible camera sector of 360° building (0.25 = 1 side, 0.45 = 2-side/curved, 0.67 = wide)
    u_offset: float = 0.25        # azimuth offset on 360° canvas
    spine_u: float = 0.475        # 360° azimuth of structural twist/fin spine
    crop_box: tuple[int, int, int, int] | None = None


def build_rig_from_option(
    night_base: np.ndarray,
    opt: FixtureRigOption,
    texture: np.ndarray | None = None,
    wrap_fraction: float | None = None,
    u_offset: float | None = None,
) -> Rig:
    """Build a fast O(1)-per-frame render Rig from a selected FixtureRigOption."""
    H, W = night_base.shape[:2]
    py, px, own, wt, halo = rasterize_oriented_splats(
        opt.xs, opt.ys, opt.lengths, opt.widths, opt.angles, H, W,
    )
    fx_map: dict[int, Fx] = {}
    for eid, idx in opt.elem_membership.items():
        if len(idx):
            ys_e, xs_e = opt.ys[idx], opt.xs[idx]
            fx_map[eid] = Fx(
                ys_e, xs_e,
                float(ys_e.min()), float(ys_e.max() + 1.0),
                float(xs_e.min()), float(xs_e.max() + 1.0),
                True,
            )

    if len(opt.u_local) == opt.fixture_count and opt.fixture_count > 0:
        u_loc = opt.u_local
        v_el = opt.v_elev
        v_cos = opt.view_cos
    elif opt.fixture_count > 0:
        x0, x1 = float(opt.xs.min()), float(opt.xs.max() + 1.0)
        y0, y1 = float(opt.ys.min()), float(opt.ys.max() + 1.0)
        u_loc = np.clip((opt.xs - x0) / max(x1 - x0, 1.0), 0.0, 1.0).astype(np.float32)
        v_el = np.clip((opt.ys - y0) / max(y1 - y0, 1.0), 0.0, 1.0).astype(np.float32)
        v_cos = np.ones(opt.fixture_count, dtype=np.float32)
    else:
        u_loc = np.empty(0, np.float32)
        v_el = np.empty(0, np.float32)
        v_cos = np.empty(0, np.float32)

    wf = float(wrap_fraction) if wrap_fraction is not None else float(getattr(opt, "default_wrap", 0.45))
    uo = float(u_offset) if u_offset is not None else float(getattr(opt, "default_u_offset", 0.25))
    local_spine = float(getattr(opt, "spine_u", 0.50))
    spine_360 = float((uo + local_spine * wf) % 1.0)
    tex = texture if texture is not None else procedural_texture(["#3A7BD5", "#B58FD6", "#E6E9F0"])
    return Rig(
        base=night_base,
        xs=opt.xs,
        ys=opt.ys,
        fx=fx_map,
        elem_membership=opt.elem_membership,
        py=py,
        px=px,
        owner=own,
        weight=wt,
        halo_mask=halo,
        n=opt.fixture_count,
        texture=tex,
        option=opt,
        u_local=u_loc,
        v_elev=v_el,
        view_cos=v_cos,
        wrap_fraction=wf,
        u_offset=uo,
        spine_u=spine_360,
        crop_box=getattr(opt, "crop_box", None),
    )


def build_rig(
    img: Image.Image,
    els: list[Element],
    scale: float | None = None,
    count: int = FIXTURES,
    survey=None,
    facets=None,
    tower=None,
) -> Rig:
    """Build a fixture Rig from image + elements (supports explicit scale/count for unit tests,
    or full architectural mullion/survey extraction for real building photos)."""
    solids = [e for e in els if e.mask is not None]
    if scale is not None and img.width <= 128 and img.height <= 128:
        W, H = max(2, int(img.width * scale)), max(2, int(img.height * scale))
        base = np.asarray(img.convert("RGB").resize((W, H), Image.LANCZOS), np.float32) * 0.35
        masks = {e.id: np.asarray(e.mask.resize((W, H), Image.NEAREST)) > 127 for e in solids}
        union = np.zeros((H, W), bool)
        for m in masks.values():
            union |= m
        step = max(2.0, float(np.sqrt(max(union.sum(), 1) / max(count, 1))))
        gx, gy = np.meshgrid(np.arange(step / 2, W, step), np.arange(step / 2, H, step))
        gx, gy = gx.ravel().astype(np.float32), gy.ravel().astype(np.float32)
        keep = union[np.clip(gy.astype(int), 0, H - 1), np.clip(gx.astype(int), 0, W - 1)]
        xs, ys = gx[keep], gy[keep]
        memb: dict[int, np.ndarray] = {}
        eids = np.zeros(len(xs), np.int32)
        for eid, m in masks.items():
            hit = np.where(m[np.clip(ys.astype(int), 0, H - 1), np.clip(xs.astype(int), 0, W - 1)])[0].astype(np.int32)
            if len(hit):
                memb[eid] = hit
                eids[hit] = eid
        opt = FixtureRigOption(
            id="synthetic", name="Synthetic", subtitle="", recommended=True,
            fixture_type="Linear", fixture_count=len(xs), pitch_desc="", est_power_kw=0.1,
            zones_used=[], xs=xs, ys=ys,
            lengths=np.full(len(xs), max(2.0, step * 0.6), np.float32),
            widths=np.full(len(xs), 1.5, np.float32),
            angles=np.full(len(xs), np.pi / 2, np.float32),
            elem_ids=eids, elem_membership=memb, render_size=(W, H),
        )
        return build_rig_from_option(base, opt)

    opts, night_base, _ = propose_fixture_rigs(img, els, survey=survey)
    chosen = next((o for o in opts if o.recommended), opts[0])
    return build_rig_from_option(night_base, chosen)


def _clean_palette(colors: list[str]) -> np.ndarray:
    cols = [_hex_rgb(c) for c in colors if c.strip().upper() not in {"#000000", "#000", "BLACK"}]
    if not cols:
        cols = [_hex_rgb(colors[0]) if colors else np.array([0.0, 195.0, 255.0], np.float32)]
    return np.stack(cols, axis=0)


def _font(size: int):
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_img(text: str, f: Fx, shape: tuple[int, int]) -> np.ndarray:
    w, h = max(8.0, f.x1 - f.x0), max(8.0, f.y1 - f.y0)
    lines = text.strip().split() if len(text.strip()) > 6 and " " in text.strip() else [text.strip()]
    label = "\n".join(lines)
    size = max(12, int(min(h * 0.75, w * 0.85)))
    font = _font(size)
    while size > 10:
        font = _font(size)
        probe = ImageDraw.Draw(Image.new("L", (16, 16), 0))
        l, t, r, b = probe.multiline_textbbox((0, 0), label, font=font, align="center", spacing=6)
        if (r - l) <= w * 0.90 and (b - t) <= h * 0.72:
            break
        size -= 3
    img = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(img).multiline_text(
        ((f.x0 + f.x1) / 2.0, (f.y0 + f.y1) / 2.0),
        label, font=font, fill=255, anchor="mm", align="center", spacing=6,
    )
    return np.asarray(img, np.float32) / 255.0


def lit(
    step: Step,
    f: Fx,
    q: float,
    t: float,
    shape: tuple[int, int],
    texture: np.ndarray | None = None,
    spine_u: float = 0.475,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (alpha N, colour Nx3) strictly at discrete fixture coordinates in `f`."""
    ys, xs, n = f.ys, f.xs, len(f.ys)
    h, w = max(1.0, f.y1 - f.y0), max(1.0, f.x1 - f.x0)
    u = f.u_360 if f.u_360 is not None else np.clip((xs - f.x0) / w, 0.0, 1.0).astype(np.float32)
    v = f.v_norm if f.v_norm is not None else np.clip((ys - f.y0) / h, 0.0, 1.0).astype(np.float32)

    if step.motion == "text" and step.text:
        cols = _clean_palette(step.colors)
        col0 = np.broadcast_to(cols[0], (n, 3)).copy()
        mask_txt = _text_img(step.text, f, shape)
        hit = mask_txt[np.clip(ys.astype(int), 0, shape[0] - 1), np.clip(xs.astype(int), 0, shape[1] - 1)]
        _, silk_bg = evaluate_360_shader("aurora", step.colors, u, v, q, t, tex=texture, spine_u=spine_u)
        alpha = np.ones(n, dtype=np.float32)
        # "Emerge, do not overlay": calm 14% negative-space veil around crisp luminous letterforms
        m_w = np.clip((hit - 0.14) / 0.55, 0.0, 1.0)[:, None]
        col_mix = silk_bg * 0.14 * (1.0 - m_w) + (col0 * 1.22) * m_w
        return alpha, col_mix.astype(np.float32)

    return evaluate_360_shader(step.motion, step.colors, u, v, q, t, tex=texture, spine_u=spine_u)


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def _step_envelope(s: Step, t: float, total_dur: float) -> tuple[float, float]:
    """Compute seamless cross-dissolve weight (never dipping to black between adjacent acts!) and progress `q`."""
    half_cf = CROSSFADE_S * 0.5
    t_in0 = s.start_s - half_cf if s.start_s > 0.05 else 0.0
    t_in1 = s.start_s + half_cf if s.start_s > 0.05 else 0.0
    t_out0 = s.end_s - half_cf if s.end_s < total_dur - 0.05 else total_dur
    t_out1 = s.end_s + half_cf if s.end_s < total_dur - 0.05 else total_dur

    if t < t_in0 or t > t_out1:
        return 0.0, 0.0

    w_in = 1.0 if t_in1 <= t_in0 else _smoothstep((t - t_in0) / (t_in1 - t_in0))
    w_out = 1.0 if t_out1 <= t_out0 else 1.0 - _smoothstep((t - t_out0) / (t_out1 - t_out0))
    dur = max(1e-4, s.end_s - s.start_s)
    q = float(np.clip((t - s.start_s) / dur, 0.0, 1.0))
    return float(w_in * w_out), q


def fixture_colours(rig: Rig, plan: Plan, t: float) -> np.ndarray:
    """Compute RGB emission (Nx3) for every discrete visible fixture by sampling the 360° show at
    `u_360 = (rig.u_offset + rig.u_local * rig.wrap_fraction) % 1.0` with seamless cross-dissolves."""
    col = np.zeros((rig.n, 3), np.float32)
    if rig.n == 0:
        return col

    shape = rig.base.shape[:2]
    u_full = ((rig.u_offset + rig.u_local * rig.wrap_fraction) % 1.0).astype(np.float32)
    cov = np.zeros(rig.n, dtype=np.float32)

    for s in plan.steps:
        env, q = _step_envelope(s, t, plan.duration_s)
        if env <= 1e-3:
            continue
        idx_parts = [rig.elem_membership[eid] for eid in s.element_ids if eid in rig.elem_membership and len(rig.elem_membership[eid])]
        if not idx_parts:
            continue
        idx = np.unique(np.concatenate(idx_parts))
        ys_g, xs_g = rig.ys[idx], rig.xs[idx]
        y0_g, y1_g = float(ys_g.min()), float(ys_g.max() + 1.0)
        v_group = np.clip((ys_g - y0_g) / max(y1_g - y0_g, 1.0), 0.0, 1.0).astype(np.float32)

        group_fx = Fx(
            ys=ys_g,
            xs=xs_g,
            y0=y0_g,
            y1=y1_g,
            x0=float(xs_g.min()),
            x1=float(xs_g.max() + 1.0),
            vertical=True,
            u_360=u_full[idx],
            v_norm=v_group,
        )
        a, c = lit(s, group_fx, q, t, shape, texture=rig.texture, spine_u=rig.spine_u)
        w = np.clip(a * env, 0.0, 1.0)
        # Screen/additive-aware cross-dissolve so overlapping accent layers brighten rather than darken
        col[idx] = col[idx] * np.maximum(0.0, 1.0 - w[:, None]) + c * w[:, None]
        cov[idx] = np.clip(cov[idx] + w, 0.0, 1.0)

    # Full-building continuity: if a multi-zone show temporarily omits part of the tower in a middle act,
    # keep those active show zones breathing with soft 42% silk harmony instead of snapping to a dead-black cutout!
    all_plan_eids = {eid for st in plan.steps for eid in st.element_ids if eid in rig.elem_membership}
    if len(all_plan_eids) >= 2 and t >= 2.4:
        active_idx = np.unique(np.concatenate([rig.elem_membership[eid] for eid in all_plan_eids if len(rig.elem_membership[eid])]))
        under_lit = active_idx[cov[active_idx] < 0.85]
        if len(under_lit):
            ref_colors = plan.steps[0].colors
            _, amb_c = evaluate_360_shader("aurora", ref_colors, u_full[under_lit], rig.v_elev[under_lit], 0.5, t, tex=rig.texture, spine_u=rig.spine_u)
            fill_w = (1.0 - cov[under_lit])[:, None] * 0.45
            col[under_lit] = col[under_lit] + amb_c * fill_w

    # Apply subtle 3D grazing cosine shading so curved/angled edges wrap naturally into depth
    if len(rig.view_cos) == rig.n:
        col *= rig.view_cos[:, None]

    return np.clip(col, 0.0, 380.0)


def frame_at(rig: Rig, plan: Plan, t: float) -> np.ndarray:
    """Render one uint8 RGB frame strictly through the rig's discrete linear LED fixtures."""
    rgb = fixture_colours(rig, plan, t)
    return composite_fixture_frame(rig.base, rig.py, rig.px, rig.owner, rig.weight, rig.halo_mask, rgb)


def frame_fixture_map(rig: Rig, plan: Plan, t: float) -> np.ndarray:
    """Render one uint8 RGB frame of the Museum-of-Light Pure Architectural Fixture-Map Preview
    (two-pass optical lens bloom on #050512 obsidian midnight, cropped cleanly to the facade)."""
    H, W = rig.base.shape[:2]
    rgb = fixture_colours(rig, plan, t)
    return composite_pure_fixture_map(rig.py, rig.px, rig.owner, rig.weight, rgb, H, W, crop_box=rig.crop_box)


def render_fixture_map_video(rig: Rig, plan: Plan, out_path: str, fps: int = 24) -> tuple[int, int]:
    """Encode an H.264 Museum-of-Light Pure Architectural Fixture-Map Preview video (#050512 canvas)."""
    first = frame_fixture_map(rig, plan, 0.0)
    H_m, W_m = first.shape[:2]
    ff = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_m}x{H_m}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", out_path,
        ],
        stdin=subprocess.PIPE,
    )
    assert ff.stdin is not None
    n_frames = max(1, int(round(plan.duration_s * fps)))
    ff.stdin.write(first.tobytes())
    for i in range(1, n_frames):
        ff.stdin.write(frame_fixture_map(rig, plan, i / fps).tobytes())
    ff.stdin.close()
    ff.wait()
    return n_frames, rig.n


def render_from_rig(rig: Rig, plan: Plan, out_path: str, fps: int = 24) -> tuple[int, int]:
    """Encode an H.264 on-building show preview from a pre-built Rig."""
    H, W = rig.base.shape[:2]
    ff = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", out_path,
        ],
        stdin=subprocess.PIPE,
    )
    assert ff.stdin is not None
    n_frames = max(1, int(round(plan.duration_s * fps)))
    for i in range(n_frames):
        ff.stdin.write(frame_at(rig, plan, i / fps).tobytes())
    ff.stdin.close()
    ff.wait()
    return n_frames, rig.n


def export_uv_mask(rig: Rig) -> Image.Image:
    """Export the 2D white-on-black fixture UV pixel mask PNG for media-server mapping."""
    H, W = rig.base.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    if len(rig.py):
        mask[rig.py, rig.px] = np.clip(rig.weight * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(mask, mode="L")


def _stitched_uv_axes(opt: FixtureRigOption | None, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Build 1D `(u_axis, v_axis)` vectors aligned to the 4 stitched elevations when `opt.panel_spans` is set."""
    if opt is None or len(getattr(opt, "panel_spans", [])) != 4 or opt.render_size[0] <= 0:
        return (
            np.linspace(0.0, 1.0, width, endpoint=False, dtype=np.float32),
            np.linspace(0.0, 1.0, height, dtype=np.float32),
        )
    W_r, H_r = opt.render_size
    spans = opt.panel_spans
    seams = [
        0.0,
        0.5 * (spans[0][1] + spans[1][0]),
        0.5 * (spans[1][1] + spans[2][0]),
        0.5 * (spans[2][1] + spans[3][0]),
        float(W_r),
    ]
    xs_q = np.linspace(0.0, float(W_r), width, endpoint=False, dtype=np.float32)
    u_axis = np.interp(xs_q, seams, [0.0, 0.25, 0.50, 0.75, 1.0]).astype(np.float32) % 1.0

    if opt.elev_y_span is not None:
        yc, yb = opt.elev_y_span
        ys_q = np.linspace(0.0, float(H_r), height, dtype=np.float32)
        v_axis = np.interp(ys_q, [0.0, float(yc), float(yb), float(H_r)], [0.0, 0.03, 0.97, 1.0]).astype(np.float32)
    else:
        v_axis = np.linspace(0.0, 1.0, height, dtype=np.float32)
    return u_axis, v_axis


def canvas_360_frame(
    plan: Plan,
    texture: np.ndarray | None,
    t: float,
    width: int = 1024,
    height: int = 576,
    u_axis: np.ndarray | None = None,
    v_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Render one frame of the UNMASKED 360° All-Sides Media Server Deliverable (`u in [0, 1)` across
    all 4 building elevations North · East · South · West, `v in [0, 1]` crown-to-base)."""
    if u_axis is None:
        u_axis = np.linspace(0.0, 1.0, width, endpoint=False, dtype=np.float32)
    if v_axis is None:
        v_axis = np.linspace(0.0, 1.0, height, dtype=np.float32)
    v_grid, u_grid = np.meshgrid(v_axis, u_axis, indexing="ij")
    acc = np.zeros((height, width, 3), dtype=np.float32)
    cov = np.zeros((height, width), dtype=np.float32)

    # Sort steps largest element-count first so full-facade base acts render before smaller crown accents
    ordered_steps = sorted(plan.steps, key=lambda s: len(s.element_ids), reverse=True)
    max_eids = max((len(s.element_ids) for s in plan.steps), default=1)

    for s in ordered_steps:
        env, q = _step_envelope(s, t, plan.duration_s)
        if env <= 1e-3:
            continue
        motion_use = "aurora" if s.motion == "text" else s.motion
        a, c = evaluate_360_shader(motion_use, s.colors, u_grid, v_grid, q, t, tex=texture)
        # Smaller accent steps (e.g. crown-only shimmer) map to the top crown band of the 360° canvas!
        if len(s.element_ids) < max(2, int(0.5 * max_eids)):
            crown_mask = np.exp(-((v_grid) / 0.22) ** 2).astype(np.float32)
            a = a * crown_mask
        w = np.clip(a * env, 0.0, 1.0)
        acc = acc * np.maximum(0.0, 1.0 - w[..., None]) + c * w[..., None]
        cov = np.clip(cov + w, 0.0, 1.0)

    if (cov < 0.95).any() and plan.steps:
        _, fallback = evaluate_360_shader("aurora", plan.steps[0].colors, u_grid, v_grid, 0.5, t, tex=texture)
        acc = acc + fallback * np.maximum(0.0, 1.0 - cov)[..., None]

    return np.clip(acc, 0.0, 255.0).astype(np.uint8)


def render_360_overview_still(
    rig: Rig,
    plan: Plan,
    t: float,
    width: int = 1280,
    height: int = 640,
) -> Image.Image:
    """Render an annotated 360° 4-Elevation Master Canvas overview highlighting the Camera Viewport slice
    (or all 4 stitched elevations when in 4-Side Stitched Full-Building mode)."""
    is_stitched = rig.option is not None and len(getattr(rig.option, "panel_spans", [])) == 4
    if is_stitched and rig.option is not None:
        W_r, H_r = rig.option.render_size
        scale = min(1.0, 1360.0 / max(W_r, 1))
        width = max(2, int(round(W_r * scale)) // 2 * 2)
        height = max(2, int(round(H_r * scale)) // 2 * 2)

    u_ax, v_ax = _stitched_uv_axes(rig.option, width, height)
    raw = canvas_360_frame(plan, rig.texture, t, width=width, height=height, u_axis=u_ax, v_axis=v_ax)
    im = Image.fromarray(raw).convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_s = _font(14 if is_stitched else 15)
    font_b = _font(16 if is_stitched else 17)

    labels = [
        "NORTH · SIDE 1 (0°–90°)" if is_stitched else "ELEVATION A (0°–90°)",
        "EAST · SIDE 2 (90°–180°)" if is_stitched else "ELEVATION B (90°–180°)",
        "SOUTH · SIDE 3 (180°–270°)" if is_stitched else "ELEVATION C (180°–270°)",
        "WEST · SIDE 4 (270°–360°)" if is_stitched else "ELEVATION D (270°–360°)",
    ]
    if is_stitched and rig.option is not None:
        W_r = rig.option.render_size[0]
        spans = rig.option.panel_spans
        divs = [
            0,
            int(round(0.5 * (spans[0][1] + spans[1][0]) * width / W_r)),
            int(round(0.5 * (spans[1][1] + spans[2][0]) * width / W_r)),
            int(round(0.5 * (spans[2][1] + spans[3][0]) * width / W_r)),
            width,
        ]
    else:
        divs = [int(round(k * width / 4)) for k in range(5)]

    for k in range(4):
        x_div = divs[k]
        if k > 0:
            draw.line([(x_div, 0), (x_div, height)], fill=(255, 255, 255, 85), width=1)
        box_w = min(232, max(140, divs[k + 1] - x_div - 20))
        draw.rounded_rectangle((x_div + 10, 12, x_div + 10 + box_w, 38), radius=6, fill=(8, 12, 24, 185))
        draw.text((x_div + 18, 25), labels[k], fill=(215, 228, 255, 240), font=font_s, anchor="lm")

    if rig.wrap_fraction >= 0.96 or is_stitched:
        draw.rectangle((3, 3, width - 3, height - 3), outline=(52, 211, 153, 245), width=3)
        badge = "STITCHED ALL 4 SIDES · FULL 360° VIDEO FACADE MAP"
        bw = len(badge) * 9 + 28
        bx = max(8, width // 2 - bw // 2)
        draw.rounded_rectangle((bx, height - 46, bx + bw, height - 14), radius=7, fill=(6, 20, 18, 230), outline=(52, 211, 153, 255), width=2)
        draw.text((bx + bw // 2, height - 30), badge, fill=(167, 243, 208, 255), font=font_b, anchor="mm")
    else:
        x0 = int(round((rig.u_offset % 1.0) * width))
        span_px = int(round(rig.wrap_fraction * width))
        x1 = min(width - 2, x0 + span_px)
        if x0 > 0:
            draw.rectangle((0, 0, x0, height), fill=(5, 8, 18, 95))
        if x1 < width:
            draw.rectangle((x1, 0, width, height), fill=(5, 8, 18, 95))

        draw.rectangle((x0, 3, x1, height - 3), outline=(56, 224, 255, 245), width=3)
        deg = int(round(rig.wrap_fraction * 360))
        badge = f"CAMERA PHOTO VIEWPORT ({deg}° VISIBLE SIDE)"
        bw = len(badge) * 9 + 24
        bx = max(x0 + 8, min((x0 + x1) // 2 - bw // 2, width - bw - 8))
        draw.rounded_rectangle((bx, height - 46, bx + bw, height - 14), radius=7, fill=(8, 16, 32, 225), outline=(56, 224, 255, 255), width=2)
        draw.text((bx + bw // 2, height - 30), badge, fill=(130, 240, 255, 255), font=font_b, anchor="mm")

    return Image.alpha_composite(im, overlay).convert("RGB")


def render_unmasked_canvas(rig: Rig, plan: Plan, out_path: str, fps: int = 24) -> None:
    """Export the clean, full-bleed UNMASKED 360° All-Sides Media Server Deliverable H.264 video.
    In 4-Side Stitched mode, renders spatially registered to the stitched 4-elevation UV mask layout!"""
    is_stitched = rig.option is not None and len(getattr(rig.option, "panel_spans", [])) == 4
    if is_stitched and rig.option is not None:
        W_r, H_r = rig.option.render_size
        scale = min(1.0, 1120.0 / max(W_r, 1))
        W_c = max(2, int(round(W_r * scale)) // 2 * 2)
        H_c = max(2, int(round(H_r * scale)) // 2 * 2)
    else:
        W_c, H_c = 960, 540

    u_ax, v_ax = _stitched_uv_axes(rig.option, W_c, H_c)
    ff = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W_c}x{H_c}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", out_path,
        ],
        stdin=subprocess.PIPE,
    )
    assert ff.stdin is not None
    n_frames = max(1, int(round(plan.duration_s * fps)))
    for i in range(n_frames):
        ff.stdin.write(canvas_360_frame(plan, rig.texture, i / fps, width=W_c, height=H_c, u_axis=u_ax, v_axis=v_ax).tobytes())
    ff.stdin.close()
    ff.wait()


def render(
    img: Image.Image,
    els: list[Element],
    plan: Plan,
    out_path: str,
    fps: int = 24,
    count: int = FIXTURES,
    survey=None,
    facets=None,
    tower=None,
) -> tuple[int, int]:
    """Write an H.264 preview. Returns (frames, fixture count)."""
    rig = build_rig(img, els, count=count, survey=survey, facets=facets, tower=tower)
    return render_from_rig(rig, plan, out_path, fps=fps)


if __name__ == "__main__":  # python -m app.motion photo.png "brief" out.mp4 [duration_s] [survey_mask.png]
    import asyncio, os, pickle
    from app.mapper import map_facade
    from app.planner import make_plan
    im = Image.open(sys.argv[1])
    dur = float(sys.argv[4]) if len(sys.argv) > 4 else 16
    surv = None
    if len(sys.argv) > 5:
        from app.survey import load_survey
        surv = load_survey(sys.argv[5], cache=sys.argv[5] + ".npz")
    cache = sys.argv[3] + ".pkl"

    async def go():
        els = await map_facade(im)
        opts, nb, _ = propose_fixture_rigs(im, els, survey=surv)
        chosen = next((o for o in opts if o.recommended), opts[0])
        fcounts = {eid: len(idx) for eid, idx in chosen.elem_membership.items()}
        plan = await make_plan(im, els, sys.argv[2], dur, elem_fixture_counts=fcounts)
        return els, plan

    els, plan = pickle.load(open(cache, "rb")) if os.path.exists(cache) else asyncio.run(go())
    pickle.dump((els, plan), open(cache, "wb"))
    print("frames, fixtures:", render(im, els, plan, sys.argv[3], survey=surv))
