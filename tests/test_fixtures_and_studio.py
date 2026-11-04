import asyncio
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from app.checker import apply_readability_fix, check_readability
from app.fixtures import propose_fixture_rigs
from app.main import app
from app.mapper import Element
from app.motion import build_rig_from_option, canvas_360_frame, export_uv_mask, fixture_colours, frame_at
from app.planner import Plan
from app.visuals import evaluate_360_shader, procedural_texture


def _synthetic_curtain_wall(w: int = 360, h: int = 540) -> tuple[Image.Image, list[Element]]:
    """Build a synthetic daytime facade with vertical mullions every 18px and floor slabs every 24px."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    # Daytime blue sky outside building
    arr[:, :] = [145, 185, 232]

    bx0, bx1, by0, by1 = 60, 300, 40, 500
    # Dark recessed curtain-wall glass bays inside building
    arr[by0:by1, bx0:bx1] = [42, 54, 70]
    # Bright vertical metallic mullions every 18px (width 2px) -> 16px wide glass bays between mullions!
    mullion_xs = list(range(bx0 + 9, bx1 - 6, 18))
    for mx in mullion_xs:
        arr[by0:by1, mx:mx + 2] = [195, 204, 216]
    # Horizontal spandrel slab lines every 24px
    for sy in range(by0 + 12, by1 - 8, 24):
        arr[sy:sy + 2, bx0:bx1] = [145, 156, 170]

    img = Image.fromarray(arr)
    m_crown = Image.new("L", (w, h), 0)
    m_crown.paste(255, (bx0, by0, bx1, 130))
    m_shaft = Image.new("L", (w, h), 0)
    m_shaft.paste(255, (bx0, 130, bx1, by1))

    els = [
        Element(1, "crown", "crown", (bx0, by0, bx1, 130), m_crown),
        Element(2, "main shaft", "tower", (bx0, 130, bx1, by1), m_shaft),
    ]
    return img, els


def test_propose_fixture_rigs_snaps_to_mullions_with_zero_glass_wash():
    img, els = _synthetic_curtain_wall()
    opts, night_base, _ = propose_fixture_rigs(img, els)
    opt_ids = [o.id for o in opts]
    assert {"mullion_matrix", "architectural_accent", "horizontal_tiers"}.issubset(opt_ids)

    opt_m = next(o for o in opts if o.id == "mullion_matrix")
    rig = build_rig_from_option(night_base, opt_m)

    plan = Plan.model_validate({
        "title": "Solid Emerald Test",
        "duration_s": 6.0,
        "steps": [{
            "start_s": 0.0,
            "end_s": 6.0,
            "element_ids": [1, 2],
            "motion": "fill",
            "colors": ["#00E676"],
            "description": "Full facade emerald fixture test",
        }],
    })
    rendered = frame_at(rig, plan, 3.0)
    diff = np.abs(rendered.astype(np.float32) - rig.base).max(axis=2)

    # Check interior facade zone: fixtures emit brightly, while mid-bay curtain-wall glass stays strictly dark (0 wash!)
    H, W = rig.base.shape[:2]
    scale = W / img.width
    ix0, ix1 = int(75 * scale), int(285 * scale)
    iy0, iy1 = int(70 * scale), int(470 * scale)
    zone_diff = diff[iy0:iy1, ix0:ix1]

    assert zone_diff.max() > 140.0, "Fixtures must emit vivid light"
    dark_glass_fraction = float((zone_diff < 2.0).mean())
    assert dark_glass_fraction > 0.45, f"Expected >45% unwashed dark glass between mullions, got {dark_glass_fraction:.2f}"

    uv = np.asarray(export_uv_mask(rig))
    assert uv.shape == (H, W)
    assert uv.max() == 255 and (uv == 0).mean() > 0.80


def test_multi_element_climb_ascends_continuously_across_zones():
    img, els = _synthetic_curtain_wall()
    opts, night_base, _ = propose_fixture_rigs(img, els)
    opt_m = next(o for o in opts if o.id == "mullion_matrix")
    rig = build_rig_from_option(night_base, opt_m)

    plan = Plan.model_validate({
        "title": "Unified Climb",
        "duration_s": 10.0,
        "steps": [{
            "start_s": 0.0,
            "end_s": 10.0,
            "element_ids": [1, 2],  # crown (top) + shaft (bottom)
            "motion": "climb",
            "colors": ["#FF2244", "#FFFFFF", "#00CC66"],
            "description": "Continuous ascent from shaft base into crown",
        }],
    })
    crown_idx = rig.elem_membership[1]
    shaft_idx = rig.elem_membership[2]

    early = fixture_colours(rig, plan, 2.0).max(axis=1) > 40.0
    late = fixture_colours(rig, plan, 9.2).max(axis=1) > 40.0

    # At t=2.0s, lower shaft fixtures are already lit while top crown has NOT started lighting yet!
    assert early[shaft_idx].sum() > 20
    assert early[crown_idx].sum() == 0
    # By t=9.2s (q=0.92 >= 0.90), the wave has climbed all the way through the crown
    assert late[crown_idx].sum() > int(0.9 * len(crown_idx))


def test_readability_audit_and_autofix_resolves_sparse_text():
    img, els = _synthetic_curtain_wall()
    opts, _, _ = propose_fixture_rigs(img, els)
    sparse_opt = next(o for o in opts if o.id == "architectural_accent")

    bad_plan = Plan.model_validate({
        "title": "Sparse Text Test",
        "duration_s": 8.0,
        "steps": [{
            "start_s": 0.0,
            "end_s": 8.0,
            "element_ids": [1],
            "motion": "text",
            "text": "CELEBRATION 2026",
            "colors": ["#FFD54F"],
            "description": "Long text on sparse accent rig",
        }],
    })
    rep1 = asyncio.run(check_readability(img, els, sparse_opt, bad_plan, use_model=False))
    assert not rep1.passed and len(rep1.flags) == 1

    fixed_plan, changes = apply_readability_fix(els, sparse_opt, bad_plan)
    assert len(changes) >= 1
    rep2 = asyncio.run(check_readability(img, els, sparse_opt, fixed_plan, use_model=False))
    assert rep2.passed and len(rep2.flags) == 0


def test_studio_routes_and_security():
    client = TestClient(app)
    r_health = client.get("/health")
    assert r_health.status_code == 200 and r_health.json()["ok"] is True

    r_ui = client.get("/studio")
    assert r_ui.status_code == 200
    assert "Facade Mapper" in r_ui.text

    r_presets = client.get("/api/presets")
    assert r_presets.status_code == 200
    assert "presets" in r_presets.json()

    r_bad = client.get("/assets/not_a_valid_id/../../pyproject.toml")
    assert r_bad.status_code in (400, 404)


def test_360_wrap_continuity_and_camera_viewport_slicing():
    palette = ["#FF2E93", "#7B2CBF", "#00F5D4", "#FFD166"]
    tex = procedural_texture(palette, size=256)
    v = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    u_zero = np.zeros_like(v)
    u_wrap = np.full_like(v, 0.99999)

    # 360° azimuth u=0 and u->1 must match seamlessly without a jump across all elevations
    for mo in ("aurora", "chase", "pulse", "sweep_left_to_right"):
        _, c0 = evaluate_360_shader(mo, palette, u_zero, v, q=0.5, t=3.2, tex=tex)
        _, c1 = evaluate_360_shader(mo, palette, u_wrap, v, q=0.5, t=3.2, tex=tex)
        assert np.abs(c0 - c1).max() < 2.0, f"Seam jump detected in 360° shader mode {mo}"

    # Camera viewport slicing: 1-side (90°) vs 2-side (162°) samples different angular sectors of the 360° show,
    # while the unmasked 360° deliverable fills all 4 elevations edge-to-edge with zero dead black margins
    img, els = _synthetic_curtain_wall()
    opts, night_base, _ = propose_fixture_rigs(img, els)
    opt_m = next(o for o in opts if o.id == "mullion_matrix")
    rig_1side = build_rig_from_option(night_base, opt_m, texture=tex, wrap_fraction=0.25)
    rig_2side = build_rig_from_option(night_base, opt_m, texture=tex, wrap_fraction=0.50)

    plan = Plan.model_validate({
        "title": "Helical Luxury Show",
        "duration_s": 12.0,
        "steps": [
            {"start_s": 0.0, "end_s": 6.0, "element_ids": [1, 2], "motion": "chase", "colors": palette, "description": "3D helical vortex"},
            {"start_s": 6.0, "end_s": 12.0, "element_ids": [1, 2], "motion": "aurora", "colors": palette, "description": "Silk aurora"},
        ],
    })
    f1 = fixture_colours(rig_1side, plan, 4.0)
    f2 = fixture_colours(rig_2side, plan, 4.0)
    assert np.abs(f1 - f2).mean() > 3.0

    # Verify step boundary at t=6.0s never flashes dead black (seamless cross-dissolve!)
    at_boundary = fixture_colours(rig_2side, plan, 6.0)
    assert at_boundary.mean() > 45.0

    full_canvas = canvas_360_frame(plan, tex, t=4.0, width=320, height=180)
    assert full_canvas.shape == (180, 320, 3)
    # All four corners of the 360° media-server deliverable carry live content (zero black margins)
    for cy, cx in ((4, 4), (4, 315), (175, 4), (175, 315)):
        assert full_canvas[cy, cx].max() > 20


def test_four_side_stitch_alignment_and_360_quadrants():
    from app.motion import render_360_overview_still
    from app.stitcher import stitch_four_elevations

    base_img, base_els = _synthetic_curtain_wall(w=320, h=520)
    # Create 4 sides with different camera distances/scales & sky brightnesses (N, E, S, W)
    s1 = base_img
    s2 = base_img.resize((260, 440), Image.LANCZOS)
    s3 = Image.fromarray(np.clip(np.asarray(base_img, np.float32) * 0.55, 0, 255).astype(np.uint8))
    s4 = base_img.resize((360, 580), Image.LANCZOS)

    async def _fake_map(im: Image.Image):
        # Scale synthetic masks proportionally to whatever input size each side has
        sx, sy = im.width / 320.0, im.height / 520.0
        scaled_els = []
        for e in base_els:
            x1, y1, x2, y2 = e.box
            nb = (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
            nm = e.mask.resize(im.size, Image.NEAREST) if e.mask is not None else None
            scaled_els.append(Element(e.id, e.label, e.kind, nb, nm))
        return scaled_els

    res = asyncio.run(stitch_four_elevations([s1, s2, s3, s4], map_fn=_fake_map))
    assert len(res.panel_spans) == 4
    assert res.side_counts is not None and all(c > 15 for c in res.side_counts)

    opt = next(o for o in res.rig_options if o.id == "mullion_matrix")
    assert opt.default_wrap == 1.0
    assert opt.default_u_offset == 0.0

    # Verify every 90° azimuth quadrant [0..0.25), [0.25..0.5), [0.5..0.75), [0.75..1.0) has active fixtures!
    for q_idx in range(4):
        in_quad = (opt.u_local >= q_idx * 0.25) & (opt.u_local < (q_idx + 1) * 0.25)
        assert int(in_quad.sum()) > 15, f"Missing fixtures in stitched quadrant {q_idx}"

    palette = ["#00E5FF", "#7B2CBF", "#FF1E8A", "#FFD166"]
    tex = procedural_texture(palette, size=256)
    rig = build_rig_from_option(res.night_base, opt, texture=tex)
    assert rig.wrap_fraction == 1.0 and rig.u_offset == 0.0

    all_ids = [e.id for e in res.elements]
    plan = Plan.model_validate({
        "title": "360 Stitched Facade Show",
        "duration_s": 10.0,
        "steps": [
            {"start_s": 0.0, "end_s": 10.0, "element_ids": all_ids, "motion": "sweep_left_to_right", "colors": palette, "description": "360 sweep across N-E-S-W"},
        ],
    })
    fr = frame_at(rig, plan, 4.2)
    assert fr.shape[:2] == (opt.render_size[1], opt.render_size[0])
    ov = render_360_overview_still(rig, plan, 4.2)
    assert ov.width > ov.height

    # Verify API validation rejects incomplete (<4 sides) stitch requests cleanly
    client = TestClient(app)
    r_ui = client.get("/studio")
    assert "mode-stitch-btn" in r_ui.text
    r_incomplete = client.post("/api/analyze-stitch", files={"side_1": ("s1.jpg", b"123", "image/jpeg")})
    assert r_incomplete.status_code == 400


def test_stencil_alpha_uv_map_and_museum_of_light_fixture_preview(tmp_path):
    from app.motion import frame_fixture_map
    from app.survey import elevations, load_survey, read_uv_mask_bool

    # 1. Stencil RGBA UV map (R=G=B=0 everywhere, fixtures encoded as alpha=0 cutouts in opaque black mask)
    W_uv, H_uv = 800, 600
    rgba = np.zeros((H_uv, W_uv, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255  # opaque black mask
    for e_idx in range(4):
        x_base = 80 + e_idx * 170
        for row_y in range(80, 500, 18):
            for col_x in range(x_base, x_base + 110, 12):
                rgba[row_y:row_y + 6, col_x:col_x + 2, 3] = 0  # stencil fixture cutout!

    uv_path = tmp_path / "stencil_uv.png"
    Image.fromarray(rgba, mode="RGBA").save(uv_path)

    mask_bool = read_uv_mask_bool(Image.open(uv_path))
    assert int(mask_bool.sum()) > 800

    sv = load_survey(str(uv_path))
    assert len(sv.cx) > 300
    elevs = elevations(sv)
    assert len(elevs) == 4, f"Expected 4 separated elevations on stencil UV map, got {len(elevs)}"

    # 2. Photo-only (NO UV map provided) Museum-of-Light #050512 Fixture-Map Preview + HDR Suhail / Night Bloom
    img, els = _synthetic_curtain_wall()
    opts, night_base, _ = propose_fixture_rigs(img, els, survey=None)
    opt_m = next(o for o in opts if o.id == "mullion_matrix")
    tex = procedural_texture(["#F0B36A", "#C76A7A", "#3B2F7A", "#15204F"], size=256)
    rig = build_rig_from_option(night_base, opt_m, texture=tex)

    plan = Plan.model_validate({
        "title": "Suhail & Night Bloom",
        "duration_s": 12.0,
        "steps": [
            {
                "start_s": 0.0, "end_s": 6.0, "element_ids": [1, 2],
                "motion": "climb", "colors": ["#F0B36A", "#C76A7A", "#3B2F7A", "#15204F"],
                "description": "Suhail rising star & nocturnal cooling",
            },
            {
                "start_s": 6.0, "end_s": 12.0, "element_ids": [1, 2],
                "motion": "pulse", "colors": ["#CFD6E6", "#8E9BBF", "#FF6F61"],
                "description": "5-lobed Night Bloom unfolding on spine",
            },
        ],
    })
    fx_frame = frame_fixture_map(rig, plan, t=4.2)
    assert fx_frame.ndim == 3 and fx_frame.shape[2] == 3
    assert fx_frame.shape[0] % 2 == 0 and fx_frame.shape[1] % 2 == 0
    # Corners sit on #050512 obsidian midnight while discrete fixture cores + optical bloom shine vividly
    assert fx_frame[2, 2, 2] >= 16 and fx_frame[2, 2, 0] <= 24
    assert fx_frame.max() > 215


def test_native_high_res_roi_resolves_dense_40_column_fins_on_wide_photo():
    """Upgrade #1: On a wide 2400x1800 photo where the tower is only 312px wide with 38 fine vertical fins
    (7.8px native pitch), native ROI extraction resolves >=34 fin columns instead of capping at ~18."""
    W_raw, H_raw = 2400, 1800
    arr = np.full((H_raw, W_raw, 3), [138, 178, 224], dtype=np.uint8)
    bx0, bx1, by0, by1 = 1044, 1356, 140, 1660
    arr[by0:by1, bx0:bx1] = [44, 54, 68]

    fin_xs = [bx0 + 8 + k * 8 for k in range(38)]  # 38 vertical fins spaced 8px apart natively!
    for fx in fin_xs:
        arr[by0:by1, fx:fx + 2] = [208, 214, 222]
    for fy in range(by0 + 14, by1 - 10, 26):
        arr[fy:fy + 2, bx0:bx1] = [132, 142, 156]

    img = Image.fromarray(arr)
    mask = Image.new("L", (W_raw, H_raw), 0)
    mask.paste(255, (bx0, by0, bx1, by1))
    els = [Element(1, "finned tower", "tower", (bx0, by0, bx1, by1), mask, void_audited=True)]

    opts, _, _ = propose_fixture_rigs(img, els)
    opt_m = next(o for o in opts if o.id == "mullion_matrix")

    # Count distinct populated vertical fin columns across mid-elevation band
    H_r, W_r = opt_m.render_size[1], opt_m.render_size[0]
    mid_band = (opt_m.ys > 0.35 * H_r) & (opt_m.ys < 0.65 * H_r)
    xs_mid = np.sort(opt_m.xs[mid_band])
    distinct_cols = 1 + int(np.sum(np.diff(xs_mid) > 3.2)) if len(xs_mid) else 0
    assert distinct_cols >= 33, f"Expected >=33 dense fin columns from native ROI extraction, got {distinct_cols}"
    assert opt_m.fixture_count >= 1400


def test_sam_void_mask_diagonal_cutout_and_cast_shadow_immunity():
    """Upgrade #2: SAM 3.1 diagonal `void_mask` carves a razor-sharp negative-space canyon with zero fixtures inside,
    tapers bars along the diagonal twist edge, and NEVER deletes cool-blue shadowed fin wings."""
    w, h = 520, 780
    arr = np.full((h, w, 3), [140, 180, 228], dtype=np.uint8)
    bx0, bx1, by0, by1 = 110, 410, 50, 730

    # Left half: warm sunlit ceramic fins (R >> B). Right half: deep cool-blue cast shadow (B >> R)!
    mid_x = (bx0 + bx1) // 2
    arr[by0:by1, bx0:mid_x] = [118, 102, 82]
    arr[by0:by1, mid_x:bx1] = [34, 48, 78]
    for fx in range(bx0 + 7, bx1 - 5, 9):
        if fx < mid_x:
            arr[by0:by1, fx:fx + 2] = [228, 206, 176]
        else:
            arr[by0:by1, fx:fx + 2] = [74, 96, 138]  # cool blue shadow fins!
    for fy in range(by0 + 12, by1 - 8, 16):
        arr[fy:fy + 1, bx0:bx1] = np.clip(arr[fy:fy + 1, bx0:bx1].astype(int) + 22, 0, 255).astype(np.uint8)

    # Diagonal S-twist SAM 3.1 negative-space glazed canyon cutting across center of tower
    yy, xx = np.mgrid[0:h, 0:w]
    canyon_cx = 195.0 + 125.0 * ((yy - by0) / float(by1 - by0))
    void_bool = (yy >= by0 + 90) & (yy <= by1 - 90) & (np.abs(xx - canyon_cx) <= 26.0)
    arr[void_bool] = [26, 36, 52]

    img = Image.fromarray(arr)
    m_tower = Image.new("L", (w, h), 0)
    m_tower.paste(255, (bx0, by0, bx1, by1))
    void_img = Image.fromarray(void_bool.astype(np.uint8) * 255)

    els = [Element(1, "twisting fin tower", "tower", (bx0, by0, bx1, by1), m_tower, void_mask=void_img, void_audited=True)]
    opts, _, _ = propose_fixture_rigs(img, els)

    W_r, H_r = opts[0].render_size
    eroded_void = np.asarray(void_img.resize((W_r, H_r), Image.NEAREST)) > 127
    from scipy import ndimage as ndi
    eroded_void = ndi.binary_erosion(eroded_void, iterations=3)

    for opt in opts:
        yi = np.clip(np.round(opt.ys).astype(int), 0, H_r - 1)
        xi = np.clip(np.round(opt.xs).astype(int), 0, W_r - 1)
        assert int(eroded_void[yi, xi].sum()) == 0, f"Rig {opt.id} placed fixtures inside SAM 3.1 void canyon!"

    opt_m = next(o for o in opts if o.id == "mullion_matrix")
    # Verify both sunlit left wing and cool-blue shadowed right wing are richly populated
    mid_r = W_r * 0.5
    left_cnt = int((opt_m.xs < mid_r - 20).sum())
    right_cnt = int((opt_m.xs > mid_r + 20).sum())
    assert left_cnt > 250 and right_cnt > 250, f"Shadowed wing lost fixtures: left={left_cnt}, right={right_cnt}"
    # Verify vertical bars taper organically near diagonal SAM void boundary
    assert float(opt_m.lengths.min()) < float(opt_m.lengths.max()) * 0.85




