"""Facade Mapper Intel — FastAPI service & standalone interactive demo studio.

Exposes:
- Legacy headless JSON endpoints: GET /health, POST /map, POST /plan
- Full interactive studio flow:
  1. POST /api/analyze        -> SAM 3.1 element map + 3+ real architectural Fixture Rig Options
  2. POST /api/concepts       -> Muse Spark 3 tailored lighting concepts + per-concept fixture mockups
  3. POST /api/generate       -> Muse Spark plan + Muse Image texture + discrete-fixture MP4 + F5 audit
  4. POST /api/fix-readability-> 1-click F5 readability auto-fix + re-render + re-audit
"""
from __future__ import annotations

import asyncio, base64, io, os, uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel

from app.checker import ReadabilityReport, apply_readability_fix, check_readability
from app.fixtures import (
    FixtureRigOption,
    propose_fixture_rigs,
    render_rig_blueprint,
)
from app.mapper import Element, element_at_point, map_facade
from app.motion import (
    Plan,
    Step,
    build_rig_from_option,
    export_uv_mask,
    frame_at,
    frame_fixture_map,
    render_360_overview_still,
    render_fixture_map_video,
    render_from_rig,
    render_unmasked_canvas,
)
from app.planner import ConceptOption, make_plan, suggest_concepts
from app.stitcher import make_demo_four_sides, stitch_four_elevations
from app.survey import load_survey
from app.visuals import generate_concept_texture, procedural_texture

app = FastAPI(title="facade-mapper-intel")

ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
OUT_DIR = ROOT_DIR / "out" / "sessions"
DATA_DIR = ROOT_DIR / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SessionState:
    id: str
    folder: Path
    img: Image.Image
    els: list[Element]
    night_base: np.ndarray
    building_mask: np.ndarray
    rig_options: dict[str, FixtureRigOption]
    concepts: list[ConceptOption] = field(default_factory=list)
    last_rig_id: str = ""
    last_active_eids: list[int] = field(default_factory=list)
    last_plan: Plan | None = None
    last_texture: np.ndarray | None = None
    last_texture_source: str = "procedural-shader"
    last_wrap_fraction: float | None = None
    is_stitched_360: bool = False
    survey: Any = None


SESSIONS: dict[str, SessionState] = {}


@app.get("/health")
def health():
    return {"ok": True}


def _elements_json(img: Image.Image, els: list[Element]) -> dict[str, Any]:
    def png(m: Image.Image) -> str:
        b = io.BytesIO()
        m.save(b, "PNG")
        return base64.b64encode(b.getvalue()).decode()

    return {
        "width": img.width,
        "height": img.height,
        "elements": [
            {
                "id": e.id,
                "label": e.label,
                "kind": e.kind,
                "role": "dynamic",
                "box": e.box,
                "mask_png_b64": png(e.mask),
            }
            for e in els
            if e.mask is not None
        ],
    }


@app.post("/map")
async def map_endpoint(photo: UploadFile = File(...)):
    """F1: one photo in -> labelled element masks out (PNG per element, white = element)."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(await photo.read()))).convert("RGB")
    return _elements_json(img, await map_facade(img))


@app.post("/plan")
async def plan_endpoint(photo: UploadFile = File(...), brief: str = Form(...), duration_s: float = Form(30)):
    """F2: photo + brief -> element map + a timed plan grounded in those elements."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(await photo.read()))).convert("RGB")
    els = await map_facade(img)
    plan = await make_plan(img, els, brief, duration_s)
    return {**_elements_json(img, els), "plan": plan.model_dump()}


# ---------------------------------------------------------------------------
# Interactive Studio UI & Endpoints
# ---------------------------------------------------------------------------

ZONE_COLOURS = [
    (56, 189, 248),
    (251, 191, 36),
    (52, 211, 153),
    (244, 114, 182),
    (167, 139, 250),
    (251, 146, 60),
    (45, 212, 191),
    (248, 113, 113),
]


def _render_zone_overlay(img: Image.Image, els: list[Element], size: tuple[int, int]) -> Image.Image:
    """Render SAM 3.1 architectural element outlines, negative-space void cutouts, & translucent tint."""
    from scipy import ndimage as ndi
    W, H = size
    orig_resized = np.asarray(img.convert("RGB").resize((W, H), Image.LANCZOS), np.float32)
    base = orig_resized * 0.55
    for idx, e in enumerate(els):
        if e.mask is None:
            continue
        m = np.asarray(e.mask.resize((W, H), Image.NEAREST)) > 127
        if not m.any():
            continue
        col = np.array(ZONE_COLOURS[idx % len(ZONE_COLOURS)], np.float32)
        edge = m & (~ndi.binary_erosion(m, iterations=2))
        base[m] = base[m] * 0.72 + col * 0.28
        base[edge] = col

    # Carve out SAM 3.1 negative-space voids (recessed glass canyons / finless cutouts) with crisp obsidian-slate contour
    void_union = np.zeros((H, W), bool)
    for e in els:
        vm = getattr(e, "void_mask", None)
        if vm is not None:
            void_union |= np.asarray(vm.resize((W, H), Image.NEAREST)) > 127
    if void_union.any():
        v_edge = void_union & (~ndi.binary_erosion(void_union, iterations=2))
        base[void_union] = orig_resized[void_union] * 0.26 + np.array([6.0, 10.0, 24.0], np.float32) * 0.74
        base[v_edge] = np.array([232.0, 121.0, 249.0], np.float32)

    out = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    sx, sy = W / img.width, H / img.height
    for idx, e in enumerate(els):
        col = ZONE_COLOURS[idx % len(ZONE_COLOURS)]
        x1, y1, x2, y2 = e.box
        cx, cy = int((x1 + x2) * 0.5 * sx), int((y1 + y2) * 0.5 * sy)
        tag = f"#{e.id} {e.label}"
        tw = len(tag) * 7 + 12
        draw.rounded_rectangle((cx - tw // 2, cy - 11, cx + tw // 2, cy + 11), radius=5, fill=(10, 15, 28), outline=col, width=2)
        draw.text((cx, cy), tag, fill=(245, 248, 255), anchor="mm")
    return out


def _filter_rig_by_elements(opt: FixtureRigOption, active_eids: list[int] | None) -> FixtureRigOption:
    """Return a copy of `opt` keeping only fixtures inside user-enabled architectural zones."""
    if not active_eids:
        return opt
    active_set = set(int(x) for x in active_eids)
    all_eids = set(opt.elem_membership.keys())
    if active_set >= all_eids:
        return opt
    keep_parts = [opt.elem_membership[eid] for eid in active_set if eid in opt.elem_membership]
    if not keep_parts:
        return opt
    keep = np.unique(np.concatenate(keep_parts))
    new_memb: dict[int, np.ndarray] = {}
    inv = np.full(opt.fixture_count, -1, dtype=np.int32)
    inv[keep] = np.arange(len(keep), dtype=np.int32)
    for eid in active_set:
        if eid in opt.elem_membership:
            mapped = inv[opt.elem_membership[eid]]
            mapped = mapped[mapped >= 0]
            if len(mapped):
                new_memb[eid] = mapped
    return FixtureRigOption(
        id=opt.id,
        name=opt.name,
        subtitle=opt.subtitle,
        recommended=opt.recommended,
        fixture_type=opt.fixture_type,
        fixture_count=len(keep),
        pitch_desc=opt.pitch_desc,
        est_power_kw=len(keep) * 0.018,
        zones_used=opt.zones_used,
        xs=opt.xs[keep],
        ys=opt.ys[keep],
        lengths=opt.lengths[keep],
        widths=opt.widths[keep],
        angles=opt.angles[keep],
        elem_ids=opt.elem_ids[keep],
        elem_membership=new_memb,
        render_size=opt.render_size,
        u_local=opt.u_local[keep] if len(opt.u_local) == opt.fixture_count else np.empty(0, np.float32),
        v_elev=opt.v_elev[keep] if len(opt.v_elev) == opt.fixture_count else np.empty(0, np.float32),
        view_cos=opt.view_cos[keep] if len(opt.view_cos) == opt.fixture_count else np.empty(0, np.float32),
        default_wrap=opt.default_wrap,
        default_u_offset=opt.default_u_offset,
        panel_spans=opt.panel_spans,
        elev_y_span=opt.elev_y_span,
        spine_u=opt.spine_u,
        crop_box=opt.crop_box,
    )


def _quick_concept_stills(
    night_base: np.ndarray,
    opt: FixtureRigOption,
    els: list[Element],
    concept: ConceptOption,
    idx: int,
) -> tuple[Image.Image, Image.Image]:
    """Synthesize both the Museum-of-Light #050512 Fixture-Map Preview and the On-Building Camera Mockup."""
    all_eids = [e.id for e in els if e.id in opt.elem_membership and len(opt.elem_membership[e.id])]
    if not all_eids:
        all_eids = [els[0].id]
    motion_cycle = [("aurora", 5.2), ("climb", 6.4), ("pulse", 5.0)]
    mo, t_hero = motion_cycle[idx % len(motion_cycle)]
    tex = procedural_texture(concept.palette)
    rig = build_rig_from_option(night_base, opt, texture=tex)
    temp_plan = Plan(
        title=concept.title,
        duration_s=10.0,
        steps=[Step(
            start_s=0.0, end_s=10.0, element_ids=all_eids,
            motion=mo, colors=concept.palette, description=concept.tagline,
        )],
    )
    return (
        Image.fromarray(frame_fixture_map(rig, temp_plan, t=t_hero)),
        Image.fromarray(frame_at(rig, temp_plan, t=t_hero)),
    )


@app.get("/", response_class=HTMLResponse)
def index_page():
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(404, "Frontend UI missing")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/assets/{session_id}/{filename}")
def session_asset(session_id: str, filename: str):
    safe = Path(filename).name
    target = OUT_DIR / Path(session_id).name / safe
    if not target.exists():
        raise HTTPException(404, "Asset not found")
    return FileResponse(target)


@app.get("/api/presets")
def list_presets():
    presets = []
    has_stitch_demo = False
    if (DATA_DIR / "wasl.png").exists():
        has_stitch_demo = True
        presets.append({
            "id": "wasl_photo_only",
            "label": "Twisting Tower — Photo Only (Auto-Detect Mullions)",
            "has_survey": False,
            "mode": "single",
        })
        if (DATA_DIR / "wasl_uv_dyn.png").exists():
            presets.append({
                "id": "wasl_with_survey",
                "label": "Twisting Tower — Photo + As-Built UV Survey",
                "has_survey": True,
                "mode": "single",
            })
    return {"presets": presets, "has_stitch_demo": has_stitch_demo}


@app.post("/api/analyze")
async def api_analyze(
    photo: UploadFile | None = File(default=None),
    survey_mask: UploadFile | None = File(default=None),
    preset: str | None = Form(default=None),
):
    """Step 1: Read building architecture (SAM 3.1 + Muse Spark) and suggest 3+ realistic Fixture Rig Options."""
    surv = None
    if preset in {"wasl_photo_only", "wasl_with_survey"} and (DATA_DIR / "wasl.png").exists():
        img = Image.open(DATA_DIR / "wasl.png").convert("RGB")
        if preset == "wasl_with_survey" and (DATA_DIR / "wasl_uv_dyn.png").exists():
            surv = load_survey(str(DATA_DIR / "wasl_uv_dyn.png"), cache=str(DATA_DIR / "wasl_uv_dyn.png.npz"))
    elif photo is not None:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(await photo.read()))).convert("RGB")
        if survey_mask is not None:
            sid_tmp = OUT_DIR / f"tmp_{uuid.uuid4().hex[:8]}.png"
            sid_tmp.write_bytes(await survey_mask.read())
            try:
                surv = load_survey(str(sid_tmp))
            finally:
                sid_tmp.unlink(missing_ok=True)
    else:
        raise HTTPException(400, "Provide a building photo or select a sample preset.")

    sid = uuid.uuid4().hex[:12]
    folder = OUT_DIR / sid
    folder.mkdir(parents=True, exist_ok=True)

    els = await map_facade(img)
    if not els:
        raise HTTPException(422, "Could not detect architectural elements in this photo.")

    opts, night_base, bmask = propose_fixture_rigs(img, els, survey=surv)
    if not opts:
        raise HTTPException(422, "Could not place architectural fixtures on the detected silhouette.")

    W, H = opts[0].render_size
    img.resize((W, H), Image.LANCZOS).save(folder / "original.jpg", quality=92)
    Image.fromarray(night_base.clip(0, 255).astype(np.uint8)).save(folder / "night_base.jpg", quality=92)
    _render_zone_overlay(img, els, (W, H)).save(folder / "zones.jpg", quality=92)

    rig_cards = []
    rig_dict: dict[str, FixtureRigOption] = {}
    for o in opts:
        rig_dict[o.id] = o
        fname = f"rig_{o.id}.jpg"
        render_rig_blueprint(night_base, o, els).save(folder / fname, quality=92)
        d = o.summary_dict()
        d["preview_url"] = f"/assets/{sid}/{fname}"
        d["element_counts"] = {str(eid): int(len(idx)) for eid, idx in o.elem_membership.items()}
        rig_cards.append(d)

    SESSIONS[sid] = SessionState(
        id=sid,
        folder=folder,
        img=img,
        els=els,
        night_base=night_base,
        building_mask=bmask,
        rig_options=rig_dict,
        last_rig_id=opts[0].id,
        last_active_eids=[e.id for e in els],
        survey=surv,
    )

    return {
        "session_id": sid,
        "is_stitched_360": False,
        "photo_size": [img.width, img.height],
        "render_size": [W, H],
        "original_url": f"/assets/{sid}/original.jpg",
        "night_url": f"/assets/{sid}/night_base.jpg",
        "zones_url": f"/assets/{sid}/zones.jpg",
        "models_used": ["muse-spark-1.3", "sam-3.1"],
        "elements": [
            {"id": e.id, "label": e.label, "kind": e.kind, "box": e.box}
            for e in els
        ],
        "rig_options": rig_cards,
    }


@app.post("/api/analyze-stitch")
async def api_analyze_stitch(
    side_1: UploadFile | None = File(default=None),
    side_2: UploadFile | None = File(default=None),
    side_3: UploadFile | None = File(default=None),
    side_4: UploadFile | None = File(default=None),
    preset: str | None = Form(default=None),
):
    """Step 1 (All-4-Sides Full Facade Map Mode): Upload & stitch 4 building elevations (North · East · South · West)
    into one crown-aligned 360° facade sheet with discrete mullions & fins mapped across all 4 quadrants."""
    if preset == "wasl_4sides_demo" and (DATA_DIR / "wasl.png").exists():
        side_imgs = make_demo_four_sides(Image.open(DATA_DIR / "wasl.png"))
    else:
        uploads = [side_1, side_2, side_3, side_4]
        if any(u is None for u in uploads):
            raise HTTPException(
                400,
                "Please upload all 4 building sides (Side 1 North, Side 2 East, Side 3 South, Side 4 West) to stitch the full 360° facade map.",
            )
        side_imgs = []
        for u in uploads:
            assert u is not None
            side_imgs.append(Image.open(io.BytesIO(await u.read())).convert("RGB"))

    sid = uuid.uuid4().hex[:12]
    folder = OUT_DIR / sid
    folder.mkdir(parents=True, exist_ok=True)

    for idx_s, s_im in enumerate(side_imgs, start=1):
        thumb = s_im.copy()
        thumb.thumbnail((480, 480), Image.LANCZOS)
        thumb.save(folder / f"side_{idx_s}.jpg", quality=88)

    stitched = await stitch_four_elevations(side_imgs)
    if not stitched.rig_options:
        raise HTTPException(422, "Could not place architectural fixtures across the 4 stitched elevations.")

    W, H = stitched.rig_options[0].render_size
    stitched.stitched_original.save(folder / "original.jpg", quality=92)
    Image.fromarray(stitched.night_base.clip(0, 255).astype(np.uint8)).save(folder / "night_base.jpg", quality=92)
    _render_zone_overlay(stitched.stitched_original, stitched.elements, (W, H)).save(folder / "zones.jpg", quality=92)

    rig_cards = []
    rig_dict: dict[str, FixtureRigOption] = {}
    for o in stitched.rig_options:
        rig_dict[o.id] = o
        fname = f"rig_{o.id}.jpg"
        render_rig_blueprint(stitched.night_base, o, stitched.elements).save(folder / fname, quality=92)
        d = o.summary_dict()
        d["preview_url"] = f"/assets/{sid}/{fname}"
        d["element_counts"] = {str(eid): int(len(idx)) for eid, idx in o.elem_membership.items()}
        rig_cards.append(d)

    SESSIONS[sid] = SessionState(
        id=sid,
        folder=folder,
        img=stitched.stitched_original,
        els=stitched.elements,
        night_base=stitched.night_base,
        building_mask=stitched.building_mask,
        rig_options=rig_dict,
        last_rig_id=stitched.rig_options[0].id,
        last_active_eids=[e.id for e in stitched.elements],
        is_stitched_360=True,
    )

    return {
        "session_id": sid,
        "is_stitched_360": True,
        "side_counts": stitched.side_counts,
        "side_urls": [f"/assets/{sid}/side_{k}.jpg" for k in range(1, 5)],
        "photo_size": [W, H],
        "render_size": [W, H],
        "original_url": f"/assets/{sid}/original.jpg",
        "night_url": f"/assets/{sid}/night_base.jpg",
        "zones_url": f"/assets/{sid}/zones.jpg",
        "models_used": ["muse-spark-1.3", "sam-3.1"],
        "elements": [
            {"id": e.id, "label": e.label, "kind": e.kind, "box": e.box}
            for e in stitched.elements
        ],
        "rig_options": rig_cards,
    }


class ElementEdit(BaseModel):
    action: str  # 'add' | 'remove'
    element_id: int | None = None
    x: int | None = None  # photo pixels
    y: int | None = None
    label: str = "element"


@app.post("/api/sessions/{sid}/elements")
async def api_edit_elements(sid: str, req: ElementEdit):
    """F1 click-to-edit: add (SAM point prompt) or remove an element; rigs/overlay are rebuilt from the edited map."""
    st = SESSIONS.get(sid)
    if st is None:
        raise HTTPException(404, "Session expired or not found; please re-upload the photo.")
    if req.action == "remove":
        if len(st.els) <= 1:
            raise HTTPException(422, "Keep at least one element.")
        st.els = [e for e in st.els if e.id != req.element_id]
    elif req.action == "add" and req.x is not None and req.y is not None:
        try:
            new = await element_at_point(st.img, req.x, req.y, req.label.strip() or "element", max((e.id for e in st.els), default=0) + 1)
        except ValueError as ex:
            raise HTTPException(422, str(ex))
        st.els.append(new)
    else:
        raise HTTPException(400, "Use {action:'remove', element_id} or {action:'add', x, y, label}.")
    # ponytail: full rig re-propose per edit (CPU, seconds); incremental membership update if it feels slow
    opts, st.night_base, st.building_mask = propose_fixture_rigs(st.img, st.els, survey=st.survey)
    st.rig_options = {o.id: o for o in opts}
    st.last_rig_id = opts[0].id
    st.last_active_eids = [e.id for e in st.els]
    W, H = opts[0].render_size
    v = uuid.uuid4().hex[:6]
    _render_zone_overlay(st.img, st.els, (W, H)).save(st.folder / f"zones_{v}.jpg", quality=92)
    cards = []
    for o in opts:
        fname = f"rig_{o.id}_{v}.jpg"
        render_rig_blueprint(st.night_base, o, st.els).save(st.folder / fname, quality=92)
        d = o.summary_dict()
        d["preview_url"] = f"/assets/{sid}/{fname}"
        d["element_counts"] = {str(eid): int(len(idx)) for eid, idx in o.elem_membership.items()}
        cards.append(d)
    return {
        "zones_url": f"/assets/{sid}/zones_{v}.jpg",
        "elements": [{"id": e.id, "label": e.label, "kind": e.kind, "box": e.box} for e in st.els],
        "rig_options": cards,
    }


class ConceptsRequest(BaseModel):
    session_id: str
    rig_id: str
    active_element_ids: list[int] | None = None


@app.post("/api/concepts")
async def api_concepts(req: ConceptsRequest):
    """Step 2: Muse Spark suggests 3 lighting concepts tailored to the chosen fixture rig + renders instant fixture mockups."""
    st = SESSIONS.get(req.session_id)
    if st is None:
        raise HTTPException(404, "Session expired or not found; please re-upload the photo.")
    base_opt = st.rig_options.get(req.rig_id)
    if base_opt is None:
        raise HTTPException(400, f"Unknown fixture rig option: {req.rig_id}")

    opt = _filter_rig_by_elements(base_opt, req.active_element_ids)
    active_els = [e for e in st.els if not req.active_element_ids or e.id in set(req.active_element_ids)] or st.els
    concepts = await suggest_concepts(st.img, active_els, opt.summary_dict())
    st.concepts = concepts
    st.last_rig_id = req.rig_id
    st.last_active_eids = [e.id for e in active_els]

    out_cards = []
    for idx, c in enumerate(concepts):
        fname_fx = f"concept_fx_{req.rig_id}_{c.id}.jpg"
        fname_cam = f"concept_{req.rig_id}_{c.id}.jpg"
        fx_still, cam_still = _quick_concept_stills(st.night_base, opt, active_els, c, idx)
        fx_still.save(st.folder / fname_fx, quality=92)
        cam_still.save(st.folder / fname_cam, quality=92)
        d = c.model_dump()
        d["preview_url"] = f"/assets/{st.id}/{fname_fx}"
        d["camera_preview_url"] = f"/assets/{st.id}/{fname_cam}"
        out_cards.append(d)

    return {
        "session_id": st.id,
        "rig_id": req.rig_id,
        "fixture_count": opt.fixture_count,
        "models_used": ["muse-spark-1.3"],
        "concepts": out_cards,
    }


class GenerateRequest(BaseModel):
    session_id: str
    rig_id: str
    brief: str
    title: str | None = None
    palette: list[str] | None = None
    visual_prompt: str | None = None
    duration_s: float = 14.0
    active_element_ids: list[int] | None = None
    use_muse_image: bool = True
    wrap_fraction: float | None = None


@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    """Step 3 & 4: Generate grounded 360° show plan + Muse Image silk plate + discrete-fixture H.264 videos + 360° deliverable + F5 check."""
    st = SESSIONS.get(req.session_id)
    if st is None:
        raise HTTPException(404, "Session expired or not found; please re-upload the photo.")
    base_opt = st.rig_options.get(req.rig_id)
    if base_opt is None:
        raise HTTPException(400, f"Unknown fixture rig option: {req.rig_id}")

    opt = _filter_rig_by_elements(base_opt, req.active_element_ids)
    active_els = [e for e in st.els if not req.active_element_ids or e.id in set(req.active_element_ids)] or st.els
    fcounts = {eid: len(idx) for eid, idx in opt.elem_membership.items()}
    dur = float(np.clip(req.duration_s, 6.0, 30.0))

    palette = req.palette or ["#3A7BD5", "#B58FD6", "#E6E9F0"]
    vprompt = req.visual_prompt or req.brief

    plan, (tex, tex_model) = await asyncio.gather(
        make_plan(st.img, active_els, req.brief, duration_s=dur, elem_fixture_counts=fcounts),
        generate_concept_texture(vprompt, palette, use_model=req.use_muse_image),
    )
    if req.title:
        plan.title = req.title

    wf = float(np.clip(req.wrap_fraction, 0.18, 1.0)) if req.wrap_fraction is not None else None
    rig = build_rig_from_option(st.night_base, opt, texture=tex, wrap_fraction=wf)
    run_tag = uuid.uuid4().hex[:6]
    vid_name = f"show_{run_tag}.mp4"
    fx_vid_name = f"fixture_preview_{run_tag}.mp4"
    still_name = f"still_{run_tag}.jpg"
    fx_still_name = f"fixture_still_{run_tag}.jpg"
    uv_name = f"uv_mask_{run_tag}.png"
    deliv_name = f"deliverable_{run_tag}.mp4"
    ov360_name = f"overview_360_{run_tag}.jpg"

    (n_frames, n_fx), _, _ = await asyncio.gather(
        asyncio.to_thread(render_from_rig, rig, plan, str(st.folder / vid_name), 24),
        asyncio.to_thread(render_fixture_map_video, rig, plan, str(st.folder / fx_vid_name), 24),
        asyncio.to_thread(render_unmasked_canvas, rig, plan, str(st.folder / deliv_name), 24),
    )

    text_steps = [s for s in plan.steps if s.motion == "text" and s.text]
    hero_t = 0.5 * (text_steps[0].start_s + text_steps[0].end_s) if text_steps else dur * 0.56
    still_img = Image.fromarray(frame_at(rig, plan, hero_t))
    still_img.save(st.folder / still_name, quality=92)
    Image.fromarray(frame_fixture_map(rig, plan, hero_t)).save(st.folder / fx_still_name, quality=94)
    render_360_overview_still(rig, plan, hero_t).save(st.folder / ov360_name, quality=92)
    export_uv_mask(rig).save(st.folder / uv_name)

    tex_name = f"texture_{run_tag}.jpg"
    Image.fromarray(tex.clip(0, 255).astype(np.uint8)).save(st.folder / tex_name, quality=90)

    audit: ReadabilityReport = await check_readability(still_img, active_els, opt, plan, use_model=True)

    st.last_rig_id = req.rig_id
    st.last_active_eids = [e.id for e in active_els]
    st.last_plan = plan
    st.last_texture = tex
    st.last_texture_source = tex_model
    st.last_wrap_fraction = rig.wrap_fraction

    return {
        "session_id": st.id,
        "is_stitched_360": st.is_stitched_360,
        "rig_id": req.rig_id,
        "fixture_count": n_fx,
        "frames": n_frames,
        "fps": 24,
        "wrap_degrees": int(round(rig.wrap_fraction * 360)),
        "texture_source": tex_model,
        "texture_url": f"/assets/{st.id}/{tex_name}",
        "video_url": f"/assets/{st.id}/{vid_name}",
        "fixture_video_url": f"/assets/{st.id}/{fx_vid_name}",
        "still_url": f"/assets/{st.id}/{still_name}",
        "fixture_still_url": f"/assets/{st.id}/{fx_still_name}",
        "overview_360_url": f"/assets/{st.id}/{ov360_name}",
        "uv_mask_url": f"/assets/{st.id}/{uv_name}",
        "deliverable_url": f"/assets/{st.id}/{deliv_name}",
        "models_used": ["muse-spark-1.3", tex_model],
        "plan": plan.model_dump(),
        "readability": audit.model_dump(),
    }


class FixRequest(BaseModel):
    session_id: str


@app.post("/api/fix-readability")
async def api_fix_readability(req: FixRequest):
    """Step 5 (F5 loop): Apply architectural readability fix, re-render on discrete fixtures, and verify pass."""
    st = SESSIONS.get(req.session_id)
    if st is None or st.last_plan is None:
        raise HTTPException(404, "No active show plan to fix.")
    base_opt = st.rig_options[st.last_rig_id]
    opt = _filter_rig_by_elements(base_opt, st.last_active_eids)
    active_els = [e for e in st.els if not st.last_active_eids or e.id in set(st.last_active_eids)] or st.els

    fixed_plan, changes = apply_readability_fix(active_els, opt, st.last_plan)
    rig = build_rig_from_option(st.night_base, opt, texture=st.last_texture, wrap_fraction=st.last_wrap_fraction)

    run_tag = f"fixed_{uuid.uuid4().hex[:6]}"
    vid_name = f"show_{run_tag}.mp4"
    fx_vid_name = f"fixture_preview_{run_tag}.mp4"
    still_name = f"still_{run_tag}.jpg"
    fx_still_name = f"fixture_still_{run_tag}.jpg"
    deliv_name = f"deliverable_{run_tag}.mp4"
    ov360_name = f"overview_360_{run_tag}.jpg"

    (n_frames, n_fx), _, _ = await asyncio.gather(
        asyncio.to_thread(render_from_rig, rig, fixed_plan, str(st.folder / vid_name), 24),
        asyncio.to_thread(render_fixture_map_video, rig, fixed_plan, str(st.folder / fx_vid_name), 24),
        asyncio.to_thread(render_unmasked_canvas, rig, fixed_plan, str(st.folder / deliv_name), 24),
    )

    hero_t = fixed_plan.duration_s * 0.56
    still_img = Image.fromarray(frame_at(rig, fixed_plan, hero_t))
    still_img.save(st.folder / still_name, quality=92)
    Image.fromarray(frame_fixture_map(rig, fixed_plan, hero_t)).save(st.folder / fx_still_name, quality=94)
    render_360_overview_still(rig, fixed_plan, hero_t).save(st.folder / ov360_name, quality=92)

    audit = await check_readability(still_img, active_els, opt, fixed_plan, use_model=False)
    st.last_plan = fixed_plan

    return {
        "session_id": st.id,
        "changes": changes,
        "fixture_count": n_fx,
        "frames": n_frames,
        "video_url": f"/assets/{st.id}/{vid_name}",
        "fixture_video_url": f"/assets/{st.id}/{fx_vid_name}",
        "still_url": f"/assets/{st.id}/{still_name}",
        "fixture_still_url": f"/assets/{st.id}/{fx_still_name}",
        "overview_360_url": f"/assets/{st.id}/{ov360_name}",
        "deliverable_url": f"/assets/{st.id}/{deliv_name}",
        "plan": fixed_plan.model_dump(),
        "readability": audit.model_dump(),
    }
