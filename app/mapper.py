"""F1: read the building. Muse Spark proposes element boxes, SAM 3.1 outlines them."""
import asyncio, base64, io, json, os
from dataclasses import dataclass

from meta_sam_parser import (FramePrompt, PromptBox, PromptObject, PromptPoint, build_object_prompt,
                             decode_mask_to_raster, image_segmentation_format, parse_responses_stream)
from openai import AsyncOpenAI
from PIL import Image, ImageChops

BASE = "https://api.meta.ai/v1"
MAX_OBJECTS = 16  # SAM limit per request

SCHEMA = {
    "type": "object",
    "properties": {"elements": {"type": "array", "items": {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "kind": {"type": "string", "enum": ["crown", "tower", "podium", "setback", "band", "fins", "glass", "other"]},
            "x1": {"type": "integer"}, "y1": {"type": "integer"},
            "x2": {"type": "integer"}, "y2": {"type": "integer"},
        },
        "required": ["label", "kind", "x1", "y1", "x2", "y2"],
        "additionalProperties": False,
    }}},
    "required": ["elements"],
    "additionalProperties": False,
}

PROMPT = (
    "You are mapping a building facade for a media-facade lighting designer. List the distinct, "
    "architecturally meaningful elements visible (e.g. crown/top, each tower or setback block, podium, "
    "a representative horizontal band or vertical fin run, glazed core). At most {n}. Each element gets a "
    "short unique label, a kind, and a tight bounding box in coordinates normalised to 0-1000 "
    "(x1,y1 = top-left, x2,y2 = bottom-right). Ignore sky, trees, street and foreground clutter."
)


VOID_SCHEMA = {
    "type": "object",
    "properties": {
        "has_voids": {"type": "boolean"},
        "voids": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "void_type": {
                        "type": "string",
                        "enum": ["recessed_glass_canyon", "lobby_glazing"],
                    },
                    "x1": {"type": "integer"},
                    "y1": {"type": "integer"},
                    "x2": {"type": "integer"},
                    "y2": {"type": "integer"},
                    "pos_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                            "required": ["x", "y"],
                            "additionalProperties": False,
                        },
                    },
                    "neg_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                            "required": ["x", "y"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["label", "void_type", "x1", "y1", "x2", "y2", "pos_points", "neg_points"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["has_voids", "voids"],
    "additionalProperties": False,
}

VOID_PROMPT = (
    "You are auditing a cropped high-resolution building facade ROI for an architectural LED media-facade engineer. "
    "Exterior vertical linear LED pixel bars are mounted ONLY on solid vertical architectural fins, ceramic/metal louvers, "
    "or protruding mullions. Look carefully for any sculptural or glazed NEGATIVE-SPACE VOID zones inside the building "
    "silhouette where exterior vertical fins/mullions are ABSENT and therefore NO exterior LEDs exist:\n"
    "- `recessed_glass_canyon`: Smooth dark recessed glass curtain-wall grooves, diagonal S-twist / V-shaped glazed canyons "
    "carving into a twisting tower, or glazed sections at dusk/night where warm lit interior office windows/floors are visible "
    "behind smooth glass WITHOUT exterior vertical LED fins.\n"
    "- `lobby_glazing`: Ground-level transparent glass lobby storefronts lacking tower mullions.\n"
    "RULES:\n"
    "1. Twisting or sculptural towers (e.g. ceramic-fin towers with a diagonal smooth-glass twist reveal or mid-tower interior "
    "office window glazing) DO have `has_voids: true` — trace that recessed smooth-glass / interior-window reveal!\n"
    "2. Do NOT mark cast shadows on finned wings (where vertical fin ribs clearly continue) or thin 1-floor horizontal seams as voids.\n"
    "3. Only return `has_voids: false` when the building is a plain rectangular box with one uniform grid from edge to edge.\n"
    "For each void (at most 5), return 0-1000 normalized ROI coordinates: a tight bounding box (`x1,y1,x2,y2`), "
    "2-4 `pos_points` strictly inside the smooth finless glass / interior-window void, and 2-4 `neg_points` "
    "on the adjacent vertical-fin-clad facade wings just outside the void so SAM 3.1 outlines the exact contour."
)


@dataclass
class Element:
    id: int
    label: str
    kind: str
    box: tuple[int, int, int, int]  # pixels
    mask: Image.Image | None = None  # L-mode, full image size, 255 = element
    void_mask: Image.Image | None = None  # L-mode, full image size, 255 = SAM 3.1 negative-space cutout
    void_audited: bool = False  # True once high-res ROI SAM 3.1 void audit has evaluated the facade


def _data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _load_dotenv() -> None:
    if "MODEL_API_KEY" not in os.environ and os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _client() -> AsyncOpenAI:
    _load_dotenv()
    return AsyncOpenAI(base_url=BASE, api_key=os.environ["MODEL_API_KEY"])


async def propose(img: Image.Image, n: int = 12) -> list[Element]:
    r = await _client().chat.completions.create(
        model="muse-spark-1.3",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT.format(n=n)},
            {"type": "image_url", "image_url": {"url": _data_url(img)}},
        ]}],
        response_format={"type": "json_schema", "json_schema": {"name": "elements", "schema": SCHEMA, "strict": True}},
    )
    w, h = img.size
    out = []
    for i, e in enumerate(json.loads(r.choices[0].message.content)["elements"][:n]):
        lbl_low = str(e["label"]).lower()
        if any(w in lbl_low for w in ("bridge", "overpass", "viaduct", "highway", "flyover", "road", "guardrail")):
            continue
        x1, x2 = sorted((e["x1"], e["x2"])); y1, y2 = sorted((e["y1"], e["y2"]))
        box = (max(0, x1 * w // 1000), max(0, y1 * h // 1000), min(w, x2 * w // 1000), min(h, y2 * h // 1000))
        if box[2] - box[0] > 2 and box[3] - box[1] > 2:
            out.append(Element(i + 1, e["label"], e["kind"], box))
    return out


async def outline_with_hero(img: Image.Image, els: list[Element], all_els: list[Element]) -> tuple[list[Element], "np.ndarray"]:
    """SAM 3.1 box prompts -> one full-size mask per solid element PLUS one overall hero-facade mask (id=999)."""
    import numpy as np
    w, h = img.size
    objs = [PromptObject(id=e.id, prompts=[FramePrompt(box=PromptBox(*e.box))]) for e in els]
    if all_els:
        ux1 = min(e.box[0] for e in all_els)
        uy1 = min(e.box[1] for e in all_els)
        ux2 = max(e.box[2] for e in all_els)
        uy2 = max(e.box[3] for e in all_els)
        objs.append(PromptObject(id=999, prompts=[FramePrompt(box=PromptBox(ux1, uy1, ux2, uy2))]))
    prompt = build_object_prompt(size=(w, h), objects=objs[:MAX_OBJECTS])
    stream = await _client().responses.create(
        model="sam-3.1", stream=True,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": _data_url(img)},
        ]}],
    )
    result = await parse_responses_stream(stream, image_segmentation_format()).final_result()
    by_id = {e.id: e for e in els}
    hero = np.zeros((h, w), bool)
    for rec in result.records:
        if type(rec).__name__ != "SegmentationMaskRecord":
            continue
        oid = int(rec.object_id)
        m = rec.mask
        raw = decode_mask_to_raster(m)
        if oid == 999:
            tile_b = np.frombuffer(raw, np.uint8).reshape(m.height, m.width) > 0
            t, l = int(rec.bounds.top), int(rec.bounds.left)
            hero[t:t + m.height, l:l + m.width] |= tile_b
            continue
        el = by_id.get(oid)
        if el is None:
            continue
        tile = Image.frombytes("L", (m.width, m.height), bytes(v * 255 for v in raw))
        full = Image.new("L", (w, h), 0)
        full.paste(tile, (int(rec.bounds.left), int(rec.bounds.top)))
        el.mask = full
    return els, hero


async def concept_masks(img: Image.Image, text: str) -> list["np.ndarray"]:
    """SAM 3.1 concept prompt (one short noun phrase) -> one bool mask per match, full image size."""
    import numpy as np
    stream = await _client().responses.create(
        model="sam-3.1", stream=True,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": text},
            {"type": "input_image", "image_url": _data_url(img)},
        ]}],
    )
    res = await parse_responses_stream(stream, image_segmentation_format()).final_result()
    out = []
    for rec in res.records:
        if type(rec).__name__ != "SegmentationMaskRecord":
            continue
        m = rec.mask
        tile = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width) > 0
        full = np.zeros((img.height, img.width), bool)
        t, l = int(rec.bounds.top), int(rec.bounds.left)
        full[t:t + m.height, l:l + m.width] = tile
        out.append(full)
    return out


async def tower_mask(img: Image.Image) -> "np.ndarray":
    """The main tower's silhouette: the largest 'skyscraper' (or 'building') match."""
    ms = await concept_masks(img, "skyscraper")
    if not ms:
        ms = await concept_masks(img, "building")
    if not ms:
        raise ValueError("SAM found no building in the photo")
    return max(ms, key=lambda m: m.sum())


async def building_silhouette(img: Image.Image, els: list[Element]) -> "np.ndarray":
    """Authoritative silhouette of the hero building(s) proposed by Muse Spark, excluding sky/bridges/neighbours."""
    import numpy as np
    H, W = img.height, img.width
    boxes = np.zeros((H, W), bool)
    for e in els:
        x1, y1, x2, y2 = e.box
        boxes[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
    ms = await concept_masks(img, "building")
    if not ms:
        ms = await concept_masks(img, "skyscraper")
    if not ms:
        return boxes
    kept = [m for m in ms if (m & boxes).sum() / max(m.sum(), 1) >= 0.25 or (m & boxes).sum() / max(boxes.sum(), 1) >= 0.2]
    if not kept:
        kept = [max(ms, key=lambda m: (m & boxes).sum())]
    sil = np.zeros((H, W), bool)
    for m in kept:
        sil |= m
    return sil


REGION_KINDS = {"band", "fins", "glass"}  # repeated detail: not a single object SAM can outline


def clip_regions(img: Image.Image, regions: list[Element], solids: list[Element]) -> None:
    """Region mask = its box clipped to the solid silhouette union (or best parent solid)."""
    import numpy as np
    if not solids:
        return
    union = np.zeros((img.height, img.width), bool)
    for s in solids:
        if s.mask is not None:
            union |= np.asarray(s.mask) > 127
    union_img = Image.fromarray(union.astype(np.uint8) * 255)
    for r in regions:
        box = Image.new("L", img.size, 0)
        box.paste(255, r.box)
        clipped = ImageChops.multiply(union_img, box)
        r.mask = clipped if any(clipped.histogram()[128:]) else None


def heal_elements(img: Image.Image, els: list[Element], sil: "np.ndarray") -> list[Element]:
    """Ensure sub-boxes of a continuous tower (waist, shaft, crown) fill the true SAM building silhouette
    without internal holes or bleeding outside the building."""
    import numpy as np
    H, W = img.height, img.width
    for e in els:
        if e.kind in REGION_KINDS:
            continue
        x1, y1, x2, y2 = e.box
        in_box = np.zeros((H, W), bool)
        in_box[max(0, y1):min(H, y2), max(0, x1):min(W, x2)] = True
        box_sil = sil & in_box
        sam = (np.asarray(e.mask) > 127) if e.mask is not None else np.zeros((H, W), bool)
        if box_sil.any():
            sam_clean = sam & sil
            if e.kind in {"crown", "tower", "setback"} or sam_clean.sum() < 0.75 * box_sil.sum():
                merged = sam_clean | box_sil
            else:
                merged = sam_clean
            e.mask = Image.fromarray(merged.astype(np.uint8) * 255)
        elif sam.any():
            e.mask = Image.fromarray(sam.astype(np.uint8) * 255)
    return els


async def outline(img: Image.Image, els: list[Element]) -> list[Element]:
    out, _ = await outline_with_hero(img, els, [])
    return out


async def detect_facade_voids(img: Image.Image, full_sil: "np.ndarray") -> tuple[Image.Image | None, bool]:
    """High-resolution Building-ROI negative-space cutout detector (Muse Spark 1.3 + SAM 3.1).

    Crops the native-resolution image tightly to `full_sil` so SAM 3.1 & Muse Spark inspect the
    tower's vertical fins vs smooth recessed glass canyons / interior office windows / mechanical bands
    at 3-5x higher effective spatial resolution, returning a full-image L-mode `void_mask` (255 = void).
    """
    import numpy as np
    from scipy import ndimage as ndi

    H, W = img.height, img.width
    if not full_sil.any():
        return None, False

    ys, xs = np.where(full_sil)
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    if bw < 32 or bh < 48:
        return None, False

    pad_x = max(8, int(round(0.035 * bw)))
    pad_y = max(8, int(round(0.025 * bh)))
    rx0 = max(0, int(xs.min()) - pad_x)
    ry0 = max(0, int(ys.min()) - pad_y)
    rx1 = min(W, int(xs.max()) + 1 + pad_x)
    ry1 = min(H, int(ys.max()) + 1 + pad_y)
    crop_w, crop_h = rx1 - rx0, ry1 - ry0

    roi_raw = img.crop((rx0, ry0, rx1, ry1)).convert("RGB")
    max_d = max(crop_w, crop_h)
    if max_d > 2048:
        sc = 2048.0 / max_d
        rw, rh = max(32, int(round(crop_w * sc))), max(32, int(round(crop_h * sc)))
        roi_img = roi_raw.resize((rw, rh), Image.LANCZOS)
    elif max_d < 960:
        sc = min(2.4, 1152.0 / max(max_d, 1))
        rw, rh = max(32, int(round(crop_w * sc))), max(32, int(round(crop_h * sc)))
        roi_img = roi_raw.resize((rw, rh), Image.LANCZOS)
    else:
        rw, rh = crop_w, crop_h
        roi_img = roi_raw

    roi_sil_crop = full_sil[ry0:ry1, rx0:rx1]
    roi_sil = (
        np.asarray(Image.fromarray(roi_sil_crop.astype(np.uint8) * 255).resize((rw, rh), Image.NEAREST)) > 127
        if (rw, rh) != (crop_w, crop_h)
        else roi_sil_crop.copy()
    )
    sil_area = max(int(roi_sil.sum()), 1)

    try:
        resp = await _client().chat.completions.create(
            model="muse-spark-1.3",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": VOID_PROMPT},
                {"type": "image_url", "image_url": {"url": _data_url(roi_img)}},
            ]}],
            response_format={"type": "json_schema", "json_schema": {"name": "facade_voids", "schema": VOID_SCHEMA, "strict": True}},
        )
        parsed = json.loads(resp.choices[0].message.content)
    except Exception:
        return None, False

    if not parsed.get("has_voids") or not parsed.get("voids"):
        return None, True

    objs: list[PromptObject] = []
    void_specs: dict[int, tuple[int, int, int, int, str, list[PromptPoint]]] = {}

    for idx, v in enumerate(parsed["voids"][:7], start=1):
        vx1, vx2 = sorted((int(v["x1"]), int(v["x2"])))
        vy1, vy2 = sorted((int(v["y1"]), int(v["y2"])))
        bx1 = int(max(0, min(rw - 2, vx1 * rw // 1000)))
        by1 = int(max(0, min(rh - 2, vy1 * rh // 1000)))
        bx2 = int(max(bx1 + 2, min(rw, vx2 * rw // 1000)))
        by2 = int(max(by1 + 2, min(rh, vy2 * rh // 1000)))

        pts: list[PromptPoint] = []
        seen_xy: set[tuple[int, int]] = set()
        for p in v.get("pos_points", [])[:4]:
            px_i = int(max(0, min(rw - 1, int(p["x"]) * rw // 1000)))
            py_i = int(max(0, min(rh - 1, int(p["y"]) * rh // 1000)))
            if (px_i, py_i) not in seen_xy:
                seen_xy.add((px_i, py_i))
                pts.append(PromptPoint(px_i, py_i, "positive"))
        for p in v.get("neg_points", [])[:4]:
            nx_i = int(max(0, min(rw - 1, int(p["x"]) * rw // 1000)))
            ny_i = int(max(0, min(rh - 1, int(p["y"]) * rh // 1000)))
            if (nx_i, ny_i) not in seen_xy:
                seen_xy.add((nx_i, ny_i))
                pts.append(PromptPoint(nx_i, ny_i, "negative"))

        vtype = str(v.get("void_type", "recessed_glass_canyon"))
        void_specs[idx] = (bx1, by1, bx2, by2, vtype, pts)
        objs.append(PromptObject(id=idx, prompts=[FramePrompt(box=PromptBox(bx1, by1, bx2, by2), points=pts)]))
        # Also query point-guided contour (id = 100 + idx) so diagonal S-twist glass grooves never over-fill a wide bounding box
        if any(pt.label == "positive" for pt in pts):
            objs.append(PromptObject(id=100 + idx, prompts=[FramePrompt(points=pts)]))

    if not objs:
        return None, True

    try:
        prompt_str = build_object_prompt(size=(int(rw), int(rh)), objects=objs[:MAX_OBJECTS])
        stream = await _client().responses.create(
            model="sam-3.1",
            stream=True,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt_str},
                {"type": "input_image", "image_url": _data_url(roi_img)},
            ]}],
        )
        res = await parse_responses_stream(stream, image_segmentation_format()).final_result()
    except Exception:
        return None, False

    raw_masks: dict[int, np.ndarray] = {}
    for rec in res.records:
        if type(rec).__name__ != "SegmentationMaskRecord":
            continue
        oid = int(rec.object_id)
        m = rec.mask
        tile = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width) > 0
        full_m = np.zeros((rh, rw), bool)
        t, l = int(rec.bounds.top), int(rec.bounds.left)
        full_m[t:t + m.height, l:l + m.width] |= tile
        raw_masks[oid] = full_m

    roi_void = np.zeros((rh, rw), bool)
    for idx, (bx1, by1, bx2, by2, vtype, pts) in void_specs.items():
        pad_bx = max(4, int(0.04 * (bx2 - bx1)))
        pad_by = max(4, int(0.04 * (by2 - by1)))
        clip_box = np.zeros((rh, rw), bool)
        clip_box[max(0, by1 - pad_by):min(rh, by2 + pad_by), max(0, bx1 - pad_bx):min(rw, bx2 + pad_bx)] = True

        pos_pts = [pt for pt in pts if pt.label == "positive"]
        neg_pts = [pt for pt in pts if pt.label == "negative"]

        def _cand_score(cm: np.ndarray) -> float:
            frac = float(cm.sum()) / sil_area
            pos_hit = float(np.mean([cm[pt.y, pt.x] for pt in pos_pts])) if pos_pts else 1.0
            neg_hit = float(np.mean([cm[pt.y, pt.x] for pt in neg_pts])) if neg_pts else 0.0
            return pos_hit - 1.35 * neg_hit - 0.55 * frac

        candidates: list[np.ndarray] = []
        for cid in (idx, 100 + idx):
            if cid in raw_masks:
                cm = raw_masks[cid] & clip_box & roi_sil
                frac = float(cm.sum()) / sil_area
                if 0.003 <= frac <= 0.48:
                    candidates.append(cm)
        if not candidates:
            continue

        best_c = max(candidates, key=_cand_score)
        c_ys = np.where(best_c.any(axis=1))[0]
        if len(c_ys) == 0:
            continue
        # Reject thin 1-2 floor horizontal shadow seams misidentified as glass canyons
        if vtype == "recessed_glass_canyon" and (c_ys.max() - c_ys.min() + 1) < int(0.08 * rh):
            continue
        roi_void |= best_c

    if not roi_void.any():
        return None, True

    roi_void = ndi.binary_opening(roi_void, structure=np.ones((3, 3), bool))
    roi_void = ndi.binary_closing(roi_void, structure=np.ones((5, 5), bool)) & roi_sil
    if float(roi_void.sum()) > 0.62 * sil_area:
        return None, True

    void_crop = (
        np.asarray(Image.fromarray(roi_void.astype(np.uint8) * 255).resize((crop_w, crop_h), Image.NEAREST)) > 127
        if (rw, rh) != (crop_w, crop_h)
        else roi_void
    )
    full_void = np.zeros((H, W), bool)
    full_void[ry0:ry1, rx0:rx1] = void_crop & roi_sil_crop
    if not full_void.any():
        return None, True
    return Image.fromarray(full_void.astype(np.uint8) * 255), True


async def map_facade(img: Image.Image) -> list[Element]:
    import numpy as np
    els = await propose(img)
    solids = [e for e in els if e.kind not in REGION_KINDS]
    (_, hero), sil = await asyncio.gather(
        outline_with_hero(img, solids, els),
        building_silhouette(img, els),
    )
    # Subtract open sky & cloud wisps from hero so shadowed building wings stay 100% intact without leaking clouds at crown
    from scipy import ndimage as ndi
    H_m, W_m = img.height, img.width
    arr_rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    luma_m = 0.299 * arr_rgb[:, :, 0] + 0.587 * arr_rgb[:, :, 1] + 0.114 * arr_rgb[:, :, 2]
    gx_abs = np.abs(np.diff(luma_m, axis=1, prepend=luma_m[:, :1]))
    gy_abs = np.abs(np.diff(luma_m, axis=0, prepend=luma_m[:1, :]))
    gx_m = gx_abs + gy_abs
    max_c = arr_rgb.max(axis=2)
    sat_m = (max_c - arr_rgb.min(axis=2)) / (max_c + 1e-3)
    blue_ratio = arr_rgb[:, :, 2] / (arr_rgb[:, :, 0] + 12.0)
    yy_norm = np.linspace(0.0, 1.0, H_m, dtype=np.float32)[:, None]

    smooth_sky = (luma_m > 95.0) & (blue_ratio > 0.86) & (gx_m < 6.0) & (~sil)
    bright_cloud_or_sky = (
        smooth_sky
        | ((luma_m > 108.0) & (blue_ratio > 0.88) & (~sil))
        | ((yy_norm < 0.62) & (luma_m > 156.0) & (sat_m < 0.19) & (~sil))
    )
    solid_sam = np.zeros_like(sil)
    for s_el in solids:
        if s_el.mask is not None and s_el.kind in {"crown", "tower", "setback", "spire", "arch"}:
            solid_sam |= (np.asarray(s_el.mask) > 127) & (~bright_cloud_or_sky)
    full_sil = sil | (hero & (~bright_cloud_or_sky)) | solid_sam

    # Shadowed-elevation recovery: when cast shadow splits a facade in half, SAM may only segment the sunlit side
    # even though Muse Spark's element box covers the full textured architectural tier. Require vertical architectural
    # texture AND spatial connectivity to `full_sil` so detached sky clouds never leak in.
    local_tex = ndi.uniform_filter(gx_m, size=15)
    vert_dom = ndi.uniform_filter(gx_abs, size=15) >= 0.72 * ndi.uniform_filter(gy_abs, size=15)
    for e in els:
        if e.kind == "glass":
            continue
        x1, y1, x2, y2 = e.box
        in_box = np.zeros((H_m, W_m), bool)
        in_box[max(0, y1):min(H_m, y2), max(0, x1):min(W_m, x2)] = True
        box_area = int(in_box.sum())
        if box_area < 400:
            continue
        structured = in_box & (~bright_cloud_or_sky) & (local_tex >= 2.8) & vert_dom
        structured = ndi.binary_closing(structured, structure=np.ones((11, 11), bool)) & in_box & (~bright_cloud_or_sky)
        if structured.sum() > 0.72 * box_area and (full_sil & in_box).sum() < 0.72 * structured.sum():
            # Keep only components physically connected to the SAM building silhouette
            lab_s, n_s = ndi.label(structured | (full_sil & in_box))
            for cid in range(1, n_s + 1):
                comp = lab_s == cid
                if (comp & full_sil).sum() >= 120:
                    full_sil |= comp & structured

    full_sil = ndi.binary_fill_holes(full_sil)

    # High-resolution Building-ROI SAM 3.1 negative-space cutout segmentation (glazed twist canyons, interior windows, louver bands)
    void_mask_img, void_audited = await detect_facade_voids(img, full_sil)

    heal_elements(img, solids, full_sil)
    solids = [e for e in solids if e.mask is not None and any(e.mask.histogram()[128:])]
    regions = [e for e in els if e.kind in REGION_KINDS]
    # Clip region elements against the full authoritative silhouette (not just solids, since Spark may label a whole shaft as 'glass' or 'fins'!)
    sil_img = Image.fromarray(full_sil.astype(np.uint8) * 255)
    for r in regions:
        box = Image.new("L", img.size, 0)
        box.paste(255, r.box)
        clipped = ImageChops.multiply(sil_img, box)
        r.mask = clipped if any(clipped.histogram()[128:]) else None
    out = solids + [e for e in regions if e.mask is not None]
    # Guarantee full silhouette coverage: if any major part of full_sil (>12%) wasn't covered by elements, add it as tower/facade body
    covered = np.zeros((img.height, img.width), bool)
    for e in out:
        covered |= np.asarray(e.mask) > 127
    missing = full_sil & (~covered)
    if missing.sum() > 0.12 * max(full_sil.sum(), 1):
        ys, xs = np.where(missing)
        nid = max((e.id for e in out), default=0) + 1
        out.append(Element(
            nid, "main facade body", "tower",
            (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            Image.fromarray(missing.astype(np.uint8) * 255),
        ))
    if out:
        out[0].void_mask = void_mask_img
        out[0].void_audited = void_audited
    return out


if __name__ == "__main__":  # python -m app.mapper photo.png outdir
    import sys
    src, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    im = Image.open(src)
    els = asyncio.run(map_facade(im))
    for e in els:
        e.mask.save(f"{outdir}/{e.id}_{e.kind}.png")
        print(e.id, e.kind, e.label, e.box)
