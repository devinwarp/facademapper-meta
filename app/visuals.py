"""F3 & 360° Museum-of-Light Architectural Shader Engine (Meta Muse Image 1.0 + Sculptural Light Art).

Embodies 3S Lighting's architectural media-facade craft (landmark-tower standard):
- "The tower is a sculpture first and a screen never; emerge, do not overlay."
- Content lives on a continuous 360° wrap-around building canvas `(u_360 in [0, 1), v in [0, 1])`
  spanning all elevations seamlessly (`u=0` wraps with zero seam to `u=1`).
- Low-frequency bicubic-upscaled periodic noise fields + restrained 3–4 stop palettes + HDR `[0..380]`
  specular focal cores (rising star, spine gathering, 5-lobed night bloom, braided silk strands,
  woven Sadu tapestry, and sparse diamond dew scintillation).
- `generate_concept_texture()` prompts Meta's `muse-image-1.0` for painterly silk & light gradients
  and palette-locks them to the concept's curated chromaticity so off-palette neon never muddies the facade.
"""
from __future__ import annotations

import base64, io
import numpy as np
from PIL import Image, ImageFilter

from app.mapper import _client

_NOISE_CACHE: dict[tuple[int, int, int, int], np.ndarray] = {}


def _hex_rgb(c: str) -> np.ndarray:
    c = c.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        return np.array([230.0, 233.0, 240.0], dtype=np.float32)
    return np.array([int(c[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def _periodic_noise(seed: int, gw: int = 20, gh: int = 20, size: int = 256) -> np.ndarray:
    """Generate a smooth bicubic-upscaled 2D noise field (size x size x 1 float32 in [0, 1])
    that wraps seamlessly in both `u` and `v` via 3x3 periodic pre-tiling."""
    key = (seed, gw, gh, size)
    cached = _NOISE_CACHE.get(key)
    if cached is not None:
        return cached
    rng = np.random.default_rng(seed)
    grid = (rng.random((gh, gw), dtype=np.float32) * 255.0).astype(np.uint8)
    tiled = np.tile(grid, (3, 3))
    up = Image.fromarray(tiled, mode="L").resize((size * 3, size * 3), Image.BICUBIC)
    center = (np.asarray(up, dtype=np.float32)[size:2 * size, size:2 * size] / 255.0)[..., None]
    _NOISE_CACHE[key] = center
    return center


def make_horizontally_seamless(arr: np.ndarray, margin: int = 56) -> np.ndarray:
    """Cross-fade left/right edges so U=0 and U=1 wrap around a 360° building without any seam."""
    H, W, C = arr.shape
    m = min(margin, W // 4)
    if m < 4:
        return arr
    out = arr.copy()
    t = np.linspace(0.0, 1.0, m, dtype=np.float32)[None, :, None]
    w = 0.5 - 0.5 * np.cos(np.pi * t)
    rolled = np.roll(arr, shift=W // 2, axis=1)
    seam_band = rolled[:, W // 2 - m:W // 2 + m, :]
    out[:, :m, :] = seam_band[:, m:, :] * (1.0 - w) + arr[:, :m, :] * w
    out[:, W - m:, :] = arr[:, W - m:, :] * (1.0 - w[:, ::-1, :]) + seam_band[:, :m, :] * w[:, ::-1, :]
    return out


def build_palette_lut(colors: list[str], n: int = 256) -> np.ndarray:
    """Build a smooth periodic 256x3 RGB colour lookup table from curated hex stops."""
    parsed = [_hex_rgb(c) for c in (colors or ["#3A7BD5", "#B58FD6", "#E6E9F0"])]
    vivid = [c for c in parsed if c.max() > 18.0] or parsed[:1]
    if len(vivid) == 1:
        base = vivid[0]
        hi = np.clip(base * 1.12 + 18.0, 0.0, 255.0)
        lo = np.clip(base * 0.78, 0.0, 255.0)
        vivid = [base, hi, base, lo]

    k = len(vivid)
    stack = np.stack(vivid, axis=0)  # (k, 3)
    pos = np.linspace(0.0, float(k), n, endpoint=False, dtype=np.float32)
    i0 = np.floor(pos).astype(np.int32) % k
    i1 = (i0 + 1) % k
    f = (pos - np.floor(pos))[:, None]
    f = f * f * (3.0 - 2.0 * f)
    return (stack[i0] * (1.0 - f) + stack[i1] * f).astype(np.float32)


def sample_lut(lut: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Linearly sample a periodic 256x3 LUT at arbitrary array `phase`."""
    n = len(lut)
    p = (phase % 1.0) * n
    i0 = np.floor(p).astype(np.int32) % n
    i1 = (i0 + 1) % n
    f = (p - np.floor(p))[..., None].astype(np.float32)
    return lut[i0] * (1.0 - f) + lut[i1] * f


def sample_bilinear_periodic(tex: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Smooth bilinear sampling of `tex` (HxWxC) with periodic wrap in both `u` and `v`."""
    th, tw = tex.shape[:2]
    uf = (u % 1.0) * tw
    vf = (v % 1.0) * th
    x0 = np.floor(uf).astype(np.int32) % tw
    y0 = np.floor(vf).astype(np.int32) % th
    x1 = (x0 + 1) % tw
    y1 = (y0 + 1) % th
    fx = (uf - np.floor(uf))[..., None].astype(np.float32)
    fy = (vf - np.floor(vf))[..., None].astype(np.float32)

    c00 = tex[y0, x0]
    c10 = tex[y0, x1]
    c01 = tex[y1, x0]
    c11 = tex[y1, x1]
    top = c00 * (1.0 - fx) + c10 * fx
    bot = c01 * (1.0 - fx) + c11 * fx
    return top * (1.0 - fy) + bot * fy


def procedural_texture(colors: list[str], size: int = 512) -> np.ndarray:
    """Museum-of-Light 360°-seamless pearlescent silk & bicubic cloud plate (HxWx3 float32, 0..255)."""
    lut = build_palette_lut(colors, 256)
    v, u = np.meshgrid(
        np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float32),
        np.linspace(0.0, 1.0, size, endpoint=False, dtype=np.float32),
        indexing="ij",
    )
    n1 = sample_bilinear_periodic(_periodic_noise(101, 16, 16, 256), u, v)[..., 0]
    n2 = sample_bilinear_periodic(_periodic_noise(202, 24, 24, 256), u + 0.25, v + 0.17)[..., 0]

    # Broad, low-frequency diagonal silk flow (reads cleanly at wide architectural pitch)
    diag = 0.55 * v + 0.35 * np.sin(2.0 * np.pi * u) + 0.42 * (n1 - 0.5) + 0.22 * (n2 - 0.5)
    silk_col = sample_lut(lut, diag)

    veil = 0.64 + 0.36 * np.sin(2.0 * np.pi * (u + 0.75 * v + 0.35 * n1))
    sheen = np.clip((n1 * 0.6 + n2 * 0.4 - 0.54) / 0.34, 0.0, 1.0) ** 2.2
    pearl = np.array([236.0, 240.0, 248.0], dtype=np.float32)
    out = silk_col * veil[..., None] + sheen[..., None] * pearl * 0.36
    return np.clip(out, 0.0, 255.0).astype(np.float32)


async def generate_concept_texture(visual_prompt: str, colors: list[str], use_model: bool = True) -> tuple[np.ndarray, str]:
    """Generate a 512x512 360°-seamless light-art plate via Meta's `muse-image-1.0` (F3),
    remapped onto the curated architectural palette so colours remain refined and restrained."""
    if use_model:
        palette_str = ", ".join(colors[:5]) if colors else "#3A7BD5, #B58FD6, #E6E9F0"
        prompt = (
            f"A flat two-dimensional minimalist architectural light-art field filling the entire frame. "
            f"Smooth low-frequency pearlescent silk gradients, soft atmospheric luminescence, and subtle "
            f"organic flow strictly in palette {palette_str}. {visual_prompt}. "
            f"Restrained museum-grade lighting aesthetic, smooth tonal transitions, edge-to-edge colour. "
            f"No harsh rainbow stripes, no border, no perspective, no building, no tower, no windows, no text."
        )
        try:
            r = await _client().images.generate(
                model="muse-image-1.0",
                prompt=prompt[:950],
                size="1024x1024",
            )
            b64 = r.data[0].b64_json
            if b64:
                im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").resize((512, 512), Image.LANCZOS)
                # Smooth out high-frequency clutter that would alias at architectural mullion pitch
                im = im.filter(ImageFilter.GaussianBlur(radius=4.0))
                arr = np.asarray(im, dtype=np.float32)
                if arr.max() > 35.0:
                    p96 = max(float(np.percentile(arr, 96)), 80.0)
                    arr = np.clip(arr * (242.0 / p96), 0.0, 255.0)
                    # Palette-lock: map Muse Image's organic luminance & chromatic flow onto `lut`
                    lut = build_palette_lut(colors, 256)
                    lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]) / 255.0
                    p_lo, p_hi = float(np.percentile(lum, 6)), float(np.percentile(lum, 94))
                    lum_norm = np.clip((lum - p_lo) / max(p_hi - p_lo, 0.12), 0.0, 1.0)
                    lut_mapped = sample_lut(lut, lum_norm * 0.88 + 0.06) * (0.52 + 0.58 * lum_norm[..., None])
                    proc = procedural_texture(colors, size=512)
                    blended = np.clip(0.64 * lut_mapped + 0.18 * arr + 0.18 * proc, 0.0, 255.0)
                    seamless = make_horizontally_seamless(blended, margin=64)
                    return seamless.astype(np.float32), "muse-image-1.0"
        except Exception as e:
            print(f"[F3] muse-image failed, using procedural-shader: {e!r}")
    return procedural_texture(colors, size=512), "procedural-shader"


def sample_texture_at_fixtures(
    tex: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    t: float,
    scroll_u: float = 0.032,
    scroll_v: float = -0.075,
) -> np.ndarray:
    """Slow, meditative curl-warped bilinear sampling of `tex` at `(u, v)` at time `t`."""
    uw = u + scroll_u * t + 0.045 * np.sin(2.0 * np.pi * (v * 1.1 - 0.11 * t))
    vw = v + scroll_v * t + 0.042 * np.sin(2.0 * np.pi * (u - 0.09 * t))
    return sample_bilinear_periodic(tex, uw, vw)


def _sstep(edge0: np.ndarray | float, edge1: np.ndarray | float, x: np.ndarray | float) -> np.ndarray:
    t = np.clip((x - edge0) / np.maximum(edge1 - edge0, 1e-5), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def evaluate_360_shader(
    motion: str,
    colors: list[str],
    u: np.ndarray,
    v: np.ndarray,
    q: float,
    t: float,
    tex: np.ndarray | None = None,
    ambient_underlay: bool = True,
    spine_u: float = 0.475,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a 3S Lighting 'Museum of Light' 360°-periodic architectural shader on `(u, v)`.

    Supports HDR highlight headroom (`rgb` in `[0, 380]`) so focal stars, spine crests, petal blooms,
    and braided strands develop crisp white-core optical lens bloom while preserving negative space.
    """
    h = 1.0 - v  # 0 at podium base -> 1 at crown top
    parsed_vivid = [_hex_rgb(c) for c in (colors or ["#3A7BD5", "#B58FD6", "#E6E9F0"]) if _hex_rgb(c).max() > 18.0]
    if not parsed_vivid:
        parsed_vivid = [np.array([230.0, 233.0, 240.0], dtype=np.float32)]
    single_solid = len(parsed_vivid) <= 1
    lut = build_palette_lut(colors, 256)

    # Periodic signed distance to the architectural spine in [-0.5, 0.5)
    d_spine = ((u - spine_u + 0.5) % 1.0) - 0.5
    abs_spine = np.abs(d_spine)

    # Smooth periodic bicubic noise fields (24x24 museum-grade scale, never noisy static)
    n_flow = sample_bilinear_periodic(_periodic_noise(17, 18, 18, 256), u + 0.025 * t, v - 0.055 * t)[..., 0]
    n_fine = sample_bilinear_periodic(_periodic_noise(53, 28, 28, 256), u - 0.018 * t, v - 0.035 * t)[..., 0]
    n_dew = sample_bilinear_periodic(_periodic_noise(89, 44, 64, 256), u, v)[..., 0]

    if tex is not None and not single_solid:
        ai_silk = sample_texture_at_fixtures(tex, u, v, t, scroll_u=0.030, scroll_v=-0.068)
    else:
        ai_silk = sample_lut(lut, 0.62 * h + 0.28 * np.sin(2.0 * np.pi * u) + 0.35 * (n_flow - 0.5) - 0.10 * t)

    pearl_white = np.array([242.0, 246.0, 255.0], dtype=np.float32)
    warm_star = np.array([255.0, 244.0, 232.0], dtype=np.float32)

    if motion == "aurora":
        # S1 — Pearlescent Iridescence & Spine Gathering (exact viz.py s1_drift + spine wcol concentration):
        c0 = parsed_vivid[0]
        c1 = parsed_vivid[1 % len(parsed_vivid)]
        c2 = parsed_vivid[2 % len(parsed_vivid)]

        phase_d = 0.85 * np.sin(2.0 * np.pi * u) + 0.65 * v + 0.055 * t + 0.28 * n_flow
        p1 = (0.5 + 0.5 * np.sin(2.0 * np.pi * phase_d))[..., None]
        p2 = (0.5 + 0.5 * np.sin(2.0 * np.pi * phase_d + 2.1))[..., None] * 0.58
        pal_col = c0 * (1.0 - p1) + c1 * p1
        pal_col = pal_col * (1.0 - p2) + c2 * p2

        # Modulate gently with Muse Image luminance variation without muddying pure azure/lilac hue
        ai_lum = (0.299 * ai_silk[..., 0] + 0.587 * ai_silk[..., 1] + 0.114 * ai_silk[..., 2]) / 255.0
        base_field = pal_col * (0.82 + 0.32 * ai_lum)[..., None]

        # Gathering to structural spine (`wcol` in viz.py): wings dim to ~28% while spine blazes at 138%
        wcol = np.exp(-((abs_spine / 0.085) ** 2))
        gather = 0.62 + 0.38 * np.sin(np.pi * np.clip(q, 0.0, 1.0))
        brightness = (1.0 - gather * (0.72 - 0.72 * wcol)) * (1.0 + gather * wcol * 0.42)
        rgb = base_field * brightness[..., None]
        return np.ones_like(u, dtype=np.float32), np.clip(rgb, 0.0, 360.0)

    if motion in ("climb", "descend"):
        # S2 — Suhail Rising Star & Thermal Horizon / Cascade:
        # Wavefront led by a blazing warm-white star on the spine; amber crest cools smoothly into rose -> indigo -> midnight
        reach = min(1.0, q / 0.88)
        ripple = 0.015 * (n_flow - 0.5) if not single_solid else 0.0
        coord = h if motion == "climb" else v
        dist = (reach * 1.04 + ripple) - coord  # >0 behind rising/descending wavefront

        alpha_wave = np.clip(0.5 + dist / 0.055, 0.0, 1.0).astype(np.float32)
        crest = np.exp(-((dist / 0.036) ** 2)).astype(np.float32)

        if single_solid:
            lut_col = sample_lut(lut, coord * 1.1 - 0.25 * t)
            body_rgb = lut_col + crest[..., None] * 40.0
            return alpha_wave, np.clip(body_rgb, 0.0, 255.0)

        # Non-periodic 4-stop thermal cooling gradient from wavefront (stop 0: warm amber) down into wake (rose -> indigo -> deep)
        c_front = parsed_vivid[0]
        c_mid1 = parsed_vivid[1 % len(parsed_vivid)]
        c_mid2 = parsed_vivid[2 % len(parsed_vivid)]
        c_cool = parsed_vivid[3 % len(parsed_vivid)] if len(parsed_vivid) >= 4 else c_mid2 * 0.55

        w_1 = _sstep(0.08, 0.26, dist)[..., None]
        w_2 = _sstep(0.24, 0.46, dist)[..., None]
        w_3 = _sstep(0.44, 0.68, dist)[..., None]
        col = c_front[None] * (1.0 - w_1) + c_mid1[None] * w_1
        col = col * (1.0 - w_2) + c_mid2[None] * w_2
        col = col * (1.0 - w_3) + c_cool[None] * w_3

        # Subtle rising heat shimmer near wavefront, calming in the cooled wake
        shim = 0.80 + 0.22 * (0.5 + 0.5 * np.sin(2.0 * np.pi * (n_fine * 4.0 + coord * 2.5 - 0.85 * t)))
        body_rgb = col * shim[..., None]

        # Suhail rising star on structural spine (`sigma_u = 0.022`, `* 2.2` incandescent white-gold!)
        star = np.exp(-((abs_spine / 0.022) ** 2 + (dist / 0.030) ** 2)).astype(np.float32)
        horizon_band = crest * (0.35 + 0.65 * np.exp(-((abs_spine / 0.14) ** 2)))

        # Sparse nocturnal diamond dew sparkles strictly in the deep cooled wake (`dist > 0.28`)
        twinkle = _sstep(0.26, 0.42, dist) * (0.5 + 0.5 * np.sin(2.0 * np.pi * (1.4 * t + n_dew * 15.0))) ** 6.0
        dew = np.where(n_dew > 0.86, ((n_dew - 0.86) / 0.14) * twinkle, 0.0)

        body_rgb = (
            body_rgb
            + horizon_band[..., None] * (0.65 * c_front + 0.35 * warm_star) * 0.55
            + star[..., None] * warm_star * 2.15
            + dew[..., None] * pearl_white * 1.45
        )

        if ambient_underlay:
            # Keep unreached crown strictly <= 22 RGB so pre-arrival facade stays dark nocturnal stone/glass
            amb = np.clip(col * 0.07, 0.0, 22.0)
            rgb = body_rgb * alpha_wave[..., None] + amb * (1.0 - alpha_wave[..., None])
            return np.ones_like(u, dtype=np.float32), np.clip(rgb, 0.0, 420.0)
        return alpha_wave, np.clip(body_rgb, 0.0, 420.0)

    if motion == "pulse":
        # S3 — Night Bloom (30% Slate Veil of Light + Pure Vivid 5-Lobed Botanical Unfolding on Spine):
        c_veil1 = parsed_vivid[0]
        c_veil2 = parsed_vivid[1 % len(parsed_vivid)] if len(parsed_vivid) >= 3 else parsed_vivid[0] * 0.72
        accent_col = parsed_vivid[-1] if len(parsed_vivid) >= 2 else np.array([255.0, 111.0, 97.0], np.float32)

        veil_grad = c_veil1 * (1.0 - v[..., None]) + c_veil2 * v[..., None]
        veil_breath = 0.24 + 0.08 * np.sin(2.0 * np.pi * (u + 0.04 * t + 0.22 * n_flow))
        veil = veil_grad * veil_breath[..., None]

        # Scale horizontal radius to single-elevation span (`dx = d_spine * 2.35`) so the 5-lobed bloom sits
        # sculpturally inside the tower silhouette surrounded by calm slate veil (and is 0 at `|d_spine|=0.5`!)
        dx = d_spine * 2.35
        dy = v - 0.42
        rad = np.sqrt(dx * dx + dy * dy)
        theta = np.arctan2(dy, dx)

        grow = 0.58 + 0.38 * (0.5 - 0.5 * np.cos(2.0 * np.pi * (0.22 * t + 0.35 * q)))
        R = 0.28 * grow
        lobe = 1.0 + 0.32 * np.cos(5.0 * theta + 0.4 - 0.25 * t) + 0.26 * np.clip(-np.sin(theta), 0.0, 1.0)
        inside = _sstep(-0.045, 0.035, R * lobe - rad)
        seed_heart = np.exp(-((rad / 0.048) ** 2))

        rgb = (
            veil * (1.0 - 0.45 * inside[..., None])
            + inside[..., None] * accent_col * 1.08
            + seed_heart[..., None] * (0.55 * accent_col + 0.45 * warm_star) * 1.55
        )
        return np.ones_like(u, dtype=np.float32), np.clip(rgb, 0.0, 400.0)

    if motion == "chase":
        # Option B — Braided Architectural Strands (exact viz.py thumb_strands contrast on deep negative space):
        # Quiet 10% nocturnal whisper between strands so 7 razor-crisp cosine silk strands pop with 12:1 contrast!
        base_tone = sample_lut(lut, 0.65 * h + 0.20 * n_flow - 0.10 * t)
        veil = np.maximum(base_tone * 0.11, np.array([6.0, 10.0, 24.0], dtype=np.float32))

        strand_acc = np.zeros(u.shape + (3,), dtype=np.float32)
        sky_cyan = parsed_vivid[1 % len(parsed_vivid)] if len(parsed_vivid) >= 2 else np.array([82.0, 201.0, 238.0], np.float32)
        for i_s in range(7):
            off = ((i_s - 3) / 3.5) * 0.135  # scaled to elevation span so all 7 braided strands weave inside view!
            phase_s = v * np.pi * 1.42 - 0.48 * t
            u_strand = (spine_u + off * np.cos(phase_s)) % 1.0
            du_s = np.abs(((u - u_strand + 0.5) % 1.0) - 0.5)
            strand_w = np.exp(-((du_s / 0.0082) ** 2))
            pulse_v = 0.84 + 0.34 * (0.5 + 0.5 * np.sin(2.0 * np.pi * (h * 1.25 - 0.42 * t + i_s * 0.19)))
            c_s = sky_cyan if i_s in (1, 5) else (parsed_vivid[0] if i_s == 3 and len(parsed_vivid) >= 3 else pearl_white)
            strand_acc = strand_acc + (strand_w * pulse_v)[..., None] * c_s * 1.32

        rgb = veil + strand_acc
        return np.ones_like(u, dtype=np.float32), np.clip(rgb, 0.0, 390.0)

    if motion in ("sweep_left_to_right", "sweep_right_to_left"):
        # Option A — Woven Sadu / Horizon Silk Weave:
        # Architectural diamond-chevron silk bands sweeping smoothly around the 360° perimeter
        direction = -1.0 if motion == "sweep_left_to_right" else 1.0
        az = u + direction * 0.26 * t
        chevron = az + 0.16 * np.cos(2.0 * np.pi * (2.0 * h - 0.18 * t))
        w1 = 0.5 + 0.5 * np.cos(2.0 * np.pi * chevron)
        w2 = 0.5 + 0.5 * np.cos(2.0 * np.pi * (2.0 * az - 1.5 * h))

        lut_col = sample_lut(lut, chevron + 0.25 * (n_flow - 0.5))
        weave = (w1 ** 2.6) * 0.72 + (w2 ** 4.5) * 0.36
        rgb = (0.35 * ai_silk + 0.65 * lut_col) * (0.36 + 0.78 * weave)[..., None] + ((w1 ** 6.0) * 0.45)[..., None] * pearl_white
        return np.ones_like(u, dtype=np.float32), np.clip(rgb, 0.0, 320.0)

    if motion == "shimmer":
        # Nocturnal Indigo & Sparse Dew Scintillation (never harsh disco strobe):
        lut_col = sample_lut(lut, 0.70 * h + 0.30 * (n_flow - 0.5) - 0.10 * t)
        velvet = (0.40 * ai_silk + 0.60 * lut_col) * (0.48 + 0.34 * n_flow)[..., None]

        # Sparse, slow-breathing starlight glints anchored to `n_dew`
        osc1 = 0.5 + 0.5 * np.sin(2.0 * np.pi * (0.95 * t + n_dew * 17.0 + u * 3.0))
        osc2 = 0.5 + 0.5 * np.cos(2.0 * np.pi * (0.72 * t - n_fine * 11.0 + h * 5.0))
        sparkle = np.where(n_dew > 0.79, ((n_dew - 0.79) / 0.21) * ((osc1 * osc2) ** 5.0), 0.0)
        spine_gleam = np.exp(-((abs_spine / 0.08) ** 2)) * 0.24
        rgb = velvet * (1.0 + spine_gleam[..., None]) + sparkle[..., None] * pearl_white * 1.45
        return np.ones_like(u, dtype=np.float32), np.clip(rgb, 0.0, 350.0)

    # fill / hold / logo: Calm Architectural Veil with Spine & Crown Presence
    if single_solid:
        lut_col = sample_lut(lut, h * 0.15)
        return np.ones_like(u, dtype=np.float32), np.clip(lut_col, 0.0, 255.0)

    lut_col = sample_lut(lut, 0.68 * h + 0.32 * (n_flow - 0.5) - 0.08 * t)
    spine_lift = 0.76 + 0.34 * np.exp(-((abs_spine / 0.13) ** 2))
    crown_rim = np.exp(-(((1.0 - h) / 0.09) ** 2)) * 0.38
    rgb = (0.42 * ai_silk + 0.58 * lut_col) * (spine_lift * (0.76 + 0.28 * n_flow))[..., None] + crown_rim[..., None] * pearl_white
    return np.ones_like(u, dtype=np.float32), np.clip(rgb, 0.0, 310.0)
