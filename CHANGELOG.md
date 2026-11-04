# Changelog

Event work only (26 Oct – 4 Nov 2026). One entry per feature, dated.

## 2026-10-31
- Scaffold: FastAPI service, smoke script for SAM 3.1 / Muse Spark / Muse Image.
- Verified all three models respond from Dubai; SAM box/point prompts work, concept prompts weak on architecture.

## 2026-10-31 (F1)
- F1 read the building: Muse Spark proposes element boxes, SAM 3.1 outlines them; band/fins/glass regions clipped to their parent. `POST /map` and `python -m app.mapper`. First test on a night photo of two curved towers: 7 elements in ~18 s, no hand-drawn zones.

## 2026-10-31 (F2)
- F2 show plan: Muse Spark writes a timed plan (JSON schema); every step must name real element ids, invalid plans are re-asked once. `POST /plan`, `python -m app.planner`.

## 2026-10-31 (F4)
- F4 element-driven motion: own code, no model. Climb/descend/sweep/pulse/shimmer/fill/hold/text run inside each element mask along its own extent, lit through the photo texture. `python -m app.motion`, H.264 preview via ffmpeg.
- F4 renders light only on linear fixtures (virtual fixtures derived from each element's edges), not as a surface wash, to match how real facades are lit.
- F4 rewritten: from a photo alone, virtual fixtures are placed on every element (dash shape and spacing calibrated on a real surveyed tower: 7,140 fixtures), each fixture lit as one unit with glow. No UV map needed at runtime; the survey is only the calibration reference.
- F4 uses a real fixture survey when given: reads every fixture from the UV mask, splits it into elevations, maps the matching elevation onto the tower silhouette (homography from the SAM mask). Without a survey it falls back to virtual fixtures.
- Facets (as in the existing app): each visible face of the building gets its own canvas-rect -> photo-quad homography. Faces come from Muse Spark, the tower silhouette from SAM ("skyscraper"), and the shared corner between two faces is searched for the best fit to that silhouette.

## 2026-11-01 (Fixture Rigs, F3 Visuals, F5 Readability & Studio UI)
- **Discrete architectural fixture engine (`app/fixtures.py`)**: traces real mullions, curved/helical fins (2D structure tensor streamlines), floor-slab spandrels, crown, and setback spines inside healed SAM 3.1 silhouettes (`app/mapper.py`) to propose 3+ fixture layout options per photo (*Full Mullion & Fin Pixel Matrix*, *Crown, Setback & Structural Spine Rig*, *Horizontal Floor-Slab Ribbons*, plus silhouette-warped *As-Built UV Survey* and *Detected Night Rig* when present). Precomputes anti-aliased linear LED bar footprints and tight 2 px mullion grazing so curtain-wall glass between mullions stays dark with zero surface wash.
- **Architecture-aware concept generation (`app/planner.py`)**: `suggest_concepts()` (`muse-spark-1.3`) proposes 3 distinct lighting concepts tailored to the building's mapped elements and selected fixture count, alongside unified multi-zone wave/climb/chase/aurora motion in `app/motion.py`.
- **F3 360° light-art synthesizer (`app/visuals.py`)**: generates horizontally-seamless `muse-image-1.0` liquid-silk plates and curl-advects them across a periodic 360° building canvas (`u_360 in [0, 1)`) with 3D helical vortex ribbons, rising comet streams, arabesque diamond prism lattices, and Burj-style champagne diamond scintillation.
- **360° full-building canvas -> visible photo side projection (`app/motion.py`, `app/fixtures.py`)**: content runs continuously across all 4 elevations of the building with smooth cross-dissolves (never dipping to black or leaving dead cutout zones), while the on-building video samples only the camera photo's visible angular sector (`90°` 1-side, `162°` curved/2-side, or `240°` wrap) with 3D cylindrical foreshortening and dusk/twilight photo preservation.
- **F5 readability check & 1-click fix (`app/checker.py`)**: combines physical fixture-pitch column×row audit with `muse-spark-1.3` vision review to flag unreadable text or sparse zones before test night, plus deterministic auto-fix & re-render.
- **Standalone Facade Mapper Studio (`app/main.py`, `app/static/index.html`)**: full interactive web UI (`GET /`) and session API (`/api/analyze`, `/api/analyze-stitch`, `/api/concepts`, `/api/generate`, `/api/fix-readability`) exporting on-building camera preview MP4, 360° 4-elevation viewport map, edge-to-edge 360° media-server deliverable MP4, and 2D UV fixture mask PNG.
- **4-Side Architectural Facade Map Stitcher (`app/stitcher.py`, `POST /api/analyze-stitch`)**: lets users upload all 4 building elevations (North · East · South · West), segments them concurrently via SAM 3.1 + Muse Spark, aligns crown-to-base height across varied camera distances, harmonizes twilight exposure across seams, and maps all 4 sides' mullions/fins into `[0..0.25..0.50..0.75..1.0)` quadrants of a pixel-registered whole-facade video map + UV mask.

## 2026-11-03 (click-to-edit, hosting measurements)
- **F1 click-to-edit:** click the element map to add an element (SAM point prompt) or remove one with the "x"; rigs are rebuilt from the edited map (`POST /api/sessions/{id}/elements`).
- **Hardening:** unreadable uploads return a clean 400; an F3 fallback to the procedural texture is logged; a missing `MODEL_API_KEY` gives a clear error.
- **Hosting measurements:** peak memory is about 650 MB for `/api/analyze` and 1.8 GB for `/api/generate`; Meta's API rejected requests from Render's Frankfurt region. Docker support kept; `render.yaml` removed. See README "Deploy".
- **Metrics:** `scripts/metrics.py` and `METRICS.md`, misses included.

## 2026-11-04 (Studio redesign, sign-in, mask and coverage fixes, samples)
- **Landing page, demo sign-in and Studio redesign:** `/` landing, `/login`, `/studio`. Sign-in is env-driven (`DEMO_PASSWORD`, `DEMO_USERS`) and gates the Studio and all `/api/*` routes; with no password set, auth is off. The Studio shows a Photo -> Layout -> Show -> Check flow, live progress, inline errors, and scales to the viewport.
- **F1 mask fix:** SAM returns each mask tile downscaled; the paste ignored that and shrank large masks toward the upper-left of the object. Tiles are now resized to their bounds before pasting (`place_tile` in `app/mapper.py`). Index Tower sample: 492 fixtures (mostly in the sky) -> 2,315 on the tower.
- **F4 survey rig coverage:** two surveyed elevations are mapped onto the two visible faces, and the short elevation onto the podium. Silhouette coverage 0.82 -> 0.96, base 0.54 -> 0.93.
- **F5 text fix:** the auto-fix keeps a readable token ("UAE NATIONAL DAY 55" -> "UAE 55", was "UAE N").
- **Auth fix:** `.env` is loaded at startup so the sign-in gate is on from the first request.
- **Samples:** bundled Index Tower sample (Aidas U., CC BY 3.0) in `app/static/samples/`; the synthetic four-sides demo is labelled as one photo reused.

## Feedback logged (4 Nov)
- User feedback recorded as roadmap, not implemented: adjustable mask size/width, movable and reshapable auto masks, selectable linear and pixel-dot fixtures. See README Roadmap.
