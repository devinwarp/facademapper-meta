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
