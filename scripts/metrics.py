"""Per-image metrics for F1 (mapper), F2 (planner), F5 (checker). Spends API credits.

Run: uv run python scripts/metrics.py data/a.png data/b.png > metrics.json
Prints one JSON object per image to stdout (progress goes to stderr).
"""
import asyncio, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image

from app import planner
from app.checker import apply_readability_fix, check_readability
from app.fixtures import propose_fixture_rigs
from app.mapper import map_facade
from app.motion import build_rig_from_option, frame_at

BRIEF = "Slow pearlescent drift rising from podium to crown, resolving into a calm crown glow"


class CountingClient:
    """Wraps planner._client so we can count chat calls (re-asks = calls - 1)."""
    calls = 0

    def __init__(self, real):
        self.chat = self
        self.completions = self
        self._real = real

    async def create(self, **kw):
        CountingClient.calls += 1
        return await self._real.chat.completions.create(**kw)


async def run(path: str) -> dict:
    out: dict = {"image": path, "stages_s": {}, "errors": []}
    img = Image.open(path).convert("RGB")
    out["size"] = list(img.size)

    def lap(name, t0):
        out["stages_s"][name] = round(time.perf_counter() - t0, 1)

    t = time.perf_counter()
    try:
        els = await map_facade(img)
    except Exception as e:
        out["errors"].append(f"F1 map_facade: {type(e).__name__}: {str(e)[:200]}")
        return out
    lap("F1_map", t)
    area = img.width * img.height
    out["elements"] = len(els)
    out["elements_no_mask"] = sum(e.mask is None for e in els)
    out["elements_tiny_mask"] = sum(
        e.mask is not None and (np.asarray(e.mask) > 127).sum() < 0.002 * area for e in els
    )
    out["kinds"] = sorted({e.kind for e in els})

    t = time.perf_counter()
    try:
        opts, night, _ = propose_fixture_rigs(img, els)
    except Exception as e:
        out["errors"].append(f"rig: {type(e).__name__}: {str(e)[:200]}")
        return out
    lap("rig_survey", t)
    opt = next((o for o in opts if o.recommended), opts[0])
    out["fixtures"] = opt.fixture_count
    fcounts = {eid: len(idx) for eid, idx in opt.elem_membership.items()}

    planner._client = (lambda real: lambda: CountingClient(real()))(planner._client)
    t = time.perf_counter()
    try:
        await planner.suggest_concepts(img, els, opt.summary_dict())
        lap("F2a_concepts", t)
    except Exception as e:
        out["errors"].append(f"F2a concepts: {type(e).__name__}: {str(e)[:200]}")

    CountingClient.calls = 0
    t = time.perf_counter()
    plan = None
    try:
        plan = await planner.make_plan(img, els, BRIEF, 16, elem_fixture_counts=fcounts)
        out["grounding_pass"] = not planner.check_grounding(plan, {e.id for e in els})
    except Exception as e:
        out["grounding_pass"] = False
        out["errors"].append(f"F2b plan: {type(e).__name__}: {str(e)[:200]}")
    lap("F2b_plan", t)
    out["plan_model_calls"] = CountingClient.calls
    out["reasks"] = max(CountingClient.calls - 1, 0)
    if plan is None:
        return out
    out["plan_steps"] = len(plan.steps)

    t = time.perf_counter()
    try:
        rig = build_rig_from_option(night, opt)
        still = Image.fromarray(frame_at(rig, plan, plan.duration_s * 0.56))
        rep = await check_readability(still, els, opt, plan, use_model=True)
        out["flags_raised"] = len(rep.flags)
        out["flags_critical"] = sum(f.severity == "critical" for f in rep.flags)
        out["flag_reasons"] = [f.reason[:140] for f in rep.flags]
        if rep.flags:
            fixed, changes = apply_readability_fix(els, opt, plan)
            rig2 = build_rig_from_option(night, opt)
            still2 = Image.fromarray(frame_at(rig2, fixed, fixed.duration_s * 0.56))
            rep2 = await check_readability(still2, els, opt, fixed, use_model=False)
            out["flags_remaining_after_fix"] = len(rep2.flags)
            out["flags_fixed"] = len(rep.flags) - len(rep2.flags)
            out["recheck_passed"] = rep2.passed
            out["fix_changes"] = len(changes)
    except Exception as e:
        out["errors"].append(f"F5 check: {type(e).__name__}: {str(e)[:200]}")
    lap("F5_check", t)
    return out


async def main(paths):
    for p in paths:  # one event loop: cached async clients break across asyncio.run calls
        print(f"== {p}", file=sys.stderr)
        print(json.dumps(await run(p), default=lambda o: o.item()), flush=True)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
