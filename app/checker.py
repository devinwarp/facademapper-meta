"""F5: Readability & fixture-pitch audit + auto-fix via Muse Spark (muse-spark-1.3) + physical pitch rules."""
from __future__ import annotations

import json
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from app.fixtures import FixtureRigOption
from app.mapper import Element, _client, _data_url
from app.planner import Plan


class ReadabilityFlag(BaseModel):
    step_index: int
    time_s: float
    element_label: str
    severity: str = Field(description="'warning' or 'critical'")
    reason: str
    suggested_fix: str


class ReadabilityReport(BaseModel):
    passed: bool
    summary: str
    flags: list[ReadabilityFlag] = Field(default_factory=list)


def _grid_resolution(opt: FixtureRigOption, idx: np.ndarray) -> tuple[int, int]:
    """Approximate distinct (columns, rows) of discrete fixtures in a subset."""
    if len(idx) < 4:
        return len(idx), 1
    xs, ys = opt.xs[idx], opt.ys[idx]
    # Bucket into ~8px bins to count independent columns & rows
    cols = len(np.unique(np.round(xs / 8.0).astype(int)))
    rows = len(np.unique(np.round(ys / 12.0).astype(int)))
    return cols, rows


async def check_readability(
    preview_img: Image.Image,
    els: list[Element],
    opt: FixtureRigOption,
    plan: Plan,
    use_model: bool = True,
) -> ReadabilityReport:
    """Audit a rendered show preview against real fixture spacing (physical rules + Muse Spark vision critique)."""
    by_id = {e.id: e for e in els}
    flags: list[ReadabilityFlag] = []

    # 1. Deterministic physical fixture-pitch audit (catches text/logos on sparse zones reliably!)
    for idx_s, s in enumerate(plan.steps):
        members = [opt.elem_membership.get(eid, np.empty(0, np.int32)) for eid in s.element_ids]
        union_idx = np.unique(np.concatenate(members)) if members else np.empty(0, np.int32)
        el_names = ", ".join(by_id[eid].label for eid in s.element_ids if eid in by_id) or "zone"
        mid_t = round(0.5 * (s.start_s + s.end_s), 1)

        if len(union_idx) == 0:
            flags.append(ReadabilityFlag(
                step_index=idx_s,
                time_s=mid_t,
                element_label=el_names,
                severity="critical",
                reason=f"Zero physical fixtures exist inside '{el_names}' under the {opt.name} layout.",
                suggested_fix="Reassign step to active facade elements carrying fixtures.",
            ))
            continue

        if s.motion == "text" and s.text:
            cols, rows = _grid_resolution(opt, union_idx)
            needed_cols = max(14, len(s.text.replace(" ", "")) * 4)
            if rows < 8 or cols < needed_cols or opt.id in {"architectural_accent", "horizontal_tiers"}:
                flags.append(ReadabilityFlag(
                    step_index=idx_s,
                    time_s=mid_t,
                    element_label=el_names,
                    severity="critical",
                    reason=(
                        f"Text '{s.text}' on '{el_names}' only spans {cols}×{rows} fixture nodes "
                        f"(needs ≥{needed_cols}×8 discrete pixel bars to read cleanly from street level)."
                    ),
                    suggested_fix=(
                        "Move text onto the tallest dense matrix zone with ≤5 bold chars, or replace text "
                        "with a bold kinetic crest/chase pulse suited to sparse architectural fixtures."
                    ),
                ))

    # 2. Optional Muse Spark visual review on the actual rendered preview frame
    if use_model and not flags:
        try:
            schema = {
                "type": "object",
                "properties": {
                    "passed": {"type": "boolean"},
                    "summary": {"type": "string"},
                    "issue": {"type": "string"},
                },
                "required": ["passed", "summary", "issue"],
                "additionalProperties": False,
            }
            r = await _client().chat.completions.create(
                model="muse-spark-1.3",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": (
                        f"Review this architectural media-facade preview rendered on discrete LED fixtures "
                        f"({opt.name}, {opt.fixture_count} fixtures). Plan title: {plan.title}. "
                        f"Does the lighting read clearly along the building's real fixtures without illegible text "
                        f"or muddled contrast? Be concise."
                    )},
                    {"type": "image_url", "image_url": {"url": _data_url(preview_img)}},
                ]}],
                response_format={"type": "json_schema", "json_schema": {"name": "audit", "schema": schema, "strict": True}},
            )
            ans = json.loads(r.choices[0].message.content)
            if not ans["passed"] and ans["issue"].strip():
                flags.append(ReadabilityFlag(
                    step_index=0,
                    time_s=round(plan.duration_s * 0.5, 1),
                    element_label=opt.zones_used[0] if opt.zones_used else "Facade",
                    severity="warning",
                    reason=ans["issue"],
                    suggested_fix="Increase colour contrast and expand effect across main structural zones.",
                ))
            return ReadabilityReport(passed=(len(flags) == 0), summary=ans["summary"], flags=flags)
        except Exception:
            pass

    if flags:
        return ReadabilityReport(
            passed=False,
            summary=f"Caught {len(flags)} fixture-spacing readability issue(s) before test night.",
            flags=flags,
        )
    return ReadabilityReport(
        passed=True,
        summary=f"All {len(plan.steps)} show steps read cleanly at {opt.name} spacing ({opt.fixture_count:,} fixtures).",
        flags=[],
    )


def apply_readability_fix(els: list[Element], opt: FixtureRigOption, plan: Plan) -> tuple[Plan, list[str]]:
    """Deterministic architectural auto-fix so flagged unreadable text or empty zones resolve cleanly."""
    changes: list[str] = []
    # Find richest element(s) with highest fixture count
    active_eids = sorted(
        [e.id for e in els if len(opt.elem_membership.get(e.id, ())) > 0],
        key=lambda eid: len(opt.elem_membership.get(eid, ())),
        reverse=True,
    )
    best_eid = active_eids[0] if active_eids else els[0].id
    by_id = {e.id: e for e in els}

    new_steps = []
    for s in plan.steps:
        st = s.model_copy(deep=True)
        # Ensure active elements
        valid_active = [eid for eid in st.element_ids if len(opt.elem_membership.get(eid, ())) > 0]
        if not valid_active:
            st.element_ids = active_eids[:2] or [best_eid]
            changes.append(f"Re-routed {st.start_s:.0f}–{st.end_s:.0f}s step onto active fixture zones.")

        if st.motion == "text" and st.text:
            best_idx = opt.elem_membership.get(best_eid, np.empty(0, np.int32))
            cols, rows = _grid_resolution(opt, best_idx)
            short_txt = st.text.strip()[:5].upper()
            if opt.id not in {"architectural_accent", "horizontal_tiers"} and rows >= 10 and cols >= max(16, len(short_txt) * 4):
                old_zone = ", ".join(by_id[i].label for i in st.element_ids if i in by_id)
                st.element_ids = [best_eid]
                st.text = short_txt
                changes.append(f"Moved text '{short_txt}' from '{old_zone}' onto '{by_id[best_eid].label}' ({cols}×{rows} matrix).")
            else:
                st.motion = "chase"
                st.text = None
                changes.append(f"Replaced illegible text on sparse '{opt.name}' with bold architectural crest chase.")
        # Strip pure black #000000 colour stops that create dead bands
        vivid = [c for c in st.colors if c.upper() not in {"#000000", "#000", "BLACK"}]
        if vivid:
            st.colors = vivid
        new_steps.append(st)

    fixed = Plan(title=plan.title, duration_s=plan.duration_s, steps=new_steps)
    return fixed, changes
