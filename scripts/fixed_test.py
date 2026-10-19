"""Fixed test (agreed with Manh 2026-10-18 22:05): does a method find the same anatomy?

Scrambled stack only (on the aligned stack "same box, nearby sections" already wins).
Correct = same anatomy, either hemisphere. For each query, the method's top 12 sections
(query section left out) are checked: a section is on target when every box it returns
covers labelled points of the structure; a section without labelled points is a false hit.

Queries, frozen, never used to tune anything:
- hippocampus, insula, Heschl's gyrus, calcarine cortex, superior temporal gyrus, cuneus,
  precentral gyrus: left + right box on the section mid-range (like the viewer's by-name search)
- Manh's failed hand box on section 3701 (14:49): two candidates, no labels there
  (subcortical), judged by eye on the sheets
- plain white matter on section 3701: correct answer = no match
Tune set (the only place a threshold is picked): lingual + supramarginal gyrus.

Methods: Nearest (control: neighbours, box kept in place) · DINOv3-S (quarter turns, no
threshold, as in the viewer) · SIFT + RANSAC alone · DINOv3-S shortlist (50) -> SIFT verify.
SIFT thresholds (min mean inliers per section) maximise on-target - 2 x off-target on the
tune set: a false alarm costs more than a miss.

Pass: >= 10 of 12 on target on every structure query, more than Nearest, and no match
on white matter. Writes results/fixed_test.md + results/fixed_test/<query>.png.

    python -I scripts/fixed_test.py
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from region_search.feature_search import FeatureIndex  # noqa: E402
from region_search.features import load_model, patch_features, pick_device  # noqa: E402
from region_search.sections import Roi, load_stack  # noqa: E402
from region_search.sift_search import SiftIndex  # noqa: E402
from region_search.structures import (  # noqa: E402
    Structure,
    box_hits,
    load_structure,
    on_section,
    query_box,
    query_section,
)
from region_search.template_match import QUARTER_TURNS  # noqa: E402

MODEL = "vit_small_patch16_dinov3.lvd1689m"
TOP = 12  # the viewer's result grid
PASS_ON_TARGET = 10
SHORTLIST = 50
THRESHOLDS = range(3, 31)
FALSE_HIT_COST = 2
TEST = ["hippocampus", "insula", "Heschl's gyrus", "calcarine cortex",
        "superior temporal gyrus", "cuneus", "precentral gyrus"]
TUNE = ["lingual gyrus", "supramarginal gyrus"]
# boxes drawn on the aligned section 3701, moved into the scrambled stack below
BY_EYE = {
    "hand box A (putamen?)": Roi(145, 205, 40, 50),
    "hand box B (dark nucleus by 3rd ventricle?)": Roi(222, 245, 30, 26),
}
NEGATIVE = ("white matter (expect no match)", Roi(140, 140, 24, 24))
HAND_SECTION = 3701


@dataclass
class Query:
    label: str
    section: int
    rois: list[Roi]
    structure: Structure | None  # None = no labels: judged by eye, or the negative query
    kind: str  # "test", "tune", "eye", "negative"


def main() -> None:
    stack_dir = ROOT / "data" / "thumbnails-scrambled"
    sections, images = load_stack(stack_dir)
    by_section = dict(zip(sections, images))
    to_stack = {int(n): np.array(m) for n, m in json.loads((stack_dir / "transforms.json").read_text())["to_stack"].items()}
    queries = build_queries(sections, by_section, to_stack)
    plot_query_boxes(queries, by_section, ROOT / "results" / "fixed_test" / "queries.png")

    device = pick_device()
    model = load_model(MODEL, device)
    dino = FeatureIndex.load(
        ROOT / "data" / "features-scrambled" / f"{MODEL}.npy", by_section,
        lambda image: patch_features(model, image, device),
    )
    started = time.perf_counter()
    sift = SiftIndex(sections, images)
    print(f"SIFT index {time.perf_counter() - started:.0f} s", flush=True)

    # per query and method: ranked [(section, score, [box or None per roi])], query section left out
    ranked: dict[str, dict[str, list]] = {}
    for q in queries:
        started = time.perf_counter()
        nearest = [(n, -abs(n - q.section), list(q.rois)) for n in sorted(sections, key=lambda n: abs(n - q.section))]
        dino_ranked = [
            (m.section, m.score, [Roi(b[0].x, b[0].y, r.w, r.h) for b, r in zip(m.boxes, q.rois)])
            for m in dino.search_boxes(q.section, q.rois, QUARTER_TURNS)
        ]
        dino_ranked = [m for m in dino_ranked if m[0] != q.section]
        shortlist = [n for n, _, _ in dino_ranked[:SHORTLIST]]
        ranked[q.label] = {
            "Nearest": [m for m in nearest if m[0] != q.section],
            "DINOv3-S": dino_ranked,
            "SIFT": sift_ranked(sift, q, [n for n in sections if n != q.section]),
            "DINOv3→SIFT": sift_ranked(sift, q, shortlist),
        }
        print(f"{q.label}: searched in {time.perf_counter() - started:.0f} s", flush=True)

    # thresholds from the tune set only
    tune = [q for q in queries if q.kind == "tune"]
    thresholds = {"Nearest": None, "DINOv3-S": None}
    tuning_lines = []
    for method in ("SIFT", "DINOv3→SIFT"):
        gains = {t: sum(tune_gain(q, ranked[q.label][method], t, to_stack) for q in tune) for t in THRESHOLDS}
        best = max(gains.values())
        thresholds[method] = max(t for t, g in gains.items() if g == best)  # tie: the stricter one
        tuning_lines.append(f"- {method}: min mean inliers {thresholds[method]} (tune gain {best}; "
                            + ", ".join(f"{t}:{g}" for t, g in gains.items()) + ")")
        print(tuning_lines[-1], flush=True)

    methods = list(thresholds)
    header = ["query", "kind", "section", *methods]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    passed = {m: True for m in methods}
    for q in queries:
        row = [q.label, q.kind, str(q.section)]
        for method in methods:
            top = returned(ranked[q.label][method], thresholds[method])[:TOP]
            if q.kind in ("test", "tune"):
                on = sum(on_target(q.structure, n, boxes, to_stack) for n, _, boxes in top)
                nearest_on = sum(
                    on_target(q.structure, n, b, to_stack) for n, _, b in ranked[q.label]["Nearest"][:TOP]
                )
                ok = on >= PASS_ON_TARGET and (method == "Nearest" or on > nearest_on)
                cell = f"{on}/{TOP} on, {len(top) - on} false"
                if q.kind == "test":
                    passed[method] &= ok
                    cell += " ✓" if ok else " ✗"
            elif q.kind == "negative":
                ok = not top
                passed[method] &= ok
                cell = ("no match ✓" if ok else f"{len(top)} matches ✗")
            else:
                cell = f"{len(top)} returned, by eye"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    lines.append("| **all test queries** | | | " + " | ".join("PASS" if passed[m] else "FAIL" for m in methods) + " |")

    out = ROOT / "results" / "fixed_test.md"
    out.write_text(
        "# Fixed test (scrambled stack)\n\n" + __doc__.split("\n\n", 1)[1].rsplit("\n\n", 1)[0] + "\n\n"
        + "Thresholds picked on the tune set:\n" + "\n".join(tuning_lines) + "\n\n" + "\n".join(lines) + "\n"
    )
    print("\n".join(lines))
    print(f"wrote {out.relative_to(ROOT)}")
    for q in queries:
        plot_sheet(q, ranked[q.label], thresholds, by_section, to_stack, ROOT / "results" / "fixed_test")


def build_queries(sections, by_section, to_stack) -> list[Query]:
    queries = []
    for kind, names in (("test", TEST), ("tune", TUNE)):
        for name in names:
            structure = load_structure(name, ROOT / "data" / "bigbrain3d")
            section = query_section(structure, sections)
            rois = [
                query_box(structure, section, left, to_stack=to_stack[section], shape=by_section[section].shape)
                for left in (True, False)
                if (on_section(structure, section) & (structure.left == left)).any()
            ]
            queries.append(Query(name, section, rois, structure, kind))
    for label, roi in BY_EYE.items():
        queries.append(Query(label, HAND_SECTION, [moved(roi, to_stack[HAND_SECTION])], None, "eye"))
    label, roi = NEGATIVE
    queries.append(Query(label, HAND_SECTION, [moved(roi, to_stack[HAND_SECTION])], None, "negative"))
    return queries


def moved(roi: Roi, matrix: np.ndarray) -> Roi:
    """A box drawn on the aligned thumbnail, same size, centred where its centre lands."""
    cx, cy = matrix @ np.array([roi.x + roi.w / 2, roi.y + roi.h / 2, 1.0])
    return Roi(round(cx - roi.w / 2), round(cy - roi.h / 2), roi.w, roi.h)


def sift_ranked(sift: SiftIndex, q: Query, only: list[int]) -> list:
    per_box = [sift.search(q.section, roi, only) for roi in q.rois]
    rows = []
    for n in only:
        hits = [box[n] for box in per_box]
        boxes = [Roi(h.x, h.y, r.w, r.h) if h.inliers else None for h, r in zip(hits, q.rois)]
        rows.append((n, float(np.mean([h.inliers for h in hits])), boxes))
    return sorted(rows, key=lambda r: r[1], reverse=True)


def returned(ranked: list, threshold: float | None) -> list:
    return ranked if threshold is None else [r for r in ranked if r[1] >= threshold]


def on_target(structure: Structure, section: int, boxes: list, to_stack) -> bool:
    found = [b for b in boxes if b is not None]
    return bool(found) and all(box_hits(structure, section, b, to_stack[section]) for b in found)


def tune_gain(q: Query, ranked: list, threshold: int, to_stack) -> int:
    top = returned(ranked, threshold)[:TOP]
    on = sum(on_target(q.structure, n, boxes, to_stack) for n, _, boxes in top)
    return on - FALSE_HIT_COST * (len(top) - on)


def plot_query_boxes(queries, by_section, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, (len(queries) + 1) // 2, figsize=(2.6 * ((len(queries) + 1) // 2), 5.6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, q in zip(axes.flat, queries):
        ax.imshow(by_section[q.section], cmap="gray", vmin=0, vmax=255)
        for r in q.rois:
            ax.add_patch(Rectangle((r.x, r.y), r.w, r.h, fill=False, edgecolor="tab:blue", lw=1.2))
        ax.set_title(f"{q.label}\n{q.kind}, section {q.section}", fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_sheet(q: Query, ranked: dict, thresholds: dict, by_section, to_stack, folder: Path) -> None:
    """Query crop + each method's top 12: green = on target, red = false hit, grey = no labels."""
    fig, axes = plt.subplots(len(ranked), TOP + 1, figsize=(1.3 * (TOP + 1), 1.5 * len(ranked)))
    for row, (method, results) in enumerate(ranked.items()):
        show_crop(axes[row, 0], by_section[q.section], q.rois, "tab:blue")
        axes[row, 0].set_title("query" if row == 0 else "", fontsize=6)
        axes[row, 0].set_ylabel(method, fontsize=7)
        top = returned(results, thresholds[method])[:TOP]
        for col in range(TOP):
            ax = axes[row, col + 1]
            if col >= len(top):
                ax.axis("off")
                continue
            n, score, boxes = top[col]
            colour = "tab:grey" if q.structure is None else (
                "tab:green" if on_target(q.structure, n, boxes, to_stack) else "tab:red")
            show_crop(ax, by_section[n], [b for b in boxes if b is not None], colour)
            ax.set_title(f"{n} {score:.2f}" if method != "Nearest" else str(n), fontsize=6, color=colour)
    fig.suptitle(f"{q.label} ({q.kind})", fontsize=9)
    fig.tight_layout()
    name = "".join(c if c.isalnum() else "-" for c in q.label.lower()).strip("-")
    fig.savefig(folder / f"{name}.png", dpi=100)
    plt.close(fig)


def show_crop(ax, image, boxes, colour) -> None:
    height, width = image.shape
    if boxes:
        x0 = max(0, min(b.x for b in boxes) - 40)
        y0 = max(0, min(b.y for b in boxes) - 40)
        x1 = min(width, max(b.x + b.w for b in boxes) + 40)
        y1 = min(height, max(b.y + b.h for b in boxes) + 40)
    else:
        x0, y0, x1, y1 = 0, 0, width, height
    ax.imshow(image[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=255)
    for b in boxes:
        ax.add_patch(Rectangle((b.x - x0, b.y - y0), b.w, b.h, fill=False, edgecolor=colour, lw=1.5))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(colour)


if __name__ == "__main__":
    main()
