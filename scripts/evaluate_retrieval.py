"""Retrieval quality per structure: precision@k, average precision and on-target rate.

For each structure in region_search.structures: the relevant sections are those inside its
labelled section range; the query is a box around the structure (left hemisphere) on the
stack section nearest the middle of that range. The query section is left out of the
ranking. On target = share of relevant sections where the best-match box covers labelled
points of the structure (either hemisphere); a ranking metric can't see where the box is.
Writes results/retrieval_eval.md (the table) and results/retrieval_queries.png (each query
box over the labelled points, to check the queries by eye).

--scrambled runs the same on the scrambled stack (scripts/scramble_stack.py): labels follow
each section's transform, both searches try quarter turns, the hand query is left out.
Writes the same files with a -scrambled suffix.

    python -I scripts/download_structures.py
    python -I scripts/evaluate_retrieval.py [--scrambled]
"""

import argparse
import json
import sys
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
from region_search.metrics import average_precision, precision_at_k  # noqa: E402
from region_search.sections import Roi, load_stack  # noqa: E402
from region_search.structures import (  # noqa: E402
    STRUCTURES,
    box_hits,
    load_structure,
    on_section,
    query_box,
    query_section,
    section_range,
    thumbnail_pixels,
)
from region_search.template_match import QUARTER_TURNS, UPRIGHT, search_stack  # noqa: E402

KS = (12, 50)  # 12 = the viewer's result grid
MODEL = "vit_small_patch16_dinov3.lvd1689m"
# The bake-off query (drawn by hand), scored against the labelled hippocampus range too.
HAND_QUERY = ("hippocampus", 3601, Roi(152, 258, 50, 44))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scrambled", action="store_true", help="use the scrambled stack")
    args = parser.parse_args()
    suffix = "-scrambled" if args.scrambled else ""

    stack_dir = ROOT / "data" / f"thumbnails{suffix}"
    sections, images = load_stack(stack_dir)
    by_section = dict(zip(sections, images))
    to_stack = {n: None for n in sections}
    angles, extract = UPRIGHT, None
    if args.scrambled:
        scramble = json.loads((stack_dir / "transforms.json").read_text())
        to_stack = {n: np.array(scramble["to_stack"][str(n)]) for n in sections}
        device = pick_device()
        model = load_model(MODEL, device)
        angles, extract = QUARTER_TURNS, lambda image: patch_features(model, image, device)
    index = FeatureIndex.load(ROOT / "data" / f"features{suffix}" / f"{MODEL}.npy", by_section, extract)
    # each method: (query section, roi) -> (section, match x, match y), best first
    methods = {
        # control: ignores the image, takes the query's neighbours and keeps the box in place
        "Nearest": lambda n, roi: [(s, roi.x, roi.y) for s in sorted(sections, key=lambda s: abs(s - n))],
        "Pixels": lambda n, roi: [(m.section, m.x, m.y)
                                  for m in search_stack(by_section[n], roi, sections, images, angles)],
        "DINOv3-S": lambda n, roi: [(m.section, m.x, m.y) for m in index.search(n, roi, angles)],
    }

    queries = []  # (label, structure, query section, roi)
    for name in STRUCTURES:
        structure = load_structure(name, ROOT / "data" / "bigbrain3d")
        query = query_section(structure, sections)
        roi = query_box(structure, query, to_stack=to_stack[query], shape=by_section[query].shape)
        queries.append((name, structure, query, roi))
    if not args.scrambled:  # drawn on the aligned stack
        hand_name, hand_section, hand_roi = HAND_QUERY
        queries.append((f"{hand_name} (hand box)", queries[0][1], hand_section, hand_roi))

    header = ["structure", "sections", "query", "relevant", "random AP"]
    for method in methods:
        header += [f"{method} P@{k}" for k in KS] + [f"{method} AP", f"{method} on target"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for label, structure, query, roi in queries:
        first, last = section_range(structure)
        relevant = {n for n in sections if first <= n <= last and n != query}
        row = [label, f"{first}–{last}", f"{query} ({roi.x},{roi.y} {roi.w}×{roi.h})",
               f"{len(relevant)}/{len(sections) - 1}", f"{len(relevant) / (len(sections) - 1):.2f}"]
        for search in methods.values():
            matches = [m for m in search(query, roi) if m[0] != query]
            ranked = [n for n, _, _ in matches]
            row += [f"{precision_at_k(ranked, relevant, k):.2f}" for k in KS]
            row.append(f"{average_precision(ranked, relevant):.2f}")
            hits = [box_hits(structure, n, Roi(x, y, roi.w, roi.h), to_stack[n])
                    for n, x, y in matches if n in relevant]
            hits = [h for h in hits if h is not None]  # sections with no labelled points can't be hit
            row.append(f"{sum(hits) / len(hits):.2f} ({len(hits)})")
        lines.append("| " + " | ".join(row) + " |")
        print(lines[-1], flush=True)

    out = ROOT / "results" / f"retrieval_eval{suffix}.md"
    out.parent.mkdir(exist_ok=True)
    scrambled_note = (
        "Scrambled stack (seed {seed}): every section turned by a random quarter turn, tilted up to ±10° and shifted "
        "up to ±40 px; Pixels and DINOv3-S search all four quarter turns. ".format(seed=scramble["seed"])
        if args.scrambled else ""
    )
    out.write_text(
        f"# Retrieval quality per structure{' (scrambled stack)' if args.scrambled else ''}\n\n"
        + scrambled_note
        + "Every 10th BigBrain 2015 section (thumbnails). Relevant = inside the structure's labelled section range "
        "(hippocampus: DeKraker 2020 surfaces; others: DKT atlas on the BigBrain white surface), query section excluded. "
        "Random AP = what a random ranking scores. Nearest = control that ranks sections by distance from the "
        "query section and keeps the query box in place, ignoring the image: a contiguous range around the query is "
        "easy for it. On target = share of relevant sections (count in brackets) where the best-match box covers "
        "labelled points of the structure, either hemisphere.\n\n" + "\n".join(lines) + "\n"
    )
    print(f"\nwrote {out.relative_to(ROOT)}")
    plot_queries(queries, by_section, to_stack, ROOT / "results" / f"retrieval_queries{suffix}.png")


def plot_queries(queries, by_section, to_stack, out: Path) -> None:
    fig, axes = plt.subplots(1, len(queries), figsize=(3.6 * len(queries), 3.8), squeeze=False)
    for ax, (label, structure, query, roi) in zip(axes[0], queries):
        cols, rows = thumbnail_pixels(structure.points[on_section(structure, query)], to_stack[query])
        ax.imshow(by_section[query], cmap="gray", vmin=0, vmax=255)
        ax.scatter(cols, rows, s=0.5, c="tab:red")
        ax.add_patch(Rectangle((roi.x, roi.y), roi.w, roi.h, fill=False, edgecolor="tab:blue", lw=1.2))
        ax.set_title(f"{label}\nsection {query}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
