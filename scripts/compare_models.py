"""Bake-off: one query through the pixel baseline and every precomputed feature model.

Pass bar (agreed 2026-10-18): every target section ranks in the top 150, the top-12 are
all the region by eye, a query takes <= 2 s. Writes results/compare_top.png (top-12 crops
per method, one row each).

    python -I scripts/compare_models.py --query-section 3601 --roi 152 258 50 44
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from region_search.feature_search import FeatureIndex  # noqa: E402
from region_search.sections import Roi, load_stack  # noqa: E402
from region_search.template_match import search_stack  # noqa: E402

TOP_BAR = 150
CROP_MARGIN = 40


def timed(search, repeats=3):
    search()  # warm-up
    times, ranked = [], None
    for _ in range(repeats):
        start = time.perf_counter()
        ranked = search()
        times.append(time.perf_counter() - start)
    return ranked, statistics.median(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--query-section", type=int, required=True)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), required=True)
    parser.add_argument("--targets", type=int, nargs=2, default=(3001, 3501), metavar=("FIRST", "LAST"))
    parser.add_argument("--region", type=int, nargs=2, default=(2800, 3700), metavar=("FIRST", "LAST"),
                        help="sections where the region is visible (eyeballed, not ground truth)")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    sections, images = load_stack(ROOT / "data" / "thumbnails")
    by_section = dict(zip(sections, images))
    roi = Roi(*args.roi)
    query_image = by_section[args.query_section]

    methods = {"Pixels (NCC)": lambda: search_stack(query_image, roi, sections, images)}
    for path in sorted((ROOT / "data" / "features").glob("*.npy")):
        if path.stem.endswith("-sections") or "-first" in path.stem:
            continue
        index = FeatureIndex.load(path, by_section)
        methods[f"DINOv3-{path.stem.split('_')[1]}"] = (  # vit_small_patch16_dinov3... -> DINOv3-small
            lambda index=index: index.search(args.query_section, roi)
        )

    rows = []
    for name, search in methods.items():
        ranked, seconds = timed(search)
        rank_of = {m.section: i for i, m in enumerate(ranked, 1)}
        targets = [rank_of[n] for n in sections if args.targets[0] <= n <= args.targets[1]]
        top = [m for m in ranked if m.section != args.query_section][: args.top]
        in_region = sum(args.region[0] <= m.section <= args.region[1] for m in top)
        print(f"\n{name}: {seconds:.2f} s/query")
        print(f"  targets {args.targets[0]}-{args.targets[1]} ({len(targets)}): ranks {min(targets)}-{max(targets)}, "
              f"median {statistics.median(targets):.0f}, {sum(r <= TOP_BAR for r in targets)}/{len(targets)} in top {TOP_BAR}")
        print(f"  top-{args.top} in {args.region[0]}-{args.region[1]}: {in_region}/{len(top)}  "
              f"sections {[m.section for m in top]}")
        rows.append((name, top))

    fig, axes = plt.subplots(len(rows), args.top + 1, figsize=(1.6 * (args.top + 1), 1.9 * len(rows)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for row_axes, (name, top) in zip(axes, rows):
        panels = [(args.query_section, roi.x, roi.y, "query")] + [(m.section, m.x, m.y, f"#{i} {m.section}") for i, m in enumerate(top, 1)]
        for ax, (section, x, y, label) in zip(row_axes, panels):
            image = by_section[section]
            x0, y0 = max(x - CROP_MARGIN, 0), max(y - CROP_MARGIN, 0)
            ax.imshow(image[y0 : y + roi.h + CROP_MARGIN, x0 : x + roi.w + CROP_MARGIN], cmap="gray", vmin=0, vmax=255)
            ax.add_patch(Rectangle((x - x0, y - y0), roi.w, roi.h, fill=False, edgecolor="red", lw=1))
            ax.set_title(label, fontsize=7)
        row_axes[0].text(-0.15, 0.5, name, transform=row_axes[0].transAxes, rotation=90, va="center", ha="right", fontsize=8)
    fig.tight_layout()
    out = ROOT / "results" / "compare_top.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
