"""Template-matching baseline: score one ROI against every section in the stack.

Writes to results/:
  baseline_scores.csv  every section with its best score and match location
  baseline_scores.png  score against section number
  baseline_top.png     the top-ranked sections with the match boxed

    python scripts/run_template_baseline.py --query-section 3601 --roi 152 258 50 44
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from region_search.sections import Roi, crop_roi, load_stack  # noqa: E402
from region_search.template_match import search_stack  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stack", type=Path, default=Path("data/thumbnails"))
    parser.add_argument("--query-section", type=int, required=True)
    parser.add_argument("--roi", type=int, nargs=4, metavar=("X", "Y", "W", "H"), required=True)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    sections, images = load_stack(args.stack)
    if args.query_section not in sections:
        raise SystemExit(f"section {args.query_section} is not in {args.stack}")
    roi = Roi(*args.roi)
    query_image = images[sections.index(args.query_section)]
    crop_roi(query_image, roi)  # validates the ROI fits the section

    ranked = search_stack(query_image, roi, sections, images)
    args.out.mkdir(exist_ok=True)

    with open(args.out / "baseline_scores.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "section", "score", "x", "y"])
        for rank, m in enumerate(ranked, 1):
            writer.writerow([rank, m.section, f"{m.score:.4f}", m.x, m.y])

    by_section = sorted(ranked, key=lambda m: m.section)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot([m.section for m in by_section], [m.score for m in by_section], lw=0.8)
    ax.axvline(args.query_section, color="red", lw=0.8, label=f"query section {args.query_section}")
    ax.set_xlabel("section number (1 = back of the brain, 7404 = front)")
    ax.set_ylabel("best match score (NCC)")
    ax.set_title(f"Template matching: ROI {roi.w}x{roi.h} px at ({roi.x}, {roi.y})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "baseline_scores.png", dpi=100)

    top = [m for m in ranked if m.section != args.query_section][: args.top]
    cols = 4
    fig, axes = plt.subplots((len(top) + cols) // cols, cols, figsize=(4 * cols, 3.6 * ((len(top) + cols) // cols)))
    panels = [(args.query_section, roi.x, roi.y, "query")] + [
        (m.section, m.x, m.y, f"#{i} score {m.score:.2f}") for i, m in enumerate(top, 1)
    ]
    for ax in axes.flat:
        ax.axis("off")
    for ax, (section, x, y, label) in zip(axes.flat, panels):
        ax.imshow(images[sections.index(section)], cmap="gray")
        ax.add_patch(Rectangle((x, y), roi.w, roi.h, fill=False, edgecolor="red", lw=1.5))
        ax.set_title(f"section {section}  {label}", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out / "baseline_top.png", dpi=70)

    print(f"{len(ranked)} sections scored; top 5:")
    for m in ranked[:5]:
        print(f"  section {m.section}  score {m.score:.3f}  at ({m.x}, {m.y})")


if __name__ == "__main__":
    main()
