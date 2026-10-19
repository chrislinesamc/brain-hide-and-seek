"""Sanity check for rotated search: plant turned copies of one section in the stack, find them.

Section 3291 goes upright on a scramble canvas (the query) and turned exactly 90, 180 and
270 degrees (the copies), inserted at random places among the scrambled stack. Searching
the hippocampus box with rotation on should rank the copies top 3, each at its own angle,
with the match box on the hippocampus.

The turned query section is pixel-identical to its copy, so this checks the wiring
(turning, box placement, angle labels), not how well the model generalises; the scramble
eval (evaluate_retrieval.py --scrambled) covers that.

    python -I scripts/check_planted_copies.py      # needs data/features-scrambled
    python -I scripts/check_planted_copies.py --save   # also write the stack for the viewer:
    python -I scripts/serve.py --stack data/thumbnails-planted \
        --features data/features-planted/vit_small_patch16_dinov3.lvd1689m.npy
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from region_search.feature_search import FeatureIndex  # noqa: E402
from region_search.features import load_model, patch_features, pick_device  # noqa: E402
from region_search.scramble import BACKGROUND, CANVAS  # noqa: E402
from region_search.sections import Roi, load_stack  # noqa: E402
from region_search.structures import box_hits, load_structure, query_box  # noqa: E402
from region_search.template_match import BACKGROUND_LEVEL, QUARTER_TURNS  # noqa: E402
from skimage.io import imsave  # noqa: E402

MODEL = "vit_small_patch16_dinov3.lvd1689m"
SECTION = 3291  # the hippocampus query section of the eval
LAST_SECTION = 7404  # BigBrain 2015; planted images take free numbers, so they sort in at random


def upright_on_canvas(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Section centred on a scramble-sized canvas, unturned; whole-pixel shift so no resampling."""
    rows, cols = np.where(image < BACKGROUND_LEVEL)
    dx = CANVAS // 2 - (cols.min() + cols.max()) // 2
    dy = CANVAS // 2 - (rows.min() + rows.max()) // 2
    height, width = image.shape
    canvas = np.full((CANVAS, CANVAS), BACKGROUND, np.uint8)
    canvas[dy : dy + height, dx : dx + width] = image
    return canvas, np.array([[1.0, 0, dx], [0, 1, dy]])


def quarter_turn_matrix(turns: int) -> np.ndarray:
    """2 x 3 pixel map of np.rot90(canvas, turns): one turn sends (x, y) to (y, CANVAS - 1 - x)."""
    one = np.array([[0.0, 1, 0], [-1, 0, CANVAS - 1], [0, 0, 1]])
    return np.linalg.matrix_power(one, turns % 4)[:2]


def then(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """2 x 3 matrix applying `first`, then `second`."""
    return (np.vstack([second, [0, 0, 1]]) @ np.vstack([first, [0, 0, 1]]))[:2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action="store_true", help="write data/thumbnails-planted + data/features-planted")
    args = parser.parse_args()

    stack_dir = ROOT / "data" / "thumbnails-scrambled"
    sections, images = load_stack(stack_dir)
    features = np.load(ROOT / "data" / "features-scrambled" / f"{MODEL}.npy")
    stored = [int(n) for n in np.load(ROOT / "data" / "features-scrambled" / f"{MODEL}-sections.npy")]
    if stored != sections:
        raise SystemExit("scrambled features and thumbnails cover different sections")
    device = pick_device()
    model = load_model(MODEL, device)
    extract = lambda image: patch_features(model, image, device)  # noqa: E731

    original = load_stack(ROOT / "data" / "thumbnails")
    query_image, to_query = upright_on_canvas(original[1][original[0].index(SECTION)])
    hippocampus = load_structure("hippocampus", ROOT / "data" / "bigbrain3d")
    roi = query_box(hippocampus, SECTION, to_stack=to_query, shape=query_image.shape)

    # planted: free section number -> (image, thumbnail -> image matrix, turn in degrees);
    # the first is the upright query
    rng = np.random.default_rng(args.seed)
    free = sorted(set(range(1, LAST_SECTION + 1)) - set(sections))
    numbers = [int(n) for n in rng.choice(free, size=4, replace=False)]
    query_id = numbers[0]
    planted = {}
    for turns, number in enumerate(numbers):
        image = np.ascontiguousarray(np.rot90(query_image, turns))
        planted[number] = (image, then(to_query, quarter_turn_matrix(turns)), 90 * turns)

    by_number = dict(zip(sections, zip(images, features)))
    by_number |= {n: (image, extract(image)) for n, (image, _, _) in planted.items()}
    ids = sorted(by_number)
    all_images = [by_number[n][0] for n in ids]
    grids = np.stack([by_number[n][1] for n in ids])
    del by_number, features
    index = FeatureIndex(grids, ids, all_images, extract)

    ranked = [m for m in index.search(query_id, roi, QUARTER_TURNS) if m.section != query_id]
    print(f"query: upright section {SECTION} as section {query_id}, box {roi}; {len(ids)} images; copies: "
          + ", ".join(f"{turn}° = section {n}" for n, (_, _, turn) in planted.items() if n != query_id))
    print("rank | image | score | found at angle | box on hippocampus")
    for rank, match in enumerate(ranked[:6], start=1):
        if match.section in planted:
            _, matrix, turn = planted[match.section]
            name = f"copy turned {turn}°"
            hit = box_hits(hippocampus, SECTION, Roi(match.x, match.y, roi.w, roi.h), matrix)
        else:
            name, hit = f"section {match.section}", "-"
        print(f"{rank} | {name} | {match.score:.3f} | {match.angle}° | {hit}")

    top3 = ranked[:3]
    passed = (
        {m.section for m in top3} == set(planted) - {query_id}
        and all(m.angle == planted[m.section][2] for m in top3)
        and all(box_hits(hippocampus, SECTION, Roi(m.x, m.y, roi.w, roi.h), planted[m.section][1]) for m in top3)
    )
    print("PASS: copies ranked top 3 at their own angle, boxes on target" if passed else "FAIL")
    if args.save:
        save(ids, all_images, grids, planted, stack_dir)
    if not passed:
        raise SystemExit(1)


def save(ids, images, grids, planted, scrambled_dir: Path) -> None:
    """Planted stack for the viewer: thumbnails + transforms.json (labels follow) + features."""
    thumbnails, features = ROOT / "data" / "thumbnails-planted", ROOT / "data" / "features-planted"
    thumbnails.mkdir(parents=True, exist_ok=True)
    features.mkdir(parents=True, exist_ok=True)
    for number, image in zip(ids, images):
        imsave(thumbnails / f"pm{number:04d}o.png", image, check_contrast=False)
    transforms = json.loads((scrambled_dir / "transforms.json").read_text())
    transforms["to_stack"] |= {str(n): matrix.tolist() for n, (_, matrix, _) in planted.items()}
    (thumbnails / "transforms.json").write_text(json.dumps(transforms))
    np.save(features / f"{MODEL}.npy", grids)
    np.save(features / f"{MODEL}-sections.npy", np.array(ids))
    print(f"saved {len(ids)} sections to {thumbnails} and {features}")


if __name__ == "__main__":
    main()
