"""Write a scrambled copy of the thumbnail stack: every section turned, tilted and shifted.

Saves the canvases as pm####o.png plus transforms.json (section -> 2 x 3 matrix from
thumbnail pixel to canvas pixel), so labels can follow. See region_search.scramble.

    python -I scripts/scramble_stack.py --seed 0
    python -I scripts/precompute_features.py --thumbnails data/thumbnails-scrambled --out data/features-scrambled
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from skimage.io import imsave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from region_search.scramble import scramble_section  # noqa: E402
from region_search.sections import load_stack  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "thumbnails-scrambled")
    args = parser.parse_args()

    sections, images = load_stack(ROOT / "data" / "thumbnails")
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    transforms = {}
    for number, image in zip(sections, images):
        canvas, matrix = scramble_section(image, rng)
        imsave(args.out / f"pm{number:04d}o.png", canvas, check_contrast=False)
        transforms[number] = matrix.tolist()
    (args.out / "transforms.json").write_text(json.dumps({"seed": args.seed, "to_stack": transforms}))
    print(f"{len(sections)} sections scrambled (seed {args.seed}) into {args.out}")


if __name__ == "__main__":
    main()
