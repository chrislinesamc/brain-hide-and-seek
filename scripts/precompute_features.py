"""Precompute DINOv3 patch features for every section and save them as one .npy stack.

Usage: .venv/bin/python -I scripts/precompute_features.py [--model NAME] [--limit N] [--thumbnails DIR --out DIR]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from region_search.features import DEFAULT_MODEL, load_model, patch_features, pick_device  # noqa: E402
from region_search.sections import load_stack  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, help="only the first N sections (for timing)")
    parser.add_argument("--thumbnails", type=Path, default=ROOT / "data" / "thumbnails")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "features")
    args = parser.parse_args()

    numbers, images = load_stack(args.thumbnails)
    if args.limit:
        numbers, images = numbers[: args.limit], images[: args.limit]

    device = pick_device()
    model = load_model(args.model, device)
    patch_features(model, images[0], device)  # warm-up: first MPS call compiles kernels

    start = time.perf_counter()
    grids = [patch_features(model, image, device) for image in images]
    elapsed = time.perf_counter() - start

    stack = np.stack(grids)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-first{args.limit}" if args.limit else ""
    np.save(out_dir / f"{args.model}{suffix}.npy", stack)
    np.save(out_dir / f"{args.model}{suffix}-sections.npy", np.array(numbers))

    per_section = elapsed / len(images)
    print(f"{args.model} on {device}: {len(images)} sections, grid {stack.shape[1:]}")
    print(f"{per_section:.3f} s/section -> full stack (741) ~{per_section * 741 / 60:.1f} min")
    print(f"features {stack.nbytes / 1e6:.0f} MB -> full stack ~{stack.nbytes / len(images) * 741 / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
