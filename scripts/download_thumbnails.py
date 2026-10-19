"""Download BigBrain 2015 coronal section thumbnails (500 x 434, 8-bit grayscale).

Sections are numbered 1..7404, 20 um apart. --step 10 keeps every 10th section
(741 files, ~58 MB), which is enough to explore and run a first baseline.

    python scripts/download_thumbnails.py --step 10
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

BASE_URL = (
    "https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015/"
    "2D_Final_Sections/Coronal/Png/Thumbnail"
)
N_SECTIONS = 7404
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def section_filename(section: int) -> str:
    return f"pm{section:04d}o.png"


def download(section: int, out_dir: Path) -> str:
    path = out_dir / section_filename(section)
    if path.exists():
        return "skipped"
    with urlopen(f"{BASE_URL}/{section_filename(section)}", timeout=60) as response:
        body = response.read()
    if not body.startswith(PNG_SIGNATURE):
        raise ValueError(f"section {section}: response is not a PNG")
    path.write_bytes(body)
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--step", type=int, default=10, help="keep every Nth section")
    parser.add_argument("--out", type=Path, default=Path("data/thumbnails"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sections = range(1, N_SECTIONS + 1, args.step)
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda s: download(s, args.out), sections))
    print(
        f"{results.count('downloaded')} downloaded, {results.count('skipped')} skipped, "
        f"{len(results)} sections in {args.out}"
    )


if __name__ == "__main__":
    main()
