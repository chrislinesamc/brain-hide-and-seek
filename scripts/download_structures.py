"""Download the BigBrain label files behind the evaluation ground truth (~45 MB).

Hippocampus mid-surfaces, the white surface with DKT atlas labels, and the 400 um volume
that scripts/check_section_mapping.py uses to place sections in 3D.

    python scripts/download_structures.py
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

BASE_URL = "https://ftp.bigbrainproject.org/bigbrain-ftp/BigBrainRelease.2015"
FILES = [
    "Hippocampus_Segmentation/gii/sub-bbhist_hemi-L_midSurf.surf.gii",
    "Hippocampus_Segmentation/gii/sub-bbhist_hemi-R_midSurf.surf.gii",
    "Surface_Parcellations/BigBrain_space/Surfaces/lh.white.anat.surf.gii",
    "Surface_Parcellations/BigBrain_space/Surfaces/rh.white.anat.surf.gii",
    "Surface_Parcellations/BigBrain_space/DKT/lh.DKTatlas40.label.gii",
    "Surface_Parcellations/BigBrain_space/DKT/rh.DKTatlas40.label.gii",
    "3D_Volumes/Histological_Space/nii/full8_400um_optbal.nii.gz",
]


def download(remote: str, out_dir: Path) -> str:
    path = out_dir / Path(remote).name
    if path.exists():
        return "skipped"
    with urlopen(f"{BASE_URL}/{remote}", timeout=120) as response:
        path.write_bytes(response.read())
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/bigbrain3d"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(len(FILES)) as pool:
        results = list(pool.map(lambda f: download(f, args.out), FILES))
    print(f"{results.count('downloaded')} downloaded, {results.count('skipped')} skipped, in {args.out}")


if __name__ == "__main__":
    main()
