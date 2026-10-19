"""Local viewer: browse the section stack, draw an ROI, rank every section.

    python scripts/serve.py            # then open http://127.0.0.1:8000

Serves web/ and the thumbnails, plus:
  GET  /api/sections   section numbers and image size
  GET  /api/structures named regions, each with the section and boxes (left and right part)
                       a by-name search starts from
  GET  /api/outline?structure=hippocampus&sections=3291,3301
                       pixels of the region's labelled points on each section (3D patch shapes)
  POST /api/search     {"section": 3601, "rois": [{"x": 152, "y": 258, "w": 50, "h": 44}],
                        "rotation": false, "mirror": false}
                       several boxes = one query, sections rank by the boxes' mean score;
                       each box returns its best hit per section

Search runs in DINOv3-S feature space (it beat pixel NCC in scripts/compare_models.py);
features come from scripts/precompute_features.py. "rotation": true also tries the query
turned 90/180/270 degrees (the model runs on the turned query section, ~4x slower).

Region names come from data/bigbrain3d (scripts/download_structures.py): each name's box
sits on the section nearest the middle of its labelled range (region_search.structures).
"""

import argparse
import json
import re
import sys
import threading
import time
from http import HTTPStatus
from urllib.parse import parse_qs, urlsplit
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from region_search.feature_search import FeatureIndex  # noqa: E402
from region_search.features import DEFAULT_MODEL, load_model, patch_features, pick_device  # noqa: E402
from region_search.sections import Roi, crop_roi, load_stack  # noqa: E402
from region_search.structures import (  # noqa: E402
    STRUCTURES,
    load_structure,
    on_section,
    query_box,
    query_section,
    thumbnail_pixels,
)
from region_search.template_match import QUARTER_TURNS, UPRIGHT  # noqa: E402

WEB_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/stack3d.js": ("stack3d.js", "text/javascript; charset=utf-8"),
    # Three.js 0.186.1 (MIT), vendored so there is no build step
    **{
        f"/vendor/three/{name}": (f"vendor/three/{name}", "text/javascript; charset=utf-8")
        for name in ("three.module.js", "three.core.js", "OrbitControls.js")
    },
}
THUMBNAIL_PATH = re.compile(r"^/thumbnails/(pm\d{4}o\.png)$")
MAX_BODY_BYTES = 1024
MIN_ROI_SIDE = 8  # one feature patch covers 8x8 thumbnail pixels
MAX_ROIS = 2  # a structure's left and right part
MAX_OUTLINE_SECTIONS = 32  # query + top matches + a margin
# One hit per box and section. A second-hit rule (score >= 0.9 and within 0.08 of the best)
# was tuned on the same 4 regions it was reported on and put 0.92 hits on plain grey tissue
# (hand box, section 3701); off until a rule passes queries it was not tuned on.


class Stack:
    def __init__(self, folder: Path, features: Path):
        self.folder = folder
        self.sections, self.images = load_stack(folder)
        self.height, self.width = self.images[0].shape
        if not features.is_file():
            raise SystemExit(f"no features at {features}; run scripts/precompute_features.py first")
        device = pick_device()
        model = load_model(features.stem, device)  # file stem = timm model name
        self.index = FeatureIndex.load(
            features,
            dict(zip(self.sections, self.images)),
            extract=lambda image: patch_features(model, image, device),
        )
        self.search_lock = threading.Lock()  # one GPU search at a time
        transforms = self.folder / "transforms.json"  # scrambled stacks: labels follow the turn
        self.to_stack = (
            {int(n): np.array(m) for n, m in json.loads(transforms.read_text())["to_stack"].items()}
            if transforms.is_file() else {}
        )
        labels = REPO / "data" / "bigbrain3d"
        if labels.is_dir():
            self.labelled = {name: load_structure(name, labels) for name in STRUCTURES}
        else:
            print(f"no region names: {labels} missing, run scripts/download_structures.py")
            self.labelled = {}
        self.structures = [self.named_query(s) for s in self.labelled.values()]

    def named_query(self, structure) -> dict:
        section = query_section(structure, self.sections)
        rois = [
            query_box(
                structure, section, left, to_stack=self.to_stack.get(section), shape=(self.height, self.width)
            ).__dict__
            for left in (True, False)
            if (on_section(structure, section) & (structure.left == left)).any()
        ]
        return {"name": structure.name, "section": section, "rois": rois}

    def outline(self, name: str, section: int) -> list[list[int]]:
        """Distinct [x, y] pixels of the structure's labelled points on the section."""
        structure = self.labelled[name]
        cols, rows = thumbnail_pixels(structure.points[on_section(structure, section)], self.to_stack.get(section))
        inside = (cols >= 0) & (cols < self.width) & (rows >= 0) & (rows < self.height)
        return np.unique(np.stack([cols[inside], rows[inside]], axis=1), axis=0).tolist()


def make_handler(stack: Stack):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in WEB_FILES:
                name, content_type = WEB_FILES[self.path]
                return self.send_body((REPO / "web" / name).read_bytes(), content_type)
            if match := THUMBNAIL_PATH.match(self.path):
                path = stack.folder / match.group(1)
                if path.is_file():
                    return self.send_body(path.read_bytes(), "image/png", cache=True)
            if self.path == "/api/sections":
                return self.send_json(
                    {"sections": stack.sections, "width": stack.width, "height": stack.height}
                )
            if self.path == "/api/structures":
                return self.send_json({"structures": stack.structures})
            url = urlsplit(self.path)
            if url.path == "/api/outline":
                try:
                    name, sections = self.read_outline_request(parse_qs(url.query))
                except ValueError as error:
                    return self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return self.send_json({"outlines": {n: stack.outline(name, n) for n in sections}})
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self):
            if self.path != "/api/search":
                return self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            started = time.perf_counter()
            try:
                section, rois, rotation, mirror = self.read_search_request()
                for roi in rois:
                    crop_roi(stack.images[stack.sections.index(section)], roi)  # validates the ROI fits
                angles = QUARTER_TURNS if rotation else UPRIGHT
                with stack.search_lock:
                    ranked = stack.index.search_boxes(section, rois, angles, mirror)
            except ValueError as error:
                return self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
            self.send_json(
                {
                    "query": {
                        "section": section, "rois": [r.__dict__ for r in rois],
                        "rotation": rotation, "mirror": mirror,
                    },
                    "seconds": round(time.perf_counter() - started, 2),
                    "matches": [
                        {"section": m.section, "score": m.score, "boxes": [[h.__dict__ for h in hits] for hits in m.boxes]}
                        for m in ranked
                    ],
                }
            )

        def read_outline_request(self, query: dict[str, list[str]]) -> tuple[str, list[int]]:
            name = query.get("structure", [""])[0]
            if name not in stack.labelled:
                raise ValueError(f"unknown structure {name!r}")
            try:
                sections = [int(n) for n in query.get("sections", [""])[0].split(",")]
            except ValueError as error:
                raise ValueError("sections must be comma-separated section numbers") from error
            if not 1 <= len(sections) <= MAX_OUTLINE_SECTIONS:
                raise ValueError(f"ask for 1 to {MAX_OUTLINE_SECTIONS} sections")
            if missing := [n for n in sections if n not in stack.sections]:
                raise ValueError(f"section {missing[0]} is not in the stack")
            return name, sections

        def read_search_request(self) -> tuple[int, list[Roi], bool, bool]:
            length = int(self.headers.get("Content-Length") or 0)
            if not 0 < length <= MAX_BODY_BYTES:
                raise ValueError("request body missing or too large")
            try:
                body = json.loads(self.rfile.read(length))
                section = body["section"]
                if type(body["rois"]) is not list:
                    raise TypeError("rois must be a list")
                rois = [Roi(*(r[k] for k in ("x", "y", "w", "h"))) for r in body["rois"]]
                rotation = body.get("rotation", False)
                mirror = body.get("mirror", False)
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
                raise ValueError(f"malformed request: {error}") from error
            if type(rotation) is not bool or type(mirror) is not bool:
                raise ValueError("rotation and mirror must be true or false")
            if not 1 <= len(rois) <= MAX_ROIS:
                raise ValueError(f"send 1 to {MAX_ROIS} boxes")
            values = (section, *(v for r in rois for v in (r.x, r.y, r.w, r.h)))
            if not all(type(v) is int for v in values):
                raise ValueError("section and ROI values must be integers")
            if section not in stack.sections:
                raise ValueError(f"section {section} is not in the stack")
            if any(r.w < MIN_ROI_SIDE or r.h < MIN_ROI_SIDE for r in rois):
                raise ValueError(f"ROI must be at least {MIN_ROI_SIDE}x{MIN_ROI_SIDE} px")
            return section, rois, rotation, mirror

        def send_json(self, payload, status=HTTPStatus.OK):
            self.send_body(json.dumps(payload).encode(), "application/json", status=status)

        def send_body(self, body: bytes, content_type: str, status=HTTPStatus.OK, cache=False):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stack", type=Path, default=REPO / "data/thumbnails")
    parser.add_argument("--features", type=Path, default=REPO / f"data/features/{DEFAULT_MODEL}.npy")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    stack = Stack(args.stack, args.features)
    print(f"{len(stack.sections)} sections + {args.features.name} features loaded")
    # The page preloads every thumbnail at once; the default backlog (5) refuses some.
    ThreadingHTTPServer.request_queue_size = 256
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(stack))
    print(f"viewer at http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
