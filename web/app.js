// Section viewer: browse the stack, draw an ROI, show where it matches on every section.

import { createStackView } from "/stack3d.js";

const SCALE = 2; // canvas pixels per image pixel, keeps boxes crisp
const MIN_ROI_SIDE = 8; // same limit as the server
const TOP_K = 12;

const els = {
  status: document.querySelector("#status"),
  canvas: document.querySelector("#section-canvas"),
  slider: document.querySelector("#slider"),
  label: document.querySelector("#section-label"),
  prev: document.querySelector("#prev"),
  next: document.querySelector("#next"),
  clear: document.querySelector("#clear"),
  rotation: document.querySelector("#rotation"),
  mirror: document.querySelector("#mirror"),
  regionPicker: document.querySelector("#region-picker"),
  region: document.querySelector("#region"),
  matches: document.querySelector("#matches"),
  strip: document.querySelector("#strip-canvas"),
  stack3d: document.querySelector("#stack-3d"),
};
const ctx = els.canvas.getContext("2d");

const state = {
  sections: [],
  width: 0,
  height: 0,
  index: 0,
  structures: [], // { name, section, rois }: where a by-name search starts (left + right part)
  query: null, // { section, rois: [{x, y, w, h}], structure }, several boxes = one query;
  // structure = region name when searched by name, else null
  drag: null, // { x0, y0, x1, y1 } in image pixels
  results: null, // { ranked, bySection: Map, seconds }
  requestId: 0,
  stack3d: null, // see stack3d.js
};
const images = new Map(); // section -> Promise<HTMLImageElement>

function imageFor(section) {
  if (!images.has(section)) {
    images.set(
      section,
      new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => {
          images.delete(section); // let the next draw retry instead of caching the failure
          reject(new Error(`section ${section} failed to load`));
        };
        img.src = `/thumbnails/pm${String(section).padStart(4, "0")}o.png`;
      }),
    );
  }
  return images.get(section);
}

const currentSection = () => state.sections[state.index];

function setStatus(text) {
  els.status.textContent = text;
}

// ---------- main view ----------

async function drawSection() {
  const section = currentSection();
  els.slider.value = state.index;
  els.label.value = `Section ${section} (${state.index + 1} of ${state.sections.length})`;
  const img = await imageFor(section).catch((error) => setStatus(error.message));
  if (section !== currentSection()) return; // user moved on while it loaded

  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, els.canvas.width, els.canvas.height);
  if (img) ctx.drawImage(img, 0, 0, els.canvas.width, els.canvas.height);

  if (state.query?.section === section) {
    state.query.rois.forEach((roi, i) => drawBox(ctx, roi, SCALE, "#4da3ff", i ? null : "query"));
  }
  if (state.results) {
    for (const box of matchBoxes(state.results.bySection.get(section))) {
      drawBox(ctx, box, SCALE, "#ff4d4d", hitText(box));
    }
  }
  if (state.drag) drawBox(ctx, dragRoi(), SCALE, "#4da3ff", null, [6, 4]);
  markCurrentMatch();
  drawStrip();
}

function drawBox(context, roi, scale, color, label, dash = []) {
  context.save();
  context.strokeStyle = color;
  context.lineWidth = Math.max(2, scale * 1.5); // thumbnails are drawn large and shown small
  context.setLineDash(dash);
  context.strokeRect(roi.x * scale, roi.y * scale, roi.w * scale, roi.h * scale);
  if (label) {
    context.font = `${12 * Math.max(1, scale / 2)}px system-ui`;
    context.fillStyle = color;
    context.fillText(label, roi.x * scale, roi.y * scale - 4);
  }
  context.restore();
}

function matchText(match) {
  const best = match.boxes.map((hits) => hits[0]);
  const angles = [...new Set(best.map((hit) => hit.angle))];
  const turn = angles.some(Boolean) ? ` ↻${angles.join("/")}°` : "";
  return `${match.score.toFixed(2)}${turn}${best.some((hit) => hit.mirrored) ? " ⇋" : ""}`;
}

function hitText(hit) {
  return `${hit.score.toFixed(2)}${hit.angle ? ` ↻${hit.angle}°` : ""}${hit.mirrored ? " ⇋" : ""}`;
}

// Every hit of every query box on the match's section, at that box's size, best hits of
// all boxes first. A hit overlapping one already listed is dropped: the left box's mirrored
// hit and the right box's own hit are the same place. On the query section, hits on the
// query boxes themselves are dropped too.
function matchBoxes(match) {
  const taken = match.section === state.query.section ? [...state.query.rois] : [];
  const boxes = [];
  const rounds = Math.max(...match.boxes.map((hits) => hits.length));
  for (let r = 0; r < rounds; r++) {
    match.boxes.forEach((hits, i) => {
      if (r >= hits.length) return;
      const box = { ...hits[r], w: state.query.rois[i].w, h: state.query.rois[i].h };
      if ([...taken, ...boxes].some((b) => overlaps(b, box))) return;
      boxes.push(box);
    });
  }
  return boxes;
}

function overlaps(a, b) {
  return Math.abs(a.x - b.x) < Math.min(a.w, b.w) && Math.abs(a.y - b.y) < Math.min(a.h, b.h);
}

function goTo(index) {
  state.index = Math.max(0, Math.min(state.sections.length - 1, index));
  drawSection();
}

// ---------- drawing the ROI ----------

function toImagePoint(event) {
  const rect = els.canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * state.width;
  const y = ((event.clientY - rect.top) / rect.height) * state.height;
  return {
    x: Math.max(0, Math.min(state.width, x)),
    y: Math.max(0, Math.min(state.height, y)),
  };
}

function dragRoi() {
  const { x0, y0, x1, y1 } = state.drag;
  const x = Math.round(Math.min(x0, x1));
  const y = Math.round(Math.min(y0, y1));
  return {
    x,
    y,
    w: Math.min(Math.round(Math.abs(x1 - x0)), state.width - x),
    h: Math.min(Math.round(Math.abs(y1 - y0)), state.height - y),
  };
}

els.canvas.addEventListener("pointerdown", (event) => {
  const p = toImagePoint(event);
  state.drag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
  els.canvas.setPointerCapture(event.pointerId);
});

els.canvas.addEventListener("pointermove", (event) => {
  if (!state.drag) return;
  const p = toImagePoint(event);
  state.drag.x1 = p.x;
  state.drag.y1 = p.y;
  drawSection();
});

els.canvas.addEventListener("pointerup", () => {
  if (!state.drag) return;
  const roi = dragRoi();
  state.drag = null;
  if (roi.w < MIN_ROI_SIDE || roi.h < MIN_ROI_SIDE) {
    setStatus(`Box too small, draw at least ${MIN_ROI_SIDE}×${MIN_ROI_SIDE} px`);
    drawSection();
    return;
  }
  els.region.value = "";
  startSearch(currentSection(), [roi], null);
});

// ---------- search by region name ----------

els.region.addEventListener("change", () => {
  const structure = state.structures.find((s) => s.name === els.region.value);
  if (!structure) return;
  state.index = state.sections.indexOf(structure.section);
  startSearch(structure.section, structure.rois, structure.name);
});

function fillRegions(structures) {
  state.structures = structures;
  const options = structures.map((s) => new Option(s.name, s.name));
  els.region.append(...options);
  els.regionPicker.hidden = structures.length === 0;
}

// ---------- search ----------

function startSearch(section, rois, structure) {
  state.query = { section, rois, structure };
  state.results = null;
  state.stack3d.clear(); // old matches would be wrong while the new search runs
  els.clear.disabled = false;
  drawSection();
  search();
}

async function search() {
  const id = ++state.requestId;
  const { section, rois } = state.query;
  const rotation = els.rotation.checked;
  const mirror = els.mirror.checked;
  const orientations = (rotation ? 4 : 1) * (mirror ? 2 : 1);
  setStatus(`Searching ${state.sections.length} sections${orientations > 1 ? ` at ${orientations} orientations` : ""}…`);
  els.matches.replaceChildren();
  try {
    const response = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ section, rois, rotation, mirror }),
    });
    const body = await response.json();
    if (id !== state.requestId) return; // a newer search replaced this one
    if (!response.ok) throw new Error(body.error ?? `HTTP ${response.status}`);
    state.results = {
      ranked: body.matches,
      bySection: new Map(body.matches.map((m) => [m.section, m])),
      seconds: body.seconds,
    };
    setStatus(`Ranked ${body.matches.length} sections in ${body.seconds} s`);
    renderMatches();
    drawSection();
    showInStack3d();
  } catch (error) {
    if (id === state.requestId) setStatus(`Search failed: ${error.message}`);
  }
}

function topMatches() {
  return state.results.ranked.filter((m) => m.section !== state.query.section).slice(0, TOP_K);
}

function renderMatches() {
  const items = topMatches().map((match, i) => {
    const li = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.section = match.section;
    const thumb = document.createElement("canvas");
    const label = document.createElement("span");
    label.textContent = `#${i + 1} · ${match.section} · ${matchText(match)}`;
    button.append(thumb, label);
    button.addEventListener("click", () => goTo(state.sections.indexOf(match.section)));
    li.append(button);

    // Zoomed crop around all boxes, with one box-width of context (half with two boxes).
    const boxes = matchBoxes(match);
    const padX = boxes[0].w / boxes.length;
    const padY = boxes[0].h / boxes.length;
    const x0 = Math.max(0, Math.min(...boxes.map((b) => b.x)) - padX);
    const y0 = Math.max(0, Math.min(...boxes.map((b) => b.y)) - padY);
    const crop = {
      x: x0,
      y: y0,
      w: Math.min(state.width, Math.max(...boxes.map((b) => b.x + b.w)) + padX) - x0,
      h: Math.min(state.height, Math.max(...boxes.map((b) => b.y + b.h)) + padY) - y0,
    };
    thumb.width = crop.w * 3;
    thumb.height = crop.h * 3;
    imageFor(match.section).then((img) => {
      const c = thumb.getContext("2d");
      c.drawImage(img, crop.x, crop.y, crop.w, crop.h, 0, 0, thumb.width, thumb.height);
      for (const b of boxes) drawBox(c, { ...b, x: b.x - crop.x, y: b.y - crop.y }, 3, "#ff4d4d", null);
    });
    return li;
  });
  els.matches.replaceChildren(...items);
  markCurrentMatch();
}

async function showInStack3d() {
  const { section, rois, structure } = state.query;
  const queryHits = matchBoxes(state.results.bySection.get(section)); // e.g. the mirrored twin
  const items = [
    { section, boxes: rois, color: "#4da3ff" },
    ...(queryHits.length ? [{ section, boxes: queryHits, color: "#ff4d4d" }] : []),
    ...topMatches().map((m) => ({ section: m.section, boxes: matchBoxes(m), color: "#ff4d4d" })),
  ];
  const id = state.requestId;
  try {
    if (structure) {
      // searched by name: patches take the region's labelled shape on each section
      const sections = [...new Set(items.map((item) => item.section))].join(",");
      const response = await fetch(`/api/outline?structure=${encodeURIComponent(structure)}&sections=${sections}`);
      const body = await response.json();
      if (!response.ok) throw new Error(body.error ?? `HTTP ${response.status}`);
      for (const item of items) item.outline = body.outlines[item.section];
    }
    if (id === state.requestId) await state.stack3d.show(items); // else a newer search took over
  } catch (error) {
    setStatus(`3D view: ${error.message}`);
  }
}

function markCurrentMatch() {
  for (const button of els.matches.querySelectorAll("button")) {
    button.setAttribute("aria-current", String(Number(button.dataset.section) === currentSection()));
  }
}

for (const toggle of [els.rotation, els.mirror]) {
  toggle.addEventListener("change", () => {
    if (state.query) search(); // re-run the current box with the new setting
  });
}

els.clear.addEventListener("click", () => {
  state.requestId++;
  state.query = null;
  state.results = null;
  els.region.value = "";
  els.clear.disabled = true;
  state.stack3d.clear();
  els.matches.replaceChildren();
  setStatus(`${state.sections.length} sections · drag on the section to search`);
  drawSection();
});

// ---------- score strip ----------

function drawStrip() {
  const dpr = window.devicePixelRatio || 1;
  const width = Math.round(els.strip.clientWidth * dpr);
  const height = Math.round(els.strip.clientHeight * dpr);
  if (els.strip.width !== width || els.strip.height !== height) {
    els.strip.width = width;
    els.strip.height = height;
  }
  const c = els.strip.getContext("2d");
  c.clearRect(0, 0, width, height);
  const n = state.sections.length;
  const barWidth = Math.max(1, width / n);

  if (state.results) {
    const top = new Set(topMatches().map((m) => m.section));
    const others = state.results.ranked.filter((m) => m.section !== state.query.section);
    const scores = others.map((m) => m.score);
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    state.sections.forEach((section, i) => {
      const match = state.results.bySection.get(section);
      const isQuery = section === state.query.section;
      const t = isQuery ? 1 : (match.score - min) / (max - min || 1);
      const barHeight = Math.max(1, t * (height - 4 * dpr));
      c.fillStyle = isQuery ? "#4da3ff" : top.has(section) ? "#ff4d4d" : "#5b6068";
      c.fillRect((i / n) * width, height - barHeight, barWidth, barHeight);
    });
  } else {
    c.fillStyle = "#2a2d31";
    c.fillRect(0, height - 2 * dpr, width, 2 * dpr);
  }

  c.fillStyle = "#e8e8e8";
  c.fillRect((state.index / n) * width, 0, Math.max(dpr, barWidth), height);
}

function stripIndex(event) {
  const rect = els.strip.getBoundingClientRect();
  return Math.floor(((event.clientX - rect.left) / rect.width) * state.sections.length);
}

els.strip.addEventListener("click", (event) => goTo(stripIndex(event)));
els.strip.addEventListener("mousemove", (event) => {
  const section = state.sections[stripIndex(event)];
  const match = state.results?.bySection.get(section);
  els.strip.title = match ? `Section ${section}: ${match.score.toFixed(3)}` : `Section ${section}`;
});

// ---------- navigation + start ----------

els.slider.addEventListener("input", () => goTo(Number(els.slider.value)));
els.prev.addEventListener("click", () => goTo(state.index - 1));
els.next.addEventListener("click", () => goTo(state.index + 1));
window.addEventListener("resize", drawStrip);

async function start() {
  const response = await fetch("/api/sections");
  if (!response.ok) throw new Error(`could not list sections (HTTP ${response.status})`);
  const { sections, width, height } = await response.json();
  Object.assign(state, { sections, width, height, index: Math.floor(sections.length / 2) });
  els.canvas.width = width * SCALE;
  els.canvas.height = height * SCALE;
  els.slider.max = sections.length - 1;
  setStatus(`${sections.length} sections · drag on the section to search`);
  drawSection();
  state.stack3d = createStackView(els.stack3d, {
    sections,
    width,
    height,
    imageFor,
    onPick: (section) => goTo(sections.indexOf(section)),
  });
  sections.forEach((section) => imageFor(section).catch(() => {})); // warm the cache for scrubbing
  const named = await fetch("/api/structures");
  if (!named.ok) throw new Error(`could not list regions (HTTP ${named.status})`);
  fillRegions((await named.json()).structures);
}

start().catch((error) => setStatus(`Failed to start: ${error.message}`));
