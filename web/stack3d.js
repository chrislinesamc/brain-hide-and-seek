// 3D stack view: every section as a faint plane, back of the brain to front; the tissue in
// the query and match boxes drawn solid, shaped by the region's labels when searched by name
// and as a soft oval otherwise, so the found region shows as a 3D shape.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const BACKGROUND_LEVEL = 250; // same as the server: brighter pixels are slide, not tissue
const FADED_SCALE = 0.25; // faded planes use quarter-size textures: 741 full ones need ~640 MB
const FADED_OPACITY = 0.02;
const FADED_TINT = 0x8a8f96; // dims the haze, so the solid sections stand out
const TINT_STRENGTH = 0.7; // 0 = plain grey tissue, 1 = fully the item's colour
const OUTLINE_RADIUS = 4; // px around each labelled point (~1.1 mm on the thumbnails)
const MIN_OUTLINE_POINTS = 5; // fewer inside a box: the box missed the region, use the oval

export function createStackView(container, { sections, width, height, imageFor, onPick }) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.setClearColor(0x111214);
  container.append(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(35, 1, 1, 10000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  // Depth = image width: close to true proportions for the BigBrain thumbnails
  // (every 10th 20 µm section over 139 mm wide images: 148 mm deep).
  const depth = width;
  const zOf = (i) => (i / (sections.length - 1) - 0.5) * depth;
  const geometry = new THREE.PlaneGeometry(width, height);

  camera.position.set(-1.3 * width, 0.45 * width, 1.25 * width);
  controls.target.set(0, 0, 0);
  addAxisLabels();

  let lit = []; // meshes of the current query + matches
  let showId = 0; // a slower, older show() must not replace a newer one
  let faded = null; // Group, built once every image has loaded

  async function buildFaded() {
    const loaded = await Promise.all(sections.map((s) => imageFor(s).catch(() => null)));
    faded = new THREE.Group();
    loaded.forEach((img, i) => {
      if (!img) return; // a failed thumbnail leaves a gap rather than breaking the view
      const material = new THREE.MeshBasicMaterial({
        map: tissueTexture(img, FADED_SCALE),
        transparent: true,
        opacity: FADED_OPACITY,
        color: FADED_TINT,
        depthWrite: false,
        side: THREE.DoubleSide,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.z = zOf(i);
      faded.add(mesh);
    });
    scene.add(faded);
  }

  // items: [{ section, boxes: [{x, y, w, h}], color, outline?: [[x, y]] }]
  async function show(items) {
    clear();
    const id = showId;
    const meshes = await Promise.all(
      items.map(async ({ section, boxes, color, outline }) => {
        const img = await imageFor(section);
        const material = new THREE.MeshBasicMaterial({
          map: tissueTexture(img, 1, patchShape(boxes, outline)),
          color: new THREE.Color("#fff").lerp(new THREE.Color(color), TINT_STRENGTH),
          alphaTest: 0.02, // keeps the oval's soft edge
          // transparent puts the patches in the same render pass as the haze, and renderOrder
          // draws them after it, so the haze doesn't wash them out
          transparent: true,
          side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.z = zOf(sections.indexOf(section));
        mesh.renderOrder = 1;
        mesh.userData.section = section;
        return mesh;
      }),
    );
    if (id !== showId) {
      meshes.forEach(dispose);
      return;
    }
    lit = meshes;
    scene.add(...lit);
  }

  function clear() {
    showId++;
    for (const mesh of lit) {
      scene.remove(mesh);
      dispose(mesh);
    }
    lit = [];
  }

  function dispose(mesh) {
    mesh.material.map.dispose();
    mesh.material.dispose();
  }

  // Opacity per pixel (RGBA data, alpha used) for the patches: inside each box, the labelled
  // points widened by OUTLINE_RADIUS, or a soft oval when there are no labels to follow.
  function patchShape(boxes, outline) {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const c = canvas.getContext("2d", { willReadFrequently: true });
    c.fillStyle = "#fff";
    for (const b of boxes) {
      c.save();
      c.beginPath();
      c.rect(b.x, b.y, b.w, b.h);
      c.clip();
      const points = (outline ?? []).filter(([x, y]) => x >= b.x && y >= b.y && x < b.x + b.w && y < b.y + b.h);
      if (points.length >= MIN_OUTLINE_POINTS) {
        c.beginPath();
        for (const [x, y] of points) {
          c.moveTo(x + 0.5 + OUTLINE_RADIUS, y + 0.5);
          c.arc(x + 0.5, y + 0.5, OUTLINE_RADIUS, 0, 2 * Math.PI);
        }
        c.fill();
      } else {
        c.translate(b.x + b.w / 2, b.y + b.h / 2);
        c.scale(b.w / 2, b.h / 2); // unit circle -> the box's oval
        const fade = c.createRadialGradient(0, 0, 0, 0, 0, 1);
        fade.addColorStop(0, "rgba(255,255,255,1)");
        fade.addColorStop(0.55, "rgba(255,255,255,1)");
        fade.addColorStop(1, "rgba(255,255,255,0)");
        c.fillStyle = fade;
        c.fillRect(-1, -1, 2, 2);
      }
      c.restore();
    }
    return c.getImageData(0, 0, width, height).data;
  }

  // Tissue on a transparent background, so stacked sections don't wash out to white;
  // with a shape (full-size alpha from patchShape), only the tissue it covers.
  function tissueTexture(img, scale, shape = null) {
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    const c = canvas.getContext("2d", { willReadFrequently: true });
    c.drawImage(img, 0, 0, canvas.width, canvas.height);
    const pixels = c.getImageData(0, 0, canvas.width, canvas.height);
    const d = pixels.data;
    for (let p = 0; p < d.length; p += 4) {
      const i = p / 4;
      const x = i % canvas.width;
      const y = (i - x) / canvas.width;
      // the outermost pixel ring holds a dark scan border on most thumbnails
      const edge = x === 0 || y === 0 || x === canvas.width - 1 || y === canvas.height - 1;
      if (edge || d[p] >= BACKGROUND_LEVEL) d[p + 3] = 0;
      else if (shape) d[p + 3] = shape[p + 3]; // shapes are only used at scale 1
    }
    c.putImageData(pixels, 0, 0);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
  }

  function addAxisLabels() {
    const label = (text, z) => {
      const canvas = document.createElement("canvas");
      canvas.width = 256;
      canvas.height = 48;
      const c = canvas.getContext("2d");
      c.font = "28px system-ui";
      c.lineWidth = 6; // dark outline keeps the label readable over tissue
      c.strokeStyle = "#111214";
      c.strokeText(text, 4, 34);
      c.fillStyle = "#e8e8e8";
      c.fillText(text, 4, 34);
      const sprite = new THREE.Sprite(
        new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false }),
      );
      sprite.scale.set(width * 0.3, width * 0.056, 1);
      sprite.position.set(0, -height * 0.42, z);
      sprite.renderOrder = 2; // over the lit patches
      scene.add(sprite);
    };
    label("back", zOf(0));
    label("front", zOf(sections.length - 1));
  }

  // ---------- picking: click a lit section to open it ----------

  const raycaster = new THREE.Raycaster();
  let down = null;
  renderer.domElement.addEventListener("pointerdown", (e) => (down = { x: e.clientX, y: e.clientY }));
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!down || Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) return; // that was an orbit
    const rect = renderer.domElement.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1,
    );
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(lit).find((h) => opaqueAt(h));
    if (hit) onPick(hit.object.userData.section);
  });

  // only count clicks on tissue, not on the transparent slide around it
  function opaqueAt(hit) {
    const canvas = hit.object.material.map.image;
    const x = Math.floor(hit.uv.x * canvas.width);
    const y = Math.floor((1 - hit.uv.y) * canvas.height);
    return canvas.getContext("2d").getImageData(x, y, 1, 1).data[3] > 0;
  }

  // ---------- render loop ----------

  let size = "";
  function resize() {
    const w = container.clientWidth;
    const h = container.clientHeight;
    if (size === `${w}x${h}`) return;
    size = `${w}x${h}`;
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = `${w}px`;
    renderer.domElement.style.height = `${h}px`;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  renderer.setAnimationLoop(() => {
    resize();
    controls.update();
    renderer.render(scene, camera);
  });

  const ready = buildFaded();
  return { show, clear, ready };
}
