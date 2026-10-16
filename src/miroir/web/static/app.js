// Miroir: the four screens of functional spec §5 (home, photo, results, record), plus login.
// Plain modules, no build step. The DSFR script handles the header menu, modals, accordions
// and the theme switch; this file handles the views and talks to /api.

import { exifPosition } from './exif.js';
import {
  $, $$, api, COLLECTION_NAMES, el, fill, grid, img, levelBadge,
} from './ui.js';
import { setupChat } from './chat.js';

/** Longest side, in pixels, of a photo sent to the server (it shrinks to 512 px again). */
const SEND_SIDE = 1024;
/** Longest side kept in the browser for display and drawing a box. */
const KEEP_SIDE = 2048;
/** Longest side, in pixels, of the photo sent to find its objects (`DETECT_SIDE`). */
const DETECT_SIDE = 1024;
/** Outlines drawn at first (`MAX_OUTLINES` in outlines.py): the server sends up to 40, biggest
 * and most central first; the rest are shown on request (walls of pictures). */
const MAX_OUTLINES = 20;
/** Smallest tap area, in CSS pixels, around an outline. */
const TAP_SIDE = 44;
/** A press that moves less than this many pixels is a tap, not a drawn box. */
const TAP_SLOP = 10;
/** Smallest box, as a share of the shorter side, that counts as drawn. */
const MIN_BOX = 0.06;
const WARM_EVERY_MS = 4 * 60 * 1000;
/** Retries while the search engine starts (each waits up to 100 s on the server). */
const MAX_RETRIES = 4;

const state = {
  photo: null,       // { canvas, url } of the current photo
  box: null,         // { x, y, w, h } in canvas pixels, or null for the whole photo
  outlines: [],      // objects found in the photo: { x0, y0, x1, y1, label } in shares of it
  outline: null,     // index of the outline searched, or null
  allOutlines: false, // true once the person asked to see every outline
  queryUrl: null,    // what the results page shows as "your photo"
  text: '',          // words of the current search
  filters: emptyFilters(),
  response: null,
  facets: null,
  lastWarm: 0,
  searching: false,
  photoPosition: null, // where the current photo was taken (its EXIF GPS), or null
  // An archive print on its card: the caption to send (null: the server reads it), and
  // whether to search the print alone (true) or the whole photo.
  archive: { caption: null, mount: true },
  // Set while the conversation view asks the visitor to frame part of their photo: called
  // with the framed part as a data URL, or null when they leave without framing.
  pick: null,
};

function emptyFilters() {
  return { collections: [], region: '', departement: '', domain: '', period: '' };
}

const TYPE_LABELS = { building: 'Bâtiment', object: 'Objet' };

function warm() {
  if (Date.now() - state.lastWarm < WARM_EVERY_MS) return;
  state.lastWarm = Date.now();
  api('warm', { method: 'POST' }).catch(() => {});
}

// --- routing ---------------------------------------------------------------------------------

function go(hash) {
  if (location.hash === hash) route(); else location.hash = hash;
}

function show(name) {
  for (const view of $$('.view')) view.hidden = view.dataset.view !== name;
  window.scrollTo({ top: 0 });
}

async function route() {
  const hash = location.hash || '#/';
  if (state.pick && hash !== '#/photo') endPick(null);
  if (hash === '#/connexion') { show('login'); $('#password').focus(); return; }
  if (!(await loggedIn())) { go('#/connexion'); return; }
  if (hash === '#/conversation') { show('conversation'); chat.enter(); return; }
  if (hash === '#/photo') {
    if (!state.photo) { go('#/'); return; }
    show('photo'); drawBox(); drawOutlines(); warm(); return;
  }
  if (hash === '#/resultats') {
    if (!state.response && !state.searching) { go('#/'); return; }
    show('results'); return;
  }
  if (hash.startsWith('#/fiche/')) {
    show('record');
    await renderRecord(decodeURIComponent(hash.slice('#/fiche/'.length)));
    return;
  }
  show('home');
}

let sessionChecked = false;
async function loggedIn() {
  if (sessionChecked) return true;
  try {
    await api('me');
    sessionChecked = true;
    $$('[data-when="logged-in"]').forEach((n) => { n.hidden = false; });
    loadAfterLogin();
    return true;
  } catch { return false; }
}

let loadedAfterLogin = false;
function loadAfterLogin() {
  if (loadedAfterLogin) return;
  loadedAfterLogin = true;
  renderExamples();
  loadFacets();
  api('stats').then((s) => {
    $('#index-size').textContent = `${s.images.toLocaleString('fr-FR')}`;
  }).catch(() => {});
}

// --- login -----------------------------------------------------------------------------------

$('#login-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const group = $('#login-group');
  const error = $('#login-error');
  try {
    await api('login', { method: 'POST', body: { password: $('#password').value } });
    group.classList.remove('fr-input-group--error');
    error.hidden = true;
    $('#password').value = '';
    sessionChecked = false;
    go('#/');
  } catch (err) {
    group.classList.add('fr-input-group--error');
    error.textContent = err.message;
    error.hidden = false;
  }
});

document.addEventListener('click', async (event) => {
  const action = event.target.closest('[data-action]')?.dataset.action;
  if (action === 'logout') {
    await api('logout', { method: 'POST' }).catch(() => {});
    sessionChecked = false;
    $$('[data-when="logged-in"]').forEach((n) => { n.hidden = true; });
    go('#/connexion');
  } else if (action === 'home') {
    go('#/');
  } else if (action === 'edit-box') {
    if (state.photo) go('#/photo');
  }
});

// --- home: photos, words, examples -----------------------------------------------------------

for (const input of [$('#camera-input'), $('#upload-input')]) {
  input.addEventListener('change', async () => {
    const file = input.files && input.files[0];
    input.value = '';
    if (!file) return;
    try {
      await usePhoto(file, await exifPosition(file));
      go('#/photo');
    } catch {
      alertHome('Cette image n’a pas pu être ouverte. Essayez une photo JPEG ou PNG.');
    }
  });
}

$('#describe-input').addEventListener('focus', warm, { once: true });
$('#describe-form').addEventListener('submit', (event) => {
  event.preventDefault();
  const text = $('#describe-input').value.trim();
  if (!text) { $('#describe-input').focus(); return; }
  state.photo = null; state.box = null; state.queryUrl = null;
  $('#refine-input').value = text;
  runSearch({ text });
});

for (const chip of document.querySelectorAll('#describe-chips button')) {
  chip.addEventListener('click', () => {
    $('#describe-input').value = chip.textContent;
    $('#describe-form').requestSubmit();
  });
}

function alertHome(message) {
  const box = el('div', { class: 'fr-alert fr-alert--error fr-alert--sm fr-mb-2w', role: 'alert' }, el('p', { text: message }));
  $('#describe-form').before(box);
  setTimeout(() => box.remove(), 6000);
}

async function renderExamples() {
  const list = $('#examples');
  const mosaic = $('#hero-mosaic');
  let examples = [];
  try { examples = await api('examples'); } catch { return; }
  list.replaceChildren(...examples.map((example) => el('li', {},
    el('button', {
      class: 'miroir-example', type: 'button',
      'aria-label': `${example.label} : ${example.frame ? 'ouvrir la photo pour encadrer une œuvre' : 'lancer la recherche'}`,
      onclick: () => useExample(example),
    },
    img(example.image_url, null, { class: 'miroir-example__img' }),
    el('div', { class: 'miroir-example__body' },
      el('div', { class: 'miroir-example__kind', text: example.kind }),
      el('div', { class: 'miroir-example__label', text: example.label }),
      el('div', { class: 'miroir-example__credit', text: `Photo : ${example.credit}` }))))));
  mosaic.replaceChildren(...examples.slice(0, 6).map((e) => img(e.image_url, null)));
}

async function useExample(example) {
  try {
    const response = await fetch(example.image_url, { credentials: 'same-origin' });
    if (!response.ok) throw new Error('example');
    await usePhoto(await response.blob());
  } catch {
    alertHome('Cet exemple n’a pas pu être chargé.');
    return;
  }
  if (example.frame) {
    go('#/photo');
    return;
  }
  state.text = '';
  $('#refine-input').value = '';
  searchPhoto(false);
}

// --- photo: load, draw a box, crop -----------------------------------------------------------

async function usePhoto(blob, position = null) {
  state.photoPosition = position;
  showPhotoPosition();
  const bitmap = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  const scale = Math.min(1, KEEP_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close?.();
  if (state.photo?.url) URL.revokeObjectURL(state.photo.url);
  const url = URL.createObjectURL(await toBlob(canvas, 0.92));
  state.photo = { canvas, url };
  state.archive = { caption: null, mount: true };
  state.box = null;
  state.outline = null;
  $('#stage-image').src = url;
  $('#box-hint').innerHTML = DRAW_HINT;
  drawBox();
  detectObjects();
  warm();
}

function toBlob(canvas, quality) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('encode'))), 'image/jpeg', quality);
  });
}

/** Part of the photo (canvas pixels) as a JPEG blob at most `side` on its long side. */
function cropBlob(box, side) {
  const source = state.photo.canvas;
  const scale = Math.min(1, side / Math.max(box.w, box.h));
  const out = document.createElement('canvas');
  out.width = Math.max(1, Math.round(box.w * scale));
  out.height = Math.max(1, Math.round(box.h * scale));
  out.getContext('2d').drawImage(source, box.x, box.y, box.w, box.h, 0, 0, out.width, out.height);
  return toBlob(out, 0.9);
}

function dataUrl(blob) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(blob);
  });
}

function wholePhoto() {
  return { x: 0, y: 0, w: state.photo.canvas.width, h: state.photo.canvas.height };
}

/** The photo, or the boxed part of it, as a JPEG data URL at most SEND_SIDE on its long side. */
async function queryImage() {
  const blob = await cropBlob(state.box || wholePhoto(), SEND_SIDE);
  if (state.queryUrl) URL.revokeObjectURL(state.queryUrl);
  state.queryUrl = URL.createObjectURL(blob);
  return dataUrl(blob);
}

const stage = $('#stage');
let dragStart = null;

function stagePoint(event) {
  const image = $('#stage-image');
  const rect = image.getBoundingClientRect();
  const x = Math.min(Math.max(event.clientX - rect.left, 0), rect.width);
  const y = Math.min(Math.max(event.clientY - rect.top, 0), rect.height);
  return { x, y, rect };
}

// A press on an outline that barely moves is a tap: search that object. Any press that moves
// draws a box, even one that starts on an outline (pointer capture sends every move here).
stage.addEventListener('pointerdown', (event) => {
  if (!state.photo) return;
  stage.setPointerCapture(event.pointerId);
  dragStart = stagePoint(event);
  const outline = event.target.closest?.('.miroir-outline');
  dragStart.outline = outline ? Number(outline.dataset.index) : null;
  dragStart.moved = false;
  event.preventDefault();
});
stage.addEventListener('pointermove', (event) => {
  if (!dragStart) return;
  const now = stagePoint(event);
  if (!dragStart.moved && Math.hypot(now.x - dragStart.x, now.y - dragStart.y) < TAP_SLOP) return;
  dragStart.moved = true;
  state.outline = null;
  setBoxFromScreen(dragStart, now, false);
});
stage.addEventListener('pointerup', (event) => {
  if (!dragStart) return;
  const start = dragStart;
  dragStart = null;
  if (!start.moved && start.outline !== null) { searchOutline(start.outline); return; }
  if (!start.moved) return;
  setBoxFromScreen(start, stagePoint(event), true);
});
stage.addEventListener('pointercancel', () => { dragStart = null; drawBox(); });

function setBoxFromScreen(a, b, done) {
  const { rect } = a;
  const canvas = state.photo.canvas;
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  const box = {
    x: Math.round(Math.min(a.x, b.x) * sx),
    y: Math.round(Math.min(a.y, b.y) * sy),
    w: Math.round(Math.abs(b.x - a.x) * sx),
    h: Math.round(Math.abs(b.y - a.y) * sy),
  };
  const small = Math.min(box.w, box.h) < MIN_BOX * Math.min(canvas.width, canvas.height);
  state.box = small && done ? null : box;
  drawBox(small && !done);
}

function drawBox(faint = false) {
  const node = $('#box');
  const image = $('#stage-image');
  const has = Boolean(state.box);
  stage.classList.toggle('has-box', has);
  $('#search-box').hidden = !has;
  $('#clear-box').hidden = !has;
  $('#search-whole').classList.toggle('fr-btn--secondary', has);
  if (!has || !state.photo || !image.clientWidth) { node.hidden = true; return; }
  const canvas = state.photo.canvas;
  const offsetX = image.offsetLeft;
  const offsetY = image.offsetTop;
  const sx = image.clientWidth / canvas.width;
  const sy = image.clientHeight / canvas.height;
  Object.assign(node.style, {
    left: `${offsetX + state.box.x * sx}px`, top: `${offsetY + state.box.y * sy}px`,
    width: `${state.box.w * sx}px`, height: `${state.box.h * sy}px`, opacity: faint ? '0.5' : '1',
  });
  node.hidden = false;
}

$('#stage-image').addEventListener('load', () => { drawBox(); drawOutlines(); });
window.addEventListener('resize', () => { drawBox(); drawOutlines(); });
$('#clear-box').addEventListener('click', () => { state.box = null; state.outline = null; drawBox(); drawOutlines(); });
$('#search-whole').addEventListener('click', () => { state.box = null; state.outline = null; searchPhoto(); });
$('#search-box').addEventListener('click', () => searchPhoto());

// --- F4: outline each object found in the photo ---------------------------------------------

const OUTLINE_LABELS = {
  painting: 'tableau', sculpture: 'sculpture', statue: 'statue', vase: 'vase',
  furniture: 'meuble', building: 'bâtiment', 'stained glass window': 'vitrail',
};
let detectId = 0;

/** Ask the server for the objects in the photo; draw them when they come. A sleeping GPU
 * answers 503 while it starts: try again quietly, the person can draw a box meanwhile. */
async function detectObjects() {
  const id = ++detectId;
  state.outlines = [];
  state.allOutlines = false;
  drawOutlines();
  setOutlineStatus('Repérage des œuvres dans la photo…', { busy: true });
  const image = await dataUrl(await cropBlob(wholePhoto(), DETECT_SIDE));
  for (let attempt = 0; ; attempt += 1) {
    try {
      const found = await api('detect', { method: 'POST', body: { image } });
      if (id !== detectId) return;
      state.outlines = found.boxes;
      drawOutlines();
      if (!state.outlines.length) setOutlineStatus('');
      return;
    } catch (err) {
      if (id !== detectId) return;
      if (err.status === 503 && attempt < MAX_RETRIES) {
        await new Promise((resolve) => { setTimeout(resolve, 2000); });
        continue;
      }
      setOutlineStatus('');
      return;
    }
  }
}

const DRAW_HINT = 'Pour chercher une partie de la photo, un tableau sur un mur ou un détail, <strong>tracez un cadre</strong> dessus avec le doigt ou la souris.';

function setOutlineStatus(text, { busy = false } = {}) {
  const node = $('#outline-status');
  node.classList.toggle('is-busy', busy);
  node.textContent = text;
}

function shownOutlines() {
  return state.allOutlines ? state.outlines : state.outlines.slice(0, MAX_OUTLINES);
}

function describeOutlines() {
  const n = shownOutlines().length;
  const more = state.outlines.length - n;
  setOutlineStatus(n ? `${n} œuvre${n > 1 ? 's' : ''} repérée${n > 1 ? 's' : ''}. Touchez un cadre bleu pour la chercher.` : '');
  $('#box-hint').innerHTML = n
    ? 'Une œuvre n’a pas de cadre ? <strong>Tracez le vôtre</strong> avec le doigt.'
    : DRAW_HINT;
  const button = $('#more-outlines');
  button.hidden = more <= 0;
  button.textContent = `Afficher ${more > 1 ? `les ${more} autres objets repérés` : 'l’autre objet repéré'}`;
}

$('#more-outlines').addEventListener('click', () => {
  state.allOutlines = true;
  drawOutlines();
  $(`#outlines .miroir-outline:nth-child(${MAX_OUTLINES + 1})`)?.focus();
});

/** An outline (shares of the photo) in canvas pixels. */
function outlineBox(o) {
  const { width, height } = state.photo.canvas;
  return {
    x: Math.round(o.x0 * width), y: Math.round(o.y0 * height),
    w: Math.max(1, Math.round((o.x1 - o.x0) * width)), h: Math.max(1, Math.round((o.y1 - o.y0) * height)),
  };
}

function searchOutline(index) {
  const outline = state.outlines[index];
  if (!outline) return;
  state.outline = index;
  state.box = outlineBox(outline);
  drawBox();
  drawOutlines();
  searchPhoto();
}

function drawOutlines() {
  const layer = $('#outlines');
  const image = $('#stage-image');
  if (state.photo && state.outlines.length) describeOutlines();
  else $('#more-outlines').hidden = true;
  if (!state.photo || !image.clientWidth || !state.outlines.length) { layer.replaceChildren(); return; }
  const left = image.offsetLeft;
  const top = image.offsetTop;
  const w = image.clientWidth;
  const h = image.clientHeight;
  layer.replaceChildren(...shownOutlines().map((o, i) => {
    const width = (o.x1 - o.x0) * w;
    const height = (o.y1 - o.y0) * h;
    const pad = Math.max(0, (TAP_SIDE - Math.min(width, height)) / 2);
    const name = OUTLINE_LABELS[o.label] || 'objet';
    return el('button', {
      class: `miroir-outline${state.outline === i ? ' is-selected' : ''}`,
      type: 'button',
      'data-index': String(i),
      'aria-label': `Chercher l’œuvre ${i + 1} (${name})`,
      style: `left:${left + o.x0 * w}px;top:${top + o.y0 * h}px;width:${width}px;height:${height}px;--pad:${pad}px`,
      // A tap arrives through the stage's pointer handlers; this click is the keyboard's.
      onclick: (event) => { if (event.detail === 0) searchOutline(i); },
    }, el('span', { class: 'miroir-outline__n', 'aria-hidden': 'true', text: String(i + 1) }));
  }));
}

// --- search ----------------------------------------------------------------------------------

async function searchPhoto(keepText = true) {
  if (state.pick) { endPick(await queryImage()); return; }
  state.text = keepText ? $('#refine-input').value.trim() : '';
  const image = await queryImage();
  runSearch({ image, text: state.text, ...state.archive });
}

$('#refine-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  state.text = $('#refine-input').value.trim();
  if (state.photo) {
    runSearch({ image: await queryImage(), text: state.text, ...state.archive });
  } else if (state.text) {
    runSearch({ text: state.text });
  }
});

let searchId = 0;
async function runSearch({ image = null, text = '', caption = null, mount = true }) {
  const id = ++searchId;
  state.searching = true;
  state.text = text;
  state.lastQuery = { image, text, caption, mount };
  go('#/resultats');
  renderQuery(null);
  renderWaiting();
  try {
    // Asked anew for each photo search while the toggle is on; never kept after it.
    const position = image ? await searchPosition() : null;
    let response = null;
    for (let attempt = 0; ; attempt += 1) {
      try {
        const body = { image, text, filters: state.filters };
        if (position) body.position = position;
        if (image && caption !== null) body.caption = caption;
        if (image && !mount) body.mount = false;
        response = await api('search', { method: 'POST', body });
        break;
      } catch (err) {
        // 503: the GPU container is still starting; the server asks to try again.
        if (err.status !== 503 || attempt >= MAX_RETRIES || id !== searchId) throw err;
        await new Promise((resolve) => { setTimeout(resolve, 2000); });
      }
    }
    if (id !== searchId) return;
    if (response.caption_read) {
      // Read once: later searches of this photo send it back instead of reading it again.
      state.archive.caption = response.caption || '';
      state.lastQuery.caption = state.archive.caption;
    }
    state.response = response;
    renderQuery(response);
    renderResults(response);
  } catch (err) {
    if (id !== searchId || err.status === 401) return;
    $('#results-status').replaceChildren(el('div', { class: 'fr-alert fr-alert--error', role: 'alert' },
      el('h2', { class: 'fr-alert__title', text: 'La recherche n’a pas abouti' }),
      el('p', { text: err.message })));
    $('#results').replaceChildren();
  } finally {
    if (id === searchId) state.searching = false;
  }
}

// --- position (H2): only when the person turns it on, only for the request ------------------

/** Where the browser says the person stands, or null when refused or unavailable. */
function currentPosition() {
  return new Promise((resolve) => {
    if (!('geolocation' in navigator)) { resolve(null); return; }
    navigator.geolocation.getCurrentPosition(
      (found) => resolve({ latitude: found.coords.latitude, longitude: found.coords.longitude }),
      () => resolve(null),
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 },
    );
  });
}

/** The position a photo search uses: the photo's own, else the person's if they turned it on. */
async function searchPosition() {
  if (state.photoPosition) {
    state.positionSource = 'photo';
    return state.photoPosition;
  }
  state.positionSource = 'device';
  return $('#use-position').checked ? currentPosition() : null;
}

/** The note "Position de la photo utilisée" replaces the toggle while the photo has one. */
function showPhotoPosition() {
  const has = Boolean(state.photoPosition);
  $('#photo-position').hidden = !has;
  $('#position-toggle').hidden = has;
}

$('#photo-position-off').addEventListener('click', () => {
  state.photoPosition = null;
  showPhotoPosition();
});

$('#use-position').addEventListener('change', async (event) => {
  const toggle = event.target;
  const hint = $('#use-position-hint');
  if (!toggle.checked) {
    hint.textContent = POSITION_HINT;
    return;
  }
  hint.textContent = 'Recherche de votre position…';
  // The browser asks permission here, on the tap, not before.
  const found = await currentPosition();
  if (!found) {
    toggle.checked = false;
    hint.textContent = 'Position indisponible. Autorisez-la dans le navigateur, ou cherchez sans.';
    return;
  }
  hint.textContent = POSITION_HINT;
});
const POSITION_HINT = $('#use-position-hint').textContent;

function rerun() {
  if (state.lastQuery) runSearch(state.lastQuery);
}

function renderQuery(response) {
  const box = $('#query-summary');
  const photo = Boolean(state.photo && state.lastQuery?.image);
  const words = state.text;
  const meta = el('p', { class: 'miroir-query__meta fr-text--sm' });
  if (response?.guessed_type) {
    const kind = TYPE_LABELS[response.guessed_type];
    meta.append(el('span', {
      class: `fr-tag fr-tag--sm ${response.guessed_type === 'building' ? 'fr-icon-building-line' : 'fr-icon-palette-line'} fr-tag--icon-left`,
      title: 'Ce que Miroir pense voir : il pèse ses deux modèles en conséquence',
      text: kind,
    }));
  }
  if (response?.nearby != null) {
    const km = String(response.near_km).replace('.', ',');
    meta.append(el('span', {
      class: 'fr-tag fr-tag--sm fr-icon-map-pin-2-line fr-tag--icon-left',
      title: response.nearby
        ? `Les monuments et œuvres les plus proches passent devant (${response.nearby} à moins de ${km} km)`
        : `Rien de connu à moins de ${km} km : résultats de toute la base`,
      text: response.nearby
        ? (state.positionSource === 'photo' ? 'Près du lieu de la photo' : 'Près de vous')
        : `Rien de connu à moins de ${km} km`,
    }));
  }
  const picture = photo ? response?.picture : null;
  if (photo && response?.mount) {
    meta.append(el('span', {
      class: 'fr-tag fr-tag--sm fr-icon-image-line fr-tag--icon-left',
      title: 'Miroir a reconnu un tirage d’archive sur son carton et a lu sa légende',
      text: response.caption ? 'Cadre et légende détectés' : 'Cadre détecté',
    }));
  }
  let title = `« ${words} »`;
  if (photo) title = state.box ? 'Votre photo (partie encadrée)' : 'Votre photo';
  if (picture) title = 'Votre photo (le tirage, sans son carton)';
  const thumb = photo ? img(state.queryUrl, null, { class: 'miroir-query__thumb', alt: 'Votre photo' }) : null;
  if (picture && !state.box) showPicture(thumb, picture);
  fill(box,
    thumb,
    el('div', { class: 'miroir-query__text' },
      el('p', { class: 'miroir-query__title', text: title }),
      photo && words ? el('p', { class: 'fr-text--sm fr-mb-0', text: `Précisé par « ${words} »` }) : null,
      meta),
    el('ul', { class: 'fr-btns-group fr-btns-group--inline fr-btns-group--sm fr-btns-group--icon-left fr-mb-0' },
      photo ? el('li', {}, el('button', { class: 'fr-btn fr-btn--sm fr-btn--tertiary fr-icon-crop-line fr-btn--icon-left', 'data-action': 'edit-box', text: 'Encadrer' })) : null,
      el('li', { class: 'fr-hidden-md' }, el('button', { class: 'fr-btn fr-btn--sm fr-btn--tertiary fr-icon-filter-line fr-btn--icon-left', type: 'button', onclick: () => $('#refine').scrollIntoView({ behavior: 'smooth' }), text: 'Affiner' })),
      el('li', {}, el('a', { class: 'fr-btn fr-btn--sm fr-btn--tertiary-no-outline fr-icon-camera-line fr-btn--icon-left', href: '#/', text: 'Nouvelle recherche' }))),
  );
  if (photo && response && (response.mount || response.caption)) box.append(captionPanel(response));
  $('#refine-label').firstChild.textContent = photo ? 'Préciser avec des mots ' : 'Modifier les mots ';
}

/** The thumbnail shows the print alone: what was searched. */
async function showPicture(thumb, [x0, y0, x1, y1]) {
  const { width, height } = state.photo.canvas;
  const box = { x: x0 * width, y: y0 * height, w: (x1 - x0) * width, h: (y1 - y0) * height };
  thumb.src = URL.createObjectURL(await cropBlob(box, 300));
}

/** What the caption says and where it places the print; the person can correct or remove it. */
function captionPanel(response) {
  const again = (archive) => {
    state.archive = { ...state.archive, ...archive };
    runSearch({ ...state.lastQuery, ...state.archive });
  };
  const input = el('input', { class: 'fr-input', type: 'text', id: 'caption-input', maxlength: '300', value: response.caption || '', autocomplete: 'off' });
  const form = el('form', { class: 'miroir-caption__form', hidden: true },
    el('label', { class: 'fr-label', for: 'caption-input', text: 'Légende du carton' }),
    el('div', { class: 'miroir-caption__row' }, input,
      el('button', { class: 'fr-btn fr-btn--sm', type: 'submit', text: 'Relancer' })));
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    again({ caption: input.value.trim() });
  });
  const button = (label, icon, onclick) => el('li', {}, el('button', {
    class: `fr-btn fr-btn--sm fr-btn--tertiary-no-outline ${icon} fr-btn--icon-left`, type: 'button', onclick, text: label,
  }));
  const edit = () => { form.hidden = false; input.focus(); };
  const read = response.caption
    ? el('p', { class: 'fr-text--sm fr-mb-0' },
      el('strong', { text: response.caption_read ? 'Légende lue : ' : 'Légende : ' }), `« ${response.caption} »`)
    : el('p', { class: 'fr-text--sm fr-mb-0', text: 'Aucune légende lue sur le carton.' });
  let place = null;
  if (response.caption_place) place = `Lieu : ${response.caption_place}. Les monuments de ce lieu passent devant.`;
  else if (response.caption) place = 'Les mots de la légende servent à la recherche, avec la photo.';
  return el('div', { class: 'miroir-caption' }, read,
    place ? el('p', { class: 'fr-text--sm fr-mb-0 miroir-caption__place', text: place }) : null,
    form,
    el('ul', { class: 'fr-btns-group fr-btns-group--inline fr-btns-group--sm fr-btns-group--icon-left fr-mb-0' },
      button(response.caption ? 'Corriger la légende' : 'Ajouter une légende', 'fr-icon-edit-line', edit),
      response.caption ? button('Retirer la légende', 'fr-icon-delete-line', () => again({ caption: '' })) : null,
      response.picture ? button('Chercher toute la photo', 'fr-icon-fullscreen-line', () => again({ mount: false })) : null));
}

let waitTimer = null;
function renderWaiting() {
  clearInterval(waitTimer);
  const started = Date.now();
  const note = el('span', { text: 'Recherche en cours…' });
  $('#results-status').replaceChildren(el('div', { class: 'miroir-waiting', role: 'status' }, el('span', { class: 'miroir-spinner', 'aria-hidden': 'true' }), note));
  waitTimer = setInterval(() => {
    if (!state.searching) { clearInterval(waitTimer); return; }
    const seconds = Math.round((Date.now() - started) / 1000);
    if (seconds >= 4) note.textContent = `Le moteur de recherche se réveille, cela peut prendre une à deux minutes la première fois… (${seconds} s)`;
  }, 1000);
  const skeleton = () => el('li', {}, el('div', { class: 'miroir-skeleton', 'aria-hidden': 'true' },
    el('div', { class: 'miroir-skeleton__img miroir-shimmer' }),
    el('div', { class: 'miroir-skeleton__line miroir-shimmer' }),
    el('div', { class: 'miroir-skeleton__line miroir-shimmer' })));
  $('#results').replaceChildren(el('ul', { class: 'miroir-grid' }, Array.from({ length: 8 }, skeleton)));
}

function renderResults(response) {
  clearInterval(waitTimer);
  const status = $('#results-status');
  const results = $('#results');
  const cards = response.cards;
  if (!cards.length) {
    status.replaceChildren(el('div', { class: 'fr-alert fr-alert--info' },
      el('h2', { class: 'fr-alert__title', text: 'Aucun résultat' }),
      el('p', { text: 'Aucune notice ne correspond avec ces filtres. Essayez d’en retirer.' })));
    results.replaceChildren();
    return;
  }
  const filtered = activeFilterCount() > 0;
  if (response.words_found === false) {
    status.replaceChildren(el('div', { class: 'fr-alert fr-alert--info fr-mb-3w' },
      el('h2', { class: 'fr-alert__title', text: 'Aucune notice ne contient ces mots' }),
      el('p', { text: 'Voici les images qui ressemblent le plus à votre description. Vérifiez l’orthographe, ou décrivez autrement : la forme, la matière, le sujet, un lieu.' })));
  } else {
    status.replaceChildren();
  }
  const photo = Boolean(response.guessed_type);
  fill(results,
    el('div', { class: 'miroir-section-title' },
      el('h2', { class: 'fr-h4', text: `Résultats${filtered ? ' filtrés' : ''}` }),
      el('p', { class: 'fr-text--sm miroir-muted fr-mb-1w', text: photo ? 'Du plus proche de votre photo au moins proche.' : 'Du plus proche de votre description au moins proche.' })),
    levelLegend(),
    cards[0].level !== 'tres-proche' ? buildingTip(response) : null,
    grid(cards));
}

/** What each level means, in a line the visitor reads once. */
function levelLegend() {
  return el('p', { class: 'fr-text--xs miroir-muted fr-mb-2w miroir-legend' },
    levelBadge('tres-proche', 'Très proche'), ' sans doute la bonne notice · ',
    levelBadge('proche', 'Proche'), ' peut-être elle · ',
    levelBadge('ressemblance', 'Ressemblance'), ' à comparer avec votre photo');
}

/** After a weak building search: what helps most, in the visitor's terms. */
function buildingTip(response) {
  if (response.guessed_type !== 'building' || state.text) return null;
  const text = response.nearby == null
    ? 'Devant un monument, activez « Utiliser ma position » ou ajoutez le nom de la commune avec « Préciser avec des mots » : Miroir cherchera d’abord ce qui est à côté. Une photo de toute la façade aide aussi.'
    : 'Ajoutez le nom du monument ou de la commune avec « Préciser avec des mots ». Une photo de toute la façade aide aussi.';
  return el('p', { class: 'fr-text--sm fr-mb-2w', text });
}

// --- filters (F3) ----------------------------------------------------------------------------

async function loadFacets() {
  try { state.facets = await api('facets'); } catch { return; }
  const f = state.facets;
  const fieldset = $('#filter-collections');
  for (const c of f.collections) {
    const id = `filter-collection-${c.value}`;
    fieldset.append(el('div', { class: 'fr-fieldset__element' },
      el('div', { class: 'fr-checkbox-group fr-checkbox-group--sm' },
        el('input', { type: 'checkbox', id, value: c.value, name: 'collection' }),
        el('label', { class: 'fr-label', for: id, text: `${COLLECTION_NAMES[c.value] || c.value}` }))));
  }
  fillSelect($('#filter-region'), f.regions, 'Toutes');
  fillDepartements('');
  fillSelect($('#filter-domain'), f.domains, 'Tous');
  fillSelect($('#filter-period'), f.periods, 'Toutes');
  $('#filter-region').addEventListener('change', (e) => fillDepartements(e.target.value));
}

function fillSelect(select, values, all) {
  select.replaceChildren(el('option', { value: '', text: all }),
    ...values.map((v) => el('option', { value: v.value, text: `${v.value} (${v.count.toLocaleString('fr-FR')})` })));
}

function fillDepartements(region) {
  const list = state.facets?.departements[region] || state.facets?.departements[''] || [];
  fillSelect($('#filter-departement'), list, 'Tous');
}

function readFilters() {
  return {
    collections: $$('#filter-collections input:checked').map((i) => i.value),
    region: $('#filter-region').value,
    departement: $('#filter-departement').value,
    domain: $('#filter-domain').value,
    period: $('#filter-period').value,
  };
}

function activeFilterCount() {
  const f = state.filters;
  return f.collections.length + ['region', 'departement', 'domain', 'period'].filter((k) => f[k]).length;
}

function showFilterCount() {
  const n = activeFilterCount();
  $('#filter-count').textContent = n ? String(n) : '';
}

$('#filters-form').addEventListener('submit', (event) => {
  event.preventDefault();
  state.filters = readFilters();
  showFilterCount();
  rerun();
});
$('#filters-reset').addEventListener('click', () => {
  $('#filters-form').reset();
  fillDepartements('');
  state.filters = emptyFilters();
  showFilterCount();
  rerun();
});

// Open the filters on wide screens, where they sit beside the results.
if (window.matchMedia('(min-width: 48em)').matches) {
  $('#filters-button').setAttribute('aria-expanded', 'true');
}

// --- record page (screen 4) ------------------------------------------------------------------

async function renderRecord(entityId) {
  const root = $('#record');
  root.replaceChildren(el('div', { class: 'miroir-waiting' }, el('span', { class: 'miroir-spinner' }), 'Chargement de la fiche…'));
  let page;
  try {
    page = await api(`entity/${encodeURIComponent(entityId)}`);
  } catch (err) {
    root.replaceChildren(el('div', { class: 'fr-alert fr-alert--error' }, el('p', { text: err.message })));
    return;
  }
  const viewer = el('div', { class: 'miroir-viewer' });
  const caption = el('p', { class: 'miroir-viewer__caption' });
  const thumbs = el('ul', { class: 'miroir-thumbs', 'aria-label': 'Toutes les images' });
  const select = (index) => {
    const image = page.images[index];
    viewer.replaceChildren(img(image.url, image.fallback_url, { alt: `${page.title}, image ${index + 1} sur ${page.images.length}`, loading: 'eager' }));
    caption.textContent = image.label || '';
    $$('.miroir-thumb', thumbs).forEach((t, i) => t.setAttribute('aria-current', String(i === index)));
  };
  thumbs.replaceChildren(...page.images.map((image, i) => el('li', {},
    el('button', { class: 'miroir-thumb', type: 'button', 'aria-label': `Image ${i + 1}`, onclick: () => select(i) },
      img(image.thumb_url, image.fallback_url)))));

  const facts = el('dl', { class: 'miroir-facts' },
    page.facts.flatMap((f) => [el('dt', { text: f.label }), el('dd', { text: f.value })]));
  const description = page.description
    ? el('p', { class: 'miroir-description is-clamped', text: page.description })
    : null;
  const more = description && page.description.length > 400
    ? el('button', { class: 'fr-link fr-mb-3w', type: 'button', text: 'Lire la suite', onclick: (e) => { description.classList.remove('is-clamped'); e.target.remove(); } })
    : null;
  const records = page.records.length > 1
    ? el('section', { class: 'fr-accordion fr-mt-3w' },
      el('h2', { class: 'fr-accordion__title' },
        el('button', { class: 'fr-accordion__btn', 'aria-expanded': 'false', 'aria-controls': 'linked-records', text: `${page.records.length} notices pour ce ${page.collection === 'memoire' ? 'monument' : 'objet'}` })),
      el('div', { class: 'fr-collapse', id: 'linked-records' },
        el('ul', { class: 'miroir-records' }, page.records.map((r) => el('li', {},
          el('a', { class: 'fr-link fr-link--sm', href: r.pop_url, target: '_blank', rel: 'noopener', title: `${r.title}, nouvelle fenêtre`, text: `${r.title} (${r.ref})` }))))))
    : null;

  root.replaceChildren(
    el('button', { class: 'fr-link fr-icon-arrow-left-line fr-link--icon-left fr-mb-3w', type: 'button', onclick: () => (state.response ? history.back() : go('#/')), text: state.response ? 'Retour aux résultats' : 'Accueil' }),
    el('div', { class: 'fr-grid-row fr-grid-row--gutters' },
      el('div', { class: 'fr-col-12 fr-col-md-7' }, viewer, caption, page.images.length > 1 ? thumbs : null),
      el('div', { class: 'fr-col-12 fr-col-md-5' },
        el('p', { class: 'fr-badge fr-badge--sm fr-badge--blue-ecume fr-badge--no-icon fr-mb-1w', title: page.collection_hint, text: page.collection_label }),
        el('h1', { class: 'fr-h2 fr-mb-1w', text: page.title }),
        page.place ? el('p', { class: 'fr-icon-map-pin-2-line fr-mb-2w miroir-muted', text: ` ${page.place}` }) : null,
        el('a', { class: 'fr-btn fr-btn--icon-right fr-icon-external-link-line', href: page.pop_url, target: '_blank', rel: 'noopener', title: 'Voir la notice complète sur POP, nouvelle fenêtre', text: 'Voir la notice complète sur POP' }),
        facts, description, more, records)),
  );
  if (page.images.length) select(0);
  else viewer.append(el('p', { class: 'miroir-card__noimage fr-text--sm', text: 'Pas d’image dans POP pour cette notice' }));
  document.title = `${page.title} · Miroir`;
}

// --- framing for the conversation view (its ask_region step) ---------------------------------

const PICK_LABELS = { box: 'Envoyer ce cadre', whole: 'Envoyer toute la photo' };
let photoLabels = null;

/** Open the photo view on `blob` so the visitor frames a part of it; resolves with the framed
 * part (a JPEG data URL), the whole photo, or null when they go back without choosing. */
async function pickRegion(blob) {
  await usePhoto(blob);
  return new Promise((resolve) => {
    state.pick = resolve;
    photoLabels = { box: $('#search-box').textContent, whole: $('#search-whole').textContent };
    $('#search-box').textContent = PICK_LABELS.box;
    $('#search-whole').textContent = PICK_LABELS.whole;
    $('#position-toggle').hidden = true;
    $('#photo-position').hidden = true;
    $('#change-photo').hidden = true;
    go('#/photo');
  });
}

function endPick(dataUrl) {
  const resolve = state.pick;
  state.pick = null;
  if (photoLabels) {
    $('#search-box').textContent = photoLabels.box;
    $('#search-whole').textContent = photoLabels.whole;
  }
  $('#change-photo').hidden = false;
  showPhotoPosition();
  if (resolve) resolve(dataUrl);
  if (dataUrl !== null && location.hash === '#/photo') go('#/conversation');
}

/** The usual search, from the conversation view (when its spending cap is reached). */
async function normalSearch(blob, text) {
  if (blob) {
    await usePhoto(blob);
    $('#refine-input').value = text || '';
    searchPhoto(true);
  } else if (text) {
    state.photo = null; state.box = null; state.queryUrl = null;
    $('#refine-input').value = text;
    runSearch({ text });
  }
}

const chat = setupChat({ go, pickRegion, normalSearch, warm });

window.addEventListener('hashchange', () => {
  if (!location.hash.startsWith('#/fiche/')) document.title = 'Miroir · retrouver une œuvre ou un monument à partir d’une photo';
  route();
});
route();
