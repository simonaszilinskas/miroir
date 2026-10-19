// The conversation view (step H5, /chat): an open vision model leads the search. The visitor
// gives a photo or words, says what they want to find, then answers the model's questions,
// frames a part of the photo when asked, and reads its picks. Each tool call shows as a
// compact step that opens to its thumbnails. Talks to /api/chat/*.

import { exifPosition } from './exif.js';
import { $, api, ApiError, el, fill, grid, img } from './ui.js';

/** Suggestions above the goal box. Change them here: a tap puts `text` in the box. */
export const GOAL_CHIPS = [
  { label: 'C’est quoi ?', text: 'Je voudrais savoir ce que c’est : quel objet ou quel monument précis.' },
  { label: 'C’est où ?', text: 'Je voudrais savoir où c’est, ou bien où cet objet est conservé.' },
  { label: 'D’autres comme ça', text: 'Je cherche d’autres œuvres ou objets qui ressemblent à celui-ci.' },
  { label: 'Qui l’a fait ?', text: 'Je voudrais savoir qui l’a fait, et à quelle époque.' },
];

/** The line above the picks, by what the model says it found. */
const FOUND_HEADS = {
  exact: 'Ce que le modèle a trouvé :',
  close: 'Les plus proches dans les collections :',
  none: 'Rien ne correspond vraiment. Les moins éloignés :',
};

/** Longest side, in pixels, of the photo sent (the server shrinks it again). */
const SEND_SIDE = 1024;
/** Longest side kept in the browser, to frame a part of the photo. */
const KEEP_SIDE = 2048;
/** Thumbnails shown in a step. */
const STEP_THUMBS = 8;

const chat = {
  id: null,          // the conversation, once opened
  photo: null,       // the photo as a Blob (KEEP_SIDE), or null for words only
  photoUrl: null,
  words: '',
  position: null,    // where the photo was taken (its EXIF GPS), sent when opening
  busy: false,
  waiting: null,     // what the model waits for: 'answer', 'region', 'message', 'limit'
  steps: new Map(),  // step id -> its <li>
  checked: false,
  enabled: true,
  starting: false,   // true while the home page opens a conversation
};

let hooks = null;

/** Wire the view once. `hooks`: go, pickRegion(blob), normalSearch(blob, text), warm. */
export function setupChat(given) {
  hooks = given;
  fill($('#chat-goal-chips'), GOAL_CHIPS.map((chip) => el('li', {},
    el('button', {
      class: 'fr-tag', type: 'button', text: chip.label,
      onclick: () => {
        const box = $('#chat-goal-input');
        box.value = chip.text;
        box.focus();
        box.setSelectionRange(box.value.length, box.value.length);
      },
    }))));
  for (const input of [$('#chat-camera'), $('#chat-upload')]) {
    input.addEventListener('change', async () => {
      const file = input.files && input.files[0];
      input.value = '';
      if (file) await choosePhoto(file);
    });
  }
  $('#chat-words-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const words = $('#chat-words').value.trim();
    if (!words) { $('#chat-words').focus(); return; }
    await open(null, words);
  });
  $('#chat-goal').addEventListener('submit', (event) => {
    event.preventDefault();
    const goal = $('#chat-goal-input').value.trim();
    if (!goal) { $('#chat-goal-input').focus(); return; }
    $('#chat-goal').hidden = true;
    if (!chat.photo && chat.words) say('visitor', `« ${chat.words} »`);
    say('visitor', goal, chat.photoUrl);
    send('goal', { text: goal });
  });
  $('#chat-composer').addEventListener('submit', (event) => {
    event.preventDefault();
    const text = $('#chat-message').value.trim();
    if (!text || chat.busy) return;
    $('#chat-message').value = '';
    say('visitor', text);
    send('message', { text });
  });
  $('#chat-restart').addEventListener('click', restart);
  return { enter, startPhoto, startWords };
}

/** Start a new conversation from the home page, with a photo (a File or Blob) or words. */
async function startPhoto(blob) {
  if (!begin()) return;
  await choosePhoto(blob);
  chat.starting = false;
}

async function startWords(words) {
  if (!begin()) return;
  await open(null, words);
  chat.starting = false;
}

/** Leave any finished conversation and show the view; false while a turn is running. */
function begin() {
  if (chat.busy) { hooks.go('#/conversation'); return false; }
  restart();
  chat.starting = true;
  $('#chat-start').hidden = true;
  hooks.go('#/conversation');
  return true;
}

async function enter() {
  hooks.warm();
  if (chat.checked) return;
  chat.checked = true;
  try {
    chat.enabled = (await api('chat/status')).enabled;
  } catch { chat.enabled = false; }
  $('#chat-off').hidden = chat.enabled;
  $('#chat-start').hidden = !chat.enabled || Boolean(chat.id) || chat.starting;
}

function restart() {
  if (chat.busy) return;
  if (chat.photoUrl) URL.revokeObjectURL(chat.photoUrl);
  Object.assign(chat, { id: null, photo: null, photoUrl: null, words: '', position: null, waiting: null });
  chat.steps.clear();
  $('#chat-thread').replaceChildren();
  $('#chat-goal').hidden = true;
  $('#chat-composer').hidden = true;
  $('#chat-goal-input').value = '';
  $('#chat-words').value = '';
  $('#chat-start').hidden = !chat.enabled;
  $('#chat-restart').hidden = true;
  setStatus('');
}

// --- opening: photo or words, then the goal --------------------------------------------------

async function choosePhoto(file) {
  let blob;
  try {
    blob = await shrink(file, KEEP_SIDE);
  } catch {
    $('#chat-start').hidden = !chat.enabled;
    alertIn($('#chat-start'), 'Cette image n’a pas pu être ouverte. Essayez une photo JPEG ou PNG.');
    return;
  }
  chat.position = await exifPosition(file);
  chat.photo = blob;
  chat.photoUrl = URL.createObjectURL(blob);
  await open(blob, $('#chat-words').value.trim());
}

/** Open the conversation: with a photo, the usual search starts now, while the visitor
 * writes what they are looking for. */
async function open(blob, words) {
  chat.words = words;
  const body = { text: words };
  if (blob) body.image = await dataUrl(await shrink(blob, SEND_SIDE));
  if (chat.position) body.position = chat.position;
  try {
    chat.id = (await api('chat/open', { method: 'POST', body })).conversation_id;
  } catch (err) {
    $('#chat-start').hidden = !chat.enabled;
    alertIn($('#chat-start'), err.message);
    return;
  }
  $('#chat-start').hidden = true;
  $('#chat-restart').hidden = false;
  $('#chat-goal-photo').replaceChildren(...(chat.photoUrl
    ? [img(chat.photoUrl, null, { alt: 'Votre photo', class: 'miroir-chat__goal-photo' })]
    : [el('p', { class: 'fr-text--sm fr-mb-0', text: `« ${words} »` })]));
  $('#chat-goal').hidden = false;
  $('#chat-goal-input').focus();
}

// --- talking to the server -------------------------------------------------------------------

async function send(kind, payload) {
  if (!chat.id) return;
  chat.busy = true;
  $('#chat-composer').hidden = true;
  setStatus('Envoi…');
  try {
    const { after } = await api('chat/send', { method: 'POST', body: { conversation_id: chat.id, kind, ...payload } });
    await read(after);
  } catch (err) {
    if (!(err instanceof ApiError && err.status === 401)) showError(err.message || 'La conversation s’est interrompue.');
    chat.busy = false;
    setStatus('');
  }
}

/** Read the turn's events as they come; the server ends a stream after a while and says
 * where to pick up. */
async function read(after) {
  let from = after;
  for (;;) {
    const response = await fetch('/api/chat/events', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: chat.id, after: from }),
      credentials: 'same-origin',
    });
    if (!response.ok) {
      let detail = `Erreur ${response.status}`;
      try { detail = (await response.json()).detail || detail; } catch { /* not JSON */ }
      throw new ApiError(response.status, detail);
    }
    let last = null;
    for await (const event of lines(response)) {
      last = event;
      handle(event);
    }
    if (last?.type !== 'continue') {
      if (chat.busy) { chat.busy = false; setStatus(''); }
      return;
    }
    from = last.after;
  }
}

async function* lines(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end;
    while ((end = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, end).trim();
      buffer = buffer.slice(end + 1);
      if (line) yield JSON.parse(line);
    }
  }
  if (buffer.trim()) yield JSON.parse(buffer);
}

function handle(event) {
  switch (event.type) {
    case 'status': setStatus(event.text); break;
    case 'step': step(event); break;
    case 'say': say('model', event.text); break;
    case 'question': question(event); break;
    case 'region': region(event); break;
    case 'results': results(event.cards, event.found); break;
    case 'limit': limit(event.text); break;
    case 'error': showError(event.text); break;
    case 'end':
      chat.busy = false;
      chat.waiting = event.waiting;
      setStatus('');
      $('#chat-composer').hidden = !['message'].includes(event.waiting);
      if (event.waiting === 'message') $('#chat-message').focus({ preventScroll: true });
      break;
    default: break;
  }
}

// --- the thread ------------------------------------------------------------------------------

function add(node) {
  $('#chat-thread').append(node);
  node.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  return node;
}

function say(who, text, photoUrl = null) {
  return add(el('li', { class: `miroir-bubble miroir-bubble--${who}` },
    photoUrl ? img(photoUrl, null, { alt: 'Votre photo', class: 'miroir-bubble__photo' }) : null,
    el('p', { class: 'fr-mb-0', text })));
}

/** One tool call: a line that opens to its thumbnails, like a log of what the model did. */
function step(event) {
  let item = chat.steps.get(event.id);
  if (!item) {
    item = add(el('li', { class: 'miroir-call' }));
    chat.steps.set(event.id, item);
  }
  const running = event.state === 'running';
  item.classList.toggle('is-running', running);
  item.classList.toggle('is-error', Boolean(event.error));
  const why = event.why ? el('p', { class: 'miroir-call__why', text: `Pourquoi : ${event.why}` }) : null;
  const head = [
    el('span', { class: running ? 'miroir-spinner' : `miroir-call__icon ${event.error ? 'fr-icon-warning-line' : 'fr-icon-search-line'}`, 'aria-hidden': 'true' }),
    el('span', { class: 'miroir-call__label', text: event.label }),
    event.summary ? el('span', { class: 'miroir-call__summary', text: ` · ${event.summary}` }) : null,
  ];
  const thumbs = (event.thumbs || []).slice(0, STEP_THUMBS);
  if (running || !thumbs.length) {
    fill(item, why, el('p', { class: 'miroir-call__line' }, head));
    return;
  }
  fill(item, why, el('details', {},
    el('summary', { class: 'miroir-call__line' }, head),
    el('ul', { class: 'miroir-call__thumbs' }, thumbs.map((t) => el('li', {},
      el('a', { href: `#/fiche/${encodeURIComponent(t.entity_id)}`, title: t.title },
        t.image_url ? img(t.image_url, null, { alt: t.title }) : el('span', { class: 'miroir-call__noimage', text: t.title })))))));
}

function question(event) {
  const input = el('input', { class: 'fr-input', type: 'text', maxlength: '300', autocomplete: 'off', 'aria-label': 'Votre réponse' });
  const card = el('li', { class: 'miroir-ask' });
  const answer = (text, position = null) => {
    if (chat.busy) return;
    card.classList.add('is-answered');
    for (const control of card.querySelectorAll('button, input')) control.disabled = true;
    say('visitor', text);
    send('answer', position ? { text, position } : { text });
  };
  const form = el('form', { class: 'miroir-ask__form' },
    input, el('button', { class: 'fr-btn', type: 'submit', text: 'Répondre' }));
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    if (input.value.trim()) answer(input.value.trim());
    else input.focus();
  });
  const choices = (event.choices || []).map((c) => el('li', {},
    el('button', { class: 'fr-btn fr-btn--sm fr-btn--secondary', type: 'button', text: c, onclick: () => answer(c) })));
  if (event.position) {
    choices.push(el('li', {}, el('button', {
      class: 'fr-btn fr-btn--sm fr-btn--secondary fr-icon-map-pin-2-line fr-btn--icon-left', type: 'button', text: 'Utiliser ma position',
      onclick: async () => {
        const found = await currentPosition();
        if (found) answer('Voici ma position.', found);
        else { input.placeholder = 'Position indisponible : écrivez le lieu'; input.focus(); }
      },
    })));
  }
  choices.push(el('li', {}, el('button', { class: 'fr-btn fr-btn--sm fr-btn--tertiary', type: 'button', text: 'Je ne sais pas', onclick: () => answer('Je ne sais pas.') })));
  fill(card,
    el('p', { class: 'miroir-ask__question', text: event.text }),
    form,
    el('p', { class: 'fr-text--xs miroir-muted fr-mb-1w', text: 'Écrivez votre réponse, ou touchez une suggestion :' }),
    el('ul', { class: 'fr-btns-group fr-btns-group--inline fr-btns-group--sm fr-btns-group--icon-left miroir-ask__choices' }, choices));
  add(card);
  input.focus({ preventScroll: true });
}

function region(event) {
  const card = el('li', { class: 'miroir-ask' });
  const done = () => {
    card.classList.add('is-answered');
    for (const control of card.querySelectorAll('button')) control.disabled = true;
  };
  fill(card,
    el('p', { class: 'miroir-ask__question', text: event.text || 'Encadrez la partie de la photo qui vous intéresse.' }),
    el('ul', { class: 'fr-btns-group fr-btns-group--inline fr-btns-group--sm fr-btns-group--icon-left' },
      el('li', {}, el('button', {
        class: 'fr-btn fr-btn--sm fr-icon-crop-line fr-btn--icon-left', type: 'button', text: 'Encadrer sur la photo',
        onclick: async () => {
          const framed = await hooks.pickRegion(chat.photo);
          if (!framed) return;
          done();
          add(el('li', { class: 'miroir-bubble miroir-bubble--visitor' },
            img(framed, null, { alt: 'La partie encadrée', class: 'miroir-bubble__photo' }),
            el('p', { class: 'fr-mb-0', text: 'Cette partie-ci.' })));
          send('region', { region: framed });
        },
      })),
      el('li', {}, el('button', {
        class: 'fr-btn fr-btn--sm fr-btn--tertiary', type: 'button', text: 'Toute la photo',
        onclick: () => { done(); say('visitor', 'Toute la photo.'); send('region', { region: null }); },
      }))));
  add(card);
}

function results(cards, found) {
  if (!cards.length) return;
  const head = FOUND_HEADS[found] || (cards.length > 1 ? 'Ce que le modèle propose, du plus probable au moins probable :' : 'Ce que le modèle propose :');
  add(el('li', { class: 'miroir-picks' },
    el('p', { class: 'fr-text--sm miroir-muted fr-mb-1w', text: head }),
    grid(cards)));
}

function limit(text) {
  add(el('li', { class: 'fr-alert fr-alert--warning fr-alert--sm miroir-chat__alert' },
    el('p', { text }),
    el('button', {
      class: 'fr-btn fr-btn--sm fr-mt-1w fr-icon-search-line fr-btn--icon-left', type: 'button', text: 'Lancer la recherche habituelle',
      onclick: () => hooks.normalSearch(chat.photo, chat.words),
    })));
}

function showError(text) {
  add(el('li', { class: 'fr-alert fr-alert--error fr-alert--sm miroir-chat__alert' }, el('p', { text })));
}

function setStatus(text) {
  const node = $('#chat-status');
  node.hidden = !text;
  fill(node, text ? [el('span', { class: 'miroir-spinner', 'aria-hidden': 'true' }), el('span', { text })] : []);
}

function alertIn(where, message) {
  const box = el('div', { class: 'fr-alert fr-alert--error fr-alert--sm fr-mb-2w', role: 'alert' }, el('p', { text: message }));
  where.prepend(box);
  setTimeout(() => box.remove(), 6000);
}

// --- small helpers ---------------------------------------------------------------------------

async function shrink(blob, side) {
  const bitmap = await createImageBitmap(blob, { imageOrientation: 'from-image' });
  const scale = Math.min(1, side / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement('canvas');
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close?.();
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('encode'))), 'image/jpeg', 0.9);
  });
}

function dataUrl(blob) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.readAsDataURL(blob);
  });
}

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
