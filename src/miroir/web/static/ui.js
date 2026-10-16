// Shared by the views of app.js and the conversation view (chat.js): DOM helpers, the API
// call, and the result card.

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

export const COLLECTION_NAMES = {
  joconde: 'Joconde (musées)',
  memoire: 'Mémoire (photographies)',
  'rose-valland': 'Rose Valland (MNR)',
};

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

/** Like `node.replaceChildren`, but skips null and false (which it would print as text). */
export function fill(node, ...children) {
  node.replaceChildren(...children.flat().filter((c) => c !== null && c !== undefined && c !== false));
}

export function img(src, fallback, attrs = {}) {
  const node = el('img', { src, alt: '', loading: 'lazy', decoding: 'async', ...attrs });
  if (fallback && fallback !== src) {
    node.addEventListener('error', () => { if (node.src !== fallback) node.src = fallback; }, { once: true });
  }
  return node;
}

export class ApiError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}

export async function api(path, { method = 'GET', body } = {}) {
  const response = await fetch(`/api/${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
    credentials: 'same-origin',
  });
  if (response.status === 401 && path !== 'login') {
    location.hash = '#/connexion';
    throw new ApiError(401, 'Connectez-vous pour continuer.');
  }
  if (!response.ok) {
    let detail = `Erreur ${response.status}`;
    try { detail = (await response.json()).detail || detail; } catch { /* not JSON */ }
    throw new ApiError(response.status, typeof detail === 'string' ? detail : 'Requête refusée.');
  }
  return response.json();
}

const LEVEL_TITLES = {
  'tres-proche': 'Très proche : sans doute la bonne notice',
  proche: 'Proche : peut-être la bonne notice, à vérifier',
  ressemblance: 'Ressemblance : une image qui ressemble, à comparer avec votre photo',
};
const LEVEL_CLASSES = {
  'tres-proche': 'fr-badge--success',
  proche: 'fr-badge--info',
  ressemblance: '',
};

export function levelBadge(level, label) {
  return el('span', { class: `fr-badge fr-badge--sm fr-badge--no-icon miroir-level ${LEVEL_CLASSES[level] ?? ''}`, title: LEVEL_TITLES[level] || '', text: label });
}

export function grid(cards) {
  return el('ul', { class: 'miroir-grid' }, cards.map((c) => el('li', {}, card(c))));
}

/** One result card. `c.level` may be missing (the conversation's picks); `c.reason` is the
 * model's reason for a pick. */
export function card(c) {
  const href = `#/fiche/${encodeURIComponent(c.entity_id)}`;
  return el('article', { class: 'miroir-card' },
    el('div', { class: 'miroir-card__media' },
      c.image_url
        ? img(c.image_url, c.image_fallback_url, { alt: '' })
        : el('p', { class: 'miroir-card__noimage fr-text--sm', text: 'Pas d’image dans POP' }),
      c.matched ? el('p', { class: 'fr-badge fr-badge--sm fr-badge--info fr-badge--no-icon miroir-card__matched', title: 'L’image qui ressemble le plus n’est pas l’image principale de la notice', text: `Correspond : ${c.matched.toLowerCase()}` }) : null,
      c.photo_count > 1 ? el('span', { class: 'miroir-card__count', text: `${c.photo_count} photos` }) : null),
    el('div', { class: 'miroir-card__body' },
      c.level ? levelBadge(c.level, c.level_label) : null,
      el('h3', { class: 'miroir-card__title' }, el('a', { href, text: c.title })),
      c.place ? el('p', { class: 'miroir-card__place', text: c.place }) : null,
      c.reason ? el('p', { class: 'miroir-card__reason fr-text--sm', text: c.reason }) : null,
      el('div', { class: 'miroir-card__foot' },
        el('p', { class: 'fr-badge fr-badge--sm fr-badge--no-icon fr-mb-0', text: COLLECTION_NAMES[c.collection]?.split(' (')[0] || c.collection_label }),
        el('a', { class: 'fr-link fr-link--sm', href: c.pop_url, target: '_blank', rel: 'noopener', title: `${c.title} sur POP, nouvelle fenêtre`, text: 'Voir sur POP' }))));
}
