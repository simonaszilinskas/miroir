// Photos that float around the home page title: cover images of publishable POP records
// (paintings, sculptures, ceramics, objects), drawn at random on each visit.

const BASE = 'https://popcorn-prd-perf-assets.s3.gra.io.cloud.ovh.net/';

const PHOTOS = [
  'joconde/06570019306/0db1efab8d66fad1f5c2.jpg', // 06570019306
  'joconde/M0650000370/047395.jpg', // M0650000370
  'joconde/06190026684/088b6ae09205c8729c6d.jpg', // 06190026684
  'joconde/06190026362/088b6ae08f4b3aab64dd.jpg', // 06190026362
  'joconde/06190026488/088b6ae0753b5bb54a11.jpg', // 06190026488
  'joconde/06190026437/088b6ae08eccf12ddd5d.jpg', // 06190026437
  'joconde/06190026377/088b6ae06dbaf2966f0d.jpg', // 06190026377
  'joconde/02650029928/0ec91262a3a451ab7f71.jpg', // 02650029928
  'joconde/M0650016686/017203.jpg', // M0650016686
  'joconde/M0650009694/007885.jpg', // M0650009694
  'joconde/09640113780/0e37e32848b214bdcb19.jpg', // 09640113780
  'joconde/M0650014897/013017.jpg', // M0650014897
  'joconde/M0650010746/008936.jpg', // M0650010746
  'joconde/M0650010093/008284.jpg', // M0650010093
  'joconde/M0650009418/007609.jpg', // M0650009418
  'joconde/M0650012515/010705.jpg', // M0650012515
  'mnr/MNR00972/MNR972_CopyrightRMN-GP_Herve-Lewandowksi_face.jpg', // MNR00972
  'mnr/MCSR0036/mcsr36n.jpg', // MCSR0036
].map((path) => BASE + path);

// Where each photo sits on wide screens: which side, how far from the page edge (0) towards
// the content (1), how far down, and how near it looks. Near photos are larger and clearer;
// all stay faint and keep clear of the content (app.css).
const SLOTS = [
  { side: 'left', x: 0.1, y: '8%', near: 1 },
  { side: 'left', x: 0.9, y: '26%', near: 0.3 },
  { side: 'left', x: 0.35, y: '45%', near: 0.7 },
  { side: 'left', x: 0.95, y: '68%', near: 0.45 },
  { side: 'left', x: 0, y: '80%', near: 0.85 },
  { side: 'right', x: 0.4, y: '5%', near: 0.75 },
  { side: 'right', x: 0.95, y: '30%', near: 0.3 },
  { side: 'right', x: 0.05, y: '40%', near: 1 },
  { side: 'right', x: 0.85, y: '64%', near: 0.5 },
  { side: 'right', x: 0.15, y: '80%', near: 0.8 },
];

const SWAP_EVERY_MS = 4000;

function shuffle(list) {
  const out = [...list];
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

function photo(src, slot, index) {
  const image = document.createElement('img');
  image.src = src;
  image.alt = '';
  image.decoding = 'async';
  const { near } = slot;
  image.style.cssText = [
    `${slot.side}:calc((50% - 27rem - var(--s)) * ${slot.x})`, `top:${slot.y}`,
    `--s:${(3.5 + near * 3).toFixed(2)}rem`,
    `--o:${(0.15 + near * 0.35).toFixed(2)}`,
    `--b:${((1 - near) * 2.5).toFixed(1)}px`,
    `--r:${(index % 2 ? 1 : -1) * (3 + ((index * 7) % 6))}deg`,
    `--t:${(7 + (1 - near) * 8 + (index % 3)).toFixed(1)}s`,
    `--d:${-index * 1.7}s`,
    `z-index:${Math.round(near * 10)}`,
  ].join(';');
  return image;
}

/** Fill the box with photos, then now and then replace one with a photo not on screen. */
export function floatPhotos(box) {
  const order = shuffle(PHOTOS);
  const waiting = order.slice(SLOTS.length);
  const images = order.slice(0, SLOTS.length).map((src, i) => photo(src, SLOTS[i], i));
  box.replaceChildren(...images);
  if (!waiting.length || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  let turn = 0;
  setInterval(() => {
    if (document.hidden || box.closest('[hidden]')) return;
    const image = images[turn % images.length];
    turn += 3;
    image.classList.add('is-leaving');
    setTimeout(() => {
      waiting.push(image.src);
      image.addEventListener('load', () => image.classList.remove('is-leaving'), { once: true });
      image.src = waiting.shift();
    }, 900);
  }, SWAP_EVERY_MS);
}
