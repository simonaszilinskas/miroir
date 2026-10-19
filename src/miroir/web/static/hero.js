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

// Where each photo sits on wide screens, and how near it looks: near photos are larger,
// sharper and opaque; far ones are small, faint and slightly blurred.
const SLOTS = [
  { x: '3%', y: '7%', near: 1 },
  { x: '14%', y: '30%', near: 0.35 },
  { x: '7%', y: '47%', near: 0.7 },
  { x: '14%', y: '73%', near: 0.45 },
  { x: '2%', y: '76%', near: 0.85 },
  { x: '81%', y: '4%', near: 0.75 },
  { x: '79%', y: '33%', near: 0.3 },
  { x: '86%', y: '36%', near: 1 },
  { x: '80%', y: '66%', near: 0.5 },
  { x: '88%', y: '75%', near: 0.8 },
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
    `--x:${slot.x}`, `--y:${slot.y}`,
    `--s:${(4.5 + near * 6).toFixed(2)}rem`,
    `--o:${(0.25 + near * 0.75).toFixed(2)}`,
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
