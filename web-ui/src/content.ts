// Fixed words and pictures of the page.

/** POP bases, as `COLLECTION_LABELS` and `COLLECTION_HINTS` in `miroir.web.catalog` name them. */
export const COLLECTIONS: Record<string, { label: string; hint: string }> = {
  merimee: { label: "Mérimée", hint: "Patrimoine architectural" },
  palissy: { label: "Palissy", hint: "Patrimoine mobilier" },
  memoire: { label: "Mémoire", hint: "Photographies du patrimoine" },
  joconde: { label: "Joconde", hint: "Collections des musées de France" },
  "rose-valland": { label: "Rose Valland", hint: "Œuvres récupérées après 1945 (MNR)" },
  museo: { label: "Muséofile", hint: "Répertoire des musées" },
};

export const collectionLabel = (value: string) => COLLECTIONS[value]?.label ?? value;

export type Level = 0 | 1;

export const LEVELS = [
  {
    name: "Rapide",
    icon: "ri-flashlight-line",
    hint: "Compare votre photo et vos mots aux images et aux notices de POP. Quelques secondes.",
  },
  {
    name: "Approfondi",
    icon: "ri-search-eye-line",
    hint: "Un modèle de langage ouvert (Mistral Large 4) regarde votre photo, cherche, lit les notices et explique ses choix. Environ une minute.",
  },
] as const;

/** What the agent is asked to do in the one-shot "Approfondi" search. */
export const DEEP_GOAL = {
  photo: "Retrouver la notice POP de ce que montre la photo.",
  words: "Retrouver la notice POP qui correspond à cette description.",
};

const POP = "https://popcorn-prd-perf-assets.s3.gra.io.cloud.ovh.net/";

/** Covers of publishable POP records (Joconde, Rose Valland) around the title. */
const HERO_PHOTOS = [
  "joconde/06570019306/0db1efab8d66fad1f5c2.jpg",
  "joconde/M0650000370/047395.jpg",
  "joconde/06190026684/088b6ae09205c8729c6d.jpg",
  "joconde/06190026362/088b6ae08f4b3aab64dd.jpg",
  "joconde/06190026488/088b6ae0753b5bb54a11.jpg",
  "joconde/06190026437/088b6ae08eccf12ddd5d.jpg",
  "joconde/06190026377/088b6ae06dbaf2966f0d.jpg",
  "joconde/02650029928/0ec91262a3a451ab7f71.jpg",
  "joconde/M0650016686/017203.jpg",
  "joconde/M0650009694/007885.jpg",
  "joconde/09640113780/0e37e32848b214bdcb19.jpg",
  "joconde/M0650014897/013017.jpg",
  "joconde/M0650010746/008936.jpg",
  "joconde/M0650010093/008284.jpg",
  "joconde/M0650009418/007609.jpg",
  "joconde/M0650012515/010705.jpg",
  "mnr/MNR00972/MNR972_CopyrightRMN-GP_Herve-Lewandowksi_face.jpg",
  "mnr/MCSR0036/mcsr36n.jpg",
].map((path) => POP + path);

/** Where each photo sits on wide screens: side, distance from the page edge (0) to the
 * content (1), height, and nearness (near photos are larger and clearer). */
const SLOTS = [
  { side: "left", x: 0.15, y: "6%", near: 1 },
  { side: "left", x: 0.85, y: "24%", near: 0.35 },
  { side: "left", x: 0.3, y: "44%", near: 0.8 },
  { side: "left", x: 0.9, y: "66%", near: 0.45 },
  { side: "left", x: 0.05, y: "80%", near: 0.7 },
  { side: "right", x: 0.45, y: "4%", near: 0.75 },
  { side: "right", x: 0.95, y: "28%", near: 0.35 },
  { side: "right", x: 0.1, y: "40%", near: 1 },
  { side: "right", x: 0.8, y: "64%", near: 0.5 },
  { side: "right", x: 0.2, y: "80%", near: 0.85 },
] as const;

/** Ten hero photos, drawn at random on each visit. */
export function heroPhotos() {
  const order = [...HERO_PHOTOS];
  for (let i = order.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return SLOTS.map((slot, i) => ({ ...slot, src: order[i] }));
}
