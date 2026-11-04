// What the visitor asks: a photo (or part of it), words, filters, the level.

import { NO_FILTERS, type Filters } from "./api";
import { collectionLabel, type Level } from "./content";
import type { Part, Photo } from "./photo";

export type Query = {
  level: Level;
  text: string;
  photo: Photo | null;
  crop: Part | null;
  filters: Filters;
  /** "Près de moi": the browser's position, asked on the tap. */
  nearMe: boolean;
  /** Use the position written in the photo, when it has one. */
  photoPosition: boolean;
};

export const EMPTY_QUERY: Query = {
  level: 0,
  text: "",
  photo: null,
  crop: null,
  filters: NO_FILTERS,
  nearMe: false,
  photoPosition: true,
};

export const canSearch = (q: Query) => Boolean(q.photo || q.text.trim());

export type Said = { label: string; remove?: (q: Query) => Query };

/** What the visitor gave, as short labels. Filters come with a way to remove them. */
export function describe(q: Query): Said[] {
  const parts: Said[] = [];
  if (q.photo) parts.push({ label: q.crop ? (q.crop.label ? `Partie : ${q.crop.label}` : "Partie tracée") : "Toute la photo" });
  if (q.text.trim()) parts.push({ label: `« ${q.text.trim()} »` });
  const f = q.filters;
  const without = (patch: Partial<Filters>) => (x: Query) => ({ ...x, filters: { ...x.filters, ...patch } });
  for (const c of f.collections) {
    parts.push({
      label: collectionLabel(c),
      remove: (x) => ({ ...x, filters: { ...x.filters, collections: x.filters.collections.filter((y) => y !== c) } }),
    });
  }
  if (f.region) parts.push({ label: f.region, remove: without({ region: "", departement: "" }) });
  if (f.departement) parts.push({ label: f.departement, remove: without({ departement: "" }) });
  if (f.domain) parts.push({ label: f.domain, remove: without({ domain: "" }) });
  if (f.period) parts.push({ label: f.period, remove: without({ period: "" }) });
  if (q.nearMe) parts.push({ label: "Près de moi", remove: (x) => ({ ...x, nearMe: false }) });
  return parts;
}
