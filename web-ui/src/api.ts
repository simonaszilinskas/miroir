// What the server sends and takes (`miroir.web.app`), and how to ask it.

export type Position = { latitude: number; longitude: number };

export type Filters = {
  collections: string[];
  region: string;
  departement: string;
  domain: string;
  period: string;
};

const POP_NOTICES = "https://pop.culture.gouv.fr/";

/** A link to a POP notice as the server gives it; nothing when it points anywhere else. */
export const popLink = (url: string | null | undefined): string | undefined =>
  url && url.startsWith(POP_NOTICES) ? url : undefined;

export const NO_FILTERS: Filters = { collections: [], region: "", departement: "", domain: "", period: "" };

export const filterCount = (f: Filters) =>
  f.collections.length + (["region", "departement", "domain", "period"] as const).filter((k) => f[k]).length;

export type FacetValue = { value: string; count: number };
export type Facets = {
  collections: FacetValue[];
  regions: FacetValue[];
  departements: Record<string, FacetValue[]>;
  domains: FacetValue[];
  periods: FacetValue[];
};

export type Card = {
  entity_id: string;
  title: string;
  place: string;
  collection: string;
  collection_label: string;
  pop_url: string;
  image_url: string | null;
  image_fallback_url: string | null;
  matched: string | null;
  photo_count: number;
  level?: string | null;
  level_label?: string | null;
  /** The agent's reason for a pick ("Approfondi"). */
  reason?: string;
};

export type SearchResponse = {
  mode: string;
  guessed_type: string | null;
  cards: Card[];
  nearby: number | null;
  words_found: boolean | null;
  mount: boolean;
  picture: [number, number, number, number] | null;
  caption: string | null;
  caption_read: boolean;
  caption_place: string | null;
};

export type Box = { x0: number; y0: number; x1: number; y1: number; score: number; label: string };

export type Example = { id: string; label: string; kind: string; image_url: string; credit: string; frame: boolean };

export type Stats = { images: number; records: number; entities: number; notice: string | null };

export type EntityPage = {
  entity_id: string;
  title: string;
  place: string;
  collection: string;
  collection_label: string;
  collection_hint: string;
  pop_url: string;
  description: string;
  facts: { label: string; value: string }[];
  images: { url: string; thumb_url: string; fallback_url: string; label: string | null; ref: string | null }[];
  records: { ref: string; title: string; pop_url: string }[];
  photo_count: number;
};

/** One thing the agent did or said (`miroir.web.chat`, `/api/chat/events`), with its number
 * in the conversation (`n`, on what the turn wrote). */
export type ChatEvent = (
  | { type: "status"; text: string }
  | { type: "step"; id: number; state?: string; label: string; summary?: string; why?: string; error?: boolean; thumbs?: { entity_id: string; title: string; image_url: string }[] }
  | { type: "say"; text: string }
  | { type: "question"; text: string; choices: string[] }
  | { type: "region"; text: string }
  | { type: "results"; cards: Card[]; found: string | null }
  | { type: "limit"; text: string }
  | { type: "error"; text: string }
  | { type: "end"; waiting: string }
  | { type: "continue"; after: number }
  | { type: "replaced" }
  | { type: "ping" }
  | { type: "visitor" }
) & { n?: number };

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/** Called on a 401: the page shows the login view. */
let onLoggedOut = () => {};
export function whenLoggedOut(fn: () => void) {
  onLoggedOut = fn;
}

async function failure(response: Response): Promise<ApiError> {
  let detail = `Erreur ${response.status}`;
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    /* not JSON */
  }
  return new ApiError(response.status, detail);
}

export async function api<T>(path: string, { method = "GET", body }: { method?: string; body?: unknown } = {}): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    method,
    headers: body === undefined ? {} : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    credentials: "same-origin",
  });
  if (response.status === 401 && path !== "login") {
    onLoggedOut();
    throw new ApiError(401, "Connectez-vous pour continuer.");
  }
  if (!response.ok) throw await failure(response);
  return (await response.json()) as T;
}

const RETRIES = 4;
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Like `api`, but tries again while the search engine starts (503 from the server). */
export async function apiPatient<T>(path: string, options: { method?: string; body?: unknown }, alive: () => boolean): Promise<T> {
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await api<T>(path, options);
    } catch (err) {
      if (!(err instanceof ApiError) || err.status !== 503 || attempt >= RETRIES || !alive()) throw err;
      await sleep(2000);
    }
  }
}

/** Tries to read the stream again after a dropped connection, and the longest wait between. */
const RECONNECTS = 5;
const MAX_WAIT_MS = 8000;

/** The agent's events from number `after` on, until its turn ends. The server closes each
 * stream after a while and says where to pick up; a stream that drops before the turn's end
 * is read again from the event after the last one seen, after a growing wait. A stream that
 * another one (another tab) replaced stops. */
export async function* chatEvents(conversationId: string, after: number, alive: () => boolean): AsyncGenerator<ChatEvent> {
  let from = after;
  let drops = 0;
  while (alive()) {
    let resume = false;
    try {
      const response = await fetch("/api/chat/events", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_id: conversationId, after: from }),
        credentials: "same-origin",
      });
      if (response.status === 401) {
        onLoggedOut();
        return;
      }
      if (!response.ok || !response.body) throw await failure(response);
      for await (const event of lines(response.body)) {
        if (!alive() || event.type === "replaced") return;
        if (event.type === "continue") {
          from = event.after;
          resume = true;
          continue;
        }
        if (event.n !== undefined) from = event.n + 1;
        drops = 0;
        yield event;
        if (event.type === "end") return;
      }
    } catch (err) {
      if (err instanceof ApiError) throw err; // the server answered: no use asking again
    }
    if (resume) continue;
    drops += 1;
    if (drops > RECONNECTS) throw new Error("La connexion au serveur a été perdue. Relancez la recherche.");
    await sleep(Math.min(500 * 2 ** drops, MAX_WAIT_MS));
  }
}

async function* lines(body: ReadableStream<Uint8Array>): AsyncGenerator<ChatEvent> {
  const reader = body.pipeThrough(new TextDecoderStream() as unknown as TransformStream<Uint8Array, string>).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end;
    while ((end = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, end).trim();
      buffer = buffer.slice(end + 1);
      if (line) yield JSON.parse(line) as ChatEvent;
    }
  }
  if (buffer.trim()) yield JSON.parse(buffer) as ChatEvent;
}
