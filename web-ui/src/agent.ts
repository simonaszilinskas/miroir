// The agentic search ("Approfondi") as the page sees it: a conversation with the model,
// read as a stream of events (`miroir.web.chat`), kept as a timeline the page draws.

import { useCallback, useEffect, useReducer, useRef } from "react";
import { api, ApiError, chatEvents, type Card, type ChatEvent, type Position, type Thumb } from "./api";

/** One thing the model did: a search, a look, a reading of records. */
export type StepItem = {
  kind: "step";
  id: number;
  tool: string;
  label: string;
  why?: string;
  summary?: string;
  thumbs: Thumb[];
  /** The model's best-so-far just after this step. */
  shortlist?: Thumb[];
  running: boolean;
  error: boolean;
};
/** What the model said to the visitor along the way. */
export type SayItem = { kind: "say"; key: number; text: string };
/** A question (or a request to frame), and the visitor's answer once given. */
export type AskItem = { kind: "ask"; key: number; text: string; about: string; answer?: string };
/** What the visitor wrote of their own accord, after results. */
export type YouItem = { kind: "you"; key: number; text: string };
export type Item = StepItem | SayItem | AskItem | YouItem;

export type Pending =
  | { kind: "question"; text: string; choices: string[]; about: string; position: boolean; region: boolean }
  | { kind: "region"; text: string };

export type Found = { cards: Card[]; found: string | null; summary: string };

export type AgentState = {
  items: Item[];
  /** What the model is doing now, in a few words. */
  status: string;
  /** A turn runs: the model works. */
  working: boolean;
  /** What the model waits for from the visitor. */
  pending: Pending | null;
  found: Found | null;
  error: string | null;
  /** The best-so-far, as the last step left it. */
  shortlist: Thumb[];
  /** Milliseconds the model has worked, waits for the visitor left out. */
  busyMs: number;
  turnStart: number | null;
};

const SEARCH_TOOLS = new Set(["image_search", "text_search", "more_like"]);
export const isSearch = (tool: string) => SEARCH_TOOLS.has(tool);

/** Searches run, looks at images, and questions asked, for the one-line summary. */
export function counts(items: Item[]) {
  let searches = 0;
  let looks = 0;
  let questions = 0;
  for (const item of items) {
    if (item.kind === "ask") questions += 1;
    else if (item.kind === "step" && !item.error && !item.running) {
      if (isSearch(item.tool) || item.tool === "first_look") searches += 1;
      else if (item.tool === "view_results") looks += 1;
    }
  }
  return { searches, looks, questions };
}

type Action =
  | { type: "event"; event: ChatEvent; now: number }
  | { type: "answered"; text: string; now: number }
  | { type: "failed"; text: string; now: number };

const INITIAL: AgentState = {
  items: [],
  status: "",
  working: true,
  pending: null,
  found: null,
  error: null,
  shortlist: [],
  busyMs: 0,
  turnStart: null,
};

let keys = 0;

function stop(state: AgentState, now: number): AgentState {
  const busy = state.turnStart === null ? 0 : now - state.turnStart;
  return { ...state, working: false, status: "", busyMs: state.busyMs + busy, turnStart: null };
}

function reduce(state: AgentState, action: Action): AgentState {
  if (action.type === "failed") return { ...stop(state, action.now), error: action.text };
  if (action.type === "answered") {
    const last = state.items[state.items.length - 1];
    const items: Item[] =
      last?.kind === "ask" && last.answer === undefined
        ? [...state.items.slice(0, -1), { ...last, answer: action.text }]
        : [...state.items, { kind: "you", key: (keys += 1), text: action.text }];
    return { ...state, items, pending: null, working: true, error: null, turnStart: action.now };
  }
  const event = action.event;
  const started = state.turnStart ?? action.now;
  switch (event.type) {
    case "status":
      return { ...state, status: event.text, turnStart: started };
    case "step": {
      const step: StepItem = {
        kind: "step",
        id: event.id,
        tool: event.tool ?? "",
        label: event.label,
        why: event.why,
        summary: event.summary,
        thumbs: event.thumbs ?? [],
        shortlist: event.shortlist,
        running: event.state === "running",
        error: Boolean(event.error),
      };
      const at = state.items.findIndex((i) => i.kind === "step" && i.id === event.id);
      const items =
        at < 0
          ? [...state.items, step]
          : state.items.map((item, i) => (i === at && item.kind === "step" ? { ...item, ...step, why: step.why ?? item.why } : item));
      return { ...state, items, shortlist: event.shortlist ?? state.shortlist, status: step.running ? step.label : state.status, turnStart: started };
    }
    case "say":
      return { ...state, items: [...state.items, { kind: "say", key: (keys += 1), text: event.text }], turnStart: started };
    case "question":
      return {
        ...state,
        items: [...state.items, { kind: "ask", key: (keys += 1), text: event.text, about: event.about ?? "other" }],
        pending: { kind: "question", text: event.text, choices: event.choices, about: event.about ?? "other", position: Boolean(event.position), region: Boolean(event.region) },
      };
    case "region":
      return {
        ...state,
        items: [...state.items, { kind: "ask", key: (keys += 1), text: event.text || "Encadrez ce qui vous intéresse dans la photo.", about: "work" }],
        pending: { kind: "region", text: event.text },
      };
    case "results": {
      // The summary comes as the model's last words, just before the cards: it heads them.
      const last = state.items[state.items.length - 1];
      const summary = last?.kind === "say" ? last.text : "";
      const items = summary ? state.items.slice(0, -1) : state.items;
      return { ...state, items, found: { cards: event.cards, found: event.found, summary } };
    }
    case "limit":
    case "error":
      return { ...state, error: event.text };
    case "end":
      return stop(state, action.now);
    default:
      return state;
  }
}

export type Reply =
  | { kind: "answer"; text: string; position?: Position }
  | { kind: "region"; region: string | null }
  | { kind: "message"; text: string };

/** The conversation for one search: opened on mount, read until each turn ends. `answer`
 * sends the visitor's reply and reads the next turn. */
export function useAgent(open: () => Promise<{ body: Record<string, unknown>; goal: string }>) {
  const [state, dispatch] = useReducer(reduce, INITIAL);
  const id = useRef<string | null>(null);
  const alive = useRef(true);

  const read = useCallback(async (from: number) => {
    const conversation = id.current!;
    for await (const event of chatEvents(conversation, from, () => alive.current)) {
      dispatch({ type: "event", event, now: Date.now() });
    }
  }, []);

  const fail = useCallback((err: unknown) => {
    if (!alive.current || (err instanceof ApiError && err.status === 401)) return;
    dispatch({ type: "failed", text: err instanceof Error ? err.message : "La recherche n’a pas abouti.", now: Date.now() });
  }, []);

  useEffect(() => {
    alive.current = true;
    (async () => {
      const { body, goal } = await open();
      const opened = await api<{ conversation_id: string }>("chat/open", { method: "POST", body });
      id.current = opened.conversation_id;
      const sent = await api<{ after: number }>("chat/send", { method: "POST", body: { conversation_id: id.current, kind: "goal", text: goal } });
      await read(sent.after);
    })().catch(fail);
    return () => {
      alive.current = false;
    };
    // One conversation per mount: a new search mounts a new Results.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const answer = useCallback(
    async (reply: Reply, said: string) => {
      if (!id.current) return;
      dispatch({ type: "answered", text: said, now: Date.now() });
      try {
        const sent = await api<{ after: number }>("chat/send", { method: "POST", body: { conversation_id: id.current, ...reply } });
        await read(sent.after);
      } catch (err) {
        fail(err);
      }
    },
    [read, fail],
  );

  return { state, answer };
}
