import { useEffect, useState, type ReactNode } from "react";
import { fr } from "@codegouvfr/react-dsfr";
import type { Thumb } from "./api";
import { counts, isSearch, type AgentState, type Item, type StepItem } from "./agent";
import { ABOUT } from "./Question";

/** Each tool's icon and its kind of work, as the timeline names it. */
const TOOLS: Record<string, { icon: string; kind: string }> = {
  first_look: { icon: "ri-flashlight-line", kind: "Recherche habituelle" },
  image_search: { icon: "ri-image-2-line", kind: "Recherche par l’image" },
  text_search: { icon: "ri-file-search-line", kind: "Recherche par les mots" },
  more_like: { icon: "ri-stack-line", kind: "Recherche par ressemblance" },
  view_results: { icon: "fr-icon-eye-line", kind: "Vérification à l’œil" },
  read_record: { icon: "ri-book-open-line", kind: "Lecture des notices" },
};
const OTHER_TOOL = { icon: "ri-sparkling-2-line", kind: "Étape" };

const THUMBS_SHOWN = 5;

function Thumbs({ thumbs, label, fresh }: { thumbs: Thumb[]; label: string; fresh?: Set<string> }) {
  const shown = thumbs.filter((t) => t.image_url).slice(0, THUMBS_SHOWN);
  if (!shown.length) return null;
  const more = thumbs.length - shown.length;
  return (
    <ul className="mr-thumbs" aria-label={label}>
      {shown.map((t, i) => (
        <li key={t.entity_id} className={fresh?.has(t.entity_id) ? "is-new" : undefined} style={{ animationDelay: `${i * 50}ms` }}>
          <img src={t.image_url!} alt="" title={t.title} decoding="async" />
        </li>
      ))}
      {more > 0 && <li className="mr-thumbs-more">+{more}</li>}
    </ul>
  );
}

function Node({ icon, state, children, tone }: { icon: string; state: "running" | "done" | "error" | "ask" | "you" | "say"; tone?: string; children: ReactNode }) {
  return (
    <li className={`mr-node is-${state}${tone ? ` mr-node--${tone}` : ""}`}>
      <span className="mr-node-dot" aria-hidden="true">
        <span className={fr.cx(icon as never, "fr-icon--sm")} />
      </span>
      <div className="mr-node-body">{children}</div>
    </li>
  );
}

function StepNode({ step, before }: { step: StepItem; before: Thumb[] | undefined }) {
  const tool = TOOLS[step.tool] ?? OTHER_TOOL;
  const state = step.running ? "running" : step.error ? "error" : "done";
  const was = new Set((before ?? []).map((t) => t.entity_id));
  const changed = step.shortlist && (step.shortlist.length !== was.size || step.shortlist.some((t) => !was.has(t.entity_id)));
  const fresh = new Set((step.shortlist ?? []).filter((t) => !was.has(t.entity_id)).map((t) => t.entity_id));
  return (
    <Node icon={tool.icon} state={state}>
      <p className="mr-node-kind">
        {tool.kind}
        {step.running && <span className="fr-sr-only"> (en cours)</span>}
      </p>
      <p className="mr-node-label">{step.label}</p>
      {step.why && <p className="mr-node-why">{step.why}</p>}
      {!step.running && step.summary && <p className="mr-node-meta">{step.summary}</p>}
      {!step.running && <Thumbs thumbs={step.thumbs} label="Ce que le modèle a vu" />}
      {!step.running && changed && step.shortlist!.length > 0 && (
        <div className="mr-node-short">
          <span className="mr-node-short-label">
            <span className={fr.cx("ri-flag-line", "fr-icon--xs")} aria-hidden="true" /> Pistes retenues · {step.shortlist!.length}
          </span>
          <Thumbs thumbs={step.shortlist!} label="Pistes retenues" fresh={before === undefined ? undefined : fresh} />
        </div>
      )}
    </Node>
  );
}

function Timeline({ items, working }: { items: Item[]; working: boolean }) {
  let before: Thumb[] | undefined;
  return (
    <ol className="mr-timeline" aria-label="Étapes de la recherche">
      {items.map((item) => {
        if (item.kind === "step") {
          const node = <StepNode key={`s${item.id}`} step={item} before={before} />;
          if (!item.running && item.shortlist) before = item.shortlist;
          return node;
        }
        if (item.kind === "say")
          return (
            <Node key={`y${item.key}`} icon="ri-sparkling-2-line" state="say">
              <p className="mr-node-say">{item.text}</p>
            </Node>
          );
        if (item.kind === "you")
          return (
            <Node key={`u${item.key}`} icon="ri-user-line" state="you">
              <p className="mr-node-kind">Vous</p>
              <p className="mr-node-answer">{item.text}</p>
            </Node>
          );
        if (item.answer === undefined) return null; // the question card stands for it
        const about = ABOUT[item.about] ?? ABOUT.other;
        return (
          <Node key={`a${item.key}`} icon={about.icon} state="ask">
            <p className="mr-node-kind">Question · {about.label}</p>
            <p className="mr-node-label">{item.text}</p>
            {item.answer !== undefined && (
              <p className="mr-node-answer">
                <span className={fr.cx("ri-user-line", "fr-icon--xs")} aria-hidden="true" /> Vous : {item.answer}
              </p>
            )}
          </Node>
        );
      })}
      {working && (
        <li className="mr-node is-thinking" aria-hidden="true">
          <span className="mr-node-dot">
            <i />
            <i />
            <i />
          </span>
        </li>
      )}
    </ol>
  );
}

/** Wide screens: the model's best-so-far beside the timeline, best first, as it changes. */
function Shortlist({ thumbs }: { thumbs: Thumb[] }) {
  return (
    <aside className="mr-run-aside" aria-label="Pistes retenues par le modèle">
      <p className="mr-run-aside-title">
        <span className={fr.cx("ri-flag-line", "fr-icon--sm")} aria-hidden="true" /> Pistes retenues
      </p>
      {thumbs.length ? (
        <ol>
          {thumbs.map((t, i) => (
            <li key={t.entity_id}>
              <span className="mr-run-aside-rank">{i + 1}</span>
              {t.image_url ? <img src={t.image_url} alt="" /> : <span className="mr-run-aside-none" />}
              <span className="mr-run-aside-name">{t.title}</span>
            </li>
          ))}
        </ol>
      ) : (
        <p className="mr-run-aside-empty">Le modèle n’a encore rien retenu.</p>
      )}
    </aside>
  );
}

/** "6 recherches, 2 questions, 41 s". */
export function runSummary(state: AgentState, seconds: number) {
  const c = counts(state.items);
  const parts = [`${c.searches} recherche${c.searches > 1 ? "s" : ""}`];
  if (c.looks) parts.push(`${c.looks} regard${c.looks > 1 ? "s" : ""} de près`);
  parts.push(`${c.questions} question${c.questions > 1 ? "s" : ""}`);
  parts.push(`${seconds} s`);
  return parts.join(", ");
}

/** What a screen reader hears as the search goes: each step once it is done, each question. */
function announcement(items: Item[]): string {
  for (let i = items.length - 1; i >= 0; i -= 1) {
    const item = items[i];
    if (item.kind === "step" && !item.running) return `${item.label}${item.summary ? ` : ${item.summary}` : ""}`;
    if (item.kind === "ask" && item.answer === undefined) return `Question du modèle : ${item.text}`;
    if (item.kind === "say") return item.text;
  }
  return "";
}

type Props = {
  state: AgentState;
  /** The visitor's photo (or framed part), shown as what the model studies. */
  photo: string | null;
  /** The question card, while the model waits. */
  question: ReactNode;
};

/** The model's research as it happens: a header with the photo and the shortlist, a timeline
 * of its steps, its questions and the visitor's answers; folded into one line once done. */
export function AgentRun({ state, photo, question }: Props) {
  const [now, setNow] = useState(Date.now());
  const [open, setOpen] = useState(false);
  const live = state.working;
  useEffect(() => {
    if (!live) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [live]);
  // A new turn opens the timeline again.
  useEffect(() => {
    if (live) setOpen(false);
  }, [live]);

  const seconds = Math.round((state.busyMs + (state.turnStart === null ? 0 : Math.max(0, now - state.turnStart))) / 1000);
  const folded = !live && !state.pending && state.found !== null;
  const c = counts(state.items);
  const searching = state.items.some((i) => i.kind === "step" && i.running && (isSearch(i.tool) || i.tool === "first_look"));

  return (
    <section className={`mr-run${live ? " is-live" : ""}${folded ? " is-folded" : ""}`} aria-label="Recherche approfondie">
      <div className="fr-sr-only" aria-live="polite">
        {announcement(state.items)}
      </div>
      <header className="mr-run-head">
        {photo ? (
          <div className={`mr-run-photo${searching ? " is-scanning" : ""}`}>
            <img src={photo} alt="" />
          </div>
        ) : (
          <div className="mr-run-photo mr-run-photo--words" aria-hidden="true">
            <span className={fr.cx("ri-chat-quote-line", "fr-icon--lg")} />
          </div>
        )}
        <div className="mr-run-meta">
          <p className="mr-run-eyebrow">
            <span className={fr.cx("ri-search-eye-line", "fr-icon--sm")} aria-hidden="true" /> Recherche approfondie
          </p>
          <p className="mr-run-status">
            {state.pending
              ? "Le modèle a une question pour vous"
              : live
                ? state.status || "Le modèle commence…"
                : state.error && !state.found
                  ? "La recherche s’est arrêtée"
                  : "Recherche terminée"}
          </p>
          <ul className="mr-run-stats" aria-label="Où en est la recherche">
            <li>
              <span className={fr.cx("ri-search-2-line", "fr-icon--xs")} aria-hidden="true" /> {c.searches} recherche{c.searches > 1 ? "s" : ""}
            </li>
            <li>
              <span className={fr.cx("fr-icon-eye-line", "fr-icon--xs")} aria-hidden="true" /> {c.looks} regard{c.looks > 1 ? "s" : ""}
            </li>
            <li>
              <span className={fr.cx("fr-icon-question-line", "fr-icon--xs")} aria-hidden="true" /> {c.questions} question{c.questions > 1 ? "s" : ""}
            </li>
            <li>
              <span className={fr.cx("ri-time-line", "fr-icon--xs")} aria-hidden="true" /> {seconds} s
            </li>
          </ul>
        </div>
        {state.shortlist.length > 0 && (
          <div className="mr-run-short">
            <p className="mr-run-short-label">Pistes</p>
            <ul aria-label="Pistes retenues par le modèle">
              {state.shortlist.slice(0, 4).map((t, i) => (
                <li key={t.entity_id} style={{ zIndex: 10 - i }}>
                  {t.image_url ? <img src={t.image_url} alt={t.title} title={t.title} /> : <span title={t.title}>?</span>}
                </li>
              ))}
            </ul>
          </div>
        )}
      </header>

      {folded ? (
        <div className="mr-run-fold">
          <button type="button" className="mr-run-toggle" aria-expanded={open} onClick={() => setOpen((o) => !o)}>
            <span className={fr.cx("fr-icon-success-line", "fr-icon--sm")} aria-hidden="true" />
            <span className="mr-run-toggle-text">
              <b>Comment le modèle a cherché</b>
              <span>{runSummary(state, seconds)}</span>
            </span>
            <span className={`${fr.cx(open ? "ri-arrow-up-s-line" : "ri-arrow-down-s-line")} mr-run-chevron`} aria-hidden="true" />
          </button>
          {open && <Timeline items={state.items} working={false} />}
        </div>
      ) : (
        <div className="mr-run-body">
          <div className="mr-run-main">
            <Timeline items={state.items} working={live && !state.pending} />
            {question}
          </div>
          <Shortlist thumbs={state.shortlist} />
        </div>
      )}
    </section>
  );
}
