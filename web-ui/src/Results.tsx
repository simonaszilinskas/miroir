import { useEffect, useId, useState } from "react";
import { fr } from "@codegouvfr/react-dsfr";
import { Alert } from "@codegouvfr/react-dsfr/Alert";
import { Badge } from "@codegouvfr/react-dsfr/Badge";
import { Button } from "@codegouvfr/react-dsfr/Button";
import { CallOut } from "@codegouvfr/react-dsfr/CallOut";
import { Tag } from "@codegouvfr/react-dsfr/Tag";
import { ApiError, apiPatient, type Card as CardData, type Position, type SearchResponse } from "./api";
import { useAgent } from "./agent";
import { AgentRun } from "./AgentRun";
import { DEEP_GOAL, LEVELS } from "./content";
import { cut, devicePosition, SEND_SIDE } from "./photo";
import { QuestionCard } from "./Question";
import { describe, type Query } from "./query";
import { LevelLegend, ResultGrid } from "./ResultCards";

type Props = {
  query: Query;
  /** False when the server has no model for "Approfondi". */
  deepEnabled: boolean;
  onEdit: () => void;
  onDeeper: () => void;
  /** Search again with a change, e.g. a filter removed. */
  onRefine: (change: (q: Query) => Query) => void;
};

/** What the model says it found, as the head of its cards. */
const FOUND: Record<string, { head: string; badge: string; severity: "success" | "info" | "warning"; icon: string }> = {
  exact: { head: "Ce que le modèle a trouvé", badge: "Trouvé", severity: "success", icon: "fr-icon-success-line" },
  close: { head: "Les plus proches dans les collections", badge: "Pistes proches", severity: "info", icon: "ri-flag-line" },
  none: { head: "Rien ne correspond vraiment. Les moins éloignés", badge: "Pas trouvé", severity: "warning", icon: "fr-icon-question-line" },
};

const TYPE_LABELS: Record<string, string> = { building: "Bâtiment", object: "Objet" };

async function positionFor(query: Query): Promise<Position | null> {
  if (query.photo?.position && query.photoPosition) return query.photo.position;
  return query.nearMe ? devicePosition() : null;
}

/** The visitor's photo, or the part they framed, as the page shows it beside the results. */
function useShownPhoto(query: Query): string | null {
  const [url, setUrl] = useState<string | null>(query.photo && !query.crop ? query.photo.url : null);
  useEffect(() => {
    if (query.photo && query.crop) cut(query.photo, query.crop, SEND_SIDE).then(setUrl).catch(() => setUrl(query.photo!.url));
  }, [query.photo, query.crop]);
  return url;
}

function QuerySummary({ query, badge, onEdit, onRefine }: { query: Query; badge: string | null; onEdit: () => void; onRefine: Props["onRefine"] }) {
  const deep = query.level === 1;
  const crop = query.crop;
  return (
    <div className="mr-query">
      {query.photo ? (
        <div className="mr-mini" style={{ backgroundImage: `url(${query.photo.url})` }}>
          {crop && <i style={{ left: `${crop.x * 100}%`, top: `${crop.y * 100}%`, width: `${crop.w * 100}%`, height: `${crop.h * 100}%` }} />}
        </div>
      ) : (
        <div className="mr-mini mr-mini--text" aria-hidden="true">
          <span className={fr.cx("ri-chat-quote-line", "fr-icon--lg")} />
        </div>
      )}
      <div className="mr-query-meta">
        <ul className="fr-badges-group fr-mb-1w">
          <li>
            <Badge small noIcon severity={deep ? "info" : "success"}>
              {LEVELS[query.level].name}
            </Badge>
          </li>
          {badge && (
            <li>
              <Badge small noIcon>
                {badge}
              </Badge>
            </li>
          )}
        </ul>
        <ul className="fr-tags-group">
          {describe(query).map((p) => (
            <li key={p.label}>
              {p.remove ? (
                <Tag
                  small
                  dismissible
                  nativeButtonProps={{ type: "button", "aria-label": `Retirer le filtre ${p.label}`, title: `Retirer le filtre ${p.label}`, onClick: () => onRefine(p.remove!) }}
                >
                  {p.label}
                </Tag>
              ) : (
                <Tag small as="span">
                  {p.label}
                </Tag>
              )}
            </li>
          ))}
        </ul>
      </div>
      <Button priority="tertiary no outline" size="small" iconId="fr-icon-edit-line" onClick={onEdit} className="mr-edit">
        Modifier
      </Button>
    </div>
  );
}

/** "Rapide": the site's usual search. */
function Quick({ query, deepEnabled, onDeeper, photo, onEdit, onRefine }: Props & { photo: string | null }) {
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const started = Date.now();
    const timer = window.setInterval(() => setSeconds(Math.round((Date.now() - started) / 1000)), 1000);
    let alive = true;
    (async () => {
      const image = query.photo ? await cut(query.photo, query.crop, SEND_SIDE) : null;
      const body: Record<string, unknown> = { image, text: query.text.trim(), filters: query.filters };
      const position = await positionFor(query);
      if (position) body.position = position;
      const found = await apiPatient<SearchResponse>("search", { method: "POST", body }, () => alive);
      if (alive) setResponse(found);
    })()
      .catch((err: unknown) => {
        if (alive && !(err instanceof ApiError && err.status === 401)) setError(err instanceof Error ? err.message : "La recherche n’a pas abouti.");
      })
      .finally(() => window.clearInterval(timer));
    return () => {
      alive = false;
      window.clearInterval(timer);
    };
    // The query is fixed for this run: a new search mounts a new Results.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const cards: CardData[] = response?.cards ?? [];
  const badge = response?.guessed_type ? (TYPE_LABELS[response.guessed_type] ?? response.guessed_type) : null;
  return (
    <>
      <QuerySummary query={query} badge={badge} onEdit={onEdit} onRefine={onRefine} />
      {!response && !error && (
        <div className="mr-wait" role="status">
          <span className="mr-spinner" aria-hidden="true" />
          {seconds >= 4 ? `Le moteur de recherche se réveille, cela peut prendre une à deux minutes la première fois… (${seconds} s)` : "Recherche dans POP…"}
        </div>
      )}
      {error && <Alert className="fr-mb-3w" severity="error" small description={error} />}
      {response && (
        <>
          {response.caption && (
            <p className="fr-text--sm mr-caption">
              <span className={fr.cx("fr-icon-file-text-line", "fr-icon--sm", "fr-mr-1v")} aria-hidden="true" />
              Cadre et légende détectés. Légende lue : « {response.caption} »
            </p>
          )}
          {response.nearby != null && <p className="fr-text--sm mr-muted">Ce qui est près de {query.nearMe ? "vous" : "l’endroit de la photo"} passe devant.</p>}
          {response.words_found === false && (
            <Alert
              className="fr-mb-3w"
              severity="info"
              small
              description="Aucune notice ne contient ces mots. Voici les images qui ressemblent le plus à votre description. Vérifiez l’orthographe, ou décrivez autrement : la forme, la matière, le sujet, un lieu."
            />
          )}
          {cards.length > 0 ? (
            <>
              <div className="mr-results-head">
                <h2 className="fr-h5 fr-mb-0">{`${cards.length} notices, de la plus proche à la moins proche`}</h2>
                <LevelLegend />
              </div>
              <ResultGrid cards={cards} picked={false} featureFirst={false} photo={photo} />
            </>
          ) : (
            <Alert severity="info" small description="Aucune notice ne correspond avec ces filtres. Essayez d’en retirer." />
          )}
          {deepEnabled && cards.length > 0 && (
            <CallOut
              className="fr-mt-4w mr-reveal"
              iconId="ri-search-eye-line"
              title="Pas trouvé ?"
              titleAs="h2"
              buttonProps={{ children: "Aller plus loin", onClick: onDeeper, iconId: "ri-arrow-right-line", iconPosition: "right" }}
            >
              La recherche approfondie reprend votre photo, votre cadre, vos mots et vos filtres : un modèle regarde, cherche, vous pose une question si besoin, lit les notices et explique ses choix.
            </CallOut>
          )}
        </>
      )}
    </>
  );
}

/** After the results: the visitor tells the model more, and it searches again. */
function FollowUp({ onSend, disabled }: { onSend: (text: string) => void; disabled: boolean }) {
  const [text, setText] = useState("");
  const id = useId();
  return (
    <form
      className="mr-follow"
      onSubmit={(e) => {
        e.preventDefault();
        if (!text.trim()) return;
        onSend(text.trim());
        setText("");
      }}
    >
      <label className="mr-follow-label" htmlFor={id}>
        <span className={fr.cx("ri-chat-quote-line", "fr-icon--sm")} aria-hidden="true" /> Pas tout à fait ? Dites-en plus au modèle
      </label>
      <div className="mr-ask-free">
        <input id={id} className="mr-ask-input" value={text} maxLength={500} placeholder="« Pas celle-ci, l’église est plus ancienne »" onChange={(e) => setText(e.target.value)} disabled={disabled} />
        <button type="submit" className={fr.cx("fr-btn", "fr-btn--sm", "fr-btn--icon-left", "ri-send-plane-2-line")} disabled={disabled || !text.trim()}>
          Relancer
        </button>
      </div>
    </form>
  );
}

/** "Approfondi": the model leads the search, may ask the visitor, then shows its picks. */
function Deep({ query, photo, onEdit, onRefine }: Props & { photo: string | null }) {
  const { state, answer } = useAgent(async () => {
    const image = query.photo ? await cut(query.photo, query.crop, SEND_SIDE) : null;
    const position = await positionFor(query);
    return {
      body: { image, text: query.text.trim(), position, filters: query.filters },
      goal: query.photo ? DEEP_GOAL.photo : DEEP_GOAL.words,
    };
  });
  const found = state.found;
  const head = FOUND[found?.found ?? ""] ?? { head: "Ce que le modèle propose", badge: "Pistes", severity: "info" as const, icon: "ri-flag-line" };
  const done = !state.working && !state.pending;

  return (
    <>
      <QuerySummary query={query} badge={null} onEdit={onEdit} onRefine={onRefine} />
      <AgentRun
        state={state}
        photo={photo}
        question={state.pending && <QuestionCard key={state.items.length} pending={state.pending} photo={query.photo} onReply={answer} />}
      />

      {state.error && (
        <Alert
          className="fr-mb-3w"
          severity={found?.cards.length ? "warning" : "error"}
          small
          description={
            <>
              {state.error}{" "}
              <button type="button" className={fr.cx("fr-link")} onClick={() => onRefine((q) => ({ ...q, level: 0 }))}>
                Lancer la recherche rapide
              </button>
            </>
          }
        />
      )}

      {found && (
        <div className={`mr-found mr-reveal${state.working ? " is-stale" : ""}`} aria-busy={state.working}>
          <div className="mr-verdict">
            <span className={`mr-verdict-icon mr-verdict-icon--${head.severity}`} aria-hidden="true">
              <span className={fr.cx(head.icon as never)} />
            </span>
            <div>
              <p className="mr-verdict-eyebrow">
                <Badge small noIcon severity={head.severity}>
                  {head.badge}
                </Badge>
              </p>
              <h2 className="mr-verdict-head">{head.head}</h2>
              {found.summary && <p className="mr-verdict-text">{found.summary}</p>}
            </div>
          </div>
          {found.cards.length > 0 ? (
            <ResultGrid cards={found.cards} picked featureFirst={found.found === "exact"} photo={photo} />
          ) : (
            <Alert severity="info" small description="Le modèle n’a rien retenu. Essayez la recherche rapide, ou dites-en plus ci-dessous." />
          )}
        </div>
      )}
      {done && (found || state.items.length > 0) && <FollowUp disabled={state.working} onSend={(text) => answer({ kind: "message", text }, text)} />}
    </>
  );
}

export function Results(props: Props) {
  const photo = useShownPhoto(props.query);
  return <div className="mr-results">{props.query.level === 1 ? <Deep {...props} photo={photo} /> : <Quick {...props} photo={photo} />}</div>;
}
