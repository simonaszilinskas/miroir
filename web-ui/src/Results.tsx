import { useEffect, useRef, useState } from "react";
import { fr } from "@codegouvfr/react-dsfr";
import { Accordion } from "@codegouvfr/react-dsfr/Accordion";
import { Alert } from "@codegouvfr/react-dsfr/Alert";
import { Badge } from "@codegouvfr/react-dsfr/Badge";
import { Button } from "@codegouvfr/react-dsfr/Button";
import { CallOut } from "@codegouvfr/react-dsfr/CallOut";
import { Card } from "@codegouvfr/react-dsfr/Card";
import { Tag } from "@codegouvfr/react-dsfr/Tag";
import { api, ApiError, apiPatient, chatEvents, type Card as CardData, type ChatEvent, type Position, type SearchResponse } from "./api";
import { collectionLabel, DEEP_GOAL, LEVELS } from "./content";
import { cut, devicePosition, SEND_SIDE } from "./photo";
import { describe, type Query } from "./query";

type Props = {
  query: Query;
  /** False when the server has no model for "Approfondi". */
  deepEnabled: boolean;
  onEdit: () => void;
  onDeeper: () => void;
  /** Search again with a change, e.g. a filter removed. */
  onRefine: (change: (q: Query) => Query) => void;
};

type Step = { id: number; label: string; running: boolean; why?: string; summary?: string; error?: boolean };

const FOUND_HEADS: Record<string, string> = {
  exact: "Ce que le modèle a trouvé",
  close: "Les plus proches dans les collections",
  none: "Rien ne correspond vraiment. Les moins éloignés",
};

const TYPE_LABELS: Record<string, string> = { building: "Bâtiment", object: "Objet" };

/** One result: the POP image that matches best, title, place, base, a link to POP. */
function ResultCard({ card, best, index }: { card: CardData; best: boolean; index: number }) {
  return (
    <li className="mr-card-in" style={{ animationDelay: `${Math.min(index, 12) * 40}ms` }}>
      <Card
        className={best ? "mr-best" : undefined}
        size="small"
        titleAs="h3"
        title={card.title}
        linkProps={{ href: `#/fiche/${encodeURIComponent(card.entity_id)}` }}
        imageComponent={
          card.image_url ? (
            <img
              className="mr-thumb"
              src={card.image_url}
              alt=""
              loading="lazy"
              decoding="async"
              onError={(e) => {
                const img = e.currentTarget;
                if (card.image_fallback_url && img.src !== card.image_fallback_url) img.src = card.image_fallback_url;
              }}
            />
          ) : (
            <div className="mr-thumb mr-thumb--none">Pas d’image dans POP</div>
          )
        }
        badge={best ? <Badge small severity="success">Meilleure piste</Badge> : undefined}
        start={
          <ul className="fr-badges-group">
            <li>
              <Badge small noIcon>
                {collectionLabel(card.collection) || card.collection_label}
              </Badge>
            </li>
            {card.matched && (
              <li>
                <Badge small noIcon severity="info">
                  {`Correspond : ${card.matched.toLowerCase()}`}
                </Badge>
              </li>
            )}
          </ul>
        }
        desc={
          <>
            {card.place}
            {card.reason && <span className="mr-why">{card.reason}</span>}
          </>
        }
        detail={card.photo_count > 1 ? `${card.entity_id} · ${card.photo_count} photos` : card.entity_id}
        footer={
          <a className={fr.cx("fr-link", "fr-link--sm", "fr-icon-external-link-line", "fr-link--icon-right")} href={card.pop_url} target="_blank" rel="noopener" title={`${card.title} sur POP, nouvelle fenêtre`}>
            Voir sur POP
          </a>
        }
      />
    </li>
  );
}

async function positionFor(query: Query): Promise<Position | null> {
  if (query.photo?.position && query.photoPosition) return query.photo.position;
  return query.nearMe ? devicePosition() : null;
}

export function Results({ query, deepEnabled, onEdit, onDeeper, onRefine }: Props) {
  const deep = query.level === 1;
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [cards, setCards] = useState<CardData[]>([]);
  const [steps, setSteps] = useState<Step[]>([]);
  const [status, setStatus] = useState("");
  const [summary, setSummary] = useState("");
  const [found, setFound] = useState<string | null>(null);
  const [seconds, setSeconds] = useState(0);
  const started = useRef(Date.now());

  useEffect(() => {
    const timer = window.setInterval(() => setSeconds(Math.round((Date.now() - started.current) / 1000)), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    let alive = true;
    const isAlive = () => alive;
    const fail = (err: unknown) => {
      if (!alive || (err instanceof ApiError && err.status === 401)) return;
      setError(err instanceof Error ? err.message : "La recherche n’a pas abouti.");
      setDone(true);
    };
    const image = query.photo ? cut(query.photo, query.crop, SEND_SIDE) : Promise.resolve(null);

    const quick = async () => {
      const body: Record<string, unknown> = { image: await image, text: query.text.trim(), filters: query.filters };
      const position = await positionFor(query);
      if (position) body.position = position;
      const found = await apiPatient<SearchResponse>("search", { method: "POST", body }, isAlive);
      if (!alive) return;
      setResponse(found);
      setCards(found.cards);
      setDone(true);
    };

    const deeper = async () => {
      let lastSay = "";
      const position = await positionFor(query);
      const opened = await api<{ conversation_id: string }>("chat/open", {
        method: "POST",
        body: { image: await image, text: query.text.trim(), position, oneshot: true, filters: query.filters },
      });
      const id = opened.conversation_id;
      const send = (body: Record<string, unknown>) => api<{ after: number }>("chat/send", { method: "POST", body: { conversation_id: id, ...body } });
      let from = (await send({ kind: "goal", text: query.photo ? DEEP_GOAL.photo : DEEP_GOAL.words })).after;
      for (;;) {
        // A reply goes once the turn has ended: sent while it runs, it is refused (409).
        let reply: Record<string, unknown> | null = null;
        for await (const event of chatEvents(id, from, isAlive) as AsyncGenerator<ChatEvent>) {
          switch (event.type) {
            case "status":
              setStatus(event.text);
              break;
            case "step":
              setSteps((all) => {
                const step: Step = { id: event.id, label: event.label, running: event.state === "running", why: event.why, summary: event.summary, error: event.error };
                const at = all.findIndex((s) => s.id === event.id);
                return at < 0 ? [...all, step] : all.map((s, i) => (i === at ? { ...s, ...step, why: step.why ?? s.why } : s));
              });
              break;
            case "say":
              lastSay = event.text;
              break;
            case "results":
              setSummary(lastSay);
              setCards(event.cards);
              setFound(event.found);
              break;
            case "limit":
            case "error":
              setError(event.text);
              break;
            case "question":
              // One-shot: nobody answers. The model was told so; if it asks anyway, say so.
              reply = { kind: "answer", text: "je ne sais pas" };
              break;
            case "region":
              reply = { kind: "region" };
              break;
            default:
              break;
          }
        }
        if (reply === null || !alive) break;
        from = (await send(reply)).after;
      }
      if (alive) {
        setStatus("");
        setDone(true);
      }
    };

    (deep ? deeper() : quick()).catch(fail);
    return () => {
      alive = false;
    };
    // The query is fixed for this run: a new search mounts a new Results.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const crop = query.crop;
  const said = describe(query);

  return (
    <div className="mr-results">
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
            {response?.guessed_type && (
              <li>
                <Badge small noIcon>
                  {TYPE_LABELS[response.guessed_type] ?? response.guessed_type}
                </Badge>
              </li>
            )}
          </ul>
          <ul className="fr-tags-group">
            {said.map((p) => (
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

      {/* Rapide: waiting */}
      {!deep && !done && (
        <div className="mr-wait" role="status">
          <span className="mr-spinner" aria-hidden="true" />
          {seconds >= 4 ? `Le moteur de recherche se réveille, cela peut prendre une à deux minutes la première fois… (${seconds} s)` : "Recherche dans POP…"}
        </div>
      )}

      {/* Approfondi: the model's steps, live, then folded */}
      {deep && !done && (
        <div className="mr-agent mr-reveal" role="status">
          <div className="mr-agent-head">
            <span className="mr-spinner" aria-hidden="true" />
            <p className="fr-mb-0">
              <b>{status || "Le modèle commence…"}</b> <span className="mr-muted">· {seconds} s</span>
            </p>
          </div>
          <ol className="mr-journal">
            {steps.map((s) => (
              <li key={s.id} className={s.running ? "is-now" : s.error ? "is-error" : ""}>
                {s.label}
                {s.summary && !s.running && <span className="mr-journal-note"> — {s.summary}</span>}
                {s.why && <span className="mr-journal-why">{s.why}</span>}
              </li>
            ))}
          </ol>
        </div>
      )}

      {error && (
        <Alert
          className="fr-mb-3w"
          severity={cards.length ? "warning" : "error"}
          small
          description={
            <>
              {error}
              {deep && (
                <>
                  {" "}
                  <button type="button" className={fr.cx("fr-link")} onClick={() => onRefine((q) => ({ ...q, level: 0 }))}>
                    Lancer la recherche rapide
                  </button>
                </>
              )}
            </>
          }
        />
      )}

      {done && (
        <>
          {deep && summary && (
            <div className="mr-summary mr-reveal">
              <p className="fr-mb-0">
                <span className={fr.cx("ri-sparkling-line", "fr-mr-1w")} aria-hidden="true" />
                <b>L’avis du modèle : </b>
                {summary}
              </p>
            </div>
          )}
          {deep && steps.length > 0 && (
            <Accordion label={`Comment le modèle a cherché (${steps.length} étape${steps.length > 1 ? "s" : ""}, ${seconds} s)`} className="fr-mb-3w mr-reveal">
              <ol className="mr-journal">
                {steps.map((s) => (
                  <li key={s.id} className={s.error ? "is-error" : ""}>
                    {s.label}
                    {s.summary && <span className="mr-journal-note"> — {s.summary}</span>}
                    {s.why && <span className="mr-journal-why">{s.why}</span>}
                  </li>
                ))}
              </ol>
            </Accordion>
          )}

          {response?.caption && (
            <p className="fr-text--sm mr-caption">
              <span className={fr.cx("fr-icon-file-text-line", "fr-icon--sm", "fr-mr-1v")} aria-hidden="true" />
              Cadre et légende détectés. Légende lue : « {response.caption} »
            </p>
          )}
          {response?.nearby != null && <p className="fr-text--sm mr-muted">Ce qui est près de {query.nearMe ? "vous" : "l’endroit de la photo"} passe devant.</p>}
          {response?.words_found === false && (
            <Alert
              className="fr-mb-3w"
              severity="info"
              small
              description="Aucune notice ne contient ces mots. Voici les images qui ressemblent le plus à votre description. Vérifiez l’orthographe, ou décrivez autrement : la forme, la matière, le sujet, un lieu."
            />
          )}

          {cards.length > 0 ? (
            <>
              <h2 className="fr-h6 fr-mb-2w">
                {deep ? FOUND_HEADS[found ?? ""] ?? "Ce que le modèle propose" : `${cards.length} notices, de la plus proche à la moins proche`}
              </h2>
              <ul className="mr-grid">
                {cards.map((c, i) => (
                  <ResultCard key={c.entity_id} card={c} best={deep && i === 0 && found === "exact"} index={i} />
                ))}
              </ul>
            </>
          ) : (
            !error && (
              <Alert
                severity="info"
                small
                description={deep ? "Le modèle n’a rien retenu. Essayez la recherche rapide, ou ajoutez des mots." : "Aucune notice ne correspond avec ces filtres. Essayez d’en retirer."}
              />
            )
          )}

          {!deep && deepEnabled && cards.length > 0 && (
            <CallOut
              className="fr-mt-4w mr-reveal"
              iconId="ri-search-eye-line"
              title="Pas trouvé ?"
              titleAs="h2"
              buttonProps={{ children: "Aller plus loin", onClick: onDeeper, iconId: "ri-arrow-right-line", iconPosition: "right" }}
            >
              La recherche approfondie reprend votre photo, votre cadre, vos mots et vos filtres : un modèle regarde, cherche, lit les notices et explique ses choix.
            </CallOut>
          )}
        </>
      )}
    </div>
  );
}
