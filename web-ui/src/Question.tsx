import { useEffect, useId, useRef, useState } from "react";
import { fr } from "@codegouvfr/react-dsfr";
import { Button } from "@codegouvfr/react-dsfr/Button";
import type { Pending, Reply } from "./agent";
import { PhotoCrop } from "./PhotoCrop";
import { cut, devicePosition, SEND_SIDE, type Part, type Photo } from "./photo";

/** What a question is about, as the card heads it. */
export const ABOUT: Record<string, { label: string; icon: string }> = {
  place: { label: "Le lieu", icon: "fr-icon-map-pin-2-line" },
  period: { label: "L’époque", icon: "ri-hourglass-line" },
  kind: { label: "Ce que c’est", icon: "ri-shapes-line" },
  work: { label: "Quelle œuvre", icon: "ri-focus-3-line" },
  text: { label: "Une inscription", icon: "ri-quill-pen-line" },
  other: { label: "Une précision", icon: "fr-icon-question-line" },
};

const NO_IDEA = "Je ne sais pas";

type Props = {
  pending: Pending;
  /** The visitor's photo, to frame a part of it. */
  photo: Photo | null;
  onReply: (reply: Reply, said: string) => void;
};

/** The model's question, waiting for the visitor: suggested answers, their own words, their
 * position, or a frame drawn on their photo. */
export function QuestionCard({ pending, photo, onReply }: Props) {
  const titleId = useId();
  const inputId = useId();
  const heading = useRef<HTMLHeadingElement>(null);
  const [text, setText] = useState("");
  const [framing, setFraming] = useState(pending.kind === "region");
  const [part, setPart] = useState<Part | null>(null);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const question = pending.kind === "question" ? pending : null;
  const about = ABOUT[question?.about ?? "work"] ?? ABOUT.other;

  // The question takes the focus, so a keyboard or screen reader user lands on it.
  useEffect(() => {
    heading.current?.focus({ preventScroll: true });
    heading.current?.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
  }, []);

  const reply = (r: Reply, said: string) => {
    setBusy(true);
    onReply(r, said);
  };

  const share = async () => {
    setNote("Votre navigateur vous demande l’autorisation…");
    const position = await devicePosition();
    if (!position) {
      setNote("Position indisponible. Écrivez plutôt la commune.");
      return;
    }
    reply({ kind: "answer", text: text.trim() || "Je partage ma position.", position }, "Position partagée");
  };

  const sendFrame = async () => {
    if (!photo) return;
    if (!part) {
      reply({ kind: "region", region: null }, "Toute la photo");
      return;
    }
    reply({ kind: "region", region: await cut(photo, part, SEND_SIDE) }, "Partie encadrée");
  };

  return (
    <section className="mr-ask" role="group" aria-labelledby={titleId}>
      <div className="mr-ask-head">
        <span className="mr-ask-icon" aria-hidden="true">
          <span className={fr.cx(about.icon as never)} />
        </span>
        <p className="mr-ask-eyebrow">
          <span className="mr-ask-wait" aria-hidden="true" />
          Le modèle attend votre réponse
          <span className="mr-ask-about">{about.label}</span>
        </p>
      </div>
      <h3 id={titleId} ref={heading} tabIndex={-1} className="mr-ask-title">
        {pending.text || "Encadrez ce qui vous intéresse dans la photo."}
      </h3>

      {framing && photo ? (
        <div className="mr-ask-frame">
          <p className="fr-text--sm fr-mb-1w mr-muted">Tracez un cadre autour de l’œuvre en glissant sur la photo.</p>
          <div className="mr-ask-photo">
            <PhotoCrop src={photo.url} alt="Votre photo, à encadrer" crop={part} onCrop={setPart} outlines={[]} />
          </div>
          <div className="mr-ask-row">
            <Button size="small" iconId="ri-crop-line" disabled={busy} onClick={sendFrame}>
              {part ? "Envoyer ce cadre" : "Toute la photo"}
            </Button>
            {question && (
              <Button size="small" priority="tertiary no outline" onClick={() => setFraming(false)}>
                Répondre autrement
              </Button>
            )}
          </div>
        </div>
      ) : (
        question && (
          <>
            <ul className="mr-chips" aria-label="Réponses proposées">
              {question.choices.map((choice) => (
                <li key={choice}>
                  <button type="button" className="mr-chip" disabled={busy} onClick={() => reply({ kind: "answer", text: choice }, choice)}>
                    {choice}
                  </button>
                </li>
              ))}
              <li>
                <button type="button" className="mr-chip mr-chip--quiet" disabled={busy} onClick={() => reply({ kind: "answer", text: NO_IDEA }, NO_IDEA)}>
                  {NO_IDEA}
                </button>
              </li>
            </ul>
            <form
              className="mr-ask-free"
              onSubmit={(e) => {
                e.preventDefault();
                if (text.trim()) reply({ kind: "answer", text: text.trim() }, text.trim());
              }}
            >
              <label className="fr-sr-only" htmlFor={inputId}>
                Votre réponse
              </label>
              <input id={inputId} className="mr-ask-input" value={text} maxLength={500} placeholder="Ou répondez avec vos mots…" onChange={(e) => setText(e.target.value)} disabled={busy} />
              <button type="submit" className={fr.cx("fr-btn", "fr-btn--sm", "fr-btn--icon-left", "ri-send-plane-2-line")} disabled={busy || !text.trim()}>
                Répondre
              </button>
            </form>
            {(question.position || (question.region && photo)) && (
              <div className="mr-ask-row">
                {question.position && (
                  <Button size="small" priority="secondary" iconId="ri-map-pin-user-line" disabled={busy} onClick={share}>
                    Partager ma position
                  </Button>
                )}
                {question.region && photo && (
                  <Button size="small" priority="secondary" iconId="ri-crop-line" disabled={busy} onClick={() => setFraming(true)}>
                    Encadrer l’œuvre
                  </Button>
                )}
              </div>
            )}
            {note && (
              <p className="fr-text--sm fr-mb-0 mr-muted" role="status">
                {note}
              </p>
            )}
          </>
        )
      )}
    </section>
  );
}
