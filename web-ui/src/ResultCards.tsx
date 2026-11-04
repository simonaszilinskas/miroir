import { useState } from "react";
import { createPortal } from "react-dom";
import { fr } from "@codegouvfr/react-dsfr";
import { createModal } from "@codegouvfr/react-dsfr/Modal";
import type { Card } from "./api";
import { collectionLabel } from "./content";

/** How close a card is, in the visitor's words, from closest to farthest (`miroir.web.levels`). */
const LEVELS: Record<string, { label: string; dots: number; hint: string }> = {
  "tres-proche": { label: "Très proche", dots: 3, hint: "sans doute la bonne notice" },
  proche: { label: "Proche", dots: 2, hint: "peut-être elle" },
  ressemblance: { label: "Ressemblance", dots: 1, hint: "à comparer avec votre photo" },
};

/** The card's level; a pick of the model with no level of the site's is the model's choice. */
function levelOf(card: Card, picked: boolean) {
  const known = card.level ? LEVELS[card.level] : undefined;
  if (known) return { key: card.level!, ...known };
  return picked ? { key: "choix", label: "Choix du modèle", dots: 0, hint: "" } : null;
}

export function LevelMeter({ card, picked }: { card: Card; picked: boolean }) {
  const level = levelOf(card, picked);
  if (!level) return null;
  return (
    <span className={`mr-level mr-level--${level.key}`} title={level.hint || undefined}>
      {level.dots > 0 && (
        <span className="mr-level-dots" aria-hidden="true">
          {[1, 2, 3].map((d) => (
            <i key={d} className={d <= level.dots ? "is-on" : undefined} />
          ))}
        </span>
      )}
      {level.dots === 0 && <span className={fr.cx("ri-sparkling-2-line", "fr-icon--xs")} aria-hidden="true" />}
      {level.label}
    </span>
  );
}

/** The legend of the three levels, above a grid of the site's results. */
export function LevelLegend() {
  return (
    <ul className="mr-legend" aria-label="Niveaux de ressemblance">
      {Object.entries(LEVELS).map(([key, l]) => (
        <li key={key}>
          <LevelMeter card={{ level: key } as Card} picked={false} />
          <span>{l.hint}</span>
        </li>
      ))}
    </ul>
  );
}

function CardImage({ card, className }: { card: Card; className: string }) {
  if (!card.image_url) return <div className={`${className} mr-art--none`}>Pas d’image dans POP</div>;
  return (
    <img
      className={className}
      src={card.image_url}
      alt=""
      loading="lazy"
      decoding="async"
      onError={(e) => {
        const img = e.currentTarget;
        if (card.image_fallback_url && img.src !== card.image_fallback_url) img.src = card.image_fallback_url;
      }}
    />
  );
}

function PopLink({ card }: { card: Card }) {
  return (
    <a className={fr.cx("fr-link", "fr-link--sm", "fr-icon-external-link-line", "fr-link--icon-right")} href={card.pop_url} target="_blank" rel="noopener" title={`${card.title} sur POP, nouvelle fenêtre`}>
      Voir sur POP
    </a>
  );
}

type CardProps = {
  card: Card;
  index: number;
  /** The model picked it ("Approfondi"): its reason is shown, and its level when the site has none. */
  picked: boolean;
  featured: boolean;
  onCompare: ((card: Card) => void) | null;
};

/** One record: its image matted like a print, how close it is, title, place, the model's
 * reason, and the ways on (compare with the photo, the record page, POP). */
function ResultCard({ card, index, picked, featured, onCompare }: CardProps) {
  const base = collectionLabel(card.collection) || card.collection_label;
  return (
    <li className={`mr-result${featured ? " mr-result--featured" : ""}`} style={{ animationDelay: `${Math.min(index, 12) * 45}ms` }}>
      <article className="mr-result-in">
        <div className="mr-mat">
          <div className="mr-mat-in">
            <CardImage card={card} className="mr-art" />
          </div>
          {featured && (
            <span className="mr-ribbon">
              <span className={fr.cx("fr-icon-success-line", "fr-icon--xs")} aria-hidden="true" /> Meilleure piste
            </span>
          )}
          {card.photo_count > 1 && <span className="mr-count-photos">{card.photo_count} photos</span>}
        </div>
        <div className="mr-result-body">
          <div className="mr-result-top">
            <LevelMeter card={card} picked={picked} />
            <span className="mr-base">{base}</span>
          </div>
          <h3 className="mr-result-title">
            <a href={`#/fiche/${encodeURIComponent(card.entity_id)}`}>{card.title || "Sans titre"}</a>
          </h3>
          {card.place && (
            <p className="mr-fact">
              <span className={fr.cx("fr-icon-map-pin-2-line", "fr-icon--xs")} aria-hidden="true" />
              {card.place}
            </p>
          )}
          {card.matched && (
            <p className="mr-fact">
              <span className={fr.cx("ri-image-2-line", "fr-icon--xs")} aria-hidden="true" />
              Correspond : {card.matched.toLowerCase()}
            </p>
          )}
          {card.reason && (
            <p className="mr-reason">
              <span className={fr.cx("ri-sparkling-2-line", "fr-icon--xs")} aria-hidden="true" />
              <span>
                <span className="fr-sr-only">Pourquoi le modèle l’a retenue : </span>
                {card.reason}
              </span>
            </p>
          )}
          <div className="mr-result-actions">
            {onCompare && (
              <button type="button" className={fr.cx("fr-btn", "fr-btn--tertiary", "fr-btn--sm", "fr-btn--icon-left", "ri-split-cells-horizontal")} onClick={() => onCompare(card)}>
                Comparer
              </button>
            )}
            <PopLink card={card} />
          </div>
        </div>
      </article>
    </li>
  );
}

const compare = createModal({ id: "mr-compare", isOpenedByDefault: false });

type GridProps = {
  cards: Card[];
  /** The model's picks ("Approfondi"). */
  picked: boolean;
  /** The first card is the very thing (`found` exact): it is shown larger. */
  featureFirst: boolean;
  /** The visitor's photo (or the part they framed), to compare a card with. */
  photo: string | null;
};

/** The results as a grid of cards; "Comparer" opens the visitor's photo beside a card's image. */
export function ResultGrid({ cards, picked, featureFirst, photo }: GridProps) {
  const [chosen, setChosen] = useState<Card | null>(null);
  const onCompare = photo
    ? (card: Card) => {
        setChosen(card);
        compare.open();
      }
    : null;
  return (
    <>
      <ul className="mr-results-grid">
        {cards.map((c, i) => (
          <ResultCard key={c.entity_id} card={c} index={i} picked={picked} featured={featureFirst && i === 0} onCompare={onCompare} />
        ))}
      </ul>
      {/* At the body's level: an animated ancestor would hold a fixed modal in its own box. */}
      {createPortal(
      <compare.Component title="Comparer avec votre photo" size="large" iconId="ri-split-cells-horizontal" className="mr-compare-modal">
        {chosen && photo && (
          <div className="mr-compare">
            <figure className="mr-compare-side">
              <div className="mr-compare-frame">
                <img src={photo} alt="Votre photo" />
              </div>
              <figcaption>
                <span className="mr-compare-tag">Votre photo</span>
              </figcaption>
            </figure>
            <figure className="mr-compare-side">
              <div className="mr-compare-frame">
                <CardImage card={chosen} className="mr-compare-img" />
              </div>
              <figcaption>
                <span className="mr-compare-tag mr-compare-tag--pop">{collectionLabel(chosen.collection) || chosen.collection_label}</span>
                <LevelMeter card={chosen} picked={picked} />
              </figcaption>
            </figure>
            <div className="mr-compare-facts">
              <p className="mr-compare-title">{chosen.title || "Sans titre"}</p>
              {chosen.place && (
                <p className="mr-fact">
                  <span className={fr.cx("fr-icon-map-pin-2-line", "fr-icon--xs")} aria-hidden="true" />
                  {chosen.place}
                </p>
              )}
              {chosen.reason && (
                <p className="mr-reason">
                  <span className={fr.cx("ri-sparkling-2-line", "fr-icon--xs")} aria-hidden="true" />
                  <span>{chosen.reason}</span>
                </p>
              )}
              <p className="fr-mb-0">
                <a className={fr.cx("fr-link", "fr-link--sm", "fr-mr-3w")} href={`#/fiche/${encodeURIComponent(chosen.entity_id)}`} onClick={() => compare.close()}>
                  Ouvrir la fiche
                </a>
                <PopLink card={chosen} />
              </p>
            </div>
          </div>
        )}
      </compare.Component>,
        document.body,
      )}
    </>
  );
}
