import { useEffect, useState } from "react";
import { fr } from "@codegouvfr/react-dsfr";
import { Alert } from "@codegouvfr/react-dsfr/Alert";
import { Badge } from "@codegouvfr/react-dsfr/Badge";
import { Button } from "@codegouvfr/react-dsfr/Button";
import { api, ApiError, popLink, type EntityPage } from "./api";

/** Record page: every POP image of the building or object, key facts, POP links. */
export function Record({ entityId, onBack }: { entityId: string; onBack: () => void }) {
  const [page, setPage] = useState<EntityPage | null>(null);
  const [error, setError] = useState("");
  const [shown, setShown] = useState(0);

  useEffect(() => {
    let alive = true;
    api<EntityPage>(`entity/${encodeURIComponent(entityId)}`)
      .then((p) => alive && setPage(p))
      .catch((err: unknown) => {
        if (alive && !(err instanceof ApiError && err.status === 401)) setError(err instanceof Error ? err.message : "Fiche introuvable.");
      });
    return () => {
      alive = false;
    };
  }, [entityId]);

  useEffect(() => {
    if (page) document.title = `${page.title} · Miroir`;
  }, [page]);

  const image = page?.images[shown];
  return (
    <div className="fr-container fr-py-4w mr-view mr-record">
      <Button priority="tertiary no outline" iconId="fr-icon-arrow-left-line" onClick={onBack} className="fr-mb-2w">
        Retour aux résultats
      </Button>
      {error && <Alert severity="error" small description={error} />}
      {!page && !error && (
        <p className="mr-wait" role="status">
          <span className="mr-spinner" aria-hidden="true" /> Chargement de la fiche…
        </p>
      )}
      {page && (
        <div className="fr-grid-row fr-grid-row--gutters">
          <div className="fr-col-12 fr-col-md-7">
            {image ? (
              <figure className="mr-record-figure">
                <img
                  src={image.url}
                  alt=""
                  onError={(e) => {
                    if (e.currentTarget.src !== image.fallback_url) e.currentTarget.src = image.fallback_url;
                  }}
                />
                {image.label && <figcaption className="fr-text--xs mr-muted">{image.label}</figcaption>}
              </figure>
            ) : (
              <p className="mr-thumb mr-thumb--none">Pas d’image dans POP pour cette notice</p>
            )}
            {page.images.length > 1 && (
              <ul className="mr-record-thumbs" aria-label={`${page.photo_count} photos dans POP`}>
                {page.images.map((im, i) => (
                  <li key={im.url}>
                    <button type="button" aria-pressed={i === shown} onClick={() => setShown(i)} title={im.label ?? `Photo ${i + 1}`}>
                      <img src={im.thumb_url} alt={`Photo ${i + 1}`} loading="lazy" />
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="fr-col-12 fr-col-md-5">
            <ul className="fr-badges-group">
              <li>
                <Badge small noIcon>
                  {page.collection_label}
                </Badge>
              </li>
            </ul>
            <h1 className="fr-h3 fr-mb-1w">{page.title}</h1>
            {page.place && <p className="fr-text--lg mr-muted">{page.place}</p>}
            <p className="fr-text--sm mr-muted">{page.collection_hint}</p>
            <a className={fr.cx("fr-btn", "fr-btn--icon-right", "fr-icon-external-link-line")} href={popLink(page.pop_url)} target="_blank" rel="noopener" title="La notice sur POP, nouvelle fenêtre">
              Voir la notice sur POP
            </a>
            <dl className="mr-facts">
              {page.facts.map((f) => (
                <div key={f.label}>
                  <dt>{f.label}</dt>
                  <dd>{f.value}</dd>
                </div>
              ))}
            </dl>
            {page.description && <p className="fr-text--sm mr-description">{page.description}</p>}
            {page.records.length > 1 && (
              <>
                <h2 className="fr-h6">{page.records.length} notices pour ce {page.collection === "memoire" ? "monument" : "objet"}</h2>
                <ul className="fr-text--sm mr-linked">
                  {page.records.map((r) => (
                    <li key={r.ref}>
                      <a href={popLink(r.pop_url)} target="_blank" rel="noopener" title={`${r.title} sur POP, nouvelle fenêtre`}>
                        {r.title}
                      </a>{" "}
                      <span className="mr-muted">{r.ref}</span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
