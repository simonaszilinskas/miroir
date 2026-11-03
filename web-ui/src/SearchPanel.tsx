import { useEffect, useId, useRef, useState, type DragEvent } from "react";
import { fr } from "@codegouvfr/react-dsfr";
import { Button } from "@codegouvfr/react-dsfr/Button";
import { SegmentedControl } from "@codegouvfr/react-dsfr/SegmentedControl";
import { Select } from "@codegouvfr/react-dsfr/Select";
import { Tag } from "@codegouvfr/react-dsfr/Tag";
import { ToggleSwitch } from "@codegouvfr/react-dsfr/ToggleSwitch";
import Pictures from "@codegouvfr/react-dsfr/picto/Pictures";
import { PhotoCrop } from "./PhotoCrop";
import { apiPatient, filterCount, NO_FILTERS, type Box, type Example, type Facets, type Filters } from "./api";
import { COLLECTIONS, LEVELS, type Level } from "./content";
import { cut, devicePosition, DETECT_SIDE, loadPhoto, type Part } from "./photo";
import { canSearch, type Query } from "./query";

type Props = {
  query: Query;
  onChange: (q: Query) => void;
  onSubmit: () => void;
  textRef: React.RefObject<HTMLInputElement | null>;
  facets: Facets | null;
  examples: Example[];
  /** False when the server has no model for "Approfondi". */
  deepEnabled: boolean;
};

/** Outlines shown at first (`MAX_OUTLINES` in outlines.py); the rest on request. */
const FIRST_OUTLINES = 20;

/** The live site's filters: collection, region, département, domain, period, and position. */
function FilterFields({ query, set, facets }: { query: Query; set: (p: Partial<Query>) => void; facets: Facets | null }) {
  const filters = query.filters;
  const setF = (patch: Partial<Filters>) => set({ filters: { ...filters, ...patch } });
  const [positionNote, setPositionNote] = useState("");
  const departements = facets?.departements[filters.region] ?? facets?.departements[""] ?? [];
  const collections = facets?.collections.map((c) => c.value) ?? Object.keys(COLLECTIONS);
  return (
    <div className="mr-filters-body">
      <fieldset className="mr-field">
        <legend className="fr-label">Collection</legend>
        <ul className="fr-tags-group">
          {collections.map((value) => {
            const on = filters.collections.includes(value);
            return (
              <li key={value}>
                <Tag
                  small
                  pressed={on}
                  nativeButtonProps={{
                    type: "button",
                    title: COLLECTIONS[value]?.hint,
                    onClick: () => setF({ collections: on ? filters.collections.filter((x) => x !== value) : [...filters.collections, value] }),
                  }}
                >
                  {COLLECTIONS[value]?.label ?? value}
                </Tag>
              </li>
            );
          })}
        </ul>
      </fieldset>
      <div className="mr-selects">
        <Select label="Région" nativeSelectProps={{ value: filters.region, onChange: (e) => setF({ region: e.target.value, departement: "" }) }}>
          <option value="">Toutes</option>
          {facets?.regions.map((r) => (
            <option key={r.value}>{r.value}</option>
          ))}
        </Select>
        <Select label="Département" nativeSelectProps={{ value: filters.departement, onChange: (e) => setF({ departement: e.target.value }) }}>
          <option value="">Tous</option>
          {departements.map((d) => (
            <option key={d.value}>{d.value}</option>
          ))}
        </Select>
        <Select label="Domaine" nativeSelectProps={{ value: filters.domain, onChange: (e) => setF({ domain: e.target.value }) }}>
          <option value="">Tous</option>
          {facets?.domains.map((d) => (
            <option key={d.value}>{d.value}</option>
          ))}
        </Select>
        <Select label="Époque" hint="De l’œuvre, ou de la photo pour Mémoire" nativeSelectProps={{ value: filters.period, onChange: (e) => setF({ period: e.target.value }) }}>
          <option value="">Toutes</option>
          {facets?.periods.map((p) => (
            <option key={p.value}>{p.value}</option>
          ))}
        </Select>
      </div>
      {query.photo?.position && query.photoPosition ? (
        <p className="fr-text--sm mr-position">
          <span className={fr.cx("fr-icon-map-pin-2-line", "fr-icon--sm", "fr-mr-1v")} aria-hidden="true" />
          Position de la photo utilisée : ce qui est à côté passe devant.{" "}
          <button type="button" className={fr.cx("fr-link", "fr-link--sm")} onClick={() => set({ photoPosition: false })}>
            Ne pas l’utiliser
          </button>
        </p>
      ) : (
        <ToggleSwitch
          label="Près de moi"
          helperText={positionNote || "Ce qui est près de vous passe devant. Votre position sert à cette recherche et n’est pas gardée."}
          checked={query.nearMe}
          showCheckedHint={false}
          onChange={async (checked) => {
            if (!checked) {
              set({ nearMe: false });
              return;
            }
            setPositionNote("Recherche de votre position…");
            // The browser asks permission here, on the tap, not before.
            const found = await devicePosition();
            setPositionNote(found ? "" : "Position indisponible. Autorisez-la dans le navigateur, ou cherchez sans.");
            set({ nearMe: Boolean(found) });
          }}
        />
      )}
    </div>
  );
}

/**
 * One search card: the photo on top (drop zone, then the photo with its outlines), the words
 * below, then filters, level and "Chercher". A photo, words, or both are enough.
 */
export function SearchPanel({ query, onChange, onSubmit, textRef, facets, examples, deepEnabled }: Props) {
  const set = (patch: Partial<Query>) => onChange({ ...query, ...patch });
  const fileRef = useRef<HTMLInputElement>(null);
  const filtersId = useId();
  const [dragging, setDragging] = useState(false);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [outlines, setOutlines] = useState<Part[]>([]);
  const [allOutlines, setAllOutlines] = useState(false);
  const [finding, setFinding] = useState(false);
  const [loadError, setLoadError] = useState("");
  const { photo, crop, level } = query;
  const ready = canSearch(query);
  const nFilters = filterCount(query.filters) + (query.nearMe ? 1 : 0);

  // Find the works in each new photo (F4); a slow answer for an old photo is dropped.
  useEffect(() => {
    setOutlines([]);
    setAllOutlines(false);
    if (!photo) return;
    let alive = true;
    setFinding(true);
    (async () => {
      try {
        const image = await cut(photo, null, DETECT_SIDE);
        const found = await apiPatient<{ boxes: Box[] }>("detect", { method: "POST", body: { image } }, () => alive);
        if (alive) setOutlines(found.boxes.map((b, i) => ({ x: b.x0, y: b.y0, w: b.x1 - b.x0, h: b.y1 - b.y0, label: `œuvre ${i + 1}` })));
      } catch {
        /* no outlines: drawing a frame still works */
      } finally {
        if (alive) setFinding(false);
      }
    })();
    return () => {
      alive = false;
    };
  }, [photo]);

  const usePhoto = async (blob: Blob, name: string, credit?: string) => {
    setLoadError("");
    try {
      const loaded = await loadPhoto(blob, name, credit);
      if (photo) URL.revokeObjectURL(photo.url);
      onChange({ ...query, photo: loaded, crop: null, photoPosition: true });
    } catch {
      setLoadError("Cette image n’a pas pu être lue. Essayez une photo JPEG ou PNG.");
    }
  };
  const addFile = (file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setLoadError("Ce fichier n’est pas une image.");
      return;
    }
    void usePhoto(file, file.name);
  };
  const useExample = async (x: Example) => {
    try {
      const response = await fetch(x.image_url, { credentials: "same-origin" });
      if (!response.ok) throw new Error("example");
      await usePhoto(await response.blob(), x.label, x.credit);
    } catch {
      setLoadError("Cet exemple n’a pas pu être chargé.");
    }
  };
  const choose = () => fileRef.current?.click();
  const shown = allOutlines ? outlines : outlines.slice(0, FIRST_OUTLINES);
  const hidden = outlines.length - shown.length;

  return (
    <form
      className={`mr-card${dragging ? " is-dragging" : ""}`}
      onSubmit={(e) => {
        e.preventDefault();
        if (ready) onSubmit();
      }}
      onDragOver={(e: DragEvent) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e: DragEvent<HTMLElement>) => {
        if (!e.currentTarget.contains(e.relatedTarget as Node)) setDragging(false);
      }}
      onDrop={(e: DragEvent) => {
        e.preventDefault();
        setDragging(false);
        addFile(e.dataTransfer.files[0]);
      }}
    >
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        hidden
        onChange={(e) => {
          addFile(e.target.files?.[0]);
          e.target.value = ""; // so the same file can be picked again
        }}
      />

      {/* Photo */}
      {!photo ? (
        <div className="mr-drop">
          <button type="button" className="mr-drop-main" onClick={choose}>
            <Pictures fontSize="4.5rem" className="mr-drop-picto" aria-hidden="true" />
            <span className="mr-drop-title">{dragging ? "Lâchez la photo ici" : "Déposez une photo d’une œuvre ou d’un monument"}</span>
            <span className="mr-drop-sub">ou cliquez pour en choisir une. Sur un téléphone, vous pouvez aussi la prendre.</span>
            <span className={`${fr.cx("fr-btn", "fr-btn--secondary", "fr-btn--icon-left", "ri-image-add-line")} mr-drop-btn`}>Choisir une photo</span>
          </button>
          {examples.length > 0 && (
            <div className="mr-examples">
              <p className="mr-examples-title">ou essayez un exemple</p>
              <ul>
                {examples.map((x, i) => (
                  <li key={x.id} style={{ animationDelay: `${200 + i * 60}ms` }}>
                    <button type="button" title={`${x.label} (${x.kind}). Photo : ${x.credit}`} onClick={() => void useExample(x)}>
                      <img src={x.image_url} alt="" loading="lazy" />
                      <span>{x.kind}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      ) : (
        <div className="mr-filled">
          <div className="mr-stage">
            <PhotoCrop src={photo.url} alt="Votre photo" crop={crop} outlines={shown} onCrop={(c) => set({ crop: c })} />
          </div>
          <div className="mr-stage-bar">
            <p className="fr-text--sm fr-mb-0" aria-live="polite">
              {crop ? (
                <>
                  <span className={`${fr.cx("fr-icon-checkbox-circle-fill", "fr-icon--sm")} mr-ok`} aria-hidden="true" /> Miroir cherchera{" "}
                  {crop.label ? <b>{crop.label}</b> : "cette partie"}.
                </>
              ) : finding ? (
                "Repérage des œuvres dans la photo…"
              ) : outlines.length ? (
                `${outlines.length} œuvre${outlines.length > 1 ? "s" : ""} repérée${outlines.length > 1 ? "s" : ""}. Touchez un cadre, ou tracez le vôtre.`
              ) : (
                "Tracez un cadre sur la photo pour viser un détail (facultatif)."
              )}
              {hidden > 0 && !crop && (
                <>
                  {" "}
                  <button type="button" className={fr.cx("fr-link", "fr-link--sm")} onClick={() => setAllOutlines(true)}>
                    Afficher les {hidden} autres
                  </button>
                </>
              )}
            </p>
            <div className="mr-stage-actions">
              {crop && (
                <Button type="button" priority="tertiary no outline" size="small" iconId="ri-fullscreen-line" onClick={() => set({ crop: null })}>
                  Toute la photo
                </Button>
              )}
              <Button type="button" priority="tertiary no outline" size="small" iconId="ri-image-edit-line" onClick={choose}>
                Changer
              </Button>
              <Button
                type="button"
                priority="tertiary no outline"
                size="small"
                iconId="fr-icon-delete-line"
                title="Retirer la photo"
                onClick={() => {
                  URL.revokeObjectURL(photo.url);
                  set({ photo: null, crop: null });
                }}
              />
            </div>
          </div>
          {photo.credit && <p className="mr-credit">Photo : {photo.credit}</p>}
        </div>
      )}
      {loadError && (
        <p className="fr-error-text mr-load-error" role="alert">
          {loadError}
        </p>
      )}

      {/* Words */}
      <div className="mr-words">
        <span className={`${fr.cx("ri-chat-quote-line")} mr-words-icon`} aria-hidden="true" />
        <label htmlFor="mr-text" className="fr-sr-only">
          {photo ? "Ajoutez des mots pour préciser (facultatif)" : "Ou décrivez ce que vous cherchez"}
        </label>
        <input
          id="mr-text"
          ref={textRef}
          type="text"
          autoComplete="off"
          maxLength={500}
          className="mr-words-input"
          placeholder={photo ? "Ajoutez des mots (facultatif)" : "Ou décrivez ce que vous cherchez…"}
          value={query.text}
          onChange={(e) => set({ text: e.target.value })}
        />
        {query.text && (
          <button type="button" className="mr-words-clear" title="Effacer le texte" onClick={() => set({ text: "" })}>
            <span className={fr.cx("fr-icon-close-line", "fr-icon--sm")} aria-hidden="true" />
          </button>
        )}
      </div>

      {/* Filters */}
      {filtersOpen && (
        <div className="mr-filters mr-reveal" id={filtersId}>
          <FilterFields query={query} set={set} facets={facets} />
          {nFilters > 0 && (
            <Button
              type="button"
              priority="tertiary no outline"
              size="small"
              iconId="fr-icon-delete-line"
              onClick={() => set({ filters: NO_FILTERS, nearMe: false })}
            >
              Tout effacer
            </Button>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="mr-actions">
        <div className="mr-actions-left">
          <Button
            type="button"
            priority="tertiary"
            size="small"
            iconId="ri-filter-3-line"
            className="mr-filters-btn"
            nativeButtonProps={{ "aria-expanded": filtersOpen, "aria-controls": filtersId }}
            onClick={() => setFiltersOpen((o) => !o)}
          >
            Filtres
            {nFilters > 0 && <span className="mr-count">{nFilters}</span>}
          </Button>
          <p className="mr-either">Photo, texte, ou les deux</p>
        </div>
        <div className="mr-actions-right">
          <SegmentedControl
            small
            hideLegend
            legend="Type de recherche"
            name="level"
            className="mr-level-toggle"
            segments={[
              { label: LEVELS[0].name, nativeInputProps: { checked: level === 0, onChange: () => set({ level: 0 as Level }) } },
              {
                label: LEVELS[1].name,
                nativeInputProps: { checked: level === 1, disabled: !deepEnabled, onChange: () => set({ level: 1 as Level }) },
              },
            ]}
          />
          <Button type="submit" size="large" iconId="ri-search-line" iconPosition="left" disabled={!ready} className="mr-go">
            Chercher
          </Button>
        </div>
      </div>
    </form>
  );
}

