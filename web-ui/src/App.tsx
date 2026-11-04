import { useCallback, useEffect, useRef, useState, type CSSProperties } from "react";
import { fr } from "@codegouvfr/react-dsfr";
import { Notice } from "@codegouvfr/react-dsfr/Notice";
import { Display, headerFooterDisplayItem } from "@codegouvfr/react-dsfr/Display";
import { api, whenLoggedOut, type Example, type Facets, type Stats } from "./api";
import { heroPhotos, LEVELS } from "./content";
import { Docs } from "./Docs";
import { SiteHeader } from "./SiteHeader";
import { Login } from "./Login";
import { Record } from "./Record";
import { Results } from "./Results";
import { SearchPanel } from "./SearchPanel";
import { EMPTY_QUERY, type Query } from "./query";

const WARM_EVERY_MS = 4 * 60 * 1000;
const smooth = (): ScrollBehavior => (window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth");

type View = { name: "home" } | { name: "login" } | { name: "docs" } | { name: "record"; id: string };

function viewOf(hash: string): View {
  if (hash === "#/connexion") return { name: "login" };
  if (hash === "#/comment-ca-marche") return { name: "docs" };
  if (hash.startsWith("#/fiche/")) return { name: "record", id: decodeURIComponent(hash.slice("#/fiche/".length)) };
  return { name: "home" };
}

/** POP photos around the title: a fan above it on phones, floating at the sides on wide screens. */
function HeroPhotos() {
  const [photos] = useState(heroPhotos);
  return (
    <div className="mr-float" aria-hidden="true">
      {photos.map((p, i) => (
        <img
          key={p.src}
          src={p.src}
          alt=""
          decoding="async"
          onError={(e) => {
            e.currentTarget.style.display = "none";
          }}
          style={
            {
              [p.side]: `calc(1rem + (50% - 31.5rem - var(--s)) * ${p.x})`,
              top: p.y,
              "--s": `${(4 + p.near * 3.5).toFixed(2)}rem`,
              "--o": (0.25 + p.near * 0.5).toFixed(2),
              "--b": `${((1 - p.near) * 2).toFixed(1)}px`,
              "--r": `${(i % 2 ? 1 : -1) * (3 + ((i * 7) % 6))}deg`,
              "--t": `${(8 + (1 - p.near) * 8 + (i % 3)).toFixed(1)}s`,
              "--d": `${-i * 1.7}s`,
              zIndex: Math.round(p.near * 10),
            } as CSSProperties
          }
        />
      ))}
    </div>
  );
}

function Footer({ stats }: { stats: Stats | null }) {
  const images = stats ? stats.images.toLocaleString("fr-FR") : "2,2 millions de";
  return (
    <footer className="fr-footer miroir-footer" role="contentinfo" id="pied">
      <div className="fr-container">
        <div className="fr-footer__body">
          <div className="fr-footer__content">
            <p className="fr-footer__content-desc">
              <strong>Miroir est un prototype</strong>, pas un service officiel. Il cherche dans les {images} photographies des notices de POP avec image, téléchargées en novembre 2026. Notices et images :{" "}
              <a href="https://pop.culture.gouv.fr/" target="_blank" rel="noopener" title="Plateforme ouverte du patrimoine, nouvelle fenêtre">
                Plateforme ouverte du patrimoine (POP)
              </a>
              , ministère de la Culture, en données ouvertes. Les droits des images restent ceux indiqués sur chaque notice.
            </p>
            <ul className="fr-footer__content-list">
              <li className="fr-footer__content-item">
                <a className="fr-footer__content-link" target="_blank" rel="noopener" href="https://pop.culture.gouv.fr/" title="pop.culture.gouv.fr, nouvelle fenêtre">
                  pop.culture.gouv.fr
                </a>
              </li>
              <li className="fr-footer__content-item">
                <a className="fr-footer__content-link" target="_blank" rel="noopener" href="https://data.culture.gouv.fr/" title="data.culture.gouv.fr, nouvelle fenêtre">
                  data.culture.gouv.fr
                </a>
              </li>
            </ul>
          </div>
        </div>
        <div className="fr-footer__bottom">
          <ul className="fr-footer__bottom-list">
            <li className="fr-footer__bottom-item">
              <span className="fr-footer__bottom-link">Accessibilité : non conforme</span>
            </li>
            <li className="fr-footer__bottom-item">
              <a className="fr-footer__bottom-link" href="#/comment-ca-marche">
                Comment ça marche
              </a>
            </li>
            <li className="fr-footer__bottom-item">
              <button className={fr.cx("fr-footer__bottom-link", "fr-icon-theme-fill", "fr-btn--icon-left")} {...headerFooterDisplayItem.buttonProps}>
                Paramètres d’affichage
              </button>
            </li>
          </ul>
          <div className="fr-footer__bottom-copy">
            <p>Prototype de recherche, novembre 2026. Modèles ouverts : EmbeddingGemma 2, MegaLoc, Grounding DINO, Qwen3-VL, et Mistral Large 4 pour la recherche approfondie. Photos des exemples : leurs auteurs, sous les licences indiquées.</p>
          </div>
        </div>
      </div>
    </footer>
  );
}

export function App() {
  const [view, setView] = useState<View>(() => viewOf(location.hash));
  const [loggedIn, setLoggedIn] = useState<boolean | null>(null);
  const [query, setQuery] = useState<Query>(EMPTY_QUERY);
  const [sent, setSent] = useState<{ query: Query; run: number } | null>(null);
  const [facets, setFacets] = useState<Facets | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [deepEnabled, setDeepEnabled] = useState(false);
  const resultsRef = useRef<HTMLElement>(null);
  const textRef = useRef<HTMLInputElement>(null);
  const lastWarm = useRef(0);

  useEffect(() => {
    const onHash = () => setView(viewOf(location.hash));
    window.addEventListener("hashchange", onHash);
    whenLoggedOut(() => {
      setLoggedIn(false);
      if (location.hash !== "#/comment-ca-marche") location.hash = "#/connexion";
    });
    api("me")
      .then(() => setLoggedIn(true))
      .catch(() => setLoggedIn(false));
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const warm = useCallback(() => {
    if (Date.now() - lastWarm.current < WARM_EVERY_MS) return;
    lastWarm.current = Date.now();
    api("warm", { method: "POST" }).catch(() => {});
  }, []);

  // After login: the menus, the examples, the index size, whether "Approfondi" is on.
  useEffect(() => {
    if (!loggedIn) return;
    if (location.hash === "#/connexion") location.hash = "#/";
    api<Facets>("facets").then(setFacets).catch(() => {});
    api<Example[]>("examples").then(setExamples).catch(() => {});
    api<Stats>("stats").then(setStats).catch(() => {});
    api<{ enabled: boolean }>("chat/status").then((s) => setDeepEnabled(s.enabled)).catch(() => {});
    warm();
  }, [loggedIn, warm]);

  useEffect(() => {
    if (query.photo) warm();
  }, [query.photo, warm]);

  useEffect(() => {
    document.title = view.name === "docs" ? "Comment ça marche · Miroir" : "Miroir · retrouver une œuvre ou un monument à partir d’une photo";
  }, [view]);

  const send = (q: Query) => setSent((prev) => ({ query: q, run: (prev?.run ?? 0) + 1 }));

  useEffect(() => {
    if (sent) resultsRef.current?.scrollIntoView({ behavior: smooth(), block: "start" });
  }, [sent]);

  const showLogin = view.name === "login" || (loggedIn === false && view.name !== "docs");
  const home = !showLogin && view.name === "home";

  return (
    <>
      <SiteHeader
        loggedIn={Boolean(loggedIn)}
        onLogout={async () => {
          await api("logout", { method: "POST" }).catch(() => {});
          setLoggedIn(false);
          location.hash = "#/connexion";
        }}
      />
      {stats?.notice && <Notice title="Mode dégradé." description={stats.notice} />}
      <main id="contenu">
        {showLogin && loggedIn !== null && <Login onDone={() => setLoggedIn(true)} />}
        {view.name === "docs" && <Docs />}
        {!showLogin && view.name === "record" && <Record entityId={view.id} onBack={() => history.back()} />}
        {loggedIn && (
          <div hidden={!home}>
            <section className="mr-hero">
              <HeroPhotos />
              <div className="fr-container mr-hero-inner mr-view">
                <h1 className="mr-title">Qu’avez-vous sous les yeux&nbsp;?</h1>
                <p className="fr-text--lead mr-lead">
                  Une photo d’un tableau, d’un objet ou d’un monument suffit&nbsp;: Miroir retrouve sa notice officielle dans la Plateforme ouverte du patrimoine.
                </p>
                <SearchPanel
                  query={query}
                  onChange={setQuery}
                  onSubmit={() => send(query)}
                  textRef={textRef}
                  facets={facets}
                  examples={examples}
                  deepEnabled={deepEnabled}
                />
                <p className="mr-under" key={query.level}>
                  <span className={fr.cx(LEVELS[query.level].icon, "fr-icon--sm", "fr-mr-1v")} aria-hidden="true" />
                  <b>{LEVELS[query.level].name}</b> : {LEVELS[query.level].hint}
                  {query.level === 1 && " Votre photo est envoyée à ce modèle."}
                </p>
              </div>
            </section>

            {sent && (
              <section ref={resultsRef} className="fr-container mr-view mr-results-section" key={sent.run} aria-label="Résultats">
                <Results
                  query={sent.query}
                  deepEnabled={deepEnabled}
                  onEdit={() => {
                    window.scrollTo({ top: 0, behavior: smooth() });
                    textRef.current?.focus({ preventScroll: true });
                  }}
                  onDeeper={() => {
                    setQuery((q) => ({ ...q, level: 1 }));
                    send({ ...sent.query, level: 1 });
                  }}
                  onRefine={(change) => {
                    setQuery(change);
                    send(change(sent.query));
                  }}
                />
              </section>
            )}
          </div>
        )}
      </main>
      <Footer stats={stats} />
      <Display />
    </>
  );
}
