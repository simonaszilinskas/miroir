import { fr } from "@codegouvfr/react-dsfr";
import { headerFooterDisplayItem } from "@codegouvfr/react-dsfr/Display";

/** The live site's header: the Miroir mark and no State logo (a prototype, not an official
 * service). The tool links show again in the phone menu. */
function Links({ loggedIn, onLogout }: { loggedIn: boolean; onLogout: () => void }) {
  return (
    <ul className="fr-btns-group">
      <li>
        <a className={fr.cx("fr-btn", "fr-icon-camera-line")} href="#/">
          Nouvelle recherche
        </a>
      </li>
      <li>
        <a className={fr.cx("fr-btn", "fr-icon-question-line")} href="#/comment-ca-marche">
          Comment ça marche
        </a>
      </li>
      <li>
        <button className={fr.cx("fr-btn", "fr-icon-theme-fill")} {...headerFooterDisplayItem.buttonProps}>
          Affichage
        </button>
      </li>
      {loggedIn && (
        <li>
          <button className={fr.cx("fr-btn", "fr-icon-logout-box-r-line")} onClick={onLogout}>
            Se déconnecter
          </button>
        </li>
      )}
    </ul>
  );
}

export function SiteHeader({ loggedIn, onLogout }: { loggedIn: boolean; onLogout: () => void }) {
  return (
    <header role="banner" className="fr-header miroir-header">
      <div className="fr-header__body">
        <div className="fr-container">
          <div className="fr-header__body-row">
            <div className="fr-header__brand">
              <div className="fr-header__brand-top">
                <div className="fr-header__service miroir-service">
                  <a href="#/" title="Miroir, accueil" className="miroir-mark" aria-hidden="true" tabIndex={-1}>
                    <span className="miroir-mark__glyph" />
                  </a>
                  <div>
                    <a href="#/" title="Miroir, accueil">
                      <p className="fr-header__service-title">
                        Miroir
                        <span className={`${fr.cx("fr-badge", "fr-badge--sm", "fr-badge--new", "fr-badge--no-icon")} miroir-badge`}>Prototype</span>
                      </p>
                    </a>
                    <p className="fr-header__service-tagline">Retrouver une œuvre ou un monument à partir d’une photo</p>
                  </div>
                </div>
                <div className="fr-header__navbar">
                  <button className={fr.cx("fr-btn--menu", "fr-btn")} data-fr-opened="false" aria-controls="menu-modal" aria-haspopup="menu" id="menu-button" title="Menu">
                    Menu
                  </button>
                </div>
              </div>
            </div>
            <div className="fr-header__tools">
              <div className="fr-header__tools-links">
                <Links loggedIn={loggedIn} onLogout={onLogout} />
              </div>
            </div>
          </div>
        </div>
      </div>
      <div className="fr-header__menu fr-modal" id="menu-modal" aria-labelledby="menu-button">
        <div className="fr-container">
          <button className={fr.cx("fr-btn--close", "fr-btn")} aria-controls="menu-modal" title="Fermer">
            Fermer
          </button>
          <div className="fr-header__menu-links">
            <Links loggedIn={loggedIn} onLogout={onLogout} />
          </div>
        </div>
      </div>
    </header>
  );
}
