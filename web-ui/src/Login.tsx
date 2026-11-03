import { useState } from "react";
import { fr } from "@codegouvfr/react-dsfr";
import { Button } from "@codegouvfr/react-dsfr/Button";
import { Input } from "@codegouvfr/react-dsfr/Input";
import { api, ApiError } from "./api";

/** Shared password, as on the live site (`miroir.web.auth`). */
export function Login({ onDone }: { onDone: () => void }) {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  return (
    <div className="fr-container fr-py-8w mr-view">
      <div className="fr-grid-row fr-grid-row--center">
        <div className="fr-col-12 fr-col-md-8 fr-col-lg-6">
          <div className="miroir-panel fr-p-4w fr-p-md-6w">
            <h1 className="fr-h3">Accès réservé</h1>
            <p>Miroir est un prototype en cours d’essai. Il est ouvert aux personnes invitées, avec le mot de passe qui leur a été remis.</p>
            <form
              noValidate
              onSubmit={async (e) => {
                e.preventDefault();
                setBusy(true);
                setError("");
                try {
                  await api("login", { method: "POST", body: { password } });
                  onDone();
                } catch (err) {
                  setError(err instanceof ApiError ? err.message : "Connexion impossible.");
                } finally {
                  setBusy(false);
                }
              }}
            >
              <Input
                label="Mot de passe"
                state={error ? "error" : "default"}
                stateRelatedMessage={error}
                nativeInputProps={{ type: "password", autoComplete: "current-password", required: true, value: password, onChange: (e) => setPassword(e.target.value), autoFocus: true }}
              />
              <Button type="submit" iconId="fr-icon-arrow-right-line" iconPosition="right" disabled={busy}>
                Se connecter
              </Button>
            </form>
            <p className="fr-mt-3w fr-mb-0">
              <a className={fr.cx("fr-link", "fr-icon-question-line", "fr-link--icon-left")} href="#/comment-ca-marche">
                Comment ça marche ?
              </a>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
