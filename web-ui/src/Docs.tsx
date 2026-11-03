import page from "./docs.html?raw";
import { DOCS_ICONS } from "./docsIcons";

void DOCS_ICONS;

/** "Comment ça marche": open without the password; it holds nothing secret. The text is
 * plain DSFR markup, kept in docs.html. */
export function Docs() {
  return <div className="miroir-doc mr-view" dangerouslySetInnerHTML={{ __html: page }} />;
}
