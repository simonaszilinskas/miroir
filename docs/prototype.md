# MIROIR: the prototype website

A working version of the product in [functional-spec.md](functional-spec.md) §5–§8, on the open models chosen in [results.md](results.md) (R1, R2). French interface, built for phones, in the State design system (DSFR 1.15.4), with no State logo: it is a prototype, not an official service.

## Use it

- **Address:** https://simonaszilinskas--miroir.us-east.modal.run
- **Password:** one shared password. It is in the main checkout's `.env` (`/Users/simonaszilinskas/miroir/.env`, line `MIROIR_APP_PASSWORD`, never committed) and in the `miroir-app` Modal secret (`APP_PASSWORD`). A login lasts a week.
- **First search of the day:** the GPU sleeps when nobody searches. The first search wakes it, which takes one to two minutes; the page says so and waits. Opening a photo or the text box wakes it in advance. Once awake, a search takes about a second; the GPU goes back to sleep after 10 minutes without a search.

What a person can do:

1. **Home.** Take a photo (the phone's camera opens), upload an image, or describe what they see in words (four example searches under the box run in one tap). Six example photos run a search in one tap: two monuments and two museum objects photographed by Wikimedia Commons users, a drawing, and a 1940s photo of a Jeu de Paume room where a blue outline marks each work.
2. **Photo.** A second after the photo appears, a blue numbered outline marks each work Miroir finds in it (F4), with a line such as "14 œuvres repérées. Touchez un cadre bleu pour la chercher." Tapping an outline searches that work alone. The page shows 20 outlines first, the biggest and most central ones (people frame what they care about); on a wall of pictures, "Afficher les N autres objets repérés" shows the rest (up to 40). A photo of a single object shows none: "Rechercher toute la photo" does the job. A work without an outline: draw a box with a finger or the mouse (a painting, a detail). Boxes and outlines are cut out in the browser.
3. **Position (step H2).** A photo that carries GPS in its EXIF uses it on its own: the photo screen says "Position de la photo utilisée" with a "Ne pas l'utiliser" link. Otherwise a toggle, "Utiliser ma position", asks the browser (and so the visitor) for the position on the tap, not before. Buildings and objects near that position rank higher (a soft prior: bonus `exp(−d / 1 km)` on the fused score); far ones stay, so nothing breaks away from anything known. The results show "Près de vous" or "Près du lieu de la photo". The position goes with that one search; it is not logged or stored. On T2 phone photos of buildings, the right building moves from 32% to 97% in the top 10 (results.md, H2). Phones often strip GPS from a photo a web page reads, so the toggle is the main way in.
4. **Results.** One card per building or object, best first: the image that matches best (marked "Correspond : revers", "détail"… when it is not the record's main image), title, place, collection, a link to POP. Mémoire photos of one building form one card ("12 photos"). Filters (collection, region, département, domain, period) narrow the search before ranking. Words refine a photo search ("à Toulouse"). The page shows, quietly, whether it took the photo for a building or an object. When one record stands out, it is shown alone under "Meilleure correspondance", marked "Correspondance sûre", with the others below. When none stands out, there is no single answer: the page says **"Voici les œuvres les plus proches"** ("les monuments" for a building), marked "Pas de réponse sûre", and asks the person to compare them with their photo.
5. **Old photo on its card (archive print).** When the photo is a black-and-white print glued on a card with a caption, as Mémoire's are, the results say **"Cadre et légende détectés"** and show the caption Miroir read ("Légende lue : « BEAUNE (Côte d'Or) Eglise Notre-Dame… »"). Its words join the search. "Corriger la légende" opens a box to fix the text and search again; "Retirer la légende" searches with the photo alone. Other photos are not affected. On T1 archive prints the right building is in the top 10 for 87% of prints, against 46% before (results.md, T1 archive).
6. **Record.** Every photo of the building or object, key facts, the description, and links to the full POP records.

Light and dark themes: "Affichage" in the header or footer.

## How it works

| Part | What runs |
|---|---|
| `Search` (Modal GPU class, L4) | EmbeddingGemma 2 (A7) and MegaLoc (A3) embed the photo with the same code and preprocessing as the R1 gallery runs (`miroir.remote.search` reuses `miroir.remote.models`); words go through EmbeddingGemma 2's "SearchQuery" prompt. Gallery vectors sit on the GPU. At most one container; it sleeps after 10 minutes |
| `web` (Modal CPU container) | FastAPI: the page, login, record pages, example photos. At most one container |
| Ranking (`miroir.web.engine`, NumPy) | The evaluation's steps: cosine per image, best image per entity, min-max per model, G4 weights. A logistic regression on the photo's A7 vector guesses building or object; buildings use A7 alone, objects 0.2 A7 + 0.8 A3 (R2 G4). Words: A7 text vector against the image vectors (0.9), mixed with French BM25 over record text (0.1), weights from T7 (results.md, step B); words refining a photo count half. When no record holds any of the words, the page says so and shows the closest images |
| Archive prints | `miroir.images.mount.find_picture` (NumPy, about 6 ms) looks for a grey print on a plain card with writing around it. Only then does Qwen3-VL 4B Instruct (open, same GPU container, loaded in a thread) copy the caption line that names the place; its words join the photo search at the photo's weight (`SITE_ARCHIVE` in `miroir.web.archive`). The page can send the caption back corrected or empty (`caption` in `POST /api/search`), and the model is not asked again |
| Outlines (F4) | Grounding DINO tiny (step C winner) in the same GPU container, loaded in a thread beside the search models so it never delays a search. It sees the photo at 1,024 px with step C's prompts; `miroir.outlines.pick_outlines` keeps sure boxes, drops repeats and the whole photo, sends the 40 surest, biggest and most central first, boxes around several works last, and none when the surest box covers 40% of the photo. `POST /api/detect` |
| "Sure" or not | Buildings: the routed mix of the top result's cosines against the bucket's cut-off in `routing.json` (at most 10% of T8 tune searches still sure). Objects: the top result must beat every other of the top 30 by a margin of 0.09 (`sure.json`, `miroir web sure`): 36% of object searches are sure and 86% of those are right on the tune halves, against 8% and 28% with the cut-off |
| Index | A bundle built on the Mac (`miroir web build-index`), stored in the `miroir-web-index` Modal volume. Which bundle is served is one line: `WEB_INDEX` in `src/miroir/config.py` |

Images on the cards come straight from the POP bucket (Rose Valland scans through the IIIF service, scaled, since some files weigh 20 MB). Query photos go to our Modal container only, never to an outside model; they are not stored. The logs keep a hash of each photo, the words, the filters and the records returned (technical spec §9).

Weights, cut-offs and buckets are data (`routing.json`), not code: a third bucket (archive building, colour building, object) needs only a new entry with `greyscale: true` or `false`; the server already measures each photo's chroma spread (`images.colour`, the statistic G5 routes on). Extra gallery photos of an entity from Commons (image ids `commons-…` in the vector runs) are picked up by `build-index` and shown with their author and licence.

## Scores of the deployed stack

Index `full` (since 2026-10-13): every image of the July exports the Modal runs could fetch, 136,731 images of 101,039 records, 63,085 buildings and objects, publishable or not (the site sits behind a login, functional spec §8). Vectors from the full-sample runs (`A7-embeddinggemma-2-512-full`, `A3-megaloc-322-full`, MegaLoc at 8,448 dims), type guess from `miroir index guess`. Tune halves, the site's own ranking (the bundle's `report.json`):

| Set | Queries | Hit@1 | Hit@10 | "Sure" answers | Right among sure |
|---|---|---|---|---|---|
| T1 archive building | 138 | 34.1 | 45.7 | 24.6% | 73% |
| T1 with its caption read (next deploy; measured on index `full`) | 138 | 70.3 | 87.0 | | |
| T3 painting in a room (crop) | 99 | 55.6 | 86.9 | 8.1% | 0% |
| T5 museum object, other view | 493 | 27.2 | 43.2 | 10.3% | 22% |
| T2 phone photo of a building (Commons) | 516 | 9.7 | 18.2 | 0% | |
| T2b museum work (Commons) | 291 | 82.1 | 94.5 | 4.1% | 75% |
| T7 words for a hidden image (French) | 500 | 7.6 | 17.6 | | |
| T7en the same words in English | 105 | 7.6 | 23.8 | | |

| Not-found set | Searches still answered "sure" |
|---|---|
| T8 (in-collection photos, entity removed) | 9.6% (target: at most 10%) |
| T8c (Commons photos with no record) | 0% |

Cut-offs: 0.934 for buildings (A7 cosine), 0.824 for objects (0.2 A7 + 0.8 A3). The gallery is six times the screening one, so every score drops (T5 most: Saint-Omer's coin and medal series look alike) and a "sure" object is often a near twin, not the right record. Since 2026-10-13 objects no longer use their cut-off: the margin rule above decides, and makes a sure object right 86% of the time on the tune halves (T3 73%, T5 73%, T2b 98%).

The first index, `screening-v2` (21,785 publishable images), scored T1 50.7, T3 89.9, T5 61.2, T2 24.8, T2b 94.3 hit@10; switch back with `WEB_INDEX = "screening-v2"`.

End-to-end check on the deployed site (2026-10-13): photos fetched from POP and Commons and posted to `/api/search`. Collection photos (T1, T3, T5) find their own record first with confidence 1.00 (the query vector equals the stored one, so preprocessing matches the gallery's); Commons photos (T2, T2b) get the confidence computed offline from their stored query vectors (0.914, 0.859, 0.55…) and the right record first or second. Words: "nature morte au lièvre" puts "Nature morte avec homard et lièvre" and "Nature morte au lièvre" first; "Pont du Gard" first; English words do less well.

Timings: outlines take 0.27 s on the GPU once warm and about 0.7 s from the Mac (upload included); a warm search takes about 200 ms on the GPU and about 0.9 s from a browser (upload included); words alone 0.5 s. An archive print adds its caption reading: 1.4 s median on the GPU (2.1 s at the 95th percentile), so about 2.3 s from a browser; other photos add 6 ms. A cold start takes 50–110 s (container start, models about 25 s, gallery 2.5 GB to the GPU, BM25 over 101,000 records).

## Run and deploy

**Archive prints, not yet deployed (2026-10-13).** On main; the next `miroir web deploy` picks them up (no new bundle file). The first wake-up after it downloads Qwen3-VL 4B (about 9 GB) into the weights volume once.

**Position (H2), not yet deployed (2026-10-13).** The code is on main; the next deploy picks it up once the bundle holds `places.npz` (one position per entity). `miroir web build-index` writes it when `~/miroir-data/places.parquet` exists (`miroir geo places`, about 30 s, already run). To add it to a built bundle without rebuilding: `miroir web places --index full --upload`, then `miroir web deploy`. A bundle without `places.npz` ignores positions.

```sh
uv sync --extra api
uv run miroir web build-index   # NumPy on the Mac, about 3 minutes for the full index: writes ~/miroir-data/web/<index>/
uv run miroir web upload        # copies the bundle to the miroir-web-index volume
uv run miroir web secret        # once: password in .env and in the miroir-app secret (--rotate for a new one)
uv run miroir web deploy        # deploys the miroir-web app and prints the address
uv run miroir web dev           # http://localhost:8000 with fake models (no model runs on the Mac)
uv run miroir web dev --real    # the same page searching with the deployed GPU (costs GPU time)
```

To switch index: add it to `INDEXES` in `src/miroir/web/bundle.py`, set `WEB_INDEX` in `src/miroir/config.py`, then build-index, upload and deploy.

## Cost

An L4 with 4 CPUs and 24 GB costs about $1.18 an hour while awake. Each wake-up costs about 12 minutes of it (start plus 10 idle minutes), about $0.22; the web container about $0.06 an hour while awake. Nothing runs when nobody uses the site. Searches are limited to 30 a minute per session, logins to 5 a minute per address.

## Known limits

- **Position.** Rose Valland works have no place (looted and moved), and Mémoire objects sit at their commune's centre, so with the position on they rank below precise neighbours. The visitor can turn the position off.

- **Index.** The 101,000 records of the July exports, not all of POP: most tester photos of places outside them will find nothing.
- **Phone photos of buildings.** T2: the right building is in the top 10 for a fifth of photos, and none reaches the "sure" bar. A3 does better than A7 on these photos (31.0 against 21.9 hit@10), but the router sends buildings to A7, which wins on archive photos; a third bucket for colour photos of buildings is being tuned.
- **"Sure" is cautious.** A quarter of in-index archive photos and a third of object photos get a sure answer; the rest see the closest records to compare.
- **Word search** finds the right record in the top 10 for about one description in six (T7: 17.6%; half and half scored 8.0%). Names work ("Pont du Gard" comes first); descriptions of common things (a church, a coin) match too many records, and Mémoire buildings are found only 7% of the time. A region in the words ("bourgogne") does not narrow the search; the region filter does. T6, written by hand, does not exist yet ([t6-guide.md](t6-guide.md)).
- **Period filter** for Mémoire reads the date of the photo, not of the building.
- **Outlines on a crowded wall.** On the Jeu de Paume room photos the right record is in the top 5 for 87% of works when every outline is shown, 78% with the 20 shown first (it was 52% with the 12 surest; results.md, step C): on the densest walls the visitor still needs "Afficher les N autres". Outlines' names come from the detector's prompts and are often wrong (it calls a statue a stained glass window); the page only uses them in labels for screen readers.
- **Keypoint re-ranking (E1) is not on the site:** it lifts T3 hit@1 by 12 points but adds 3 s per search (6 s at the 95th percentile), mostly reading candidate images.
- **Cold start** of about two minutes after 10 idle minutes. A longer idle window or one warm container would remove it, at about $27 a day.
- **Type guess** read from the index agent's full-sample file; a three-bucket router (G5) can replace the two G4 buckets in `routing.json`.
