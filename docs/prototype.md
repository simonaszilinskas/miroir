# MIROIR: the prototype website

A working version of the product in [functional-spec.md](functional-spec.md) §5–§8, on the open models chosen in [results.md](results.md) (R1, R2). French interface, built for phones, in the State design system (DSFR 1.15.4), with no State logo: it is a prototype, not an official service.

## Use it

- **Address:** https://simonaszilinskas--miroir.us-east.modal.run
- **Password:** one shared password. It is in the main checkout's `.env` (`/Users/simonaszilinskas/miroir/.env`, line `MIROIR_APP_PASSWORD`, never committed) and in the `miroir-app` Modal secret (`APP_PASSWORD`). A login lasts a week.
- **First search of the day:** the GPU sleeps when nobody searches. The first search wakes it, which takes one to two minutes; the page says so and waits. Opening a photo or the text box wakes it in advance. Once awake, a search takes about a second; the GPU goes back to sleep after 10 minutes without a search.

What a person can do:

1. **Home.** Take a photo (the phone's camera opens), upload an image, or describe what they see in words. Six example photos run a search in one tap: two monuments and two museum objects photographed by Wikimedia Commons users, a drawing, and a 1940s photo of a Jeu de Paume room where they draw a box around one tapestry.
2. **Photo.** Search the whole photo, or draw a box with a finger or the mouse to search one part (a painting on a wall, a detail). The box is cut out in the browser.
3. **Position (step H2).** A photo that carries GPS in its EXIF uses it on its own: the photo screen says "Position de la photo utilisée" with a "Ne pas l'utiliser" link. Otherwise a toggle, "Utiliser ma position", asks the browser (and so the visitor) for the position on the tap, not before. Buildings and objects near that position rank higher (a soft prior: bonus `exp(−d / 1 km)` on the fused score); far ones stay, so nothing breaks away from anything known. The results show "Près de vous" or "Près du lieu de la photo". The position goes with that one search; it is not logged or stored. On T2 phone photos of buildings, the right building moves from 32% to 97% in the top 10 (results.md, H2). Phones often strip GPS from a photo a web page reads, so the toggle is the main way in.
4. **Results.** One card per building or object, best first: the image that matches best (marked "Correspond : revers", "détail"… when it is not the record's main image), title, place, collection, a link to POP. Mémoire photos of one building form one card ("12 photos"). Filters (collection, region, département, domain, period) narrow the search before ranking. Words refine a photo search ("à Toulouse"). The page shows, quietly, whether it took the photo for a building or an object. When the best result is not close enough, it says **"Pas de correspondance sûre dans les collections"** and lists what the photo "ressemble à".
5. **Record.** Every photo of the building or object, key facts, the description, and links to the full POP records.

Light and dark themes: "Affichage" in the header or footer.

## How it works

| Part | What runs |
|---|---|
| `Search` (Modal GPU class, L4) | EmbeddingGemma 2 (A7) and MegaLoc (A3) embed the photo with the same code and preprocessing as the R1 gallery runs (`miroir.remote.search` reuses `miroir.remote.models`); words go through EmbeddingGemma 2's "SearchQuery" prompt. Gallery vectors sit on the GPU. At most one container; it sleeps after 10 minutes |
| `web` (Modal CPU container) | FastAPI: the page, login, record pages, example photos. At most one container |
| Ranking (`miroir.web.engine`, NumPy) | The evaluation's steps: cosine per image, best image per entity, min-max per model, G4 weights. A logistic regression on the photo's A7 vector guesses building or object; buildings use A7 alone, objects 0.2 A7 + 0.8 A3 (R2 G4). Words: A7 text vector against the image vectors, mixed half and half with French BM25 over record text (the H1 index); words refining a photo count half |
| "Not found" | Confidence is the routed mix of the top result's cosines, as T8 measures it. One cut-off per guessed type, set so that at most 10% of T8 tune searches (whole entity removed from the index) still look sure. The values and their source are in the bundle's `routing.json` |
| Index | A bundle built on the Mac (`miroir web build-index`), stored in the `miroir-web-index` Modal volume. Which bundle is served is one line: `WEB_INDEX` in `src/miroir/config.py` |

Images on the cards come straight from the POP bucket (Rose Valland scans through the IIIF service, scaled, since some files weigh 20 MB). Query photos go to our Modal container only, never to an outside model; they are not stored. The logs keep a hash of each photo, the words, the filters and the records returned (technical spec §9).

Weights, cut-offs and buckets are data (`routing.json`), not code: a third bucket (archive building, colour building, object) needs only a new entry with `greyscale: true` or `false`; the server already measures each photo's chroma spread (`images.colour`, the statistic G5 routes on). Extra gallery photos of an entity from Commons (image ids `commons-…` in the vector runs) are picked up by `build-index` and shown with their author and licence.

## Scores of the deployed stack

Index `full` (since 2026-10-13): every image of the July exports the Modal runs could fetch, 136,731 images of 101,039 records, 63,085 buildings and objects, publishable or not (the site sits behind a login, functional spec §8). Vectors from the full-sample runs (`A7-embeddinggemma-2-512-full`, `A3-megaloc-322-full`, MegaLoc at 8,448 dims), type guess from `miroir index guess`. Tune halves, the site's own ranking (the bundle's `report.json`):

| Set | Queries | Hit@1 | Hit@10 | "Sure" answers | Right among sure |
|---|---|---|---|---|---|
| T1 archive building | 138 | 34.1 | 45.7 | 24.6% | 73% |
| T3 painting in a room (crop) | 99 | 55.6 | 86.9 | 8.1% | 0% |
| T5 museum object, other view | 493 | 27.2 | 43.2 | 10.3% | 22% |
| T2 phone photo of a building (Commons) | 516 | 9.7 | 18.2 | 0% | |
| T2b museum work (Commons) | 291 | 82.1 | 94.5 | 4.1% | 75% |

| Not-found set | Searches still answered "sure" |
|---|---|
| T8 (in-collection photos, entity removed) | 9.6% (target: at most 10%) |
| T8c (Commons photos with no record) | 0% |

Cut-offs: 0.934 for buildings (A7 cosine), 0.824 for objects (0.2 A7 + 0.8 A3). The gallery is six times the screening one, so every score drops (T5 most: Saint-Omer's coin and medal series look alike) and a "sure" object is often a near twin, not the right record.

The first index, `screening-v2` (21,785 publishable images), scored T1 50.7, T3 89.9, T5 61.2, T2 24.8, T2b 94.3 hit@10; switch back with `WEB_INDEX = "screening-v2"`.

End-to-end check on the deployed site (2026-10-13): photos fetched from POP and Commons and posted to `/api/search`. Collection photos (T1, T3, T5) find their own record first with confidence 1.00 (the query vector equals the stored one, so preprocessing matches the gallery's); Commons photos (T2, T2b) get the confidence computed offline from their stored query vectors (0.914, 0.859, 0.55…) and the right record first or second. Words: "nature morte au lièvre" puts "Nature morte avec homard et lièvre" and "Nature morte au lièvre" first; "Pont du Gard" first; English words do less well.

Timings: a warm search takes about 200 ms on the GPU and about 0.9 s from a browser (upload included); words alone 0.5 s. A cold start takes 50–110 s (container start, models about 25 s, gallery 2.5 GB to the GPU, BM25 over 101,000 records).

## Run and deploy

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

An L4 with 4 CPUs and 16 GB costs about $1.12 an hour while awake. Each wake-up costs about 12 minutes of it (start plus 10 idle minutes), about $0.22; the web container about $0.06 an hour while awake. Nothing runs when nobody uses the site. Searches are limited to 30 a minute per session, logins to 5 a minute per address.

## Known limits

- **Position.** Rose Valland works have no place (looted and moved), and Mémoire objects sit at their commune's centre, so with the position on they rank below precise neighbours. The visitor can turn the position off.

- **Index.** The 101,000 records of the July exports, not all of POP: most tester photos of places outside them will find nothing.
- **Phone photos of buildings.** T2: the right building is in the top 10 for a fifth of photos, and none reaches the "sure" bar. A3 does better than A7 on these photos (31.0 against 21.9 hit@10), but the router sends buildings to A7, which wins on archive photos; a third bucket for colour photos of buildings is being tuned.
- **"Not found" is cautious, and weak for objects.** At 10% wrong-but-confident, a quarter of in-index archive photos and a tenth of objects get a "sure" answer, and only a fifth of those objects are the right record.
- **Word search** weights (half A7, half BM25) are not tuned: T6 and T7 do not exist yet.
- **Period filter** for Mémoire reads the date of the photo, not of the building.
- **No object outlines (F4) yet:** the photo screen has the hook (`detectObjects` in `static/app.js`); people draw boxes by hand.
- **Cold start** of about two minutes after 10 idle minutes. A longer idle window or one warm container would remove it, at about $27 a day.
- **Type guess** read from the index agent's full-sample file; a three-bucket router (G5) can replace the two G4 buckets in `routing.json`.
