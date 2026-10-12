# MIROIR: the prototype website

A working version of the product in [functional-spec.md](functional-spec.md) §5–§8, on the open models chosen in [results.md](results.md) (R1, R2). French interface, built for phones, in the State design system (DSFR 1.15.4), with no State logo: it is a prototype, not an official service.

## Use it

- **Address:** https://simonaszilinskas--miroir.us-east.modal.run
- **Password:** one shared password. It is in the main checkout's `.env` (`/Users/simonaszilinskas/miroir/.env`, line `MIROIR_APP_PASSWORD`, never committed) and in the `miroir-app` Modal secret (`APP_PASSWORD`). A login lasts a week.
- **First search of the day:** the GPU sleeps when nobody searches. The first search wakes it, which takes about two minutes; the page says so and waits. Opening a photo or the text box wakes it in advance. Once awake, a search takes under a second; the GPU goes back to sleep after 10 minutes without a search.

What a person can do:

1. **Home.** Take a photo (the phone's camera opens), upload an image, or describe what they see in words. Six example photos run a search in one tap: two monuments and two museum objects photographed by Wikimedia Commons users, a drawing, and a 1940s photo of a Jeu de Paume room where they draw a box around one tapestry.
2. **Photo.** Search the whole photo, or draw a box with a finger or the mouse to search one part (a painting on a wall, a detail). The box is cut out in the browser.
3. **Results.** One card per building or object, best first: the image that matches best (marked "Correspond : revers", "détail"… when it is not the record's main image), title, place, collection, a link to POP. Mémoire photos of one building form one card ("12 photos"). Filters (collection, region, département, domain, period) narrow the search before ranking. Words refine a photo search ("à Toulouse"). The page shows, quietly, whether it took the photo for a building or an object. When the best result is not close enough, it says **"Pas de correspondance sûre dans les collections"** and lists what the photo "ressemble à".
4. **Record.** Every photo of the building or object, key facts, the description, and links to the full POP records.

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

Weights, cut-offs and buckets are data (`routing.json`), not code: a third bucket (archive building, colour building, object) needs only a new entry with `greyscale: true` or `false`; the server already measures each photo's colourfulness. Extra gallery photos of an entity from Commons (image ids `commons-…` in the vector runs) are picked up by `build-index` and shown with their author and licence.

## Scores of the deployed stack

Index `screening-v2`: the publishable screening gallery, 21,785 images, 19,883 records, 5,811 buildings and objects. Tune halves, the site's own ranking (from the bundle's `report.json`):

| Set | Queries | Hit@1 | Hit@10 | "Sure" answers | Right among sure |
|---|---|---|---|---|---|
| T1 archive building | 138 | 37.0 | 50.7 | 32.6% | 78% |
| T3 painting in a room (crop) | 99 | 56.6 | 89.9 | 12.1% | 33% |
| T5 museum object, other view | 443 | 39.7 | 61.2 | 16.7% | 57% |
| T2 phone photo of a building (Commons) | 516 | 13.8 | 24.8 | 0% | |
| T2b museum work (Commons) | 122 with an answer in the index | 86.9 | 94.3 | 8.2% | 100% |

| Not-found set | Searches still answered "sure" |
|---|---|
| T8 (in-collection photos, entity removed) | 9.5% (target: at most 10%) |
| T8c (Commons photos with no record) | 0% |

Cut-offs: 0.921 for buildings (A7 cosine), 0.750 for objects (0.2 A7 + 0.8 A3).

End-to-end check on the deployed site (2026-10-12): photos fetched from POP and Commons and posted to `/api/search`. Collection photos find their own record first with confidence 1.00 (the query vector equals the stored one, so preprocessing matches); Commons photos get exactly the confidence computed offline from the stored query vectors (0.914, 0.852, 0.341…), and the right record first. Text: "nature morte au lièvre" puts "Nature morte avec homard et lièvre" first; "a still life with a dead hare" finds it second; "Pont du Gard" first.

Timings: a warm search takes 170–200 ms on the GPU and about 0.9 s from a browser (upload included); words alone 0.5 s. A cold start takes about 110 s (container start, then models 21 s, gallery 2 s, BM25 1 s).

## Run and deploy

```sh
uv sync --extra api
uv run miroir web build-index   # NumPy on the Mac, about 1.5 minutes: writes ~/miroir-data/web/<index>/
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

- **Index.** About 22,000 publishable images, not the 137,775 of the July exports nor all of POP. Most tester photos will find nothing until the full index is in.
- **Phone photos of buildings.** T2: the right building is in the top 10 for a quarter of photos, and none reaches the "sure" bar. A3 does better than A7 on these photos (31.0 against 21.9 hit@10), but the router sends buildings to A7, which wins on archive photos; a third bucket for colour photos of buildings is being tuned.
- **"Not found" is cautious.** At 10% wrong-but-confident, only a third of in-index archive photos and a sixth of objects get a "sure" answer.
- **Word search** weights (half A7, half BM25) are not tuned: T6 and T7 do not exist yet.
- **Period filter** for Mémoire reads the date of the photo, not of the building.
- **No object outlines (F4) yet:** the photo screen has the hook (`detectObjects` in `static/app.js`); people draw boxes by hand.
- **Cold start** of about two minutes after 10 idle minutes. A longer idle window or one warm container would remove it, at about $27 a day.
- **Type guess** trained on the screening gallery with tune-query entities left out; retrain on the full index.
