# MIROIR: the prototype website

A working version of the product in [functional-spec.md](functional-spec.md) §5–§8, on the open models chosen in [results.md](results.md) (R1, R2). French interface, built for phones, in the State design system (DSFR 1.15.4), with no State logo: it is a prototype, not an official service.

## Use it

- **Address:** https://simonaszilinskas--miroir.us-east.modal.run
- **Access:** one shared password, created with `miroir web secret` and kept out of the repository (in `.env` as `MIROIR_APP_PASSWORD` and in the `miroir-app` Modal secret as `APP_PASSWORD`). Ask the project owner for it. A login lasts a week.
- **First search of the day:** the GPU sleeps when nobody searches. The first search wakes it, which takes one to two minutes; the page says so and waits. Opening a photo or the text box wakes it in advance. Once awake, a search takes about a second; the GPU goes back to sleep after 10 minutes without a search.

What a person can do:

1. **Home** (`#/`). The title "Qu'avez-vous sous les yeux ?" and two DSFR tiles: **"Recherche simple"** (`#/simple`, "Stable", a few seconds) and **"Recherche agentique"** (`#/agentique`, "Expérimental", about a minute). On wide screens, photos of POP works float in the side margins, at different depths, and change one at a time (`static/hero.js`: covers of publishable Joconde and Rose Valland records, shuffled on each visit; still when the visitor asks for less motion). On phones a few of them sit above the title.
   **Simple search** (`#/simple`). Take a photo (the phone's camera opens), upload an image, or describe what they see in words (four example searches under the box run in one tap). Six example photos run a search in one tap: two monuments and two museum objects photographed by Wikimedia Commons users, a drawing, and a 1940s photo of a Jeu de Paume room where a blue outline marks each work. Steps 2 to 6 below belong to this search.
2. **Photo.** A second after the photo appears, a blue numbered outline marks each work Miroir finds in it (F4), with a line such as "14 œuvres repérées. Touchez un cadre bleu pour la chercher." Tapping an outline searches that work alone. The page shows 20 outlines first, the biggest and most central ones (people frame what they care about); on a wall of pictures, "Afficher les N autres objets repérés" shows the rest (up to 40). A photo of a single object shows none: "Rechercher toute la photo" does the job. A work without an outline: draw a box with a finger or the mouse (a painting, a detail). Boxes and outlines are cut out in the browser.
3. **Position (step H2).** A photo that carries GPS in its EXIF uses it on its own: the photo screen says "Position de la photo utilisée" with a "Ne pas l'utiliser" link. Otherwise a toggle, "Utiliser ma position", asks the browser (and so the visitor) for the position on the tap, not before. Buildings and objects near that position rank higher (a soft prior: bonus `exp(−d / 1 km)` on the fused score); far ones stay, so nothing breaks away from anything known. The results show "Près de vous" or "Près du lieu de la photo". The position goes with that one search; it is not logged or stored. On T2 phone photos of buildings, the right building moves from 32% to 97% in the top 10 (results.md, H2). Phones often strip GPS from a photo a web page reads, so the toggle is the main way in.
4. **Results.** One list of cards, one per building or object, from the closest to the least close. Each card shows a small badge with its level: **"Très proche"** (green: most likely the right record), **"Proche"** (blue: maybe it, check) or **"Ressemblance"** (grey: looks alike, compare with your photo); a line above the list says what each means. Only the first card can be "Très proche" or "Proche" (see "Card levels" below). Then the POP image that matches best (marked "Correspond : revers", "détail"… when it is not the record's main image), title, place, collection, a link to POP. Mémoire photos of one building form one card ("12 photos", POP photos only). Filters (collection, region, département, domain, period) narrow the search before ranking. Words refine a photo search ("à Toulouse"). The page shows, quietly, whether it took the photo for a building or an object. When a building photo's first card is not "Très proche", the page says what helps: "Utiliser ma position", the commune with "Préciser avec des mots", a photo of the whole façade.
5. **Old photo on its card (archive print).** When the photo is a black-and-white print glued on a card with a caption, as Mémoire's are, the results say **"Cadre et légende détectés"** and show the caption Miroir read ("Légende lue : « BEAUNE (Côte d'Or) Eglise Notre-Dame… »"). Its words join the search. "Corriger la légende" opens a box to fix the text and search again; "Retirer la légende" searches with the photo alone. Other photos are not affected. On T1 archive prints the right building is in the top 10 for 89% of prints, against 46% without the caption (results.md, T1 archive).
6. **Record.** Every POP photo of the building or object, key facts, the description, and links to the full POP records. Commons photos are not shown ("Pas d'image dans POP pour cette notice" for the six buildings that have none).
7. **Agentic search** (`#/agentique`; `/chat` and the old `#/conversation` lead there). A photo, words, or one of the same six examples. The page asks "Que cherchez-vous ?" with four chips ("C'est quoi ?", "C'est où ?", "D'autres comme ça", "Qui l'a fait ?") and a free-text box. Mistral Large 4, an open model reached through OpenRouter, then leads the search over the live index: each step shows what it searched and why, and opens to its thumbnails. It may ask a question (with suggested answers, a free-text box and "Je ne sais pas") or ask the visitor to frame part of the photo. It ends with result cards, each with a short reason, or says plainly that the thing is not in the collections. A line under the title says the photo goes to that model. Spending caps per conversation and per day; past them the page offers the simple search. How it works: technical spec §8.1.

Wikimedia Commons photos stay behind the scenes: 10,221 of them in the index help find 2,726 buildings, but a card whose best match is a Commons photo shows the building's best-matching POP image instead, with no Commons label or credit. The home page's example photos from Commons are queries, not results: they keep their credit.

Light and dark themes: "Affichage" in the header or footer.

## How it works

| Part | What runs |
|---|---|
| `Search` (Modal GPU class, L4) | EmbeddingGemma 2 (A7) and MegaLoc (A3) embed the photo with the same code and preprocessing as the R1 gallery runs (`miroir.remote.search` reuses `miroir.remote.models`); words go through EmbeddingGemma 2's "SearchQuery" prompt. Gallery vectors sit on the GPU. At most one container; it sleeps after 10 minutes |
| `web` (Modal CPU container) | FastAPI: the page, login, record pages, example photos. At most one container |
| Ranking (`miroir.web.engine`, NumPy) | The evaluation's steps: cosine per image, best image per entity, min-max per model, G4 weights. A logistic regression on the photo's A7 vector guesses building or object; buildings use A7 alone, objects 0.2 A7 + 0.8 A3 (R2 G4). Words: A7 text vector against the image vectors (0.9), mixed with French BM25 over record text (0.1), weights from T7 (results.md, step B); words refining a photo count half. When no record holds any of the words, the page says so and shows the closest images |
| Archive prints | `miroir.images.mount.find_picture` (NumPy, about 6 ms) looks for a grey print on a plain card with writing around it. Only then does Qwen3-VL 4B Instruct (open, same GPU container, loaded in a thread) copy the caption line that names the place; its words join the photo search at the photo's weight (`SITE_ARCHIVE` in `miroir.web.archive`), searched with their own mix, text vector 0.3 and BM25 0.7 (`caption_a7`, `caption_bm25` in `routing.json`), since captions are mostly place names. The page can send the caption back corrected or empty (`caption` in `POST /api/search`), and the model is not asked again |
| Outlines (F4) | Grounding DINO tiny (step C winner) in the same GPU container, loaded in a thread beside the search models so it never delays a search. It sees the photo at 1,024 px with step C's prompts; `miroir.outlines.pick_outlines` keeps sure boxes, drops repeats and the whole photo, sends the 40 surest, biggest and most central first, boxes around several works last, and none when the surest box covers 40% of the photo. `POST /api/detect` |
| Card levels | `miroir.web.levels`: the first card's score (photo: the routed mix of its raw cosines; words: the fused score) and its gap to the best other card of the top 30, against two pairs of thresholds per route in the bundle's `levels.json` (`miroir web levels`). Every other card is "Ressemblance". A bundle without `levels.json` falls back on the older sure rule (`routing.json` cut-off, `sure.json` margin) for the first card |
| Agentic search (H5) | `miroir.web.chat` on the web container: Mistral Large 4 through OpenRouter, with the tools of `miroir.web.chat_tools`; the searches run in the GPU container (`Search.tool_search`). Technical spec §8.1 |
| Index | A bundle built on the Mac (`miroir web build-index`), stored in the `miroir-web-index` Modal volume. Which bundle is served is one line: `WEB_INDEX` in `src/miroir/config.py` |

Images on the cards come straight from the POP bucket (Rose Valland scans through the IIIF service, scaled, since some files weigh 20 MB). Query photos go to our Modal container only, never to an outside model; they are not stored. The logs keep a hash of each photo, the words, the filters and the records returned (technical spec §9).

Weights, cut-offs and buckets are data (`routing.json`), not code: a third bucket (archive building, colour building, object) needs only a new entry with `greyscale: true` or `false`; the server already measures each photo's chroma spread (`images.colour`, the statistic G5 routes on). Extra gallery photos of an entity from Commons (image ids `commons-…` in the vector runs) are picked up by `build-index`; they count in the ranking but are never shown.

## Scores of the deployed stack

Index `full-f2` (since 2026-10-13): every image of the July exports the Modal runs could fetch (136,731 images of 101,039 records), plus 10,221 Commons photos of 2,726 Mérimée buildings (step F2, `miroir testsets commons-gallery --gallery full`): 146,952 images, 63,091 buildings and objects. No Commons photo used as a test query (T2, T2b, T8c, tune or sealed), no near-copy of one and no other frame of its visit is among them. Routing G5 (`routing.json`): a black-and-white building photo goes to A7 alone, a colour building photo to A3 alone, an object 0.2 A7 + 0.8 A3. Tune halves, the site's own ranking (the bundle's `report.json`), without the photo's position:

| Set | Queries | Hit@1 | Hit@3 | Hit@10 |
|---|---|---|---|---|
| T1 archive building | 138 | 34.1 | 38.4 | 45.7 |
| T1 with its caption read | 138 | 71.0 | | 89.1 |
| T3 painting in a room (crop) | 99 | 55.6 | 76.8 | 86.9 |
| T5 museum object, other view | 493 | 27.2 | 35.1 | 43.2 |
| **T2 phone photo of a building (Commons)** | 516 | **25.8** | **32.6** | **36.8** |
| T2b museum work (Commons) | 291 | 82.1 | 88.7 | 94.5 |
| T7 words for a hidden image (French) | 500 | 7.6 | | 17.6 |
| T7en the same words in English | 105 | 7.6 | | 23.8 |

T2 before F2 and G5 (index `full`, G4 routing): hit@1 9.7, hit@10 18.2, no sure answer. G5 alone, without the Commons photos: hit@1 19.4, hit@10 26.9.

T1 with its caption read: caption words now have their own weights, text vector 0.3 and BM25 0.7, while typed words keep 0.9 / 0.1. With the typed words' weights captions scored 62.3 hit@1 and 79.7 hit@10; caption words are mostly place names, which BM25 finds and the text vector does not (results.md, caption words weighed apart).

### Card levels

Set 2026-10-13 by `miroir web levels` on the tune halves, index `full-f2`, from the stored query vectors (NumPy on the Mac, no model, no spend; per-search detail in `eval/results/R3/G-detail/levels.json`). Each route's thresholds apply to the first card: score at least *s* and gap to the best other card of the top 30 at least *g*. Of the pairs keeping the promises ("Très proche" right at least 80% of the time on the route and 70% on each set with ten or more of them, on at most 10% of each not-found set; "Proche" right at least 40%, with "Proche" or better on at most 20% of each not-found set), the two chosen label the most right first cards. Why only the first card: on every route, cards 2 to 24 are the right record 0.2–4% of the time whatever their score.

| Route | "Très proche" (s, g) | Shown on | Right | "Proche" (s, g) | Shown on | Right | First card "Ressemblance", right |
|---|---|---|---|---|---|---|---|
| Building, black and white (T1) | 0.920, 0.010 | 30% | 81% | 0.850, 0.010 | 14% | 40% | 7% |
| Building, colour (T2) | 0.281, 0.005 | 30% | 80% | 0.141, 0.025 | 18% | 43% | 3% |
| Object (T3, T5, T2b, T2 guessed object) | any, 0.090 | 30% | 85% | any, 0.045 | 11% | 59% | 15% |
| Words (T7) | never | 0% | | 0.900, 0.070 | 5% | 46% | 6% |

Objects per set: "Très proche" right 98% on T2b, 81% on T2, 73% on T3 and T5. No gap reaches 80% for words (the biggest gaps, 0.08 and more, are right 58% of the time on T7 descriptions), so a word search never shows "Très proche"; T7 holds descriptions, not names, so a name such as "Pont du Gard" may well deserve more than "Proche". The object "Très proche" rule is the old sure margin (0.09) again. Scores per search are on disk for every photo route, so no route needed a stand-in rule. These are the halves the thresholds were set on: expect a little less on new photos.

| Not-found set (record not in the index) | First card "Très proche" | "Proche" or better |
|---|---|---|
| T8 (in-collection photos, entity removed; 249) | **8.4%** (archive 8.1%, object 8.7%) | 19% |
| T8c (Commons photos with no record; 150) | 2.0% (colour building 3.8%, object 1.0%) | 11% |

Before: one "sure" or "not sure" answer for the whole search (cut-offs 0.934 archive, 0.247 colour; object margin 0.09), with T8 9.6% and T8c 3.3% still "sure".

The first index, `screening-v2` (21,785 publishable images), scored T1 50.7, T3 89.9, T5 61.2, T2 24.8, T2b 94.3 hit@10; switch back with `WEB_INDEX = "screening-v2"`.

End-to-end check on the deployed site (2026-10-13): photos fetched from POP and Commons and posted to `/api/search`. Collection photos (T1, T3, T5) find their own record first with confidence 1.00 (the query vector equals the stored one, so preprocessing matches the gallery's); Commons photos (T2, T2b) get the confidence computed offline from their stored query vectors (0.914, 0.859, 0.55…) and the right record first or second. Words: "nature morte au lièvre" puts "Nature morte avec homard et lièvre" and "Nature morte au lièvre" first; "Pont du Gard" first; English words do less well.

Visitor check of phone photos of buildings (2026-10-13, on the deploy of `ccb4142`: G5 and `full-f2`, before captions). Ten T2 tune photos sent to the deployed search, seven without GPS, three with it (seeded draw). Before (index `full`, G4): the right building first for 2 of 10, in the top 10 for 3. After: **first for 7 of 10, top 3 for 7, top 10 for 8**; four of the seven were "sure" (the page then said so; card levels replaced that wording since). The three photos with GPS were all found first, with 1, 5 and 51 known places within the search radius. The two misses (a Paris hôtel particulier and an Avignon hôtel, both street façades among many alike) got no sure answer and the closest buildings, nearly all in the right town. After a missed building photo the page now says what helps: turn on "Utiliser ma position" or add the commune with "Préciser avec des mots", and photograph the whole façade. Words: "Pont du Gard", "moulin à vent", "nature morte au lièvre" and "cathedral rose window" (the four chips) all put a fitting record first. The Jeu de Paume example drew 14 outlines (the detector found 14; up to 20 are shown).

Timings: outlines take 0.27 s on the GPU once warm and about 0.7 s from the Mac (upload included); a warm search takes about 200 ms on the GPU and about 0.9 s from a browser (upload included); words alone 0.5 s. An archive print adds its caption reading: 1.4 s median on the GPU (2.1 s at the 95th percentile), so about 2.3 s from a browser; other photos add 6 ms. A cold start takes 50–110 s (container start, models about 25 s, gallery 2.5 GB to the GPU, BM25 over 101,000 records).

### Photos from outside, tried on the live site

Two photos that are in neither the index nor any test set (neither is among the 12,094 Commons files of `commons.parquet` and `commons-gallery-full.parquet`), loaded through each page's upload box on 2026-10-19:

| Photo | Simple search | Agentique search |
|---|---|---|
| West front of Notre-Dame-la-Grande, Poitiers ([Benjism89, 2025, CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Poitiers_-_%C3%89glise_Notre-Dame_la_Grande_-_02.jpg)) | PA00105596 first, "Très proche"; then 23 Romanesque churches of the region as "Ressemblance" | Same record, alone, in about 30 s after looking at 4 thumbnails; names the church and the 12th century, but calls the carved mandorla an oval rose window |
| Detail of Corot's *Château de Wagnonville* (gatehouse and bridge, about a quarter of the canvas, no frame), cut from a museum visitor's photo ([Edelseider, CC BY-SA 2.0](https://commons.wikimedia.org/wiki/File:Douai_(Nord)_-_Mus%C3%A9e_de_la_Chartreuse_-_%22Le_ch%C3%A2teau_de_Wagnonville%22_(Jean-Baptiste_Camille_Corot,_1796-1875).jpg)) | Joconde 06190026809 first, "Très proche"; then Douai and Saint-Omer landscapes | Same record, alone, after 3 searches (image, "more like this", words); answers Corot, third quarter of the 19th century, musée de la Chartreuse |

Both first searches met a sleeping engine and took over a minute.

## Run and deploy

**Order when code and bundle change together.** Deploy first, then upload: `miroir web deploy`, then `miroir web upload`. New code reads older bundles (a bundle without `levels.json` falls back on the older sure rule; a `routing.json` without `caption_a7` and `caption_bm25` takes the caption weights from `miroir.web.routing`, 0.3 / 0.7), but older code refuses a `routing.json` that holds the caption weights. After a deploy, check one photo search on the site. The first wake-up after a deploy that adds a model downloads its weights into the weights volume once (Qwen3-VL 4B: about 9 GB).

**Card levels.** To set them again after a rebuild or on another index: `miroir web levels` (writes `levels.json` into the bundle and a copy `~/miroir-data/index/levels-<index>.json` that `build-index` puts back), then `miroir web upload`.

**Position (H2).** The `full-f2` bundle holds `places.npz` (one position per entity), so the site uses the photo's GPS position or the visitor's. To add positions to another built bundle without rebuilding: `miroir web places --index <name> --upload`, then `miroir web deploy`. A bundle without `places.npz` ignores positions.

**Caption reader.** It loads in a thread after the search models and resets torch's default dtype afterwards; loading it beside EmbeddingGemma 2 without that reset left the default at bfloat16 and broke every photo search ("expected mat1 and mat2 to have the same dtype").

```sh
uv sync
uv run miroir web build-index   # NumPy on the Mac, about 3 minutes for the full index: writes ~/miroir-data/web/<index>/
uv run miroir web upload        # copies the bundle to the miroir-web-index volume
uv run miroir web secret        # once: password in .env and in the miroir-app secret (--rotate for a new one)
uv run miroir web deploy        # deploys the miroir-web app and prints the address
uv run miroir web dev           # http://localhost:8000 with fake models (no model runs on the Mac)
uv run miroir web dev --real    # the same page searching with the deployed GPU (costs GPU time)
```

To switch index: add it to `INDEXES` in `src/miroir/web/bundle.py`, set `WEB_INDEX` in `src/miroir/config.py`, then build-index, upload and deploy.

**Agentic search (H5, `#/agentique`).** Needs the OpenRouter key in its own Modal secret and the list of images that may go to the model in the bundle:

```sh
uv run miroir web publishable --upload   # writes publishable.txt.gz into the bundle, copies it to the volume
uv run miroir web chat-secret            # copies OPENROUTER_API_KEY from .env into the miroir-openrouter secret
uv run miroir web deploy
```

Each conversation is logged as JSON lines with no images (technical spec §8.1). To read the deployed site's logs on the Mac:

```sh
uv run miroir web chat-logs   # copies chat-logs/ from the miroir-web-index volume to ~/miroir-data/chat-logs/hosted/
```

The deploy fails until the `miroir-openrouter` secret exists. Caps: `MIROIR_CHAT_CONVERSATION_USD` (default 0.10) and `MIROIR_CHAT_DAY_USD` (default 2), read from the environment; to change them on Modal, add them to the `miroir-openrouter` secret. Locally, `miroir web dev` runs the agentic search with Mistral when `OPENROUTER_API_KEY` is in `.env` (same caps), or with a scripted stand-in with `--fake-llm` or without a key.

## Cost

An L4 with 4 CPUs and 24 GB costs about $1.18 an hour while awake. Each wake-up costs about 12 minutes of it (start plus 10 idle minutes), about $0.22; the web container about $0.06 an hour while awake. Nothing runs when nobody uses the site. Searches are limited to 30 a minute per session, logins to 5 a minute per address.

The agentic search also pays OpenRouter: a turn costs about $0.02–0.04. It stops at $0.10 a conversation and $2 a day for all visitors (`MIROIR_CHAT_CONVERSATION_USD`, `MIROIR_CHAT_DAY_USD`).

## Known limits

- **Position.** Rose Valland works have no place (looted and moved), and Mémoire objects sit at their commune's centre, so with the position on they rank below precise neighbours. The visitor can turn the position off.

- **Index.** The 101,000 records of the July exports, not all of POP: most tester photos of places outside them will find nothing.
- **Phone photos of buildings.** Without the photo's position, the right building comes first for a quarter of photos (25.8%), in the top 3 for a third (32.6%) and in the top 10 for 36.8%. A building whose Commons photos are in the index is found far more often than one without (58.6% against 24.9% hit@10 on the screening gallery): 1,198 of 3,924 buildings have none. Interiors and close details stay hard. With the photo's GPS position (H2) the right building is in the top 10 for most photos.
- **Levels are cautious.** About 30% of in-index photo searches get a "Très proche" first card and 11–18% a "Proche" one; the rest see "Ressemblance" cards to compare. Words never reach "Très proche". A caption read on an archive print raises T1 hit@1 from 34% to 71%, but the level reads the photo's cosines only, so it undersells those searches.
- **Word search** finds the right record in the top 10 for about one description in six (T7: 17.6%; half and half scored 8.0%). Names work ("Pont du Gard" comes first); descriptions of common things (a church, a coin) match too many records, and Mémoire buildings are found only 7% of the time. A region in the words ("bourgogne") does not narrow the search; the region filter does. T6, written by hand, does not exist yet ([t6-guide.md](t6-guide.md)).
- **Period filter** for Mémoire reads the date of the photo, not of the building.
- **Outlines on a crowded wall.** On the Jeu de Paume room photos the right record is in the top 5 for 87% of works when every outline is shown, 78% with the 20 shown first (it was 52% with the 12 surest; results.md, step C): on the densest walls the visitor still needs "Afficher les N autres". Outlines' names come from the detector's prompts and are often wrong (it calls a statue a stained glass window); the page only uses them in labels for screen readers.
- **Keypoint re-ranking (E1) is not on the site:** it lifts T3 hit@1 by 12 points but adds 3 s per search (6 s at the 95th percentile), mostly reading candidate images.
- **Cold start** of about two minutes after 10 idle minutes. A longer idle window or one warm container would remove it, at about $27 a day.
- **Type guess** trained on the full-sample index (`miroir.index.guess`). It calls more than half of phone photos of buildings objects (it partly learned colour); they then get 0.2 A7 + 0.8 A3, close to what a colour building gets.
