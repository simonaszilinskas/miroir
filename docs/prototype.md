# MIROIR: the prototype website

A working version of the product in [functional-spec.md](functional-spec.md) §5–§8, on the open models chosen in [results.md](results.md) (R1, R2). French interface, built for phones, in the State design system (DSFR 1.15.4), with no State logo: it is a prototype, not an official service.

## Use it

- **Address:** the site runs on a Scaleway GPU machine, powered on for demos (since 2026-11-03; before, on Modal). Address and how it runs: see "Hosting" below.
- **Access:** one shared password, created with `miroir web secret` and kept out of the repository (in `.env` as `MIROIR_APP_PASSWORD`; `miroir demo install` sends it to the machine as `APP_PASSWORD`). Ask the project owner for it. A login lasts a week.

What a person can do:

1. **Home = search** (`#/`, since 2026-11-03). The title "Qu'avez-vous sous les yeux ?", one sentence on what Miroir does, then one search card: a large photo drop zone (drop, click, or take a photo on a phone; six example photos under it, credited), a text field under it ("Ou décrivez ce que vous cherchez…"; a photo, words or both are enough), a "Filtres" button (collection, région, département, domaine, époque, "Près de moi"; a count when some are set), a small **Rapide / Approfondi** switch and a large "Chercher" button. Results come on the same page, under the card; the filters show there as tags a tap removes (the search runs again). "Rapide" is the usual search (steps 2 to 6 below). "Approfondi" is the agentic search in one go (step 7). After a Rapide search, "Aller plus loin" runs Approfondi on the same photo, part, words and filters. On wide screens, photos of POP works float in the side margins (covers of publishable Joconde and Rose Valland records, drawn on each visit; still when the visitor asks for less motion); on phones a few sit above the title. The page is a React app in `web-ui/` with the DSFR through `@codegouvfr/react-dsfr` (see "Run and deploy").
   The earlier page (two tiles, `#/simple` and `#/agentique`, plain JavaScript in `src/miroir/web/static/`) is gone; `/chat` leads to the home.
2. **Photo.** A second after the photo appears, a blue numbered outline marks each work Miroir finds in it (F4), with a line such as "14 œuvres repérées. Touchez un cadre bleu pour la chercher." Tapping an outline searches that work alone. The page shows 20 outlines first, the biggest and most central ones (people frame what they care about); on a wall of pictures, "Afficher les N autres objets repérés" shows the rest (up to 40). A photo of a single object shows none: "Rechercher toute la photo" does the job. A work without an outline: draw a box with a finger or the mouse (a painting, a detail). Boxes and outlines are cut out in the browser.
3. **Position (step H2).** A photo that carries GPS in its EXIF uses it on its own: the photo screen says "Position de la photo utilisée" with a "Ne pas l'utiliser" link. Otherwise a toggle, "Utiliser ma position", asks the browser (and so the visitor) for the position on the tap, not before. Buildings and objects near that position rank higher (a soft prior: bonus `exp(−d / 1 km)` on the fused score); far ones stay, so nothing breaks away from anything known. The results show "Près de vous" or "Près du lieu de la photo". The position goes with that one search; it is not logged or stored. On T2 phone photos of buildings, the right building moves from 32% to 97% in the top 10 (results.md, H2). Phones often strip GPS from a photo a web page reads, so the toggle is the main way in.
4. **Results.** One list of cards, one per building or object, from the closest to the least close. Each card shows a small badge with its level: **"Très proche"** (green: most likely the right record), **"Proche"** (blue: maybe it, check) or **"Ressemblance"** (grey: looks alike, compare with your photo); a line above the list says what each means. Only the first card can be "Très proche" or "Proche" (see "Card levels" below). Then the POP image that matches best (marked "Correspond : revers", "détail"… when it is not the record's main image), title, place, collection, a link to POP. Mémoire photos of one building form one card ("12 photos", POP photos only). Filters (collection, region, département, domain, period) narrow the search before ranking. Words refine a photo search ("à Toulouse"). The page shows, quietly, whether it took the photo for a building or an object. When a building photo's first card is not "Très proche", the page says what helps: "Utiliser ma position", the commune with "Préciser avec des mots", a photo of the whole façade.
5. **Old photo on its card (archive print).** When the photo is a black-and-white print glued on a card with a caption, as Mémoire's are, the results say **"Cadre et légende détectés"** and show the caption Miroir read ("Légende lue : « BEAUNE (Côte d'Or) Eglise Notre-Dame… »"). Its words join the search. "Corriger la légende" opens a box to fix the text and search again; "Retirer la légende" searches with the photo alone. Other photos are not affected. On T1 archive prints the right building is in the top 10 for 89% of prints, against 46% without the caption (results.md, T1 archive).
6. **Record.** Every POP photo of the building or object, key facts, the description, and links to the full POP records. Commons photos are not shown ("Pas d'image dans POP pour cette notice" for the six buildings that have none).
7. **Approfondi (agentic search, one go).** The same photo (or the part the visitor framed), words and filters go to Mistral Large 4 (Mistral's API, or OpenRouter), which leads the search over the live index without asking the visitor anything (`oneshot` in `POST /api/chat/open`: the tools that wait for the visitor are not offered, and the model is told so; the page's filters reach the first look and the model, which keeps them in its own searches). Its image search can work on a transformed copy of the photo, made on the server: cropped to one of the detector's numbered outlines (listed to it with their boxes), to a box or to a cell of a 3 by 3 grid; turned, mirrored, or turned black and white to meet Mémoire's archive prints; each counted as a search, and the model sees each transformed image after its search (`miroir.images.transform`, `transform` in `image_search`). Each step it takes shows live under the card (« Recadre sur le contour 3 (sculpture), passe en noir et blanc, puis cherche par image »), then folds into "Comment le modèle a cherché". It ends with its summary and result cards, each with a short reason, or says plainly that nothing matches. A line under the card says the photo goes to that model. Spending caps per search and per day; past them the page offers the Rapide search. How it works: technical spec §8.1.

Wikimedia Commons photos stay behind the scenes: 10,221 of them in the index help find 2,726 buildings, but a card whose best match is a Commons photo shows the building's best-matching POP image instead, with no Commons label or credit. The home page's example photos from Commons are queries, not results: they keep their credit.

Light and dark themes: "Affichage" in the header or footer.

## How it works

| Part | What runs |
|---|---|
| Search (`miroir.remote.search`, on the machine's L4 GPU) | EmbeddingGemma 2 (A7) and MegaLoc (A3) embed the photo with the same code and preprocessing as the R1 gallery runs (`miroir.remote.search` reuses `miroir.remote.models`); words go through EmbeddingGemma 2's "SearchQuery" prompt. Gallery vectors sit on the GPU. One GPU call at a time |
| Site (`miroir.remote.site`, same process) | FastAPI: the page, login, record pages, example photos |
| Ranking (`miroir.web.engine`, NumPy) | The evaluation's steps: cosine per image, best image per entity, min-max per model, G4 weights. A logistic regression on the photo's A7 vector guesses building or object; buildings use A7 alone, objects 0.2 A7 + 0.8 A3 (R2 G4). Words: A7 text vector against the image vectors (0.9), mixed with French BM25 over record text (0.1), weights from T7 (results.md, step B); words refining a photo count half. When no record holds any of the words, the page says so and shows the closest images |
| Archive prints | `miroir.images.mount.find_picture` (NumPy, about 6 ms) looks for a grey print on a plain card with writing around it. Only then does Qwen3-VL 4B Instruct (open, same GPU, loaded in a thread) copy the caption line that names the place; its words join the photo search at the photo's weight (`SITE_ARCHIVE` in `miroir.web.archive`), searched with their own mix, text vector 0.3 and BM25 0.7 (`caption_a7`, `caption_bm25` in `routing.json`), since captions are mostly place names. The page can send the caption back corrected or empty (`caption` in `POST /api/search`), and the model is not asked again |
| Outlines (F4) | Grounding DINO tiny (step C winner) on the same GPU, loaded in a thread beside the search models so it never delays a search. It sees the photo at 1,024 px with step C's prompts; `miroir.outlines.pick_outlines` keeps sure boxes, drops repeats and the whole photo, sends the 40 surest, biggest and most central first, boxes around several works last, and none when the surest box covers 40% of the photo. `POST /api/detect` |
| Card levels | `miroir.web.levels`: the first card's score (photo: the routed mix of its raw cosines; words: the fused score) and its gap to the best other card of the top 30, against two pairs of thresholds per route in the bundle's `levels.json` (`miroir web levels`). Every other card is "Ressemblance". A bundle without `levels.json` falls back on the older sure rule (`routing.json` cut-off, `sure.json` margin) for the first card |
| Agentic search (H5) | `miroir.web.chat` in the site's process: Mistral Large 4 on Mistral's API (through OpenRouter when only that key is set), with the tools of `miroir.web.chat_tools`, `read_record` among them; the searches run on the GPU (`SearchService.tool_search`). Technical spec §8.1 |
| Index | A bundle built on the Mac (`miroir web build-index`) or on a Scaleway machine (`pop-all`), stored in the index bucket (`miroir web upload`). Which bundle the demo machine serves is one line: `DEMO_INDEX` in `src/miroir/config.py`; `WEB_INDEX` is the one the Mac reads |

Images on the cards come straight from the POP bucket (Rose Valland scans through the IIIF service, scaled, since some files weigh 20 MB). Query photos go to our own GPU machine only, never to an outside model; they are not stored. The logs keep a hash of each photo, the words, the filters and the records returned (technical spec §9).

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

### Index of all of POP (`pop-all`, built and served since 2026-11-03)

Every POP image the models could read: 2,257,225 image rows (Commons photos included), 2,135,437 records, 885,596 buildings and objects; 99.86% of records with an image have vectors (docs/search-levels.md, "What step 3 built"). Built on Scaleway with `miroir scw pop-index` (docs/developpement.md, "Scaleway"), 27.4 GB in the `miroir-index` bucket under `pop-all/`. MegaLoc is kept as a 1,024-number shortlist on the GPU (4.6 GB) and whole in int8 in memory (19.1 GB), so serving it needs about 20 GB of memory for the int8 vectors alone. The card levels and the sure rule are those of `full-f2`, not set again on this index.

Tune halves, the bundle's own `report.json` (the July index `full-f2` in brackets):

| Set | Hit@1 | Hit@3 | Hit@10 |
|---|---|---|---|
| T1 archive building | 45.7 (34.1) | 52.2 (38.4) | 59.4 (45.7) |
| T3 painting in a room | 54.5 (55.6) | 72.7 (76.8) | 82.8 (86.9) |
| T5 museum object, other view | 26.9 (27.2) | 34.7 (35.1) | 41.8 (43.2) |
| T2 phone photo of a building | 53.7 (25.8) | 59.1 (32.6) | 64.9 (36.8) |
| T2b museum work (Commons) | 78.0 (82.1) | 83.5 (88.7) | 88.0 (94.5) |

Not-found sets, first card wrongly sure: T8 9.6%, T8c 4.7%. Buildings gain most, since many more of their photos are now in the index; objects lose a little among 15 times more images.

Visitor check on the demo machine (2026-11-03, `pop-all`, over HTTPS from the Mac, upload included):

- **Vétheuil (VET).** The target, PM95000723 (photo APMH047981), comes **first** for the two sale-catalogue pages (VET-1 "Très proche", VET-3 "Proche") and **fourth** for today's colour photo of the altarpiece (VET-2), behind three other churches. Tapping the outline of the carved group on VET-1 also finds it first; none of VET-2's outlines does.
- **Other photos.** Five PARTS details find their whole first (the detail photo is itself in the index, so this checks only that the path works); two T2 phone photos of buildings find their building first; two T2b museum works come first and second; two T1 archive prints come first, their captions read ("VIEUX (Tarn) Église…").
- **Time.** A photo search takes 0.3 s, a print with its caption 1.6–2.3 s, words 0.25 s, outlines 0.3–0.4 s. The record page of PM95000723 shows 24 images and 40 records.
- **Agentic search.** One conversation on VET-1 ("Ce groupe sculpté a été volé…") took 2 minutes and about 2 cents: it read the record, searched words, and ended with PM95000723 first (exact) and a looted *Mise au tombeau* (MNR 961) to check.

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

## Hosting

Since 2026-11-03 the site runs on one Scaleway GPU machine, `miroir-demo`, that is on only for demos (`src/miroir/jobs/demo.py`). Before, it ran on Modal.

- **Address:** <https://51-159-130-27.sslip.io>. A flexible IP keeps the address while the machine is off, so the URL never changes. sslip.io turns the name into that address; Caddy, on the machine, gets and renews the Let's Encrypt certificate.
- **Machine:** an L4-1-24G (one L4 GPU with 24 GB, 8 CPUs, 48 GB of memory) in fr-par-2, with one 100 GB block volume that holds the system, the Docker image, the model weights and the `pop-all` bundle. Nothing is downloaded at start.
- **Inside:** three containers that Docker starts at every boot. `miroir-site` loads the models and the bundle, then serves the page on the machine's loopback (`src/miroir/remote/site.py`); `caddy` takes HTTPS on ports 80 and 443; `miroir-watch` powers the machine off (below). Loaded, the site uses 30 of the 48 GB of memory (the MegaLoc vectors in int8 are 19 GB) and 19 of the GPU's 23 GB.

**Crops (not deployed yet).** The bundle `pop-all-crops` adds crop vectors, so a photo of a part finds the whole it belongs to (PARTS hit@10 58 → 71 over all of POP; search-levels.md, "Crops of all of POP"). Its whole crop vectors need a 200 GB volume beside the root one (about €19 a month). With the machine on, `uv run miroir demo crops` makes, attaches and mounts it, then serves that bundle; `demo up` keeps it; `uv run miroir demo install --index pop-all` goes back.

```sh
uv run miroir demo up        # power on, wait until the page answers, print the address
uv run miroir demo down      # power off: only the disk and the address are billed
uv run miroir demo status    # on or off, time on, this session's cost, last visitor, spend
uv run miroir demo logs      # the site's last lines (--container caddy or miroir-watch)
uv run miroir demo install   # after a code change: rebuild on the machine and restart (machine on)
uv run miroir demo chat-logs # copy the agentic search's logs to ~/miroir-data/chat-logs/hosted/
```

`miroir demo create` made the machine once (IP, machine, image, bundle, HTTPS: 12 minutes); `miroir demo destroy` deletes it with its disk and address.

**Power.** On an L4, `demo up` takes about 4 min 40 s from the command to a page that answers (measured 2026-11-03: 282 s): the boot and SSH take under a minute, then the site loads (the catalog of 2.26 million images 50 s, the 19 GB of int8 vectors, the facts the agentic search filters on 50 s, the caption reader 10 s). The first search after that takes 1 s, the next ones 0.3 s. The machine powers itself off after 60 minutes with no visitor, even if the Mac is closed: the watcher looks once a minute at the last request that carried a session (robots that probe the address, and the page `demo up` waits for, do not count). It also stops after 8 hours on, visitors or not. `demo up --idle-minutes N` sets another wait for that session (`MIROIR_DEMO_IDLE_MINUTES` and `MIROIR_DEMO_MAX_HOURS` change the defaults, then `demo install`). Checked 2026-11-03: with 8 minutes, the machine powered off 8 minutes after the last search.

**No GPU free.** Scaleway often has no L4 free in fr-par-2, even for a machine that was just on: on 2026-11-03 every L4 and L40S in Paris showed "shortage" for over an hour. `demo up` then changes the machine's type while it is off, on the same disk, and tries in order (`DEMO_OFFERS`): L4 (€0.79 an hour), L40S (€1.47), then two machines without a GPU, POP2-HM-12C-96G (€0.62) and POP2-HM-16C-128G (€0.83). A type whose 8 hours would pass the cap is skipped. When none starts it tries again every 30 s for 5 minutes. The next `demo up` goes back to an L4 when one is free. `--gpu-only` waits for a GPU instead; `--type <type>` asks for one type.

Without a GPU (measured on POP2-HM-12C-96G, 2026-11-03): `demo up` takes about 6 minutes (352 s); a photo search 3.9–4.1 s instead of 0.3 s, with the same cards and levels on the checks below; outlines 4–5 s; words 0.7 s. The caption reader is left out (Qwen3-VL 4B would take most of a minute per print), so archive prints are searched without their caption, and the page says so at the top ("Mode dégradé…"). The models run in float32 there (bfloat16 took 12.6 s a photo). Left out of the list: RENDER-S (a P100 with 16 GB: the site needs 19 GB of GPU memory, and EmbeddingGemma 2 needs bfloat16, which that GPU lacks; €1.22 an hour) and H100 (€2.73 an hour). The machine cannot move to another zone without moving its disk.

**Money.** Prices from Scaleway's public catalogue (`api.scaleway.com/product-catalog/v2alpha1/public-catalog/products`, 2026-11-03), without VAT:

| What | Price | When |
|---|---|---|
| L4-1-24G | €0.013125 a minute (€0.79 an hour) | while on |
| Block volume, 100 GB, 5K IOPS | €0.00013 per GB an hour (€9.49 a month) | always |
| Flexible IPv4 | €0.005 an hour (€3.65 a month) | always |

So a demo hour costs €0.79 (€0.81 with the disk and address), and the machine at rest €13.14 a month. Each session and the storage between commands go to the machines' ledger (`~/miroir-data/scaleway/spend.jsonl`) when a `demo` command sees them end: a session the watcher ended is billed up to the time Scaleway gives for the stop. `demo up` refuses to start when 8 hours on would take the ledger past `MIROIR_SCW_BUDGET_EUR`.

**Secrets.** `demo install` sends the site's password and the model keys from `.env` to root-only files on the machine, over SSH, on standard input: never on a command line, in cloud-init or in a log. The site's container gets the password and the OpenRouter and Mistral keys; the session key is made on the machine once and never leaves it. The watcher's container holds the project's Scaleway key, the only one that can power the machine off: the project key may not create a narrower IAM key (2026-11-03, `permissions_denied` on IAM). Anyone root on the machine could read it; only the Mac's own SSH key gets in.

**Agentic search.** Its spending per day is kept on the machine's disk (`/opt/miroir/state/chat-days.json`) and its logs beside it (`state/chat-logs/`); `demo chat-logs` copies them to the Mac. The logs of the Modal site (2026-10-19 to 10-27) are in `~/miroir-data/chat-logs/hosted/`.

## Run and deploy

**Bundles and code.** New code reads older bundles (a bundle without `levels.json` falls back on the older sure rule; a `routing.json` without `caption_a7` and `caption_bm25` takes the caption weights from `miroir.web.routing`, 0.3 / 0.7), but older code refuses a `routing.json` that holds the caption weights.

**Card levels.** To set them again after a rebuild or on another index: `miroir web levels` (writes `levels.json` into the bundle and a copy `~/miroir-data/index/levels-<index>.json` that `build-index` puts back), then `miroir web upload`.

**Position (H2).** The `full-f2` bundle holds `places.npz` (one position per entity), so the site uses the photo's GPS position or the visitor's. To add positions to another built bundle without rebuilding: `miroir web places --index <name> --upload`, then `miroir demo install`. A bundle without `places.npz` ignores positions.

**Caption reader.** It loads in a thread after the search models and resets torch's default dtype afterwards; loading it beside EmbeddingGemma 2 without that reset left the default at bfloat16 and broke every photo search ("expected mat1 and mat2 to have the same dtype").

**The page.** Its sources are in `web-ui/` (React, Vite, `@codegouvfr/react-dsfr`); `make ui` (Node 20 or later on the Mac) builds it into `src/miroir/web/ui/`, outside git, with `source.sha256`, a digest of the sources it came from. The package carries that folder, so the demo machine needs no Node: `miroir demo install` refuses to build the image when the page is missing or older than its sources (`miroir.web.pagebuild`). For work on the page, `npm --prefix web-ui run dev` serves it with hot reload against `miroir web dev` on port 8000.

```sh
make ui                         # the page: npm ci and Vite build into src/miroir/web/ui/
uv sync
uv run miroir web build-index   # NumPy on the Mac, about 3 minutes for the full index: writes ~/miroir-data/web/<index>/
uv run miroir web upload        # copies the bundle to the index bucket
uv run miroir web secret        # once: password in .env (--rotate for a new one); `miroir demo install` sends it
uv run miroir web dev           # http://localhost:8000 with fake models (no model runs on the Mac)
uv run miroir web dev --scw     # the same page searching on a Scaleway GPU started for it (costs GPU time)
```

To switch index: add it to `INDEXES` in `src/miroir/web/bundle.py`, set `DEMO_INDEX` (the demo machine) or `WEB_INDEX` (the Mac) in `src/miroir/config.py`, then build-index, upload and `miroir demo install`.

**Agentic search (H5, `#/agentique`).** Needs the Mistral key (or the OpenRouter key) on the machine and the list of images that may go to the model in the bundle:

```sh
uv run miroir web publishable --upload   # writes publishable.txt.gz into the bundle, copies it to the index bucket
uv run miroir demo install               # sends the keys from .env and restarts the site
```

Each conversation is logged as JSON lines with no images (technical spec §8.1). To read the demo site's logs on the Mac:

```sh
uv run miroir demo chat-logs   # copies the machine's chat logs to ~/miroir-data/chat-logs/hosted/
```

Caps: `MIROIR_CHAT_CONVERSATION_USD` (default 0.10) and `MIROIR_CHAT_DAY_USD` (default 2), read from the environment. When both keys are set, OpenRouter wins. Locally, `miroir web dev` runs the agentic search through OpenRouter when `OPENROUTER_API_KEY` is in `.env`, else on Mistral's API when `MISTRAL_API_KEY` is (same caps), or with a scripted stand-in with `--fake-llm` or without a key.

## Utiliser MIROIR depuis ChatGPT ou Claude

The site is also a remote MCP server (`miroir.web.mcp`), so ChatGPT or Claude can search POP through MIROIR on their own. It runs in the site's process, on the same engine, and answers at `<address>/mcp/<token>` (Streamable HTTP, stateless) only when the machine is on (`miroir demo up`).

| Tool | Does |
|---|---|
| `search` | The simple search: `query` (words) and/or `image_url` (a public https link; `image_base64` for clients that hold the bytes), filters (`collections`, `region`, `departement`, `domain`, `period`, matched to the site's menus whatever the case and accents; an unknown value is an error listing the valid ones), `latitude`/`longitude` (H2's prior), `top_k` up to 20. Each result: id, title, POP url, base, place, period, authors, level ("tres-proche", "proche", "ressemblance", as the site), a thumbnail link |
| `fetch` | One result in full by its id (or a POP reference): description and facts as text, url, its POP records and images |
| `delegate_search` | A turn of the site's own agent (§7 above, technical spec §8.1): `request` in the user's words, photo, filters, position. Returns the agent's summary, its picks with a reason each, the steps it took, and `status`: `done`, `question` (relay it, call again with `conversation_id` and `reply`), `frame` (send `region_image_url`), `running` (after `wait_seconds`, 50 by default: call again with only `conversation_id`), `stopped` (a spending cap) |

`search` and `fetch` return structured content and the same JSON as text, the shape ChatGPT's deep research reads; `delegate_search` returns a short text and structured content. A commune is not a filter (the site has none): put it in the words or give a position. The agent's fine-grained tools (`more_like`, `view_results`) are not exposed: they work on a conversation's state and send thumbnails to the agent's model, and `delegate_search` already runs them.

**Access.** One secret token per client, `name=token` pairs in `MIROIR_MCP_TOKENS` (`.env`), made by `miroir web mcp-token <name>` and sent to the machine by `miroir demo install` (as `MCP_TOKENS`). The token goes in the URL path (ChatGPT takes only OAuth or no authentication) or as `Authorization: Bearer` (Claude can send a fixed header). Each token has its own limits: 30 searches or fetches a minute, 4 agent turns a minute, and its own agent conversations. The agent keeps the site's caps, $0.10 a conversation and $2 a day shared with the site's visitors. The site's password is not used. The access log masks the token. OAuth would need an authorization server (accounts, consent, token issue and refresh); not built.

**Images and privacy.** An image link is fetched by the server: https on port 443 only, every address of the name public (no private, loopback, link-local or reserved range), the request sent to the address checked (so DNS cannot change it in between), at most 3 redirects each checked again, 8 MB, 20 s, 50 megapixels (`miroir.web.mcp_photo`). `search` sends the photo to our GPU machine only. `delegate_search` sends it, and the words, to the agent's model through OpenRouter (or Mistral's API), as the site's Approfondi search does; its description says so. Thumbnails and images in results are publishable ones only (`publishable.txt.gz`; none without it): a link given to ChatGPT or Claude leaves our hands. Neither client documents passing a photo uploaded into the chat to a remote tool: the user gives a link.

**Deploy.**

```sh
uv run miroir web mcp-token chatgpt --show   # once per client; prints /mcp/<token>
uv run miroir demo install                   # sends the tokens; the site serves /mcp
uv run miroir demo up                        # the connector answers while the machine is on
```

`miroir web dev` serves it too, at `http://localhost:8000/mcp/<token>`, when `MIROIR_MCP_TOKENS` is set (a client in the cloud cannot reach a laptop).

**ChatGPT** (Plus, Pro, Business, Enterprise, Edu; on the web): Settings → Apps → Advanced settings → turn on Developer mode. Then "Create app": a name (MIROIR), the URL `https://51-159-130-27.sslip.io/mcp/<token>`, authentication "No authentication", tick that you trust it. In a chat, pick Developer mode in the "+" menu and turn MIROIR on. ChatGPT asks before a call unless the tool says it only reads; all three do (`readOnlyHint`).

**Claude** (any plan; Free allows one custom connector): Customize → Connectors → "+ Add" → "Add custom connector": name MIROIR, the URL `https://51-159-130-27.sslip.io/mcp/<token>`, Authentication "No sign in". Or the URL without the token, `…/mcp`, and under Request headers `Authorization: Bearer <token>`. Turn it on in a chat with "+" → Connectors. Claude calls from Anthropic's servers, so the site must be on and public, as it is.

**What the specs say** (read 2026-11-04; OpenAI's and Anthropic's developer pages could not be opened from the build machine, so their points come from their search snippets and are worth checking):

- MCP Python SDK 2.3 ([docs](https://py.sdk.modelcontextprotocol.io/run/asgi/)): `FastMCP` is now `MCPServer`; a mounted server's session manager must run in the host app's lifespan, once per start; DNS-rebinding protection allows only localhost unless told otherwise (421 behind a real name). Structured output: the return type is the output schema, the text carries the same JSON ([structured output](https://py.sdk.modelcontextprotocol.io/servers/structured-output/)).
- ChatGPT developer mode ([OpenAI](https://developers.openai.com/api/docs/guides/developer-mode)): remote servers over SSE or Streamable HTTP; OAuth, no authentication, or mixed; `search`/`fetch` not required; tools without `readOnlyHint` count as write actions and are confirmed. Deep research and company knowledge want `search` (results of id, title, url) and `fetch` (id, title, text, url, metadata), in structured content and as JSON text ([OpenAI, MCP servers](https://developers.openai.com/api/docs/mcp)).
- Claude custom connectors ([help centre](https://support.claude.com/en/articles/11175166-getting-started-with-custom-connectors-using-remote-mcp)): a remote MCP URL; authentication "Sign in" (OAuth: Claude's published identity, automatic registration or your own client) or "No sign in"; fixed request headers for an API key or bearer token; reached from Anthropic's IP ranges.

Not tried with the real clients yet: the tests drive the server with the SDK's own client over in-process HTTP (`tests/test_web_mcp.py`). To check after deploying: both clients connect and list the three tools; a `search` with a link; a `delegate_search` that asks, and its answer; how long each client waits for a tool before it gives up (the 50 s wait is a guess).

## Cost

The demo machine: €0.79 an hour while on, €13.14 a month at rest (see "Hosting"). Before 2026-11-03, on Modal, an L4 with 4 CPUs and 24 GB cost about $1.18 an hour while awake and each wake-up about $0.22. Searches are limited to 30 a minute per session, logins to 5 a minute per address.

The agentic search also pays for the model: a turn cost about $0.02–0.04 through OpenRouter. Mistral sends no price with its answers, so the site works out each call's cost from its tokens at prices typed into `miroir.jobs.mistral.PRICES` ($2 per million input tokens, $6 output). Check them against Mistral's console. It stops at $0.10 a conversation and $2 a day for all visitors (`MIROIR_CHAT_CONVERSATION_USD`, `MIROIR_CHAT_DAY_USD`).

## Known limits

- **Position.** Rose Valland works have no place (looted and moved), and Mémoire objects sit at their commune's centre, so with the position on they rank below precise neighbours. The visitor can turn the position off.

- **Index.** The site serves all of POP (`pop-all`, above). Its card levels and sure rule were set on the July index and not set again; scores below that are not marked `pop-all` were measured on the July index.
- **Phone photos of buildings.** Without the photo's position, the right building comes first for a quarter of photos (25.8%), in the top 3 for a third (32.6%) and in the top 10 for 36.8%. A building whose Commons photos are in the index is found far more often than one without (58.6% against 24.9% hit@10 on the screening gallery): 1,198 of 3,924 buildings have none. Interiors and close details stay hard. With the photo's GPS position (H2) the right building is in the top 10 for most photos.
- **Levels are cautious.** About 30% of in-index photo searches get a "Très proche" first card and 11–18% a "Proche" one; the rest see "Ressemblance" cards to compare. Words never reach "Très proche". A caption read on an archive print raises T1 hit@1 from 34% to 71%, but the level reads the photo's cosines only, so it undersells those searches.
- **Word search** finds the right record in the top 10 for about one description in six (T7: 17.6%; half and half scored 8.0%). Names work ("Pont du Gard" comes first); descriptions of common things (a church, a coin) match too many records, and Mémoire buildings are found only 7% of the time. A region in the words ("bourgogne") does not narrow the search; the region filter does. T6, written by hand, does not exist yet ([t6-guide.md](t6-guide.md)).
- **Period filter** for Mémoire reads the date of the photo, not of the building.
- **Outlines on a crowded wall.** On the Jeu de Paume room photos the right record is in the top 5 for 87% of works when every outline is shown, 78% with the 20 shown first (it was 52% with the 12 surest; results.md, step C): on the densest walls the visitor still needs "Afficher les N autres". Outlines' names come from the detector's prompts and are often wrong (it calls a statue a stained glass window); the page only uses them in labels for screen readers.
- **Keypoint re-ranking (E1) is not on the site:** it lifts T3 hit@1 by 12 points but adds 3 s per search (6 s at the 95th percentile), mostly reading candidate images.
- **Off between demos.** Someone must run `miroir demo up` (about 5 minutes, 6 without a GPU) before a demo; a visitor who finds it off sees no page. It powers itself off after an hour without a visitor. Scaleway is sometimes out of L4s (see "Hosting").
- **Type guess** trained on the full-sample index (`miroir.index.guess`). It calls more than half of phone photos of buildings objects (it partly learned colour); they then get 0.2 A7 + 0.8 A3, close to what a colour building gets.
