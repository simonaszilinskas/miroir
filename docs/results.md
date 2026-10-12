# MIROIR: results

One section per round of [experiments.md](experiments.md). Before each round, write what we expect and why. After it, add the table and what we learned.

## R0: harness check

Expected (written 2026-10-12, before any run): DINOv2 ViT-g/14 (`facebook/dinov2-giant`), class token after the final LayerNorm, shortest side 256 then centre crop 224, cosine, top 128, lands within 2 points of 48.9 mAP (DINOv3 paper, Tables 9 and 23) in one of the three directions. Our guess: "all" (both directions pooled) lands nearest, near 49, with archive → street view and street view → archive a few points either side of it. Archive photos are greyscale and often odd crops, so we expect street-view queries (searching the archive) to score lower. If no direction lands within 2 points, suspect preprocessing first (resize, crop, colour), then the AP formula.

Second check, expected (written 2026-10-12, before any run): DINOv3 ViT-7B/16 (`facebook/dinov3-vit7b16-pretrain-lvd1689m`), same protocol, lands within 2 points of 56.5 mAP (same tables). It is the only DINOv3 size with a published AmsterTime score; the distilled ViT-L and H+ have none. It runs in bf16 on an L4, since float32 needs a 48 GB GPU.

Results (2026-10-12). DINOv2 ViT-g/14 on Modal (L4, float32), run `R0-dinov2-giant-224-amstertime`, $0.36 (17 minutes, most of it reading files one by one from the volume; fixed since). mAP with 95% intervals:

| Direction | Queries | mAP | Hit@1 | vs 48.9 |
|---|---|---|---|---|
| old: archive → street view | 1,231 | **50.7** [48.4, 53.3] | 40.4 | +1.8, within 2: **pass** |
| all: both pooled | 2,462 | 46.7 [45.0, 48.4] | 36.6 | −2.2 |
| new: street view → archive | 1,231 | 42.7 [40.2, 45.1] | 32.7 | −6.2 |

What we learned:

- The harness passes: archive queries land 1.8 points from the published 48.9. That is the direction the AmsterTime paper's text describes, so the DINOv3 paper likely reports it.
- Our guess on direction was wrong: "all" is not nearest, it sits 2.2 below. Street-view queries searching the archive do score lowest, as guessed, by 8 points.
- "all" equals the DINOv2 paper's own ViT-g figure (46.7). That may be the protocol behind the two papers' 2.2-point gap.

DINOv3 ViT-7B/16 on Modal (L4, bf16), run `R0-dinov3-vit7b16-224-amstertime`, $0.18:

| Direction | Queries | mAP | Hit@1 | vs 56.5 |
|---|---|---|---|---|
| new: street view → archive | 1,231 | **57.1** [54.7, 59.3] | 44.8 | +0.6, within 2: **pass** |
| all: both pooled | 2,462 | 59.0 [57.4, 60.6] | 47.1 | +2.5 |
| old: archive → street view | 1,231 | 61.0 [58.5, 63.4] | 49.3 | +4.5 |

- Both checks pass, but not in the same direction: DINOv2 matches its figure on archive queries, DINOv3 on street-view queries. So the paper's direction stays unknown, and a 2-point bar on one direction is looser than it looks. The harness itself (preprocessing, cosine, AP cut at 128) is not in doubt: both models land within 2 points in some direction, with the expected 8–12-point gap between them.
- DINOv3 7B beats DINOv2 g by 12–14 points in every direction, as the paper reports. It ran in bf16, not float32; that may move scores by a fraction of a point.

## R1: image models alone (step A)

Expected (written 2026-10-12, before any R1 score): on the 19,212-image publishable screening gallery, scored on queries a closed model may see, the DINO family (A1, A2) should win T1 and T5: published instance-retrieval scores put DINOv2 and DINOv3 well ahead of text-aligned models (Oxford-H, Met, AmsterTime). We expect the two closed embedders, Gemini Embedding 2 (A9) and Voyage multimodal 3.5 (A10a), to be weaker on T1's archive building photos, which are greyscale shots of one building from different angles, where meaning-level likeness pulls in other buildings of the same kind. They may do better on T3, where query and answers are crops of the same negative, and on T5 museum objects. Between the two, no strong guess; Gemini has the larger vectors (3,072 against 1,024). T8 waits for the not-found measure.

Results: A9 and A10a first; open models join when their runs exist. Each closed-model score uses `--gallery screening-20k-publishable --publishable-queries` (T1 500, T3 99, T5 449 queries); score open models the same way to compare them.

### T1 rule changed after seeing a score (2026-10-12)

We changed T1 after seeing A9's score on it, and say so plainly: the new T1 is not a set chosen blind. A9 (Gemini Embedding 2) scored 90.8% hit@1 on the old T1 tune half. Looking at its hits, 443 of 453 were a photo from the query's own shoot (same photographer and year), 298 of them the very next ref: the next frame on the same film, the same card layout, the same light. 357 of the 499 queries had no answer outside that shoot. So the old T1 measured finding a sibling print, not knowing a building from another visit.

New rule (technical spec §7): a T1 query no longer sees any photo of its building from its own shoot, and buildings photographed in one shoot only are dropped. T1 goes from 1,000 queries (500 / 500) to 273 (139 / 134); halves and galleries are unchanged, so A9's vectors still serve. T3, T5 and T8 are unchanged: their closest pairs are other faces of the same object or other buildings, not copies.

A9, tune half, gallery `screening-20k-publishable-fetched` (19,105 images), queries a closed model may see:

| Set | Rule | Queries | Hit@1 | Hit@10 | mAP |
|---|---|---|---|---|---|
| T1 | old: only the query's own record hidden | 499 | 90.8 [88.2, 93.2] | 94.6 [92.6, 96.6] | 92.1 [89.9, 94.3] |
| T1 | old, on the 138 queries the new rule keeps | 138 | 92.0 | 94.2 | |
| T1 | new: the query's whole shoot hidden | 138 | **33.3** [26.1, 40.6] | **51.4** [44.2, 59.4] | 38.9 [32.1, 46.0] |
| T5 | unchanged | 443 | 34.8 [30.5, 39.1] | 51.2 [46.3, 55.5] | 41.0 [36.9, 45.1] |

What we learned:

- On the same 138 queries, hit@10 falls from 94% to 51% once the shoot is hidden. Nearly all of A9's old T1 score came from same-shoot siblings.
- A9 now sits far below T1's 80% top-10 bar. The bar was set against the old set; keep it until an open model has run on the new one.
- 139 tune queries give intervals about ±8 points. Only 957 of the 3,163 buildings have photos from two or more shoots, so a larger T1 means drawing from those 957, which moves the screening gallery and needs every run embedded again.

A10a (Voyage multimodal 3.5) against A9 (Gemini Embedding 2), same gallery and queries, paired bootstrap (`miroir eval compare`):

| Set | Queries | A10a hit@1 | A10a hit@10 | A9 − A10a, hit@10 | A9 − A10a, mAP |
|---|---|---|---|---|---|
| T1 | 138 | 29.7 [22.5, 37.7] | 40.6 [33.3, 49.3] | +10.9 [+5.1, +16.7] | +5.3 [+0.6, +10.2] |
| T3 | 99 | 28.3 [20.2, 37.4] | 64.6 [55.6, 74.7] | +19.2 [+11.1, +28.3] | +24.9 [+18.1, +31.4] |
| T5 | 443 | 32.1 [27.5, 36.6] | 43.6 [38.8, 48.1] | +7.7 [+4.5, +11.1] | +4.4 [+2.0, +6.7] |

What we learned:

- A9 beats A10a on every set beyond the error bar for hit@10 and mAP. At hit@1 they tie on T1 and T5. Gemini is the closed reference to beat; Voyage adds nothing over it.
- Both runs cost about the same: $2.22 (A9) and $2.28 (A10a) for 19,212 images. A9's run file says $3.25 because an earlier attempt died at 8,004 images, before runs saved their progress.
- 107 of the 19,212 images could not be fetched (84 missing from the bucket and the IIIF service, 23 broken files), so both runs score on `screening-20k-publishable-fetched` (19,105 images). Score open models on that list with `--publishable-queries` to pair with these.

### Open models on Modal (2026-10-12)

Same gallery (`screening-20k-publishable-fetched`, 19,105 images) and queries as A9 and A10a (T1 138, T3 99, T5 443), tune halves. Each cell: hit@1 / hit@10, 95% interval in brackets. Cost: the Modal run, by the published price.

| Option | Cost | T1 | T3 | T5 |
|---|---|---|---|---|
| A7 EmbeddingGemma 2, 512 px | $1.05 | **36.2** [29–44] / **50.0** [43–58] | **56.6** [47–67] / 84.8 [78–92] | 33.4 [29–38] / 48.1 [43–53] |
| A3 MegaLoc, 322 px | $0.27 | 20.3 [14–27] / 29.7 [22–37] | 55.6 [46–66] / **85.9** [79–92] | **39.5** [35–44] / **62.5** [58–67] |
| A4 SigLIP 2 so400m, 384 | $0.27 | 23.2 [17–30] / 37.0 [29–45] | 49.5 [39–60] / 80.8 [73–88] | 29.3 [25–34] / 47.6 [43–52] |
| A5 PE-Core L, 336 | $0.45 | 23.2 [17–30] / 35.5 [28–43] | 45.5 [35–55] / 79.8 [71–88] | 30.7 [26–35] / 49.7 [45–54] |
| A8 Qwen3-VL-Embedding 2B, 512 | $0.47 | 24.6 [18–32] / 31.9 [25–39] | 39.4 [30–49] / 75.8 [68–84] | 26.4 [22–30] / 42.0 [37–47] |
| A2 DINOv3 L/16, 256, CLS | $0.35 | 20.3 [14–28] / 29.0 [22–36] | 37.4 [27–47] / 62.6 [54–72] | 29.8 [26–34] / 42.7 [38–47] |
| A2 DINOv3 L/16, 256, GeM | $0.55 | 12.3 [7–18] / 21.7 [15–28] | 34.3 [24–43] / 61.6 [53–71] | 27.8 [24–32] / 46.3 [42–51] |
| A2 DINOv3 H+/16, 256, CLS | $0.81 | 24.6 [18–32] / 33.3 [25–41] | 30.3 [21–39] / 55.6 [45–65] | 31.2 [27–35] / 41.1 [36–46] |
| A2 DINOv3 H+/16, 256, GeM | $0.82 | 15.2 [9–22] / 26.8 [20–34] | 30.3 [21–39] / 59.6 [51–69] | 27.8 [24–32] / 44.0 [39–49] |
| A2 DINOv3 7B/16, 256, CLS (bf16) | $0.63 | 22.5 [16–30] / 33.3 [26–41] | 32.3 [23–41] / 59.6 [49–69] | 30.9 [26–35] / 42.9 [38–47] |
| A1 DINOv2 L/14, 224, CLS | $0.45 | 14.5 [9–20] / 25.4 [18–33] | 33.3 [24–42] / 50.5 [40–60] | 29.6 [25–34] / 39.5 [35–44] |
| A1 DINOv2 L/14, 224, GeM | $0.34 | 7.2 [4–12] / 19.6 [14–26] | 29.3 [20–38] / 54.5 [45–65] | 28.9 [25–33] / 43.8 [39–49] |
| A1 DINOv2 g/14, 224, CLS | $0.63 | 13.0 [8–19] / 24.6 [17–32] | 31.3 [23–41] / 52.5 [43–62] | 27.8 [23–32] / 39.7 [35–45] |
| A1 DINOv2 g/14, 224, GeM | $0.73 | 7.2 [4–12] / 19.6 [14–26] | 30.3 [21–39] / 57.6 [48–68] | 32.3 [28–37] / 51.0 [46–56] |
| A9 Gemini Embedding 2 (closed) | $2.22 | 33.3 [26–41] / 51.4 [44–59] | 54.5 [44–65] / 83.8 [76–91] | 34.8 [30–39] / 51.2 [46–56] |

What we learned:

- Our guess was wrong. The DINO family does not win T1 or T5; it comes last. DINO runs use the published eval crop (shorter side to 256 × crop/224, then a centre square), which cuts off the sides of every wide photo; the other models see the whole image. That crop is the first suspect, before concluding anything about DINO itself.
- EmbeddingGemma 2 (open, 740M) ties Gemini Embedding 2 on T1 and T3 (paired bootstrap, every interval crosses 0) and trails it on T5 by 3 points hit@10 [−5.6, −0.5].
- MegaLoc wins T5 outright: hit@10 62.5 against A9's 51.2, +11.3 [+7.4, +14.9] paired. It also ties the best on T3.
- GeM pooling never beats CLS on T1, and helps a little on T5 hit@10.
- The 14 runs cost $7.82 together; probes, the image volume and R0 make up the rest of the Modal spend. Project spend after R1: $14.74 (OpenRouter $5.55, Modal $9.18).
- T8 is not scored: the harness refuses queries with no right answer until the not-found measure exists. T6 and T7 do not exist yet, so EmbeddingGemma 2's text side waits.

### DINO on the whole photo (2026-10-12)

Which R1 models crop, from our code and each model's published preprocessing config:

| Option | What the model sees |
|---|---|
| A1 DINOv2, A2 DINOv3 | **A centre square.** Shorter side to crop × 256/224, then a crop × crop square: a 4:3 photo keeps 58% of its area, a 2:1 photo 38% |
| A3 MegaLoc | The whole image, stretched to 322 × 322 |
| A4 SigLIP 2 | The whole image, stretched to 384 × 384 (processor: resize to 384 × 384, no crop) |
| A5 PE-Core | The whole image, stretched (open_clip config: `resize_mode` "squash") |
| A7 EmbeddingGemma 2 | The whole image, aspect kept (Gemma 4 image processor, up to 280 image tokens) |
| A8 Qwen3-VL-Embedding | The whole image, aspect kept, within 512² pixels |

Only DINO crops, so only DINO was rerun. New runs see the whole image: shorter side 256 (DINOv3, patch 16) or 224 (DINOv2, patch 14), longer side rounded to the patch grid and capped at twice the shorter. CLS and GeM come from one pass. These runs are in bf16; R1's L, H+ and g ran in float32 (the 7B was bf16 in both). Same gallery and queries as above. Each cell: hit@1 / hit@10 without the crop, then R1's hit@10 with it.

| Option | Cost | T1 | T3 | T5 |
|---|---|---|---|---|
| A2 DINOv3 L/16, CLS | $0.16 (both) | 20.3 / 29.7 (29.0) | 26.3 / 59.6 (62.6) | 29.1 / 40.9 (42.7) |
| A2 DINOv3 L/16, GeM | | 11.6 / 21.7 (21.7) | 28.3 / 64.6 (61.6) | 31.4 / **51.2** (46.3) |
| A2 DINOv3 H+/16, CLS | $0.35 (both) | 21.0 / 34.1 (33.3) | 30.3 / 60.6 (55.6) | 29.1 / 41.5 (41.1) |
| A2 DINOv3 H+/16, GeM | | 13.0 / 26.1 (26.8) | 30.3 / 60.6 (59.6) | 27.3 / 45.4 (44.0) |
| A2 DINOv3 7B/16, CLS | $0.88 (both) | 23.2 / 36.2 (33.3) | 31.3 / 65.7 (59.6) | 30.7 / 42.4 (42.9) |
| A2 DINOv3 7B/16, GeM | | 13.8 / 26.8 (none) | 35.4 / 70.7 (none) | 29.8 / 51.2 (none) |
| A1 DINOv2 g/14, CLS | $0.29 (both) | 15.2 / 29.7 (24.6) | 27.3 / 53.5 (52.5) | 28.2 / 39.5 (39.7) |
| A1 DINOv2 g/14, GeM | | 10.9 / 18.1 (19.6) | 27.3 / 56.6 (57.6) | 34.1 / **58.2** (51.0) |

What we learned:

- The crop is not why DINO comes last. On T1 and T3 no run moves beyond the error bar at hit@10 (paired bootstrap against the cropped run), and the best uncropped DINO run on T1 (7B CLS, 23.2 hit@1) stays far behind EmbeddingGemma 2 (36.2).
- GeM on T5 gains from the whole object: DINOv2 g +7.2 hit@10 [+4.1, +10.6], DINOv3 L +5.0 [+2.3, +7.9]. DINOv2 g GeM now sits between MegaLoc (62.5) and Gemini (51.2) on T5, but still trails on T1 and T3.
- One loss beyond the bar: DINOv3 L CLS on T3, hit@1 −11.1 [−19.2, −3.0]. We have no explanation yet; bf16 is the other change in these runs.
- The four runs cost $1.67, and their probes $0.09.

### Commons photos: T2, T2b, T8c (2026-10-12)

Expected (written 2026-10-12, before any score). Queries are recent colour photos from Commons (technical spec §7); the gallery is `screening-v2-publishable`, so every model sees the same images and T2b keeps only works with a publishable image.

- T2 (a visitor's photo of a building, answers mostly black-and-white Mémoire photos): A7 and A9 lead, as on T1, with hit@10 near 40%, below the 60% target. MegaLoc (A3) is trained on street-level colour photos, so it should do better here than on T1's archive queries, but still trail A7. Exterior photos score at least 10 points above interior and detail ones: most Mémoire photos show the outside.
- T2b (Roman sculpture, lamps and coins of one museum, shot on stands): A3 leads, as on T5, near 50% hit@10; A7 and A9 near 40%. Coins shot on the other face are misses for every model.
- G1-A7+A3 sits between its two models on each set and edges the best single model on the mean of the two, but not beyond the error bar on either set (as on T1, T3, T5).
- T8c: a quarter or more of not-found photos still get a confident wrong answer at the cut-off that answers half of T2 and T2b, as on T8.

Results (2026-10-12). Gallery `screening-v2-publishable-fetched` (21,785 images: the 107 images no run could fetch on `screening-20k` and 2 of the 2,682 new ones are left out), tune halves, `--publishable-queries`: T2 516 queries, T2b 122 (the tune half's works with a publishable image), T8c 150. Each model's 19,105 old vectors were reused; only the 2,682 new images and the 957 Commons photos were embedded. Hit@10, 95% interval in brackets:

| Option | T2 | T2b | T2 exterior (346) | T2 interior (73) | T2 detail (97) |
|---|---|---|---|---|---|
| A3 MegaLoc | **31.0** [27.3, 35.1] | 92.6 [87.7, 96.7] | **32.7** | 13.7 | **38.1** |
| A9 Gemini Embedding 2 (closed) | 24.8 [21.3, 28.7] | 97.5 [94.3, 100] | 25.4 | 12.3 | 32.0 |
| A7 EmbeddingGemma 2 | 21.9 [18.4, 25.6] | 97.5 [94.3, 100] | 24.0 | 11.0 | 22.7 |
| A5 PE-Core L | 19.4 [16.1, 23.1] | 94.3 [89.3, 97.5] | 20.2 | 12.3 | 21.6 |
| A4 SigLIP 2 so400m | 9.1 [6.8, 11.6] | 95.1 [91.0, 98.4] | 9.5 | 4.1 | 11.3 |
| G1-A7+A3 (0.7 / 0.3) | 31.2 [27.5, 35.5] | **98.4** [95.9, 100] | | | |

Paired bootstrap, hit@10: on T2, G1-A7+A3 ties A3 (+0.2 [−1.9, +2.3]) and loses on mAP (−2.4 [−4.0, −0.9]); A3 beats A9 by +6.2 [+3.3, +9.3] and A9 beats A7 by +2.9 [+1.0, +5.0]. On T2b, G1 ties A9 (+0.8 [−1.6, +3.3]) and beats A3 (+5.7 [+1.6, +10.7]).

T8c, against T2 and T2b tune (638 in index, 150 not found):

| Run | Wrong but confident, cut-off at 50% matched | Matched, cut-off at 5% wrong |
|---|---|---|
| A9 (closed) | **26.0** [17.3, 33.3] | **25.1** [21.6, 31.5] |
| G1-A7+A3 | 35.3 [27.3, 43.3] | 21.8 [16.9, 27.6] |
| A7 | 36.0 [26.7, 44.0] | 19.6 [16.1, 25.9] |
| A5 | 38.7 [31.3, 50.0] | 17.2 [14.3, 22.7] |
| A3 | 40.0 [32.0, 48.0] | 21.5 [17.1, 25.1] |
| A4 | 45.3 [36.0, 54.7] | 22.3 [18.2, 25.9] |

What we learned:

- Our T2 guess was wrong twice. Every model sits far below the 60% target, lower than we guessed (best 31%), and MegaLoc, not EmbeddingGemma 2, finds buildings best from a visitor's photo: the street-level colour photos it was trained on look like these queries, not like T1's archive prints. A7, the T1 winner, comes third.
- Interior photos are hard for every model (11–14% hit@10), as guessed: few Mémoire photos show the inside. Detail photos score as well as or better than exterior ones, against our guess; many show a portal or a sculpted part that Mémoire also photographed.
- T2b is easy, against our guess: 92–98% hit@10 for every model. Its works are distinctive Roman objects of one museum, and the gallery holds few look-alikes. A look at hits found real visitor photos, not leftover copies of the museum's image, but many come from the museum's own Commons uploads. T2b does not separate models; the label-noise check of 100 tune queries (technical spec §7) should come before we read more into it.
- Fusion again buys a compromise, not a step up: G1-A7+A3 equals the best single model on each set (A3 on T2, A9 on T2b) and beats neither beyond the error bar. The grid again picks its edge weight, 0.7 on A7.
- T8c: a confident wrong answer for a quarter (A9) to almost half (A4) of photos with no answer in the index, at the cut-off that answers half of T2 and T2b. As on T8, a top-1 cosine is not a "not found" signal.
- Cost: OpenRouter $0.43 (A9: probe, 2,680 new images, 957 Commons photos), Modal $0.37 (image volume and A3, A4, A5, A7 on new images and Commons photos). Fetching the 957 Commons thumbnails took about 10 minutes at Wikimedia's pace; every later model read them from the cache.

## R2: fusion (step G), re-ranking (E1), query expansion (E5), "not found" (T8)

G belongs to R3 in experiments.md §5. We pulled it forward because it costs nothing: it only combines vectors we already have, with NumPy, on the Mac.

Same gallery (`screening-20k-publishable-fetched`, 19,105 images) and queries as R1 (T1 138, T3 99, T5 443), tune halves. Run with `miroir eval stack` (technical spec §7.2). Every fusion picks one first-model weight from 0.3–0.7 by mean hit@10 over T1, T3 and T5 together, never one weight per set; with three models, the other two share the rest equally.

### G1 and G2: fusing image models (2026-10-12)

Hit@10, 95% interval in brackets. Best single models from R1 for reference.

| Run | Weights | T1 | T3 | T5 | Mean |
|---|---|---|---|---|---|
| G1-A7+A3 | 0.7 / 0.3 | 45.7 [38–54] | **89.9** [84–96] | 54.9 [50–59] | **63.5** |
| G1-A9+A3 (closed) | 0.6 / 0.4 | 41.3 [33–49] | 88.9 [82–95] | 60.5 [56–65] | 63.6 |
| G1-A7+A3+A5 | 0.7 / 0.15 / 0.15 | 47.8 [41–56] | 88.9 [83–95] | 52.8 [48–57] | 63.2 |
| G2-A9+A3 (closed) | 0.7 / 0.3 | 44.9 [38–53] | 88.9 [83–95] | 55.8 [51–60] | 63.2 |
| G2-A7+A3 | 0.7 / 0.3 | 44.2 [37–52] | 89.9 [84–96] | 51.9 [48–57] | 62.0 |
| G1-A7+A4 | 0.7 / 0.3 | **51.4** [44–59] | 84.8 [78–92] | 49.0 [44–54] | 61.8 |
| G2-A7+A3+A5 | 0.5 / 0.25 / 0.25 | 42.0 [35–50] | 88.9 [83–95] | 52.6 [48–57] | 61.2 |
| G2-A7+A4 | 0.6 / 0.4 | 46.4 [39–55] | 84.8 [77–92] | 48.1 [44–53] | 59.8 |
| A9 alone (R1) | | 51.4 | 83.8 | 51.2 | 62.1 |
| A7 alone (R1) | | 50.0 | 84.8 | 48.1 | 61.0 |
| A3 alone (R1) | | 29.7 | 85.9 | **62.5** | 59.4 |

G1-A7+A3 against the best single model on each set, paired bootstrap, hit@10:

| Set | Best single | G1-A7+A3 − best |
|---|---|---|
| T1 | A9 51.4 (A7 50.0) | −5.8 [−10.9, −0.7] (vs A7: −4.3 [−8.7, −0.7]) |
| T3 | A3 85.9 | +4.0 [+0.0, +9.1] |
| T5 | A3 62.5 | −7.7 [−10.8, −4.3] |

What we learned:

- Fusion does not beat the best single model on any set beyond the error bar. On T3 it gains 4 points, and the interval only just touches zero. On T1 and T5 it loses to the best single model, beyond the error bar.
- Averaged over the three sets, G1-A7+A3 (63.5) edges out every single model (A9 62.1), because no one model is best on every set. The gain is a compromise, not a step up.
- One weight for all sets is the catch: T1 wants EmbeddingGemma 2, T5 wants MegaLoc. The grid picks the weight at its edge (0.7) for most pairs, so a higher weight might score a little better, but it would only move closer to A7 alone.
- G1 beats G2 on every pair. Min-max sums keep how far ahead a model's top answer is; ranks throw that away.
- Adding PE-Core L (A5) or swapping MegaLoc for SigLIP 2 (A4) does not help.

### E5: query expansion (2026-10-12)

The query plus its top 3 images (hidden images never used), weighted by similarity to the power alpha, then searched again. Alpha 0 is a plain average, as experiments.md words it; alpha 3 is the usual α-QE setting. Paired against the same stack without expansion, hit@10:

| Stack | Alpha | T1 | T3 | T5 |
|---|---|---|---|---|
| A7 | 0 | −5.1 [−8.7, −1.4] | −7.1 [−12.1, −2.0] | −4.7 [−7.0, −2.9] |
| A7 | 3 | −4.3 [−8.0, −0.7] | −6.1 [−11.1, −2.0] | −4.3 [−6.3, −2.5] |
| A3 | 0 | −4.3 [−8.7, −0.7] | −6.1 [−11.1, −2.0] | −8.8 [−11.7, −6.1] |
| A3 | 3 | −2.2 [−5.1, 0.0] | 0.0 | −5.6 [−8.1, −3.4] |
| G1-A7+A3 | 0 | −2.2 [−6.5, +1.5] | −5.1 [−9.1, −1.0] | −3.8 [−6.1, −1.8] |
| G1-A7+A3 | 3 | −1.4 [−5.8, +2.2] | −2.0 [−5.1, 0.0] | −3.6 [−5.9, −1.6] |

- Expansion hurts, or at best does nothing, on every stack and set. With only about a third of top-1 answers right, the top 3 hits are mostly wrong, and the second search follows them. Drop E5 unless a re-ranker first makes the top hits right.

### T8: "not found" (2026-10-12)

T8 tune holds 250 queries from T1 and T5 tune (125 each) with their whole building or object taken out of the gallery, so any answer is wrong; 232 have a publishable image in the gallery list. In-index queries are the 581 T1 and T5 tune queries (138 + 443). Confidence is the top entity's cosine; for fusion, the weighted cosines of the entity fusion put first.

| Run | Wrong but confident, cut-off at 50% matched | Matched, cut-off at 5% wrong |
|---|---|---|
| A7 | **25.0** [18.5, 31.5] | 12.4 [8.6, 17.9] |
| G1-A7+A3 | 27.6 [21.6, 34.1] | **15.0** [10.2, 19.4] |
| A9 (closed) | 29.7 [23.3, 35.8] | 8.8 [3.3, 17.6] |
| A3 | 30.2 [23.7, 36.2] | 11.9 [7.1, 20.7] |

- A top-1 cosine is a poor "not found" signal for every model. With a cut-off that answers half the in-index searches, a quarter or more of searches with no answer still get a confident wrong one. Keeping wrong answers under 5% (the spec target) leaves only 9–15% of in-index searches answered.
- The intervals overlap: no model is clearly better at it. G3 (a learned score over several signals, R3) and a geometric re-ranker (E1) are the next things to try.

### H1 and H3: ask a vision model, search its answer (2026-10-12)

H belongs to R3. We pulled H1 forward because it is cheap: one question per query photo.

Expected (written after the 100-photo probe of Qwen, before the full runs): the probe answers showed that most T1 archive photos are mounted on a card with a typed caption ("JÉZEAU (Hautes-Pyrénées). Eglise. Nef Sud…"), and the model copies it. So we expected H1 to beat every image model on T1 by reading, and to fail on T3, whose queries are blurred black-and-white crops of room photos with no text (checked: every T3 query is a crop, `jdp-<room>-<item>-<n>.jpg`). T5 museum objects carry labels and inventory numbers now and then; we expected a little help there.

Method. `miroir vlm ask` sends each publishable query photo of T1, T3 and T5 tune (138 + 99 + 443 = 680, the same queries as R1) to a vision model, asking in French for JSON: type, likely name, commune, department, region, period or style, author, any visible text, a one-line description and a confidence from 0 to 1. Answers are kept in `~/miroir-data/vlm/<model>.jsonl`. `miroir vlm score` turns each answer into a query (name and commune said twice) and searches it with BM25 over the record text of the gallery (`screening-20k-publishable-fetched`: title, description/LEG, denomination, domain, authors, period, commune, department, region; the record model has no materials field). A record's score goes to each of its images, hidden images leave, an entity scores its best image: the same ranking as the image runs. H3 mixes H1 with A7 (EmbeddingGemma 2) as G1 does: weighted sum of min-max scores, one A7 weight for all three sets, picked from 0.3–0.9 by mean hit@10. Both pick 0.7.

Models: Qwen3-VL 30B-A3B instruct (open; no batch variant on OpenRouter, so standard price) and Gemini 3.5 Flash Lite (closed) through the half-price Batch API. Cost, probes included: Qwen $0.13, Gemini $0.26, $0.39 together. The Batch API refuses images sent as data URLs and Google could not fetch 99 of 100 IIIF URLs in time, so batch requests send the asset bucket URL.

Hit@10 with 95% intervals; paired difference against A7 alone on the same queries:

| Run | T1 | T3 | T5 |
|---|---|---|---|
| A7 alone (R1) | 50.0 [43–58] | **84.8** [78–92] | 48.1 [43–53] |
| H1 Qwen3-VL 30B-A3B | 87.0 [81–92]; +37.0 [+28.3, +44.9] | 9.1 [4–14]; −75.8 | 20.1 [17–24]; −28.0 |
| H1 Gemini 3.5 Flash Lite (closed) | **97.1** [94–99]; +47.1 [+39.9, +54.3] | 22.2 [14–31]; −62.6 | 38.1 [34–43]; −9.9 [−15.8, −4.1] |
| H3 A7 0.7 + H1 Qwen 0.3 | 79.7 [73–86]; +29.7 [+22.5, +37.0] | 82.8 [75–91]; −2.0 [−7.1, +3.0] | 47.0 [42–52]; −1.1 [−4.5, +2.3] |
| H3 A7 0.7 + H1 Gemini 0.3 | 89.1 [84–94]; +39.1 [+31.2, +46.4] | 82.8 [75–90]; −2.0 [−6.1, +1.0] | **53.7** [49–59]; +5.6 [+2.0, +9.3] |

What the models said (from `miroir vlm score`):

| | Qwen T1 | Gemini T1 | Qwen T5 | Gemini T5 |
|---|---|---|---|---|
| Right commune (of 138 / 443) | 79 | 123 | 0 | 9 |
| … of which read in the image | 76 | 113 | 0 | 2 |
| Gave a name / name matches the record | 103 / 84 | 135 / 118 | 36 / 16 | 82 / 41 |
| Read some text | 136 | 136 | 276 | 269 |
| Hit@10, confidence ≥ 0.8 | 90% (134 q) | 99% (136 q) | 25% (147 q) | 69% (91 q) |
| Hit@10, confidence < 0.5 | 0% (2 q) | 0% (2 q) | 11% (123 q) | 28% (318 q) |

What we learned:

- On T1, H1 wins by reading, not by knowing buildings. 113 of Gemini's 123 right communes, and 76 of Qwen's 79, were copied from the caption typed on the photo's mount. Every one of the 54 T1 queries that H3 Gemini gains over A7 has such a caption. This holds for archive prints in Mémoire; it says nothing about phone photos of a building (T2), where there is no caption.
- H1 alone fails on T3 (no text, blurred crops) and trails A7 on T5. Mixed with A7, it costs nothing beyond the error bar on T3 and, with Gemini, gains 5.6 points on T5 [+2.0, +9.3]. 41 of those 45 T5 gains come from read text: medal legends, signatures, inventory labels, captions on the photo.
- Gemini beats Qwen on every set beyond the error bar (H1 hit@10: T1 +10.1, T3 +13.1, T5 +18.1). By experiments.md rule 1 that is a closed model winning; Qwen still makes H3 a large T1 gain at no loss elsewhere.
- Gemini's confidence means something on T5 (69% hit@10 when ≥ 0.8, 28% below 0.5); Qwen says 0.9 to nearly everything. A G3 score could use Gemini's confidence to decide how much to trust the text.
- One weight for all sets is again the catch: on T1, H1 Gemini alone (97.1) beats the mix (89.1), because A7 at 0.7 drowns a sure text match. Weighting by the model's confidence, or by whether it read a place name, is the obvious next step (G3).
- Text the model reads but our index cannot use: inventory numbers on museum labels ("2019.0.998") are not in the record model. Adding the Joconde inventory field to the records would let BM25 match them exactly.

### G4: weights chosen per photo by a building-or-object guess (2026-10-12)

G1 gives every photo the same weight, but T1 (archive photos of buildings) wants EmbeddingGemma 2 (A7) and T5 (museum objects) wants MegaLoc (A3). G4 guesses from the photo's own vectors whether it shows a building or an object, then sets A7's weight: `p(building) × w_building + (1 − p) × w_object`, or a hard switch at p > 0.5. Run with `miroir eval route` (technical spec §7.2).

Expected (written before the run): the guess should be easy, since buildings and objects look nothing alike; the gain should come on T1 and T5, where G1 lost to the best single model.

**The guess.** A logistic regression on A7 vectors (no model runs; NumPy on the Mac). Labels come from the records: Joconde, Rose Valland and Mémoire photos linked to a Palissy object (PM) are objects; Mémoire photos linked to a Mérimée building (PA) are buildings; 538 Mémoire photos linked to neither are left out. Every building or object of a T1, T3, T5 or T8 tune query (826) is kept out of training, leaving 12,980 gallery images. The penalty (0.1) was picked on a fifth of the gallery's buildings and objects held out.

| Guess reads | Held-out gallery, balanced | T1 (138) | T3 (99) | T5 (443) | T8 (232) |
|---|---|---|---|---|---|
| A7 | 98.7% | 98.6% | 100% | 99.8% | 99.6% |
| A7 + A3 side by side | 98.7% | 98.6% | 100% | 100% | 99.6% |

Adding MegaLoc's 8,448 numbers buys nothing: one more T5 photo right, at about 20 times the training time. Routing on it (`G4-router-A7+A3-guessA7A3`, weights 1.0 / 0.1) scores T1 51.4, T3 89.9, T5 62.1, within a point of the A7 guess. The rest of this section uses the A7 guess.

**Ranking.** The grid (A7 weight 0 to 1 in steps of 0.1, one pair for T1, T3 and T5 together) picks A7 1.0 for buildings and 0.2 for objects; the hard switch picks 0.9 and 0.2. Hit@10, 95% interval:

| Run | Weights (buildings / objects) | T1 | T3 | T5 | Mean |
|---|---|---|---|---|---|
| G4-switch-A7+A3 | 0.9 / 0.2, switch | **52.2** [45–60] | **89.9** [84–96] | 61.4 [57–66] | **67.8** |
| G4-router-A7+A3 | 1.0 / 0.2, blend | 51.4 [44–59] | **89.9** [84–96] | 61.4 [57–66] | 67.6 |
| G1-A7+A3 | 0.7 for all | 45.7 [38–54] | **89.9** [84–96] | 54.9 [50–59] | 63.5 |
| best single (R1) | | A9 51.4 / A7 50.0 | A3 85.9 | A3 **62.5** | |

Paired bootstrap, G4-router-A7+A3 minus the other run, hit@10 (the switch gives the same verdicts):

| Set | − G1-A7+A3 | − best single |
|---|---|---|
| T1 | **+5.8** [+2.2, +10.1] | A7: +1.4 [0.0, +3.6]; A9: 0.0 [−5.1, +5.1] |
| T3 | 0.0 [−3.0, +3.0] | A3: **+4.0** [+1.0, +9.1] |
| T5 | **+6.5** [+3.8, +9.3] | A3: −1.1 [−3.2, +0.9] |

With one building set, the building weight is in effect tuned on T1 alone, which flatters G4. A two-fold check picks the weights on half of each set's buildings and objects and scores the other half: G4 T1 50.7, T3 86.9, T5 61.2, against G1 on the same grid 46.4, 88.9, 53.0. The gain holds.

**Not found (T8).** Confidence is the routed mix of the top entity's cosines. With one cut-off, 66.8% [59.9, 73.7] of T8 searches still get a confident wrong answer at 50% matched (G1: 27.6%), because A7 and A3 cosines sit on different scales and routing puts buildings on one and objects on the other. A cut-off per guessed type (each confidence replaced by its rank among in-index confidences of the same type; this is the T8 row in the summary) brings it to 42.2% [35.3, 50.4] for the blend and 36.2% [30.2, 43.5] for the switch, still worse than G1 and A7 (25.0%). Matched at 5% wrong: 14.8% and 16.5%, against G1's 15.0%.

What we learned:

- Our guess held: telling buildings from objects is near-perfect from A7 vectors alone (98.6–100% on tune queries).
- G4 removes G1's compromise. It beats G1 on T1 and T5 beyond the error bar, ties it on T3, and matches the best single model on every set (beats A3 on T3). Mean hit@10 rises from 63.5 to 67.6.
- The grid gives buildings to A7 alone and objects mostly to A3. This is the reverse of the idea that started G4 (MegaLoc for streets, EmbeddingGemma for objects): MegaLoc wins on museum objects here, EmbeddingGemma on archive building photos.
- Blend and switch tie: the guess is almost always sure, and the top 10 differs on one query in 680. Keep the switch; it is simpler to explain.
- The guess does not help "not found". Keep a separate confidence score (A7's or G1's cosines) for that, and leave the cut-off to G3.
- Before R4: the guess learned from gallery images of entities that may be in sealed queries (we cannot see which). Retrain it in R4 with every sealed-query entity left out too.

### E1: re-ranking the top 50 by keypoint matches (2026-10-12)

We wrote no guess before these runs; the reasons below came after the scores.

Set-up: for each tune query (the same queries as R1), the top 50 entities by cosine, each with its best image. ALIKED keypoints (2,048 at most, on the 512 px images; BSD licence, where SuperPoint's weights forbid commercial use), LightGlue matching, then OpenCV MAGSAC fits one homography (4 px). Two ways to re-rank, the second fixed before scoring:

- **inliers**: matches that agree on the homography, most first;
- **mix**: cosine + inliers / 100.

Matching ran on a Modal L4, one pair after another. It adds 1.2 s per search at the median (0.7 to 1.4 s by set, 95th percentile at most 1.6 s), with the gallery's keypoints computed ahead. Cost: $0.71 for the A7 shortlists, $0.52 for A3.

Each cell: hit@1 / hit@10. Last two columns: paired difference, mix minus the shortlist's model alone, with 95% intervals.

| Shortlist | Set | Alone | Inliers | Mix | Mix − alone, hit@1 | Mix − alone, hit@10 |
|---|---|---|---|---|---|---|
| A7 EmbeddingGemma 2 | T1 | 36.2 / 50.0 | 23.2 / 44.9 | 25.4 / 46.4 | −10.9 [−17.4, −4.3] | −3.6 [−10.1, +2.9] |
| A7 EmbeddingGemma 2 | T3 | 56.6 / 84.8 | 71.7 / 91.9 | **72.7 / 92.9** | +16.2 [+8.1, +24.2] | +8.1 [+2.0, +14.1] |
| A7 EmbeddingGemma 2 | T5 | 33.4 / 48.1 | 31.8 / 45.4 | 37.9 / 50.3 | +4.5 [+0.7, +8.4] | +2.3 [−0.7, +5.4] |
| A3 MegaLoc | T1 | 20.3 / 29.7 | 20.3 / 29.0 | 21.0 / 29.7 | +0.7 [−4.3, +5.8] | +0.0 [−4.3, +4.3] |
| A3 MegaLoc | T3 | 55.6 / 85.9 | 67.7 / 90.9 | 66.7 / 92.9 | +11.1 [+3.0, +19.2] | +7.1 [+3.0, +12.1] |
| A3 MegaLoc | T5 | 39.5 / 62.5 | 35.2 / 54.0 | **44.7 / 63.2** | +5.2 [+2.0, +8.6] | +0.7 [−2.0, +3.8] |

What we learned:

- T3 gains most: crops of one negative share every detail, so the right record gets 195 inliers at the median against 6 for wrong ones. A7 then mix reaches 72.7 hit@1, 16 points above any model alone.
- T1 loses with A7. Wrong buildings still get 27 inliers at the median (right ones 51): archive prints share borders, mounts and repeated stonework. With 49 wrong candidates, one often beats the right one. On A3's shortlist the gains and losses cancel.
- On T5, inliers alone hurt; the mix lifts hit@1 by about 5 points on both shortlists, and A3 + mix is the best T5 result so far (44.7 / 63.2).
- The mix scores at or above inliers alone everywhere but A3 on T3 at hit@1 (one point lower). Its weight was not tuned.
- 1.2 s per search is slow for a website; matching the 50 pairs in one batch would cut it.

## Commons label check (2026-10-12)

The hand check of 100 tune queries that technical spec §7 asks for: 70 drawn from T2 (516 tune queries), 30 from T2b (291). For 79 pairs we set the Commons photo beside a POP image of the answer. For the 21 whose answer is not publishable (1 T2, 20 T2b) we judged from the photo, the record's title and description, and the Commons file title and category. One verdict per line, with its reason and how it was judged: `eval/labels/commons-check-100.jsonl`.

Verdicts: **same**, the photo shows the labelled building or work; **part**, it does, but only an interior, a small detail, the other face of a coin or the side of a sarcophagus: a fair label, a hard match; **wrong-nearby**, a neighbouring building, or the right one far in the background; **wrong-other**, something else; **unsure**.

| Set | Pairs | Same | Part | Wrong-nearby | Wrong-other | Unsure | Wrong labels, 95% interval |
|---|---|---|---|---|---|---|---|
| T2 | 70 | 44 | 23 | 1 | 2 | 0 | **4.3%** [1.5, 11.9] |
| T2b | 30 | 24 | 5 | 0 | 0 | 1 | **0%** [0, 11.4]; 3.3% [0.6, 16.7] if the unsure one is wrong |

Wilson score intervals. The three wrong T2 labels: a tomb now in the Agen museum, filed in the Château de Lafox category ("Agen - Musée des Beaux-Arts - Tombeau"); a Canada goose on the lawn of Plessis-Bourré; "Torsac 3", a village view with the church spire behind trees and a château. The unsure T2b pair is a gladiator lamp: title and scene match the record, but its cracks differ from the POP image's.

What we learned:

- Wrong labels come only from files read out of the item's category: 3 of 45 (6.7%). The 19 P18 photos and the 6 from interior subcategories hold none. Two of the three have a title that names only the commune, or another place.
- A third of T2 is "part" (23 of 70), mostly interiors and details. The view guess misses many: 5 of the 48 photos guessed exterior are interiors (a nave, a cellar, a council room, a painted vault), and 3 more are close details. Of photos guessed interior or detail, 15 of 22 are part. These labels are right, but they test a different task from finding a building by its outside.
- T2b labels are clean. Its Commons categories carry the Musée Saint-Raymond inventory number, and every one we checked agrees with its record. The hard cases are views: two coins show the other face (both files end in `-R`, though a third `-R` file shows the face the record describes), two sarcophagi a short side, one head its back.

Proposed filters for the T2 builder (the sets are unchanged here). Shares are of the T2 tune half; "sample after" counts what stays of the 70 checked pairs.

| Rule | T2 tune kept | Sample after: same / part / wrong | Wrong labels |
|---|---|---|---|
| None (today) | 516 | 44 / 23 / 3 | 4.3% [1.5, 11.9] |
| A. Drop a category file whose title shares no distinctive word with the category name (`names_place` false) | 490 (95%) | 44 / 22 / 2 | 2.9% |
| B. Drop a title that holds only the commune and numbers ("Torsac 3", "Mazangé (11)") | 474 (92%) | 42 / 18 / 2 | 3.2% |
| A and B | **451 (87%)** | 42 / 17 / 1 | **1.7%** [0.3, 8.9] |
| A and B, exterior guess only | 302 (59%) | 36 / 6 / 1 | 2.3% [0.4, 12.1]; part falls from 33% to 14% |

- Apply A and B: they cut the wrong labels by more than half and cost 13% of the set. B also drops 2 good photos and 5 part ones in the sample.
- Keep interiors and details, but score them as their own slice rather than drop them: the view guess is wrong for about 1 photo in 10, so the slice needs a better guess before it means much.
- No word rule catches the goose: its title, "Branta canadensis Plessis", names the place. One such photo in 70 is noise we can live with.
- T2b needs no label filter. Its coin and sarcophagus views are hard but right; report them, do not drop them.

## R2: phone photos of buildings (G5, F2, F1)

T2, a visitor's colour photo of a building, sits far below its 60% target, and the router of G4 sends every building photo to EmbeddingGemma 2 (A7), the archive-print winner. Three cheap fixes, in order of cost. Gallery `screening-v2-publishable-fetched` (21,785 images) unless named; tune halves; queries a closed model may see (T1 138, T2 516, T3 99, T5 443). The label check above found T2's labels 96% right, so the gap is real.

### G5: three buckets, archive building, colour building, object (2026-10-12)

Expected (written before the run): T2 photos are buildings, so G4 gives them A7 weight 0.9 and loses to MegaLoc (A3) alone. A third bucket for colour building photos should take a low A7 weight, lift T2 to A3's level or a little above, and leave T1, T3 and T5 as they are.

Method. `miroir eval route3` (technical spec §7.2). A building photo (G4's guess, p > 0.5) is an archive building when its chroma spread, the standard deviation of Cb and Cr over the pixels at 128 px, is under 2; else a colour building. No model: Mémoire prints are stored grey and score 0; 1% of T2 photos score under 3. One A7 weight per bucket (A3 takes the rest; min-max weighted sum, hard switch), picked from the 0–1 grid by mean hit@10 over T1, T2, T3 and T5 together. Each set is ranked once per grid weight, so G1, G4 and G5 are all read from the same rankings. The guess leaves out T2 tune buildings too.

Buckets found: T1 136 archive, 2 object; T3 and T5 all object; **T2 2 archive, 209 colour building, 305 object**. The building-or-object guess, trained on grey Mémoire buildings and mostly colour museum objects, calls 59% of phone photos of buildings objects: it partly learned colour.

Hit@10, 95% interval in brackets:

| Run | A7 weight: archive / colour / object | T1 | T2 | T3 | T5 | Mean |
|---|---|---|---|---|---|---|
| **G5-router3-A7+A3** | 1.0 / 0.4 / 0.2 | **50.7** [43.5, 58.7] | **32.4** [28.7, 36.4] | 89.9 [83.8, 96.0] | 61.2 [56.4, 65.7] | **58.6** |
| G4 (picked on the same four sets) | 0.9 / 0.9 / 0.2 | **50.7** | 28.5 [24.8, 32.6] | 89.9 | 61.2 | 57.6 |
| G1 | 0.6 for all | 41.3 | 32.0 | **90.9** | 56.4 | 55.2 |
| A3 alone | | 28.3 | 31.0 [27.3, 35.1] | 85.9 | **62.1** | 51.8 |
| A7 alone | | 50.0 | 21.9 | 84.8 | 47.9 | 51.2 |

The G4 grid picks its old weights (0.9 / 0.2) again with T2 added. Paired bootstrap, hit@10, G5 minus:

| Set | G4 | Best single model |
|---|---|---|
| T1 | 0.0 [−2.2, +2.2] | A7: +0.7 [0.0, +2.2] |
| T2 | **+3.9** [+2.1, +6.0] | A3: **+1.4** [+0.4, +2.5] |
| T3 | 0.0 | A3: **+4.0** [+1.0, +9.1] |
| T5 | 0.0 | A3: −0.9 [−3.2, +1.1] |

Split-half check (weights picked on half the buildings and objects, scored on the other half): G5 T1 50.0, T2 31.8, T3 89.9, T5 60.5; G4 48.6, 29.7, 89.9, 60.5; G1 42.8, 31.8, 89.9, 55.3. T2 exterior photos (346): G5 34.4, A3 32.7, A7 24.0.

What we learned:

- The guess held in direction, not in size: a colour bucket fixes G4's T2 loss (+3.9 points) at no cost on T1, T3 or T5, and G5 now matches or beats the best single model on every set. But T2 only edges past A3 alone (+1.4): routing cannot make a phone photo easier, only stop sending it to the wrong model.
- The grid gives colour buildings A7 0.4 and objects 0.2, so 514 of 516 T2 photos lean on MegaLoc either way. Whether the guess calls a phone photo a building or an object matters little here; it would matter for a mixed stack where objects should get a different model.
- For the prototype: route on colour first. A grey photo of a building goes to A7 alone; anything in colour gets A7 0.2–0.4 and A3 the rest.

### F1: grey query photos, then grey everything (2026-10-13)

Expected (written before the run): Mémoire prints are grey, phone photos are colour; greying the query should help T2 a little, more for A7 (trained on colour pairs) than for A3. T1 should not move: its queries and answers are already grey.

Method. `miroir embed modal-grey` turns every image with colour (any pixel's R, G or B more than 1 level from its luma on average) grey with Pillow on the Mac and sends the bytes to Modal: 5,162 of 22,302 images (gallery plus T1 and T2 tune queries), $0.27 for both models. Images already grey keep their vectors: greying them changes no pixel beyond JPEG noise. Query only: grey query vectors against the usual gallery (`F-F1-query-*`). Both: the gallery's colour images greyed too (`F-F1-both-*`). Baselines `F-base-A7-v2`, `F-base-A3-v2` are the same models on the same gallery and queries. Hit@10, paired difference with the baseline:

| Run | T1 | T2 | T5 (side check) |
|---|---|---|---|
| A3, grey query | 28.3; 0.0 | **32.9; +1.9** [+0.4, +3.7] | 55.3; −6.8 |
| A3, grey query and gallery | 28.3; 0.0 | 32.4; +1.4 [−0.2, +3.1] | 58.0; −4.1 |
| A7, grey query | 50.0; 0.0 | 20.5; −1.4 [−2.7, 0.0] | 40.0; −7.9 |
| A7, grey query and gallery | 49.3; −0.7 [−2.2, 0.0] | 20.2; −1.7 [−3.3, −0.4] | 44.5; −3.4 |

What we learned:

- Our guess was half wrong. Grey queries help MegaLoc a little on T2 (+1.9, just beyond the error bar) and hurt EmbeddingGemma 2; colour is not what keeps A7 from finding buildings in phone photos.
- T1 does not move, as expected: its queries and answers are grey already.
- Greying museum objects costs T5 4–8 points: colour tells objects apart. If used at all, grey only a colour building photo's MegaLoc query, never the gallery. The gain is too small to be worth a second pass in the prototype.

### F2: Commons photos as gallery images of each building (2026-10-13)

Expected (written before the run): a phone photo is closer to another visitor's colour photo of the same building than to a 1950s print. Adding a few Commons photos per building should lift T2 by several points, most with MegaLoc, and leave T1 and T5 alone (their queries are grey prints and museum objects).

Method. `miroir testsets commons-gallery --gallery screening-v2-publishable-fetched` takes each Mérimée building of the gallery (1,840; 1,495 with one unambiguous Wikidata item), reads its P18 file and the 50 newest files of its Commons category, keeps those that pass T2's photo rules plus filters A and B of the label check (category files only), puts exteriors and new visits first, and keeps up to 5. Left out so no test query reaches the gallery: every T2, T2b and T8c photo, tune and sealed (1,107 files), any file within 12 bits of one by perceptual hash (48), any file by the author and day of a query photo of the same building (1,812), and any file within 12 bits of a POP image of the same building (none). Result: **4,588 photos for 1,208 buildings**, read as 500 px copies; 5,809 requests to Wikimedia at about 1.4 a second (70 minutes). Each counts as an image of its building (`commons-gallery-*.parquet`, gallery `screening-v2-publishable-fetched+F2`, 26,373 images). Embedded on Modal: A3 $0.11, A7 $0.15. Same queries as above.

Hit@10, paired difference with the same run without F2 photos:

| Run | T1 | T2 | T2 exterior (346) | T3 | T5 |
|---|---|---|---|---|---|
| A3 + F2 | 28.3; 0.0 | **40.5; +9.5** [+5.8, +13.0] | 42.8 (was 32.7) | | 62.1; 0.0 |
| A7 + F2 | 50.0; 0.0 | 21.9; 0.0 [−4.5, +4.3] | 23.1 (was 24.0) | | 47.9; 0.0 |
| **G5 + F2**, weights re-picked (1.0 / 0.0 / 0.2) | **50.7**; 0.0 | **40.5; +8.1** [+4.7, +11.8] | 42.8 (was 34.4) | 89.9 | 61.2; 0.0 |
| G5 + F2, weights of G5 above (1.0 / 0.4 / 0.2) | 50.7; 0.0 | 38.4; +6.0 [+2.5, +10.1] | | 89.9 | 61.2; 0.0 |

With F2 the grid sends colour buildings to MegaLoc alone (A7 weight 0.0). Split-half: G5 T1 50.0, T2 40.3, T3 89.9, T5 60.5.

The gain sits where the photos are. Only 80 of the 193 T2 tune buildings got F2 photos: the T2 builder had already taken their best Commons files, and other frames of those visits are left out. On the 239 queries of those 80 buildings, G5 hit@10 goes **from 30.5 to 58.6**; on the 277 others it falls from 33.9 to 24.9, since 4,588 colour photos of other buildings now compete with their grey answers.

T8c, against T2 and T2b tune on the F2 gallery (R1 numbers without F2 in brackets):

| Run | Wrong but confident, cut-off at 50% matched | Matched, cut-off at 5% wrong |
|---|---|---|
| A3 + F2 | 26.7 [17.3, 34.7] (40.0) | 35.4 [29.5, 40.1] (21.5) |
| A7 + F2 | 11.3 [6.0, 16.7] (36.0) | 31.5 [26.3, 43.1] (19.6) |

What we learned:

- F2 is the largest T2 gain so far: +9.5 points for MegaLoc, +8.1 for G5, beyond the error bar, at no cost on T1, T3 or T5. A building with Commons photos is found from a phone photo 59% of the time, at the 60% target.
- A building without them loses ground, because the gallery's colour photos all belong to others. So F2 must cover every building, not some: in the full index, give every Mérimée building its Commons photos.
- EmbeddingGemma 2 gains nothing: it ranks colour photos of the wrong building above the right one (F2 photos sit close together in its space, mean cosine 0.72 against 0.04 for MegaLoc). Phone photos of buildings belong to MegaLoc.
- "Not found" improves for both models: right answers now match a near-identical colour photo, so their confidences rise above those of photos with no answer.
- Full sample: 3,924 buildings, about 13,000 requests, about 2.6 hours at the pace measured here (0.9 hours at the 4-a-second cap) and about 9,800 photos; embedding them with A3 and A7 costs about $0.60.

## Summary of all runs

<!-- summary:start (generated by `miroir eval summary`; do not edit) -->
Each cell: value in %, then the 95% bootstrap interval (1,000 resamples over queries).

| Round | Set | Half | Gallery | Run | Model | Px | Pooling | Queries | Hit@1 | Hit@5 | Hit@10 | mAP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | amstertime-all | all | amstertime | A-dinov3-vit7b16-224 | facebook/dinov3-vit7b16-pretrain-lvd1689m | 224 | cls | 2,462 | 47.1 [45.1, 49.1] | 73.6 [71.9, 75.4] | 81.1 [79.5, 82.5] | 59.0 [57.4, 60.6] |
| R0 | amstertime-all | all | amstertime | A-dinov2-giant-224 | facebook/dinov2-giant | 224 | cls | 2,462 | 36.6 [34.7, 38.4] | 58.0 [56.2, 59.9] | 65.6 [63.9, 67.4] | 46.7 [45.0, 48.4] |
| R0 | amstertime-new | all | amstertime | A-dinov3-vit7b16-224 | facebook/dinov3-vit7b16-pretrain-lvd1689m | 224 | cls | 1,231 | 44.8 [42.0, 47.7] | 72.2 [69.6, 74.8] | 80.0 [77.6, 82.3] | 57.1 [54.7, 59.3] |
| R0 | amstertime-new | all | amstertime | A-dinov2-giant-224 | facebook/dinov2-giant | 224 | cls | 1,231 | 32.7 [30.0, 35.4] | 53.9 [51.0, 56.8] | 61.3 [58.5, 64.0] | 42.7 [40.2, 45.1] |
| R0 | amstertime-old | all | amstertime | A-dinov3-vit7b16-224 | facebook/dinov3-vit7b16-pretrain-lvd1689m | 224 | cls | 1,231 | 49.3 [46.3, 52.2] | 75.1 [72.5, 77.5] | 82.1 [79.9, 84.2] | 61.0 [58.5, 63.4] |
| R0 | amstertime-old | all | amstertime | A-dinov2-giant-224 | facebook/dinov2-giant | 224 | cls | 1,231 | 40.4 [37.6, 43.1] | 62.1 [59.5, 64.8] | 70.0 [67.6, 72.5] | 50.7 [48.4, 53.3] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A7-embeddinggemma-2-512 | google/embeddinggemma-2 | 512 | head | 138 | 36.2 [29.0, 44.2] | 49.3 [42.0, 57.2] | 50.0 [42.8, 58.0] | 42.2 [35.4, 49.5] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A9 | google/gemini-embedding-2 | 512 | api | 138 | 33.3 [26.1, 40.6] | 45.7 [37.7, 53.6] | 51.4 [44.2, 59.4] | 38.9 [32.1, 46.0] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A10a | voyageai/voyage-multimodal-3.5 | 512 | api | 138 | 29.7 [22.5, 37.7] | 35.5 [28.3, 43.5] | 40.6 [33.3, 49.3] | 33.6 [26.8, 41.4] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A4-siglip2-so400m-384 | google/siglip2-so400m-patch14-384 | 384 | head | 138 | 23.2 [16.7, 30.4] | 34.1 [26.8, 42.0] | 37.0 [29.0, 44.9] | 28.4 [22.3, 35.0] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A8-qwen3-vl-embedding-2b-512 | Qwen/Qwen3-VL-Embedding-2B | 512 | head | 138 | 24.6 [18.1, 31.9] | 29.7 [23.2, 37.0] | 31.9 [24.6, 39.1] | 27.7 [21.1, 34.9] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-7b-256-nocrop-cls | facebook/dinov3-vit7b16-pretrain-lvd1689m | 256 | cls | 138 | 23.2 [15.9, 29.7] | 31.2 [23.2, 39.1] | 36.2 [29.0, 44.2] | 27.6 [21.1, 34.5] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-cls | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | cls | 138 | 24.6 [18.1, 31.9] | 30.4 [23.2, 37.7] | 33.3 [25.4, 40.6] | 27.6 [20.8, 34.6] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A5-pe-core-l-336 | timm/PE-Core-L-14-336 | 336 | head | 138 | 23.2 [16.7, 29.7] | 32.6 [25.4, 40.6] | 35.5 [28.3, 43.5] | 27.1 [20.8, 33.9] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-7b-256-cls | facebook/dinov3-vit7b16-pretrain-lvd1689m | 256 | cls | 138 | 22.5 [15.9, 29.7] | 31.2 [23.2, 38.4] | 33.3 [26.1, 40.6] | 26.5 [20.0, 33.5] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-nocrop-cls | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | cls | 138 | 21.0 [14.5, 28.3] | 29.7 [22.5, 37.0] | 34.1 [26.8, 41.3] | 25.2 [18.8, 31.8] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-cls | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | cls | 138 | 20.3 [13.8, 27.5] | 27.5 [20.3, 34.8] | 29.0 [21.7, 36.2] | 24.2 [17.5, 31.3] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-nocrop-cls | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | cls | 138 | 20.3 [14.5, 26.8] | 26.8 [19.6, 34.8] | 29.7 [22.5, 37.0] | 24.0 [18.1, 30.8] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322 | gmberton/MegaLoc | 322 | salad | 138 | 20.3 [14.5, 26.8] | 26.1 [18.8, 33.3] | 29.7 [22.5, 37.0] | 23.3 [17.3, 30.0] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-nocrop-cls | facebook/dinov2-giant | 224 | cls | 138 | 15.2 [9.4, 21.0] | 24.6 [18.1, 31.9] | 29.7 [22.5, 37.0] | 20.0 [14.6, 25.8] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-gem | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | gem | 138 | 15.2 [9.4, 21.7] | 23.9 [17.4, 31.2] | 26.8 [19.6, 34.1] | 19.8 [14.2, 25.8] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A1-dinov2-l-224-cls | facebook/dinov2-large | 224 | cls | 138 | 14.5 [8.7, 20.3] | 21.0 [14.5, 28.3] | 25.4 [18.1, 32.6] | 18.0 [12.5, 23.7] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-cls | facebook/dinov2-giant | 224 | cls | 138 | 13.0 [8.0, 18.8] | 22.5 [15.9, 29.7] | 24.6 [17.4, 31.9] | 17.8 [12.5, 23.4] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-nocrop-gem | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | gem | 138 | 13.0 [8.0, 18.8] | 21.0 [15.2, 27.5] | 26.1 [18.8, 32.6] | 17.1 [12.4, 22.5] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-7b-256-nocrop-gem | facebook/dinov3-vit7b16-pretrain-lvd1689m | 256 | gem | 138 | 13.8 [8.7, 19.6] | 18.1 [12.3, 24.6] | 26.8 [19.6, 34.1] | 16.8 [11.8, 22.5] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-gem | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | gem | 138 | 12.3 [7.2, 18.1] | 18.1 [12.3, 24.6] | 21.7 [15.2, 28.3] | 15.1 [10.2, 21.0] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-nocrop-gem | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | gem | 138 | 11.6 [6.5, 16.7] | 16.7 [10.9, 23.2] | 21.7 [15.2, 29.0] | 14.4 [9.8, 19.7] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-nocrop-gem | facebook/dinov2-giant | 224 | gem | 138 | 10.9 [5.8, 15.9] | 14.5 [9.4, 20.3] | 18.1 [12.3, 25.4] | 13.4 [8.6, 18.4] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A1-dinov2-l-224-gem | facebook/dinov2-large | 224 | gem | 138 | 7.2 [3.6, 12.3] | 15.2 [9.4, 21.0] | 19.6 [13.8, 26.1] | 11.5 [7.4, 16.1] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-gem | facebook/dinov2-giant | 224 | gem | 138 | 7.2 [3.6, 11.6] | 13.8 [8.0, 19.6] | 19.6 [13.8, 26.1] | 11.4 [7.4, 16.2] |
| R1 | T2 | tune | screening-v2-publishable-fetched | A-A3-megaloc-322-v2 | gmberton/MegaLoc | 322 | salad | 516 | 23.8 [20.5, 27.7] | 28.9 [25.4, 32.9] | 31.0 [27.3, 35.1] | 26.6 [23.2, 30.3] |
| R1 | T2 | tune | screening-v2-publishable-fetched | A-A9-v2 | google/gemini-embedding-2 | 512 | api | 516 | 10.3 [7.9, 13.0] | 21.7 [18.4, 25.2] | 24.8 [21.3, 28.7] | 16.1 [13.6, 18.9] |
| R1 | T2 | tune | screening-v2-publishable-fetched | A-A7-embeddinggemma-2-512-v2 | google/embeddinggemma-2 | 512 | head | 516 | 9.1 [6.8, 11.8] | 18.2 [15.1, 21.7] | 21.9 [18.4, 25.6] | 13.6 [11.3, 16.4] |
| R1 | T2 | tune | screening-v2-publishable-fetched | A-A5-pe-core-l-336-v2 | timm/PE-Core-L-14-336 | 336 | head | 516 | 6.6 [4.5, 8.7] | 15.5 [12.4, 18.6] | 19.4 [16.1, 23.1] | 11.2 [9.0, 13.6] |
| R1 | T2 | tune | screening-v2-publishable-fetched | A-A4-siglip2-so400m-384-v2 | google/siglip2-so400m-patch14-384 | 384 | head | 516 | 1.6 [0.6, 2.7] | 6.0 [4.1, 8.1] | 9.1 [6.8, 11.6] | 4.2 [2.9, 5.4] |
| R1 | T2b | tune | screening-v2-publishable-fetched | A-A9-v2 | google/gemini-embedding-2 | 512 | api | 122 | 87.7 [82.0, 93.4] | 95.9 [91.8, 99.2] | 97.5 [94.3, 100.0] | 91.1 [86.7, 95.5] |
| R1 | T2b | tune | screening-v2-publishable-fetched | A-A7-embeddinggemma-2-512-v2 | google/embeddinggemma-2 | 512 | head | 122 | 86.1 [79.5, 91.8] | 92.6 [87.7, 96.7] | 97.5 [94.3, 100.0] | 89.5 [84.7, 94.1] |
| R1 | T2b | tune | screening-v2-publishable-fetched | A-A3-megaloc-322-v2 | gmberton/MegaLoc | 322 | salad | 122 | 82.8 [75.4, 89.3] | 92.6 [87.7, 96.7] | 92.6 [87.7, 96.7] | 87.2 [81.9, 92.3] |
| R1 | T2b | tune | screening-v2-publishable-fetched | A-A5-pe-core-l-336-v2 | timm/PE-Core-L-14-336 | 336 | head | 122 | 79.5 [72.1, 86.9] | 89.3 [83.6, 94.3] | 94.3 [89.3, 97.5] | 84.4 [78.3, 90.0] |
| R1 | T2b | tune | screening-v2-publishable-fetched | A-A4-siglip2-so400m-384-v2 | google/siglip2-so400m-patch14-384 | 384 | head | 122 | 76.2 [68.9, 83.6] | 91.0 [85.2, 95.1] | 95.1 [91.0, 98.4] | 83.0 [77.2, 88.5] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322 | gmberton/MegaLoc | 322 | salad | 99 | 55.6 [46.5, 65.7] | 83.8 [76.7, 90.9] | 85.9 [78.8, 91.9] | 67.5 [59.8, 75.6] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A7-embeddinggemma-2-512 | google/embeddinggemma-2 | 512 | head | 99 | 56.6 [47.4, 66.7] | 78.8 [70.7, 86.9] | 84.8 [77.8, 91.9] | 66.8 [59.0, 75.1] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A9 | google/gemini-embedding-2 | 512 | api | 99 | 54.5 [44.4, 64.6] | 77.8 [69.7, 85.9] | 83.8 [75.8, 90.9] | 65.2 [57.4, 73.6] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A4-siglip2-so400m-384 | google/siglip2-so400m-patch14-384 | 384 | head | 99 | 49.5 [39.4, 59.6] | 73.7 [64.6, 81.8] | 80.8 [72.7, 87.9] | 60.2 [52.2, 68.2] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A5-pe-core-l-336 | timm/PE-Core-L-14-336 | 336 | head | 99 | 45.5 [35.4, 54.6] | 68.7 [59.6, 76.8] | 79.8 [70.7, 87.9] | 55.7 [47.6, 63.9] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A8-qwen3-vl-embedding-2b-512 | Qwen/Qwen3-VL-Embedding-2B | 512 | head | 99 | 39.4 [30.3, 49.5] | 65.7 [56.6, 74.7] | 75.8 [67.7, 83.8] | 50.5 [42.9, 59.0] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-7b-256-nocrop-gem | facebook/dinov3-vit7b16-pretrain-lvd1689m | 256 | gem | 99 | 35.4 [26.3, 45.5] | 63.6 [53.5, 72.7] | 70.7 [61.6, 79.8] | 48.5 [40.2, 57.3] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-cls | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | cls | 99 | 37.4 [27.3, 47.5] | 55.6 [46.5, 65.7] | 62.6 [53.5, 71.7] | 46.5 [38.2, 55.2] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-gem | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | gem | 99 | 34.3 [24.2, 43.4] | 56.6 [47.5, 66.7] | 61.6 [52.5, 70.7] | 44.5 [36.1, 52.9] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-7b-256-nocrop-cls | facebook/dinov3-vit7b16-pretrain-lvd1689m | 256 | cls | 99 | 31.3 [22.2, 40.4] | 53.5 [43.4, 63.6] | 65.7 [56.6, 74.7] | 43.1 [35.1, 51.2] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-7b-256-cls | facebook/dinov3-vit7b16-pretrain-lvd1689m | 256 | cls | 99 | 32.3 [23.2, 41.4] | 50.5 [40.4, 59.6] | 59.6 [49.5, 68.7] | 42.1 [33.5, 50.5] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-nocrop-gem | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | gem | 99 | 30.3 [21.2, 39.4] | 51.5 [42.4, 61.6] | 60.6 [51.5, 69.7] | 41.3 [33.5, 49.4] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-cls | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | cls | 99 | 30.3 [21.2, 39.4] | 51.5 [41.4, 60.6] | 55.6 [45.5, 64.6] | 40.7 [32.5, 49.1] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-nocrop-cls | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | cls | 99 | 30.3 [21.2, 38.4] | 50.5 [40.4, 59.6] | 60.6 [51.5, 69.7] | 40.6 [32.5, 48.4] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-gem | facebook/dinov2-giant | 224 | gem | 99 | 30.3 [21.2, 39.4] | 50.5 [41.4, 60.6] | 57.6 [48.5, 67.7] | 40.5 [32.7, 48.7] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-gem | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | gem | 99 | 30.3 [21.2, 39.4] | 48.5 [39.4, 57.6] | 59.6 [50.5, 68.7] | 40.5 [32.1, 48.4] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A1-dinov2-l-224-cls | facebook/dinov2-large | 224 | cls | 99 | 33.3 [24.2, 42.4] | 46.5 [36.4, 55.6] | 50.5 [40.4, 59.6] | 40.3 [31.4, 49.0] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-nocrop-gem | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | gem | 99 | 28.3 [19.2, 37.4] | 53.5 [44.4, 63.6] | 64.6 [55.6, 73.7] | 40.3 [32.7, 49.0] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A10a | voyageai/voyage-multimodal-3.5 | 512 | api | 99 | 28.3 [20.2, 37.4] | 54.5 [45.5, 64.6] | 64.6 [55.6, 74.7] | 40.2 [32.7, 48.5] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-cls | facebook/dinov2-giant | 224 | cls | 99 | 31.3 [23.2, 41.4] | 49.5 [40.4, 58.6] | 52.5 [43.4, 61.6] | 39.9 [31.8, 48.7] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-nocrop-cls | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | cls | 99 | 26.3 [17.2, 35.4] | 54.5 [44.4, 63.6] | 59.6 [49.5, 69.7] | 39.6 [31.8, 47.5] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-nocrop-gem | facebook/dinov2-giant | 224 | gem | 99 | 27.3 [19.2, 36.4] | 51.5 [41.4, 61.6] | 56.6 [46.5, 65.7] | 38.5 [30.6, 47.1] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A1-dinov2-l-224-gem | facebook/dinov2-large | 224 | gem | 99 | 29.3 [20.2, 38.4] | 46.5 [37.4, 55.6] | 54.5 [45.5, 64.6] | 37.4 [29.4, 46.0] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-nocrop-cls | facebook/dinov2-giant | 224 | cls | 99 | 27.3 [19.2, 36.4] | 48.5 [39.4, 58.6] | 53.5 [44.4, 62.6] | 37.1 [29.4, 45.2] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322 | gmberton/MegaLoc | 322 | salad | 443 | 39.5 [35.0, 44.0] | 53.7 [49.0, 58.2] | 62.5 [58.5, 67.0] | 46.7 [42.5, 50.8] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-nocrop-gem | facebook/dinov2-giant | 224 | gem | 443 | 34.1 [29.8, 38.4] | 49.0 [44.0, 54.0] | 58.2 [54.0, 63.0] | 42.0 [38.0, 46.0] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A9 | google/gemini-embedding-2 | 512 | api | 443 | 34.8 [30.5, 39.1] | 47.0 [42.2, 51.2] | 51.2 [46.3, 55.5] | 41.0 [36.9, 45.1] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-gem | facebook/dinov2-giant | 224 | gem | 443 | 32.3 [28.0, 36.6] | 45.4 [40.6, 50.1] | 51.0 [46.0, 55.8] | 39.1 [34.8, 43.2] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A7-embeddinggemma-2-512 | google/embeddinggemma-2 | 512 | head | 443 | 33.4 [29.1, 37.7] | 43.3 [38.6, 48.1] | 48.1 [43.3, 52.6] | 38.7 [34.7, 42.8] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-nocrop-gem | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | gem | 443 | 31.4 [26.9, 35.7] | 42.9 [37.9, 47.6] | 51.2 [46.5, 56.0] | 37.8 [33.6, 41.8] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A5-pe-core-l-336 | timm/PE-Core-L-14-336 | 336 | head | 443 | 30.7 [26.4, 34.8] | 44.2 [39.1, 49.0] | 49.7 [44.9, 54.2] | 37.5 [33.2, 41.3] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-7b-256-nocrop-gem | facebook/dinov3-vit7b16-pretrain-lvd1689m | 256 | gem | 443 | 29.8 [25.5, 34.3] | 44.5 [40.0, 49.2] | 51.2 [46.5, 55.8] | 37.0 [33.1, 41.0] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A10a | voyageai/voyage-multimodal-3.5 | 512 | api | 443 | 32.1 [27.5, 36.6] | 40.6 [35.9, 45.1] | 43.6 [38.8, 48.1] | 36.5 [32.3, 40.7] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A4-siglip2-so400m-384 | google/siglip2-so400m-patch14-384 | 384 | head | 443 | 29.3 [25.3, 33.6] | 41.5 [36.8, 45.8] | 47.6 [42.9, 52.1] | 35.5 [31.5, 39.6] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-7b-256-cls | facebook/dinov3-vit7b16-pretrain-lvd1689m | 256 | cls | 443 | 30.9 [26.2, 35.2] | 39.5 [34.5, 44.2] | 42.9 [37.9, 47.4] | 34.9 [30.5, 39.2] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-7b-256-nocrop-cls | facebook/dinov3-vit7b16-pretrain-lvd1689m | 256 | cls | 443 | 30.7 [26.4, 35.0] | 39.1 [34.3, 43.8] | 42.4 [37.7, 47.0] | 34.9 [30.6, 39.2] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-cls | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | cls | 443 | 31.2 [26.9, 35.4] | 38.8 [33.9, 43.6] | 41.1 [36.1, 45.6] | 34.8 [30.5, 39.1] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A1-dinov2-l-224-gem | facebook/dinov2-large | 224 | gem | 443 | 28.9 [24.8, 33.2] | 40.4 [35.7, 44.9] | 43.8 [39.1, 48.5] | 34.6 [30.7, 38.8] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-gem | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | gem | 443 | 27.8 [23.9, 31.8] | 41.5 [36.8, 46.3] | 46.3 [41.5, 51.0] | 34.1 [30.2, 38.1] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-cls | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | cls | 443 | 29.8 [25.5, 34.1] | 38.6 [33.9, 43.1] | 42.7 [38.1, 47.0] | 33.7 [29.6, 37.9] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-nocrop-gem | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | gem | 443 | 27.3 [23.0, 31.8] | 40.2 [35.2, 44.7] | 45.4 [40.4, 49.9] | 33.7 [29.6, 37.8] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-gem | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | gem | 443 | 27.8 [23.9, 32.1] | 39.5 [34.8, 44.0] | 44.0 [39.3, 49.0] | 33.6 [29.7, 37.8] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-nocrop-cls | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | cls | 443 | 29.1 [24.8, 33.6] | 38.6 [33.6, 43.3] | 41.5 [37.0, 46.5] | 33.5 [29.2, 37.8] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-nocrop-cls | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | cls | 443 | 29.1 [24.8, 33.4] | 37.9 [33.4, 42.4] | 40.9 [36.1, 45.6] | 33.5 [29.4, 37.5] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A1-dinov2-l-224-cls | facebook/dinov2-large | 224 | cls | 443 | 29.6 [25.3, 33.6] | 36.3 [31.6, 40.9] | 39.5 [34.8, 44.2] | 33.1 [28.8, 37.0] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A8-qwen3-vl-embedding-2b-512 | Qwen/Qwen3-VL-Embedding-2B | 512 | head | 443 | 26.4 [22.3, 30.5] | 37.7 [33.0, 42.2] | 42.0 [37.2, 46.5] | 32.4 [28.4, 36.5] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-nocrop-cls | facebook/dinov2-giant | 224 | cls | 443 | 28.2 [23.9, 32.5] | 35.4 [31.2, 40.2] | 39.5 [34.8, 44.2] | 32.3 [28.2, 36.6] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-cls | facebook/dinov2-giant | 224 | cls | 443 | 27.8 [23.5, 31.8] | 36.1 [31.8, 40.6] | 39.7 [34.8, 44.7] | 31.9 [27.8, 35.9] |
| R2 | T1 | tune | screening-20k-publishable-fetched | H-H1-gemini-3.5-flash-lite | google/gemini-3.5-flash-lite | 512 | bm25 | 138 | 86.2 [80.4, 92.0] | 96.4 [93.5, 99.3] | 97.1 [94.2, 99.3] | 90.9 [87.0, 94.8] |
| R2 | T1 | tune | screening-20k-publishable-fetched | H-H3-A7+H1-gemini-3.5-flash-lite | google/embeddinggemma-2 + google/gemini-3.5-flash-lite | 512 | fusion | 138 | 75.4 [68.1, 82.6] | 89.1 [84.1, 94.2] | 89.1 [84.1, 94.2] | 81.0 [75.3, 86.5] |
| R2 | T1 | tune | screening-20k-publishable-fetched | H-H1-qwen3-vl-30b-a3b | qwen/qwen3-vl-30b-a3b-instruct | 512 | bm25 | 138 | 65.2 [57.2, 73.2] | 82.6 [76.1, 88.4] | 87.0 [81.2, 92.0] | 72.7 [66.2, 79.1] |
| R2 | T1 | tune | screening-20k-publishable-fetched | H-H3-A7+H1-qwen3-vl-30b-a3b | google/embeddinggemma-2 + qwen/qwen3-vl-30b-a3b-instruct | 512 | fusion | 138 | 65.2 [57.2, 73.2] | 76.1 [68.8, 83.3] | 79.7 [73.2, 86.2] | 70.7 [63.7, 77.7] |
| R2 | T1 | tune | screening-20k-publishable-fetched | G-G4-router-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 38.4 [31.2, 46.4] | 48.6 [41.3, 56.5] | 51.4 [44.2, 59.4] | 43.4 [36.3, 50.9] |
| R2 | T1 | tune | screening-20k-publishable-fetched | G-G4-router-A7+A3-guessA7A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 38.4 [31.2, 46.4] | 48.6 [41.3, 56.5] | 51.4 [44.2, 59.4] | 43.1 [35.9, 50.7] |
| R2 | T1 | tune | screening-20k-publishable-fetched | G-G4-switch-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 38.4 [31.2, 46.4] | 47.1 [39.9, 55.1] | 52.2 [44.9, 60.1] | 42.8 [35.8, 50.4] |
| R2 | T1 | tune | screening-20k-publishable-fetched | G-G1-A7+A4 | google/embeddinggemma-2 + google/siglip2-so400m-patch14-384 | 512 | fusion | 138 | 36.2 [29.0, 44.2] | 47.1 [39.9, 55.8] | 51.4 [44.2, 59.4] | 41.3 [34.5, 48.7] |
| R2 | T1 | tune | screening-20k-publishable-fetched | E-E5-A7-alpha3 | google/embeddinggemma-2 | 512 | head | 138 | 35.5 [27.5, 42.8] | 44.2 [37.0, 52.2] | 45.7 [38.4, 53.6] | 39.8 [32.6, 47.3] |
| R2 | T1 | tune | screening-20k-publishable-fetched | G-G1-A7+A3+A5 | google/embeddinggemma-2 + gmberton/MegaLoc + timm/PE-Core-L-14-336 | 512 | fusion | 138 | 35.5 [28.3, 43.5] | 42.8 [35.5, 50.7] | 47.8 [40.6, 55.8] | 39.8 [32.8, 47.5] |
| R2 | T1 | tune | screening-20k-publishable-fetched | E-E5-A7 | google/embeddinggemma-2 | 512 | head | 138 | 34.1 [26.1, 41.3] | 44.2 [37.0, 52.2] | 44.9 [37.7, 52.9] | 38.9 [31.8, 46.5] |
| R2 | T1 | tune | screening-20k-publishable-fetched | E-E5-G1-A7+A3-alpha3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 35.5 [27.5, 43.5] | 41.3 [33.3, 49.3] | 44.2 [37.0, 52.2] | 38.6 [31.0, 46.1] |
| R2 | T1 | tune | screening-20k-publishable-fetched | E-E5-G1-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 34.8 [26.8, 42.0] | 40.6 [32.6, 49.3] | 43.5 [36.2, 51.4] | 37.8 [30.6, 45.3] |
| R2 | T1 | tune | screening-20k-publishable-fetched | G-G1-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 32.6 [25.4, 40.6] | 40.6 [33.3, 48.6] | 45.7 [38.4, 54.3] | 37.6 [30.4, 45.0] |
| R2 | T1 | tune | screening-20k-publishable-fetched | G-G2-A7+A4 | google/embeddinggemma-2 + google/siglip2-so400m-patch14-384 | 512 | fusion | 138 | 29.7 [22.5, 37.0] | 42.0 [34.8, 50.0] | 46.4 [39.1, 55.1] | 36.0 [29.4, 43.1] |
| R2 | T1 | tune | screening-20k-publishable-fetched | G-G2-A9+A3 | google/gemini-embedding-2 + gmberton/MegaLoc | 512 | fusion | 138 | 29.7 [22.5, 37.7] | 37.0 [29.7, 44.2] | 44.9 [37.7, 52.9] | 34.2 [27.4, 41.7] |
| R2 | T1 | tune | screening-20k-publishable-fetched | G-G1-A9+A3 | google/gemini-embedding-2 + gmberton/MegaLoc | 512 | fusion | 138 | 29.7 [22.5, 37.7] | 37.0 [29.7, 44.9] | 41.3 [33.3, 49.3] | 33.8 [27.2, 41.4] |
| R2 | T1 | tune | screening-20k-publishable-fetched | G-G2-A7+A3+A5 | google/embeddinggemma-2 + gmberton/MegaLoc + timm/PE-Core-L-14-336 | 512 | fusion | 138 | 29.0 [22.5, 36.2] | 36.2 [29.0, 44.2] | 42.0 [34.8, 50.0] | 33.5 [26.8, 40.8] |
| R2 | T1 | tune | screening-20k-publishable-fetched | G-G2-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 28.3 [21.0, 35.5] | 34.8 [27.5, 42.8] | 44.2 [36.9, 52.2] | 33.3 [26.6, 40.1] |
| R2 | T1 | tune | screening-20k-publishable-fetched | E-E1-A7-embeddinggemma-2-512-mix | google/embeddinggemma-2 + ALIKED/LightGlue | 512 | head | 138 | 25.4 [18.8, 32.6] | 37.7 [30.4, 45.7] | 46.4 [38.4, 53.6] | 32.1 [25.6, 39.0] |
| R2 | T1 | tune | screening-20k-publishable-fetched | E-E1-A7-embeddinggemma-2-512-inliers | google/embeddinggemma-2 + ALIKED/LightGlue | 512 | head | 138 | 23.2 [16.7, 30.4] | 34.1 [26.8, 42.0] | 44.9 [37.0, 52.9] | 29.6 [23.6, 36.2] |
| R2 | T1 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322-pca1024 | gmberton/MegaLoc | 322 | salad | 138 | 21.0 [15.2, 27.5] | 27.5 [20.3, 35.5] | 31.2 [23.2, 38.4] | 24.7 [18.7, 31.2] |
| R2 | T1 | tune | screening-20k-publishable-fetched | E-E1-A3-megaloc-322-mix | gmberton/MegaLoc + ALIKED/LightGlue | 322 | salad | 138 | 21.0 [14.5, 27.5] | 26.8 [20.3, 34.8] | 29.7 [22.5, 37.7] | 24.3 [18.1, 30.7] |
| R2 | T1 | tune | screening-20k-publishable-fetched | E-E1-A3-megaloc-322-inliers | gmberton/MegaLoc + ALIKED/LightGlue | 322 | salad | 138 | 20.3 [13.8, 26.8] | 26.1 [19.6, 33.3] | 29.0 [22.5, 37.0] | 23.7 [17.4, 30.1] |
| R2 | T1 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322-dims8448 | gmberton/MegaLoc | 322 | salad | 138 | 20.3 [14.5, 26.8] | 26.1 [18.8, 33.3] | 29.7 [22.5, 37.0] | 23.3 [17.3, 30.0] |
| R2 | T1 | tune | screening-20k-publishable-fetched | E-E5-A3-alpha3 | gmberton/MegaLoc | 322 | salad | 138 | 20.3 [14.5, 26.8] | 23.9 [17.4, 31.2] | 27.5 [20.3, 34.8] | 22.7 [16.7, 29.3] |
| R2 | T1 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322-pca512 | gmberton/MegaLoc | 322 | salad | 138 | 17.4 [11.6, 23.9] | 27.5 [21.0, 35.5] | 30.4 [23.2, 38.4] | 22.3 [16.4, 28.6] |
| R2 | T1 | tune | screening-20k-publishable-fetched | E-E5-A3 | gmberton/MegaLoc | 322 | salad | 138 | 15.2 [9.4, 21.0] | 23.9 [17.4, 31.2] | 25.4 [18.1, 32.6] | 19.5 [14.0, 25.4] |
| R2 | T1 | tune | screening-v2-publishable-fetched | G-G5-router3-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 37.0 [29.0, 44.9] | 47.8 [40.6, 55.8] | 50.7 [43.5, 58.7] | 42.6 [35.7, 49.9] |
| R2 | T1 | tune | screening-v2-publishable-fetched | F-F1-query-A7 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 36.2 [29.0, 44.2] | 47.1 [39.9, 55.1] | 50.0 [42.8, 58.0] | 41.9 [35.0, 49.1] |
| R2 | T1 | tune | screening-v2-publishable-fetched | F-base-A7-v2 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 36.2 [29.0, 44.2] | 47.1 [39.9, 55.1] | 50.0 [42.8, 58.0] | 41.9 [35.0, 49.1] |
| R2 | T1 | tune | screening-v2-publishable-fetched | F-F1-both-A7 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 36.2 [29.0, 44.2] | 47.1 [39.9, 55.1] | 49.3 [42.0, 57.2] | 41.8 [35.0, 49.1] |
| R2 | T1 | tune | screening-v2-publishable-fetched | F-F1-query-A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 20.3 [14.5, 26.8] | 25.4 [18.1, 32.6] | 28.3 [21.0, 35.5] | 22.9 [16.9, 29.5] |
| R2 | T1 | tune | screening-v2-publishable-fetched | F-base-A3-v2 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 20.3 [14.5, 26.8] | 25.4 [18.1, 32.6] | 28.3 [21.0, 35.5] | 22.9 [16.9, 29.5] |
| R2 | T1 | tune | screening-v2-publishable-fetched | F-F1-both-A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 20.3 [14.5, 26.8] | 25.4 [18.1, 32.6] | 28.3 [21.0, 35.5] | 22.9 [16.9, 29.5] |
| R2 | T1 | tune | screening-v2-publishable-fetched+F2 | F-F2-G5-router3-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 37.0 [29.0, 44.9] | 47.8 [40.6, 55.8] | 50.7 [43.5, 58.7] | 42.6 [35.7, 49.9] |
| R2 | T1 | tune | screening-v2-publishable-fetched+F2 | F-F2-G5-router3-A7+A3-fixed | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 37.0 [29.0, 44.9] | 47.8 [40.6, 55.8] | 50.7 [43.5, 58.7] | 42.6 [35.7, 49.9] |
| R2 | T1 | tune | screening-v2-publishable-fetched+F2 | F-F2-A7 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 36.2 [29.0, 44.2] | 47.1 [39.9, 55.1] | 50.0 [42.8, 58.0] | 41.9 [35.0, 49.1] |
| R2 | T1 | tune | screening-v2-publishable-fetched+F2 | F-F2-A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 138 | 20.3 [14.5, 26.8] | 25.4 [18.1, 32.6] | 28.3 [21.0, 35.5] | 22.9 [16.9, 29.6] |
| R2 | T2 | tune | screening-v2-publishable-fetched | G-G5-router3-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 516 | 24.2 [20.7, 28.1] | 29.3 [25.8, 33.3] | 32.4 [28.7, 36.4] | 27.1 [23.7, 30.8] |
| R2 | T2 | tune | screening-v2-publishable-fetched | F-F1-query-A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 516 | 24.2 [20.7, 28.1] | 28.9 [25.2, 32.9] | 32.9 [29.1, 37.0] | 26.9 [23.5, 30.6] |
| R2 | T2 | tune | screening-v2-publishable-fetched | F-F1-both-A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 516 | 24.0 [20.5, 27.9] | 28.7 [25.0, 32.8] | 32.4 [28.7, 36.6] | 26.7 [23.3, 30.4] |
| R2 | T2 | tune | screening-v2-publishable-fetched | A-A3-megaloc-322-dims8448-T2 | gmberton/MegaLoc | 322 | salad | 516 | 23.8 [20.5, 27.7] | 28.9 [25.4, 32.9] | 31.0 [27.3, 35.1] | 26.6 [23.2, 30.3] |
| R2 | T2 | tune | screening-v2-publishable-fetched | F-base-A3-v2 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 516 | 23.8 [20.5, 27.7] | 28.9 [25.4, 32.9] | 31.0 [27.3, 35.1] | 26.6 [23.2, 30.3] |
| R2 | T2 | tune | screening-v2-publishable-fetched | G-G1-A7+A3-v2 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 516 | 20.5 [17.4, 24.4] | 26.9 [23.4, 30.8] | 31.2 [27.5, 35.5] | 24.1 [21.1, 27.8] |
| R2 | T2 | tune | screening-v2-publishable-fetched | F-base-A7-v2 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 516 | 9.1 [6.8, 11.8] | 18.2 [15.1, 21.7] | 21.9 [18.4, 25.6] | 13.6 [11.3, 16.4] |
| R2 | T2 | tune | screening-v2-publishable-fetched | F-F1-query-A7 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 516 | 8.7 [6.6, 11.2] | 17.6 [14.7, 21.1] | 20.5 [17.1, 24.0] | 13.1 [10.8, 15.8] |
| R2 | T2 | tune | screening-v2-publishable-fetched | F-F1-both-A7 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 516 | 8.3 [6.2, 10.9] | 17.4 [14.5, 20.9] | 20.2 [16.7, 23.6] | 12.7 [10.4, 15.4] |
| R2 | T2 | tune | screening-v2-publishable-fetched | A-A3-megaloc-322-pca1024-T2 | gmberton/MegaLoc | 322 | salad | 516 | 9.3 [7.0, 11.8] | 12.4 [9.7, 15.3] | 13.2 [10.3, 16.1] | 10.6 [8.2, 13.2] |
| R2 | T2 | tune | screening-v2-publishable-fetched | A-A3-megaloc-322-pca512-T2 | gmberton/MegaLoc | 322 | salad | 516 | 4.1 [2.5, 5.6] | 7.0 [4.8, 9.1] | 7.9 [5.8, 10.3] | 5.5 [3.8, 7.2] |
| R2 | T2 | tune | screening-v2-publishable-fetched+F2 | F-F2-A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 516 | 31.6 [28.1, 35.7] | 38.0 [34.1, 42.2] | 40.5 [36.6, 45.0] | 34.9 [31.4, 39.0] |
| R2 | T2 | tune | screening-v2-publishable-fetched+F2 | F-F2-A3-megaloc-322-T8c | gmberton/MegaLoc | 322 | salad | 516 | 31.6 [28.1, 35.7] | 38.0 [34.1, 42.2] | 40.5 [36.6, 45.0] | 34.9 [31.4, 39.0] |
| R2 | T2 | tune | screening-v2-publishable-fetched+F2 | F-F2-G5-router3-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 516 | 31.2 [27.7, 35.3] | 38.6 [34.7, 42.8] | 40.5 [36.6, 45.0] | 34.6 [31.2, 38.6] |
| R2 | T2 | tune | screening-v2-publishable-fetched+F2 | F-F2-G5-router3-A7+A3-fixed | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 516 | 29.8 [26.4, 33.7] | 37.0 [33.1, 41.5] | 38.4 [34.7, 42.6] | 33.1 [29.6, 36.9] |
| R2 | T2 | tune | screening-v2-publishable-fetched+F2 | F-F2-A7 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 516 | 14.3 [11.4, 17.4] | 19.4 [16.1, 22.9] | 21.9 [18.4, 25.6] | 17.1 [14.2, 20.1] |
| R2 | T2 | tune | screening-v2-publishable-fetched+F2 | F-F2-A7-embeddinggemma-2-512-T8c | google/embeddinggemma-2 | 512 | head | 516 | 14.3 [11.4, 17.4] | 19.4 [16.1, 22.9] | 21.9 [18.4, 25.6] | 17.1 [14.2, 20.1] |
| R2 | T2b | tune | screening-v2-publishable-fetched | G-G1-A7+A3-v2 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 122 | 90.2 [85.2, 95.9] | 95.9 [91.8, 99.2] | 98.4 [95.9, 100.0] | 92.5 [88.3, 96.8] |
| R2 | T2b | tune | screening-v2-publishable-fetched+F2 | F-F2-A7-embeddinggemma-2-512-T8c | google/embeddinggemma-2 | 512 | head | 122 | 86.1 [79.5, 91.8] | 92.6 [87.7, 96.7] | 97.5 [94.3, 100.0] | 89.5 [84.7, 94.1] |
| R2 | T2b | tune | screening-v2-publishable-fetched+F2 | F-F2-A3-megaloc-322-T8c | gmberton/MegaLoc | 322 | salad | 122 | 82.8 [75.4, 89.3] | 92.6 [87.7, 96.7] | 92.6 [87.7, 96.7] | 87.2 [81.9, 92.3] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E1-A7-embeddinggemma-2-512-mix | google/embeddinggemma-2 + ALIKED/LightGlue | 512 | head | 99 | 72.7 [63.6, 81.8] | 92.9 [87.9, 98.0] | 92.9 [87.9, 98.0] | 82.0 [75.6, 87.9] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E1-A7-embeddinggemma-2-512-inliers | google/embeddinggemma-2 + ALIKED/LightGlue | 512 | head | 99 | 71.7 [62.6, 80.8] | 91.9 [85.9, 97.0] | 91.9 [85.9, 97.0] | 80.9 [74.1, 86.8] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E1-A3-megaloc-322-mix | gmberton/MegaLoc + ALIKED/LightGlue | 322 | salad | 99 | 66.7 [57.6, 75.8] | 92.9 [87.9, 97.0] | 92.9 [87.9, 97.0] | 78.7 [72.0, 84.8] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E1-A3-megaloc-322-inliers | gmberton/MegaLoc + ALIKED/LightGlue | 322 | salad | 99 | 67.7 [58.6, 77.8] | 90.9 [84.8, 96.0] | 90.9 [84.8, 96.0] | 78.0 [70.8, 84.6] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G1-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 99 | 60.6 [51.5, 70.7] | 83.8 [76.8, 90.9] | 89.9 [83.8, 96.0] | 71.5 [64.2, 79.3] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G1-A7+A3+A5 | google/embeddinggemma-2 + gmberton/MegaLoc + timm/PE-Core-L-14-336 | 512 | fusion | 99 | 59.6 [50.5, 69.7] | 81.8 [73.7, 89.9] | 88.9 [82.8, 94.9] | 70.0 [62.8, 78.1] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G1-A9+A3 | google/gemini-embedding-2 + gmberton/MegaLoc | 512 | fusion | 99 | 55.6 [46.5, 65.7] | 85.9 [78.8, 92.9] | 88.9 [81.8, 94.9] | 69.0 [62.0, 76.4] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E5-G1-A7+A3-alpha3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 99 | 59.6 [50.5, 69.7] | 79.8 [71.7, 87.9] | 87.9 [80.8, 93.9] | 69.0 [61.6, 77.1] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G4-router-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 99 | 56.6 [47.4, 66.7] | 85.9 [78.8, 91.9] | 89.9 [83.8, 96.0] | 68.8 [61.4, 76.7] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G4-switch-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 99 | 56.6 [47.4, 66.7] | 85.9 [78.8, 91.9] | 89.9 [83.8, 96.0] | 68.7 [61.3, 76.4] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G4-router-A7+A3-guessA7A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 99 | 56.6 [47.4, 66.7] | 85.9 [78.8, 91.9] | 89.9 [83.8, 96.0] | 68.5 [61.2, 76.2] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E5-A3-alpha3 | gmberton/MegaLoc | 322 | salad | 99 | 57.6 [48.5, 67.7] | 83.8 [76.7, 90.9] | 85.9 [78.8, 91.9] | 68.4 [60.5, 76.8] |
| R2 | T3 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322-dims8448 | gmberton/MegaLoc | 322 | salad | 99 | 55.6 [46.5, 65.7] | 83.8 [76.7, 90.9] | 85.9 [78.8, 91.9] | 67.5 [59.8, 75.6] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G2-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 99 | 55.6 [45.5, 65.7] | 80.8 [72.7, 88.9] | 89.9 [83.8, 96.0] | 67.2 [59.9, 75.3] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G2-A9+A3 | google/gemini-embedding-2 + gmberton/MegaLoc | 512 | fusion | 99 | 56.6 [47.4, 66.7] | 80.8 [72.7, 88.9] | 88.9 [82.8, 94.9] | 67.1 [59.5, 75.5] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G1-A7+A4 | google/embeddinggemma-2 + google/siglip2-so400m-patch14-384 | 512 | fusion | 99 | 55.6 [45.5, 65.7] | 79.8 [71.7, 87.9] | 84.8 [77.8, 91.9] | 66.6 [58.6, 74.8] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G2-A7+A3+A5 | google/embeddinggemma-2 + gmberton/MegaLoc + timm/PE-Core-L-14-336 | 512 | fusion | 99 | 51.5 [41.4, 61.6] | 80.8 [72.7, 87.9] | 88.9 [82.8, 94.9] | 64.1 [56.6, 72.0] |
| R2 | T3 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322-pca1024 | gmberton/MegaLoc | 322 | salad | 99 | 52.5 [42.4, 62.6] | 80.8 [72.7, 88.9] | 85.9 [78.8, 92.9] | 64.0 [56.4, 72.1] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E5-G1-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 99 | 50.5 [40.4, 60.6] | 82.8 [74.7, 89.9] | 84.8 [77.8, 91.9] | 63.8 [56.2, 71.5] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G2-A7+A4 | google/embeddinggemma-2 + google/siglip2-so400m-patch14-384 | 512 | fusion | 99 | 52.5 [42.4, 62.6] | 76.8 [67.7, 84.8] | 84.8 [76.8, 91.9] | 63.7 [55.8, 72.0] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E5-A7-alpha3 | google/embeddinggemma-2 | 512 | head | 99 | 51.5 [42.4, 61.6] | 74.7 [65.7, 83.8] | 78.8 [70.7, 86.9] | 63.1 [55.0, 71.6] |
| R2 | T3 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322-pca512 | gmberton/MegaLoc | 322 | salad | 99 | 48.5 [39.4, 58.6] | 75.8 [66.7, 83.8] | 80.8 [72.7, 88.9] | 59.7 [51.9, 68.1] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E5-A7 | google/embeddinggemma-2 | 512 | head | 99 | 45.5 [35.4, 55.6] | 74.7 [65.7, 83.8] | 77.8 [69.7, 85.9] | 59.6 [51.7, 68.0] |
| R2 | T3 | tune | screening-20k-publishable-fetched | H-H3-A7+H1-gemini-3.5-flash-lite | google/embeddinggemma-2 + google/gemini-3.5-flash-lite | 512 | fusion | 99 | 45.5 [36.4, 55.6] | 74.7 [65.7, 82.8] | 82.8 [74.7, 89.9] | 58.6 [50.9, 66.6] |
| R2 | T3 | tune | screening-20k-publishable-fetched | H-H3-A7+H1-qwen3-vl-30b-a3b | google/embeddinggemma-2 + qwen/qwen3-vl-30b-a3b-instruct | 512 | fusion | 99 | 42.4 [33.3, 51.5] | 75.8 [67.7, 83.8] | 82.8 [74.7, 90.9] | 57.2 [49.7, 64.8] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E5-A3 | gmberton/MegaLoc | 322 | salad | 99 | 30.3 [21.2, 40.4] | 78.8 [70.7, 86.9] | 79.8 [71.7, 87.9] | 51.3 [44.7, 59.0] |
| R2 | T3 | tune | screening-20k-publishable-fetched | H-H1-gemini-3.5-flash-lite | google/gemini-3.5-flash-lite | 512 | bm25 | 99 | 4.0 [1.0, 8.1] | 15.2 [9.1, 22.2] | 22.2 [14.1, 31.3] | 10.0 [6.2, 14.5] |
| R2 | T3 | tune | screening-20k-publishable-fetched | H-H1-qwen3-vl-30b-a3b | qwen/qwen3-vl-30b-a3b-instruct | 512 | bm25 | 99 | 1.0 [0.0, 3.0] | 5.1 [1.0, 9.1] | 9.1 [4.0, 14.1] | 4.6 [2.2, 7.5] |
| R2 | T3 | tune | screening-v2-publishable-fetched | G-G5-router3-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 99 | 56.6 [47.4, 66.7] | 85.9 [78.8, 91.9] | 89.9 [83.8, 96.0] | 68.7 [61.3, 76.4] |
| R2 | T3 | tune | screening-v2-publishable-fetched+F2 | F-F2-G5-router3-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 99 | 56.6 [47.4, 66.7] | 85.9 [78.8, 91.9] | 89.9 [83.8, 96.0] | 68.7 [61.3, 76.4] |
| R2 | T3 | tune | screening-v2-publishable-fetched+F2 | F-F2-G5-router3-A7+A3-fixed | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 99 | 56.6 [47.4, 66.7] | 85.9 [78.8, 91.9] | 89.9 [83.8, 96.0] | 68.7 [61.3, 76.4] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E1-A3-megaloc-322-mix | gmberton/MegaLoc + ALIKED/LightGlue | 322 | salad | 443 | 44.7 [40.2, 49.2] | 56.7 [51.9, 61.2] | 63.2 [58.9, 67.5] | 51.1 [46.9, 55.4] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G4-router-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 40.4 [35.7, 44.7] | 53.0 [48.3, 57.3] | 61.4 [56.9, 65.9] | 47.3 [43.0, 51.2] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G4-switch-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 40.2 [35.4, 44.5] | 53.0 [48.3, 57.3] | 61.4 [56.9, 65.9] | 47.1 [42.8, 51.1] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G1-A9+A3 | google/gemini-embedding-2 + gmberton/MegaLoc | 512 | fusion | 443 | 40.2 [35.7, 44.7] | 54.4 [49.9, 58.9] | 60.5 [56.0, 64.8] | 47.1 [42.9, 50.9] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G4-router-A7+A3-guessA7A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 39.5 [35.0, 44.0] | 53.7 [49.0, 58.2] | 62.1 [57.8, 66.6] | 46.9 [42.8, 51.1] |
| R2 | T5 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322-dims8448 | gmberton/MegaLoc | 322 | salad | 443 | 39.5 [35.0, 44.0] | 53.7 [49.0, 58.2] | 62.5 [58.5, 67.0] | 46.7 [42.5, 50.8] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E5-A3-alpha3 | gmberton/MegaLoc | 322 | salad | 443 | 38.4 [34.1, 42.7] | 51.7 [47.2, 56.2] | 56.9 [52.1, 61.4] | 45.1 [41.0, 49.3] |
| R2 | T5 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322-pca1024 | gmberton/MegaLoc | 322 | salad | 443 | 37.5 [33.2, 42.0] | 51.2 [46.5, 55.5] | 57.3 [52.8, 61.6] | 44.2 [40.0, 48.3] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G1-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 37.9 [33.2, 42.4] | 49.9 [45.4, 54.4] | 54.9 [50.3, 59.1] | 43.8 [39.5, 47.8] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E1-A7-embeddinggemma-2-512-mix | google/embeddinggemma-2 + ALIKED/LightGlue | 512 | head | 443 | 37.9 [33.4, 42.4] | 47.9 [42.9, 52.4] | 50.3 [45.6, 54.6] | 42.8 [38.4, 47.0] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G2-A9+A3 | google/gemini-embedding-2 + gmberton/MegaLoc | 512 | fusion | 443 | 34.5 [30.2, 38.8] | 50.3 [45.4, 54.9] | 55.8 [51.0, 60.3] | 42.3 [38.3, 46.3] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G1-A7+A3+A5 | google/embeddinggemma-2 + gmberton/MegaLoc + timm/PE-Core-L-14-336 | 512 | fusion | 443 | 35.7 [30.9, 40.0] | 47.6 [42.9, 52.2] | 52.8 [48.3, 57.3] | 41.9 [37.7, 46.0] |
| R2 | T5 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322-pca512 | gmberton/MegaLoc | 322 | salad | 443 | 35.4 [31.1, 39.5] | 47.2 [42.4, 51.7] | 54.4 [49.9, 59.1] | 41.6 [37.6, 45.5] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E1-A3-megaloc-322-inliers | gmberton/MegaLoc + ALIKED/LightGlue | 322 | salad | 443 | 35.2 [30.7, 39.7] | 47.2 [42.4, 51.7] | 54.0 [49.2, 58.5] | 41.5 [37.3, 45.6] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G2-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 34.8 [30.5, 39.1] | 47.6 [43.1, 52.1] | 51.9 [47.6, 56.7] | 41.1 [36.9, 45.1] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G2-A7+A3+A5 | google/embeddinggemma-2 + gmberton/MegaLoc + timm/PE-Core-L-14-336 | 512 | fusion | 443 | 35.0 [30.5, 39.3] | 46.3 [41.5, 51.0] | 52.6 [48.1, 57.1] | 41.0 [36.9, 45.2] |
| R2 | T5 | tune | screening-20k-publishable-fetched | H-H3-A7+H1-gemini-3.5-flash-lite | google/embeddinggemma-2 + google/gemini-3.5-flash-lite | 512 | fusion | 443 | 33.2 [28.9, 37.5] | 47.9 [43.1, 52.6] | 53.7 [49.2, 58.5] | 40.7 [36.7, 44.6] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E5-G1-A7+A3-alpha3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 35.0 [30.7, 39.1] | 45.8 [40.9, 50.1] | 51.2 [46.5, 55.8] | 40.4 [36.1, 44.3] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E5-A3 | gmberton/MegaLoc | 322 | salad | 443 | 31.2 [26.9, 35.4] | 50.3 [45.8, 54.9] | 53.7 [49.0, 58.0] | 40.0 [35.9, 43.9] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E5-G1-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 33.4 [29.1, 37.5] | 45.8 [40.9, 50.1] | 51.0 [46.3, 55.3] | 39.1 [34.8, 42.9] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G1-A7+A4 | google/embeddinggemma-2 + google/siglip2-so400m-patch14-384 | 512 | fusion | 443 | 33.2 [28.9, 37.5] | 45.1 [40.4, 49.7] | 49.0 [44.2, 53.5] | 39.1 [34.9, 43.2] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G2-A7+A4 | google/embeddinggemma-2 + google/siglip2-so400m-patch14-384 | 512 | fusion | 443 | 32.5 [28.0, 36.8] | 43.8 [39.3, 48.5] | 48.1 [43.6, 52.6] | 38.1 [33.8, 42.3] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E1-A7-embeddinggemma-2-512-inliers | google/embeddinggemma-2 + ALIKED/LightGlue | 512 | head | 443 | 31.8 [27.5, 36.6] | 41.8 [37.0, 46.3] | 45.4 [40.6, 49.7] | 36.8 [32.5, 41.2] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E5-A7-alpha3 | google/embeddinggemma-2 | 512 | head | 443 | 26.6 [22.6, 30.7] | 41.8 [37.0, 46.3] | 43.8 [39.1, 48.3] | 33.7 [29.9, 37.6] |
| R2 | T5 | tune | screening-20k-publishable-fetched | H-H3-A7+H1-qwen3-vl-30b-a3b | google/embeddinggemma-2 + qwen/qwen3-vl-30b-a3b-instruct | 512 | fusion | 443 | 25.7 [21.9, 29.6] | 40.2 [35.7, 44.7] | 47.0 [42.2, 51.5] | 33.5 [29.6, 37.2] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E5-A7 | google/embeddinggemma-2 | 512 | head | 443 | 22.6 [18.7, 26.4] | 41.5 [36.8, 46.0] | 43.3 [38.6, 47.9] | 31.2 [27.4, 34.9] |
| R2 | T5 | tune | screening-20k-publishable-fetched | H-H1-gemini-3.5-flash-lite | google/gemini-3.5-flash-lite | 512 | bm25 | 443 | 15.8 [12.4, 19.4] | 30.7 [26.4, 35.0] | 38.1 [33.6, 43.1] | 23.4 [20.0, 26.9] |
| R2 | T5 | tune | screening-20k-publishable-fetched | H-H1-qwen3-vl-30b-a3b | qwen/qwen3-vl-30b-a3b-instruct | 512 | bm25 | 443 | 8.8 [6.1, 11.5] | 16.3 [12.9, 19.9] | 20.1 [16.7, 23.9] | 12.5 [9.6, 15.3] |
| R2 | T5 | tune | screening-v2-publishable-fetched | G-G5-router3-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 39.7 [35.0, 44.0] | 53.0 [48.3, 57.3] | 61.2 [56.4, 65.7] | 46.8 [42.5, 50.8] |
| R2 | T5 | tune | screening-v2-publishable-fetched | F-base-A3-v2 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 39.1 [34.5, 43.6] | 53.7 [49.0, 58.2] | 62.1 [57.8, 66.6] | 46.4 [42.2, 50.5] |
| R2 | T5 | tune | screening-v2-publishable-fetched | F-F1-both-A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 38.8 [34.3, 43.3] | 49.9 [45.4, 54.2] | 58.0 [53.7, 62.3] | 44.9 [40.9, 49.2] |
| R2 | T5 | tune | screening-v2-publishable-fetched | F-F1-query-A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 36.1 [31.8, 40.4] | 47.9 [43.6, 52.6] | 55.3 [50.8, 60.0] | 42.4 [38.5, 46.5] |
| R2 | T5 | tune | screening-v2-publishable-fetched | F-base-A7-v2 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 33.2 [28.9, 37.5] | 43.1 [38.4, 47.6] | 47.9 [43.1, 52.4] | 38.4 [34.4, 42.5] |
| R2 | T5 | tune | screening-v2-publishable-fetched | F-F1-both-A7 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 26.4 [22.3, 30.5] | 38.6 [33.9, 42.9] | 44.5 [39.7, 49.2] | 32.6 [28.5, 36.6] |
| R2 | T5 | tune | screening-v2-publishable-fetched | F-F1-query-A7 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 26.2 [22.1, 30.7] | 35.4 [31.2, 40.2] | 40.0 [35.4, 44.5] | 31.3 [27.2, 35.3] |
| R2 | T5 | tune | screening-v2-publishable-fetched+F2 | F-F2-G5-router3-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 39.7 [35.0, 44.0] | 53.0 [48.3, 57.3] | 61.2 [56.4, 65.7] | 46.7 [42.5, 50.8] |
| R2 | T5 | tune | screening-v2-publishable-fetched+F2 | F-F2-G5-router3-A7+A3-fixed | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 39.7 [35.0, 44.0] | 53.0 [48.3, 57.3] | 61.2 [56.4, 65.7] | 46.7 [42.5, 50.8] |
| R2 | T5 | tune | screening-v2-publishable-fetched+F2 | F-F2-A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 39.1 [34.5, 43.6] | 53.5 [48.8, 58.0] | 62.1 [57.8, 66.6] | 46.3 [42.1, 50.4] |
| R2 | T5 | tune | screening-v2-publishable-fetched+F2 | F-F2-A7 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 33.2 [28.9, 37.5] | 43.1 [38.4, 47.6] | 47.9 [43.1, 52.4] | 38.4 [34.4, 42.5] |

Not found (T8): confidence is the top entity's score. Lower is better for "wrong but confident", higher for "matched".

| Round | Set | Half | Gallery | Run | In-index sets | In-index | Not found | Wrong but confident, 50% matched | Matched, 5% wrong |
|---|---|---|---|---|---|---|---|---|---|
| R1 | T8c | tune | screening-v2-publishable-fetched | A-A9-v2 | T2 + T2b | 638 | 150 | 26.0 [17.3, 33.3] | 25.1 [21.6, 31.5] |
| R1 | T8c | tune | screening-v2-publishable-fetched | A-A7-embeddinggemma-2-512-v2 | T2 + T2b | 638 | 150 | 36.0 [26.7, 44.0] | 19.6 [16.1, 25.9] |
| R1 | T8c | tune | screening-v2-publishable-fetched | A-A5-pe-core-l-336-v2 | T2 + T2b | 638 | 150 | 38.7 [31.3, 50.0] | 17.2 [14.3, 22.7] |
| R1 | T8c | tune | screening-v2-publishable-fetched | A-A3-megaloc-322-v2 | T2 + T2b | 638 | 150 | 40.0 [32.0, 48.0] | 21.5 [17.1, 25.1] |
| R1 | T8c | tune | screening-v2-publishable-fetched | A-A4-siglip2-so400m-384-v2 | T2 + T2b | 638 | 150 | 45.3 [36.0, 54.7] | 22.3 [18.2, 25.9] |
| R2 | T8 | tune | screening-20k-publishable-fetched | A-A7-embeddinggemma-2-512 | T1 + T5 | 581 | 232 | 25.0 [18.5, 31.5] | 12.4 [8.6, 17.9] |
| R2 | T8 | tune | screening-20k-publishable-fetched | G-G1-A7+A3 | T1 + T5 | 581 | 232 | 27.6 [21.6, 34.1] | 15.0 [10.2, 19.4] |
| R2 | T8 | tune | screening-20k-publishable-fetched | A-A9 | T1 + T5 | 581 | 232 | 29.7 [23.3, 35.8] | 8.8 [3.3, 17.6] |
| R2 | T8 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322 | T1 + T5 | 581 | 232 | 30.2 [23.7, 36.2] | 11.9 [7.1, 20.7] |
| R2 | T8 | tune | screening-20k-publishable-fetched | G-G4-switch-A7+A3 | T1 + T5 | 581 | 232 | 36.2 [30.2, 43.5] | 16.5 [8.6, 22.4] |
| R2 | T8 | tune | screening-20k-publishable-fetched | G-G4-router-A7+A3-guessA7A3 | T1 + T5 | 581 | 232 | 40.1 [31.5, 47.8] | 14.8 [9.0, 19.3] |
| R2 | T8 | tune | screening-20k-publishable-fetched | G-G4-router-A7+A3 | T1 + T5 | 581 | 232 | 42.2 [35.3, 50.4] | 14.8 [7.9, 20.0] |
| R2 | T8c | tune | screening-v2-publishable-fetched | G-G1-A7+A3-v2 | T2 + T2b | 638 | 150 | 35.3 [27.3, 43.3] | 21.8 [16.9, 27.6] |
| R2 | T8c | tune | screening-v2-publishable-fetched+F2 | F-F2-A7-embeddinggemma-2-512-T8c | T2 + T2b | 638 | 150 | 11.3 [6.0, 16.7] | 31.5 [26.3, 43.1] |
| R2 | T8c | tune | screening-v2-publishable-fetched+F2 | F-F2-A3-megaloc-322-T8c | T2 + T2b | 638 | 150 | 26.7 [17.3, 34.7] | 35.4 [29.5, 40.1] |
<!-- summary:end -->
