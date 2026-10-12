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

## R2: fusion (step G), query expansion (E5), "not found" (T8)

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
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-cls | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | cls | 138 | 24.6 [18.1, 31.9] | 30.4 [23.2, 37.7] | 33.3 [25.4, 40.6] | 27.6 [20.8, 34.6] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A5-pe-core-l-336 | timm/PE-Core-L-14-336 | 336 | head | 138 | 23.2 [16.7, 29.7] | 32.6 [25.4, 40.6] | 35.5 [28.3, 43.5] | 27.1 [20.8, 33.9] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-7b-256-cls | facebook/dinov3-vit7b16-pretrain-lvd1689m | 256 | cls | 138 | 22.5 [15.9, 29.7] | 31.2 [23.2, 38.4] | 33.3 [26.1, 40.6] | 26.5 [20.0, 33.5] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-cls | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | cls | 138 | 20.3 [13.8, 27.5] | 27.5 [20.3, 34.8] | 29.0 [21.7, 36.2] | 24.2 [17.5, 31.3] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322 | gmberton/MegaLoc | 322 | salad | 138 | 20.3 [14.5, 26.8] | 26.1 [18.8, 33.3] | 29.7 [22.5, 37.0] | 23.3 [17.3, 30.0] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-gem | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | gem | 138 | 15.2 [9.4, 21.7] | 23.9 [17.4, 31.2] | 26.8 [19.6, 34.1] | 19.8 [14.2, 25.8] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A1-dinov2-l-224-cls | facebook/dinov2-large | 224 | cls | 138 | 14.5 [8.7, 20.3] | 21.0 [14.5, 28.3] | 25.4 [18.1, 32.6] | 18.0 [12.5, 23.7] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-cls | facebook/dinov2-giant | 224 | cls | 138 | 13.0 [8.0, 18.8] | 22.5 [15.9, 29.7] | 24.6 [17.4, 31.9] | 17.8 [12.5, 23.4] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-gem | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | gem | 138 | 12.3 [7.2, 18.1] | 18.1 [12.3, 24.6] | 21.7 [15.2, 28.3] | 15.1 [10.2, 21.0] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A1-dinov2-l-224-gem | facebook/dinov2-large | 224 | gem | 138 | 7.2 [3.6, 12.3] | 15.2 [9.4, 21.0] | 19.6 [13.8, 26.1] | 11.5 [7.4, 16.1] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-gem | facebook/dinov2-giant | 224 | gem | 138 | 7.2 [3.6, 11.6] | 13.8 [8.0, 19.6] | 19.6 [13.8, 26.1] | 11.4 [7.4, 16.2] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322 | gmberton/MegaLoc | 322 | salad | 99 | 55.6 [46.5, 65.7] | 83.8 [76.7, 90.9] | 85.9 [78.8, 91.9] | 67.5 [59.8, 75.6] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A7-embeddinggemma-2-512 | google/embeddinggemma-2 | 512 | head | 99 | 56.6 [47.4, 66.7] | 78.8 [70.7, 86.9] | 84.8 [77.8, 91.9] | 66.8 [59.0, 75.1] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A9 | google/gemini-embedding-2 | 512 | api | 99 | 54.5 [44.4, 64.6] | 77.8 [69.7, 85.9] | 83.8 [75.8, 90.9] | 65.2 [57.4, 73.6] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A4-siglip2-so400m-384 | google/siglip2-so400m-patch14-384 | 384 | head | 99 | 49.5 [39.4, 59.6] | 73.7 [64.6, 81.8] | 80.8 [72.7, 87.9] | 60.2 [52.2, 68.2] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A5-pe-core-l-336 | timm/PE-Core-L-14-336 | 336 | head | 99 | 45.5 [35.4, 54.6] | 68.7 [59.6, 76.8] | 79.8 [70.7, 87.9] | 55.7 [47.6, 63.9] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A8-qwen3-vl-embedding-2b-512 | Qwen/Qwen3-VL-Embedding-2B | 512 | head | 99 | 39.4 [30.3, 49.5] | 65.7 [56.6, 74.7] | 75.8 [67.7, 83.8] | 50.5 [42.9, 59.0] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-cls | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | cls | 99 | 37.4 [27.3, 47.5] | 55.6 [46.5, 65.7] | 62.6 [53.5, 71.7] | 46.5 [38.2, 55.2] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-gem | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | gem | 99 | 34.3 [24.2, 43.4] | 56.6 [47.5, 66.7] | 61.6 [52.5, 70.7] | 44.5 [36.1, 52.9] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-7b-256-cls | facebook/dinov3-vit7b16-pretrain-lvd1689m | 256 | cls | 99 | 32.3 [23.2, 41.4] | 50.5 [40.4, 59.6] | 59.6 [49.5, 68.7] | 42.1 [33.5, 50.5] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-cls | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | cls | 99 | 30.3 [21.2, 39.4] | 51.5 [41.4, 60.6] | 55.6 [45.5, 64.6] | 40.7 [32.5, 49.1] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-gem | facebook/dinov2-giant | 224 | gem | 99 | 30.3 [21.2, 39.4] | 50.5 [41.4, 60.6] | 57.6 [48.5, 67.7] | 40.5 [32.7, 48.7] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-gem | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | gem | 99 | 30.3 [21.2, 39.4] | 48.5 [39.4, 57.6] | 59.6 [50.5, 68.7] | 40.5 [32.1, 48.4] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A1-dinov2-l-224-cls | facebook/dinov2-large | 224 | cls | 99 | 33.3 [24.2, 42.4] | 46.5 [36.4, 55.6] | 50.5 [40.4, 59.6] | 40.3 [31.4, 49.0] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A10a | voyageai/voyage-multimodal-3.5 | 512 | api | 99 | 28.3 [20.2, 37.4] | 54.5 [45.5, 64.6] | 64.6 [55.6, 74.7] | 40.2 [32.7, 48.5] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-cls | facebook/dinov2-giant | 224 | cls | 99 | 31.3 [23.2, 41.4] | 49.5 [40.4, 58.6] | 52.5 [43.4, 61.6] | 39.9 [31.8, 48.7] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A1-dinov2-l-224-gem | facebook/dinov2-large | 224 | gem | 99 | 29.3 [20.2, 38.4] | 46.5 [37.4, 55.6] | 54.5 [45.5, 64.6] | 37.4 [29.4, 46.0] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322 | gmberton/MegaLoc | 322 | salad | 443 | 39.5 [35.0, 44.0] | 53.7 [49.0, 58.2] | 62.5 [58.5, 67.0] | 46.7 [42.5, 50.8] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A9 | google/gemini-embedding-2 | 512 | api | 443 | 34.8 [30.5, 39.1] | 47.0 [42.2, 51.2] | 51.2 [46.3, 55.5] | 41.0 [36.9, 45.1] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-gem | facebook/dinov2-giant | 224 | gem | 443 | 32.3 [28.0, 36.6] | 45.4 [40.6, 50.1] | 51.0 [46.0, 55.8] | 39.1 [34.8, 43.2] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A7-embeddinggemma-2-512 | google/embeddinggemma-2 | 512 | head | 443 | 33.4 [29.1, 37.7] | 43.3 [38.6, 48.1] | 48.1 [43.3, 52.6] | 38.7 [34.7, 42.8] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A5-pe-core-l-336 | timm/PE-Core-L-14-336 | 336 | head | 443 | 30.7 [26.4, 34.8] | 44.2 [39.1, 49.0] | 49.7 [44.9, 54.2] | 37.5 [33.2, 41.3] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A10a | voyageai/voyage-multimodal-3.5 | 512 | api | 443 | 32.1 [27.5, 36.6] | 40.6 [35.9, 45.1] | 43.6 [38.8, 48.1] | 36.5 [32.3, 40.7] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A4-siglip2-so400m-384 | google/siglip2-so400m-patch14-384 | 384 | head | 443 | 29.3 [25.3, 33.6] | 41.5 [36.8, 45.8] | 47.6 [42.9, 52.1] | 35.5 [31.5, 39.6] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-7b-256-cls | facebook/dinov3-vit7b16-pretrain-lvd1689m | 256 | cls | 443 | 30.9 [26.2, 35.2] | 39.5 [34.5, 44.2] | 42.9 [37.9, 47.4] | 34.9 [30.5, 39.2] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-cls | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | cls | 443 | 31.2 [26.9, 35.4] | 38.8 [33.9, 43.6] | 41.1 [36.1, 45.6] | 34.8 [30.5, 39.1] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A1-dinov2-l-224-gem | facebook/dinov2-large | 224 | gem | 443 | 28.9 [24.8, 33.2] | 40.4 [35.7, 44.9] | 43.8 [39.1, 48.5] | 34.6 [30.7, 38.8] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-gem | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | gem | 443 | 27.8 [23.9, 31.8] | 41.5 [36.8, 46.3] | 46.3 [41.5, 51.0] | 34.1 [30.2, 38.1] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-l-256-cls | facebook/dinov3-vitl16-pretrain-lvd1689m | 256 | cls | 443 | 29.8 [25.5, 34.1] | 38.6 [33.9, 43.1] | 42.7 [38.1, 47.0] | 33.7 [29.6, 37.9] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A2-dinov3-hplus-256-gem | facebook/dinov3-vith16plus-pretrain-lvd1689m | 256 | gem | 443 | 27.8 [23.9, 32.1] | 39.5 [34.8, 44.0] | 44.0 [39.3, 49.0] | 33.6 [29.7, 37.8] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A1-dinov2-l-224-cls | facebook/dinov2-large | 224 | cls | 443 | 29.6 [25.3, 33.6] | 36.3 [31.6, 40.9] | 39.5 [34.8, 44.2] | 33.1 [28.8, 37.0] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A8-qwen3-vl-embedding-2b-512 | Qwen/Qwen3-VL-Embedding-2B | 512 | head | 443 | 26.4 [22.3, 30.5] | 37.7 [33.0, 42.2] | 42.0 [37.2, 46.5] | 32.4 [28.4, 36.5] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A1-dinov2-g-224-cls | facebook/dinov2-giant | 224 | cls | 443 | 27.8 [23.5, 31.8] | 36.1 [31.8, 40.6] | 39.7 [34.8, 44.7] | 31.9 [27.8, 35.9] |
| R2 | T1 | tune | screening-20k-publishable-fetched | H-H1-gemini-3.5-flash-lite | google/gemini-3.5-flash-lite | 512 | bm25 | 138 | 86.2 [80.4, 92.0] | 96.4 [93.5, 99.3] | 97.1 [94.2, 99.3] | 90.9 [87.0, 94.8] |
| R2 | T1 | tune | screening-20k-publishable-fetched | H-H3-A7+H1-gemini-3.5-flash-lite | google/embeddinggemma-2 + google/gemini-3.5-flash-lite | 512 | fusion | 138 | 75.4 [68.1, 82.6] | 89.1 [84.1, 94.2] | 89.1 [84.1, 94.2] | 81.0 [75.3, 86.5] |
| R2 | T1 | tune | screening-20k-publishable-fetched | H-H1-qwen3-vl-30b-a3b | qwen/qwen3-vl-30b-a3b-instruct | 512 | bm25 | 138 | 65.2 [57.2, 73.2] | 82.6 [76.1, 88.4] | 87.0 [81.2, 92.0] | 72.7 [66.2, 79.1] |
| R2 | T1 | tune | screening-20k-publishable-fetched | H-H3-A7+H1-qwen3-vl-30b-a3b | google/embeddinggemma-2 + qwen/qwen3-vl-30b-a3b-instruct | 512 | fusion | 138 | 65.2 [57.2, 73.2] | 76.1 [68.8, 83.3] | 79.7 [73.2, 86.2] | 70.7 [63.7, 77.7] |
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
| R2 | T1 | tune | screening-20k-publishable-fetched | E-E5-A3-alpha3 | gmberton/MegaLoc | 322 | salad | 138 | 20.3 [14.5, 26.8] | 23.9 [17.4, 31.2] | 27.5 [20.3, 34.8] | 22.7 [16.7, 29.3] |
| R2 | T1 | tune | screening-20k-publishable-fetched | E-E5-A3 | gmberton/MegaLoc | 322 | salad | 138 | 15.2 [9.4, 21.0] | 23.9 [17.4, 31.2] | 25.4 [18.1, 32.6] | 19.5 [14.0, 25.4] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G1-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 99 | 60.6 [51.5, 70.7] | 83.8 [76.8, 90.9] | 89.9 [83.8, 96.0] | 71.5 [64.2, 79.3] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G1-A7+A3+A5 | google/embeddinggemma-2 + gmberton/MegaLoc + timm/PE-Core-L-14-336 | 512 | fusion | 99 | 59.6 [50.5, 69.7] | 81.8 [73.7, 89.9] | 88.9 [82.8, 94.9] | 70.0 [62.8, 78.1] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G1-A9+A3 | google/gemini-embedding-2 + gmberton/MegaLoc | 512 | fusion | 99 | 55.6 [46.5, 65.7] | 85.9 [78.8, 92.9] | 88.9 [81.8, 94.9] | 69.0 [62.0, 76.4] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E5-G1-A7+A3-alpha3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 99 | 59.6 [50.5, 69.7] | 79.8 [71.7, 87.9] | 87.9 [80.8, 93.9] | 69.0 [61.6, 77.1] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E5-A3-alpha3 | gmberton/MegaLoc | 322 | salad | 99 | 57.6 [48.5, 67.7] | 83.8 [76.7, 90.9] | 85.9 [78.8, 91.9] | 68.4 [60.5, 76.8] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G2-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 99 | 55.6 [45.5, 65.7] | 80.8 [72.7, 88.9] | 89.9 [83.8, 96.0] | 67.2 [59.9, 75.3] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G2-A9+A3 | google/gemini-embedding-2 + gmberton/MegaLoc | 512 | fusion | 99 | 56.6 [47.4, 66.7] | 80.8 [72.7, 88.9] | 88.9 [82.8, 94.9] | 67.1 [59.5, 75.5] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G1-A7+A4 | google/embeddinggemma-2 + google/siglip2-so400m-patch14-384 | 512 | fusion | 99 | 55.6 [45.5, 65.7] | 79.8 [71.7, 87.9] | 84.8 [77.8, 91.9] | 66.6 [58.6, 74.8] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G2-A7+A3+A5 | google/embeddinggemma-2 + gmberton/MegaLoc + timm/PE-Core-L-14-336 | 512 | fusion | 99 | 51.5 [41.4, 61.6] | 80.8 [72.7, 87.9] | 88.9 [82.8, 94.9] | 64.1 [56.6, 72.0] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E5-G1-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 99 | 50.5 [40.4, 60.6] | 82.8 [74.7, 89.9] | 84.8 [77.8, 91.9] | 63.8 [56.2, 71.5] |
| R2 | T3 | tune | screening-20k-publishable-fetched | G-G2-A7+A4 | google/embeddinggemma-2 + google/siglip2-so400m-patch14-384 | 512 | fusion | 99 | 52.5 [42.4, 62.6] | 76.8 [67.7, 84.8] | 84.8 [76.8, 91.9] | 63.7 [55.8, 72.0] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E5-A7-alpha3 | google/embeddinggemma-2 | 512 | head | 99 | 51.5 [42.4, 61.6] | 74.7 [65.7, 83.8] | 78.8 [70.7, 86.9] | 63.1 [55.0, 71.6] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E5-A7 | google/embeddinggemma-2 | 512 | head | 99 | 45.5 [35.4, 55.6] | 74.7 [65.7, 83.8] | 77.8 [69.7, 85.9] | 59.6 [51.7, 68.0] |
| R2 | T3 | tune | screening-20k-publishable-fetched | H-H3-A7+H1-gemini-3.5-flash-lite | google/embeddinggemma-2 + google/gemini-3.5-flash-lite | 512 | fusion | 99 | 45.5 [36.4, 55.6] | 74.7 [65.7, 82.8] | 82.8 [74.7, 89.9] | 58.6 [50.9, 66.6] |
| R2 | T3 | tune | screening-20k-publishable-fetched | H-H3-A7+H1-qwen3-vl-30b-a3b | google/embeddinggemma-2 + qwen/qwen3-vl-30b-a3b-instruct | 512 | fusion | 99 | 42.4 [33.3, 51.5] | 75.8 [67.7, 83.8] | 82.8 [74.7, 90.9] | 57.2 [49.7, 64.8] |
| R2 | T3 | tune | screening-20k-publishable-fetched | E-E5-A3 | gmberton/MegaLoc | 322 | salad | 99 | 30.3 [21.2, 40.4] | 78.8 [70.7, 86.9] | 79.8 [71.7, 87.9] | 51.3 [44.7, 59.0] |
| R2 | T3 | tune | screening-20k-publishable-fetched | H-H1-gemini-3.5-flash-lite | google/gemini-3.5-flash-lite | 512 | bm25 | 99 | 4.0 [1.0, 8.1] | 15.2 [9.1, 22.2] | 22.2 [14.1, 31.3] | 10.0 [6.2, 14.5] |
| R2 | T3 | tune | screening-20k-publishable-fetched | H-H1-qwen3-vl-30b-a3b | qwen/qwen3-vl-30b-a3b-instruct | 512 | bm25 | 99 | 1.0 [0.0, 3.0] | 5.1 [1.0, 9.1] | 9.1 [4.0, 14.1] | 4.6 [2.2, 7.5] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G1-A9+A3 | google/gemini-embedding-2 + gmberton/MegaLoc | 512 | fusion | 443 | 40.2 [35.7, 44.7] | 54.4 [49.9, 58.9] | 60.5 [56.0, 64.8] | 47.1 [42.9, 50.9] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E5-A3-alpha3 | gmberton/MegaLoc | 322 | salad | 443 | 38.4 [34.1, 42.7] | 51.7 [47.2, 56.2] | 56.9 [52.1, 61.4] | 45.1 [41.0, 49.3] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G1-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 37.9 [33.2, 42.4] | 49.9 [45.4, 54.4] | 54.9 [50.3, 59.1] | 43.8 [39.5, 47.8] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G2-A9+A3 | google/gemini-embedding-2 + gmberton/MegaLoc | 512 | fusion | 443 | 34.5 [30.2, 38.8] | 50.3 [45.4, 54.9] | 55.8 [51.0, 60.3] | 42.3 [38.3, 46.3] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G1-A7+A3+A5 | google/embeddinggemma-2 + gmberton/MegaLoc + timm/PE-Core-L-14-336 | 512 | fusion | 443 | 35.7 [30.9, 40.0] | 47.6 [42.9, 52.2] | 52.8 [48.3, 57.3] | 41.9 [37.7, 46.0] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G2-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 34.8 [30.5, 39.1] | 47.6 [43.1, 52.1] | 51.9 [47.6, 56.7] | 41.1 [36.9, 45.1] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G2-A7+A3+A5 | google/embeddinggemma-2 + gmberton/MegaLoc + timm/PE-Core-L-14-336 | 512 | fusion | 443 | 35.0 [30.5, 39.3] | 46.3 [41.5, 51.0] | 52.6 [48.1, 57.1] | 41.0 [36.9, 45.2] |
| R2 | T5 | tune | screening-20k-publishable-fetched | H-H3-A7+H1-gemini-3.5-flash-lite | google/embeddinggemma-2 + google/gemini-3.5-flash-lite | 512 | fusion | 443 | 33.2 [28.9, 37.5] | 47.9 [43.1, 52.6] | 53.7 [49.2, 58.5] | 40.7 [36.7, 44.6] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E5-G1-A7+A3-alpha3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 35.0 [30.7, 39.1] | 45.8 [40.9, 50.1] | 51.2 [46.5, 55.8] | 40.4 [36.1, 44.3] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E5-A3 | gmberton/MegaLoc | 322 | salad | 443 | 31.2 [26.9, 35.4] | 50.3 [45.8, 54.9] | 53.7 [49.0, 58.0] | 40.0 [35.9, 43.9] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E5-G1-A7+A3 | google/embeddinggemma-2 + gmberton/MegaLoc | 512 | fusion | 443 | 33.4 [29.1, 37.5] | 45.8 [40.9, 50.1] | 51.0 [46.3, 55.3] | 39.1 [34.8, 42.9] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G1-A7+A4 | google/embeddinggemma-2 + google/siglip2-so400m-patch14-384 | 512 | fusion | 443 | 33.2 [28.9, 37.5] | 45.1 [40.4, 49.7] | 49.0 [44.2, 53.5] | 39.1 [34.9, 43.2] |
| R2 | T5 | tune | screening-20k-publishable-fetched | G-G2-A7+A4 | google/embeddinggemma-2 + google/siglip2-so400m-patch14-384 | 512 | fusion | 443 | 32.5 [28.0, 36.8] | 43.8 [39.3, 48.5] | 48.1 [43.6, 52.6] | 38.1 [33.8, 42.3] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E5-A7-alpha3 | google/embeddinggemma-2 | 512 | head | 443 | 26.6 [22.6, 30.7] | 41.8 [37.0, 46.3] | 43.8 [39.1, 48.3] | 33.7 [29.9, 37.6] |
| R2 | T5 | tune | screening-20k-publishable-fetched | H-H3-A7+H1-qwen3-vl-30b-a3b | google/embeddinggemma-2 + qwen/qwen3-vl-30b-a3b-instruct | 512 | fusion | 443 | 25.7 [21.9, 29.6] | 40.2 [35.7, 44.7] | 47.0 [42.2, 51.5] | 33.5 [29.6, 37.2] |
| R2 | T5 | tune | screening-20k-publishable-fetched | E-E5-A7 | google/embeddinggemma-2 | 512 | head | 443 | 22.6 [18.7, 26.4] | 41.5 [36.8, 46.0] | 43.3 [38.6, 47.9] | 31.2 [27.4, 34.9] |
| R2 | T5 | tune | screening-20k-publishable-fetched | H-H1-gemini-3.5-flash-lite | google/gemini-3.5-flash-lite | 512 | bm25 | 443 | 15.8 [12.4, 19.4] | 30.7 [26.4, 35.0] | 38.1 [33.6, 43.1] | 23.4 [20.0, 26.9] |
| R2 | T5 | tune | screening-20k-publishable-fetched | H-H1-qwen3-vl-30b-a3b | qwen/qwen3-vl-30b-a3b-instruct | 512 | bm25 | 443 | 8.8 [6.1, 11.5] | 16.3 [12.9, 19.9] | 20.1 [16.7, 23.9] | 12.5 [9.6, 15.3] |

Not found (T8): confidence is the top entity's score. Lower is better for "wrong but confident", higher for "matched".

| Round | Set | Half | Gallery | Run | In-index sets | In-index | Not found | Wrong but confident, 50% matched | Matched, 5% wrong |
|---|---|---|---|---|---|---|---|---|---|
| R2 | T8 | tune | screening-20k-publishable-fetched | A-A7-embeddinggemma-2-512 | T1 + T5 | 581 | 232 | 25.0 [18.5, 31.5] | 12.4 [8.6, 17.9] |
| R2 | T8 | tune | screening-20k-publishable-fetched | G-G1-A7+A3 | T1 + T5 | 581 | 232 | 27.6 [21.6, 34.1] | 15.0 [10.2, 19.4] |
| R2 | T8 | tune | screening-20k-publishable-fetched | A-A9 | T1 + T5 | 581 | 232 | 29.7 [23.3, 35.8] | 8.8 [3.3, 17.6] |
| R2 | T8 | tune | screening-20k-publishable-fetched | A-A3-megaloc-322 | T1 + T5 | 581 | 232 | 30.2 [23.7, 36.2] | 11.9 [7.1, 20.7] |
<!-- summary:end -->
