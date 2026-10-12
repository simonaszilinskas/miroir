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

DINOv3 7B: not run. The code is ready (`uv run miroir r0 run --model dinov3-vit7b16-224`), but the weights are gated and Meta has not yet approved access.

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

## Summary of all runs

<!-- summary:start (generated by `miroir eval summary`; do not edit) -->
Each cell: value in %, then the 95% bootstrap interval (1,000 resamples over queries).

| Round | Set | Half | Gallery | Run | Model | Px | Pooling | Queries | Hit@1 | Hit@5 | Hit@10 | mAP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R0 | amstertime-all | all | amstertime | A-dinov2-giant-224 | facebook/dinov2-giant | 224 | cls | 2,462 | 36.6 [34.7, 38.4] | 58.0 [56.2, 59.9] | 65.6 [63.9, 67.4] | 46.7 [45.0, 48.4] |
| R0 | amstertime-new | all | amstertime | A-dinov2-giant-224 | facebook/dinov2-giant | 224 | cls | 1,231 | 32.7 [30.0, 35.4] | 53.9 [51.0, 56.8] | 61.3 [58.5, 64.0] | 42.7 [40.2, 45.1] |
| R0 | amstertime-old | all | amstertime | A-dinov2-giant-224 | facebook/dinov2-giant | 224 | cls | 1,231 | 40.4 [37.6, 43.1] | 62.1 [59.5, 64.8] | 70.0 [67.6, 72.5] | 50.7 [48.4, 53.3] |
| R1 | T1 | tune | screening-20k-publishable-fetched | A-A9 | google/gemini-embedding-2 | 512 | api | 138 | 33.3 [26.1, 40.6] | 45.7 [37.7, 53.6] | 51.4 [44.2, 59.4] | 38.9 [32.1, 46.0] |
| R1 | T3 | tune | screening-20k-publishable-fetched | A-A9 | google/gemini-embedding-2 | 512 | api | 99 | 54.5 [44.4, 64.6] | 77.8 [69.7, 85.9] | 83.8 [75.8, 90.9] | 65.2 [57.4, 73.6] |
| R1 | T5 | tune | screening-20k-publishable-fetched | A-A9 | google/gemini-embedding-2 | 512 | api | 443 | 34.8 [30.5, 39.1] | 47.0 [42.2, 51.2] | 51.2 [46.3, 55.5] | 41.0 [36.9, 45.1] |
<!-- summary:end -->
