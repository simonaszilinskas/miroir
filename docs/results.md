# MIROIR: results

One section per round of [experiments.md](experiments.md). Before each round, write what we expect and why. After it, add the table and what we learned.

## R0: harness check

Expected (written 2026-10-12, before any run): DINOv2 ViT-g/14 (`facebook/dinov2-giant`), class token after the final LayerNorm, shortest side 256 then centre crop 224, cosine, top 128, lands within 2 points of 48.9 mAP (DINOv3 paper, Tables 9 and 23) in one of the three directions. Our guess: "all" (both directions pooled) lands nearest, near 49, with archive → street view and street view → archive a few points either side of it. Archive photos are greyscale and often odd crops, so we expect street-view queries (searching the archive) to score lower. If no direction lands within 2 points, suspect preprocessing first (resize, crop, colour), then the AP formula.

Results: pending. The embedding needs a hosted route (no model runs on the Mac); once a vector run of the 2,462 images exists, `uv run miroir eval amstertime score --run <run>` scores it.

## R1: image models alone (step A)

Expected (written 2026-10-12, before any R1 score): on the 19,212-image publishable screening gallery, scored on queries a closed model may see, the DINO family (A1, A2) should win T1 and T5: published instance-retrieval scores put DINOv2 and DINOv3 well ahead of text-aligned models (Oxford-H, Met, AmsterTime). We expect the two closed embedders, Gemini Embedding 2 (A9) and Voyage multimodal 3.5 (A10a), to be weaker on T1's archive building photos, which are greyscale shots of one building from different angles, where meaning-level likeness pulls in other buildings of the same kind. They may do better on T3, where query and answers are crops of the same negative, and on T5 museum objects. Between the two, no strong guess; Gemini has the larger vectors (3,072 against 1,024). T8 waits for the not-found measure.

Results: A9 and A10a first; open models join when their runs exist. Each closed-model score uses `--gallery screening-20k-publishable --publishable-queries` (T1 500, T3 99, T5 449 queries); score open models the same way to compare them.
