# MIROIR: results

One section per round of [experiments.md](experiments.md). Before each round, write what we expect and why. After it, add the table and what we learned.

## R0: harness check

Expected (written 2026-10-12, before any run): DINOv2 ViT-g/14 (`facebook/dinov2-giant`), class token after the final LayerNorm, shortest side 256 then centre crop 224, cosine, top 128, lands within 2 points of 48.9 mAP (DINOv3 paper, Tables 9 and 23) in one of the three directions. Our guess: "all" (both directions pooled) lands nearest, near 49, with archive → street view and street view → archive a few points either side of it. Archive photos are greyscale and often odd crops, so we expect street-view queries (searching the archive) to score lower. If no direction lands within 2 points, suspect preprocessing first (resize, crop, colour), then the AP formula.

Results: pending. The embedding needs a hosted route (no model runs on the Mac); once a vector run of the 2,462 images exists, `uv run miroir eval amstertime score --run <run>` scores it.
