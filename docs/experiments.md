# MIROIR: experiments

Status: plan for weeks 1–2, with later rounds running into week 3. Last updated 2026-10-12.

Finding the right heritage record from a photo is a research problem before it is a product problem. We don't pick models by reputation. We split search into steps (A–H), list every serious option for each step, and score them all with the same harness on the same test sets ([technical-spec.md §7](technical-spec.md#7-evaluation)).

## 1. Rules

1. **Open models first.** Closed models (through OpenRouter or their own API) run as reference points. One enters the final stack only if it beats the best open option by more than the error bar (§4).
2. **Simplest wins a tie.** When two options' error bars overlap, keep the cheaper, faster, open one.
3. **Same queries for every option.** Comparisons are paired: every option answers the same queries against the same index.
4. **The sealed half stays sealed.** We tune on the tuning half only and open the sealed half once, in week 6.
5. **Paid runs have a cap.** Before running a paid model on the full index, measure its cost on 100 images. If the full run would cost more than €200, ask the project lead first.
6. **Every result is committed.** Each run writes `eval/results/<round>/<step>-<option>.json`, and `docs/results.md` holds the summary table.

## 2. Options by step

### A. Image to vector (photo → photo search)

| ID | Option | Open? | Why it's in the list |
|---|---|---|---|
| A1 | DINOv2 ViT-L/14 and ViT-g/14 | yes | The current default. Everything else is measured against it |
| A2 | DINOv3 ViT-L and larger distilled sizes | yes | Published results beat DINOv2 clearly on Met (museum art) and AmsterTime (modern vs archive photos of Amsterdam) |
| A3 | MegaLoc (DINOv2 + SALAD) | yes | Built to recognise places from any angle |
| A4 | SigLIP 2 so400m and giant | yes | Text and image in one space |
| A5 | Meta Perception Encoder, PE-Core L and G | yes | A strong rival to SigLIP 2 |
| A6 | MetaCLIP 2 | yes | Trained on many languages |
| A7 | EmbeddingGemma 2 (740M, 768-d) | yes | Small, multimodal, runs on a phone |
| A8 | Qwen3-VL-Embedding 2B and 8B | yes | Top of MMEB-V2 among open models |
| A9 | Gemini Embedding 2 | no | Closed reference point |
| A10 | Voyage multimodal 3.5, Nemotron Embed VL 1B v2 | no / yes | More reference points, both on OpenRouter |
| A11 | RootSIFT + VLAD, and perceptual hash | yes | Classic floor. The hash also flags exact duplicates |

For A1–A3, also compare three ways to turn an image into one vector: the class token, GeM pooling over patch tokens, and VLAD over patch tokens (as in AnyLoc).

### B. Words → records (F2)

| ID | Option |
|---|---|
| B1 | The text tower of A4–A6, searching image vectors directly |
| B2 | The unified embedders A7–A9, searching image vectors directly |
| B3 | Caption first: a vision model writes a French description of every collection image. We search those with BM25 plus a text embedder (EmbeddingGemma 2 text or Qwen3-Embedding). Captioners to compare: Qwen3-VL 8B and 32B on our machines, and one closed model through OpenRouter |
| B4 | BM25 on POP text fields only (the floor) |
| B5 | An AI model rewrites the query into filters and search terms before B1–B4 |

### C. Finding objects in the query photo (F4)

| ID | Option |
|---|---|
| C1 | OWLv2, prompted with "painting", "sculpture", "statue", "vase", "furniture", "building", "stained glass window" |
| C2 | Grounding DINO with the same prompts |
| C3 | SAM 3 with the same prompts, using box and mask output |
| C4 | Qwen3-VL asked to return a box for every artwork |
| C5 | Classic frame finding (line detection, then four-sided shapes) |
| C6 | Sliding windows at 3 scales (the floor) |
| C+ | Straightening: undo the slant of any four-sided box before embedding. Tested on top of the best of C1–C5 |

Scored on: box recall against the 231 paintings in the room photos, then the T3 search score.

### D. Detail → whole (F5)

| ID | Option |
|---|---|
| D1 | Index tiles at 3 scales, 14 per image |
| D2 | Store DINO patch vectors and score by best patch-to-patch match (ColBERT-style late interaction) |
| D3 | Local features with an inverted index (DISK or RootSIFT with ASMK) |
| D4 | SAM 3 cuts each collection image into regions, and each region gets a vector |
| D5 | Embed the crop alone against whole-image vectors (the floor) |

Scored on: T4, plus index size and search time. Run on Rose Valland first. Extend to everything only if the winner fits in memory.

### E. Re-ranking the top 50

| ID | Option |
|---|---|
| E1 | SuperPoint or ALIKED keypoints, LightGlue matching, RANSAC. Score = number of matches that agree on one geometry |
| E2 | Dense matching with RoMa or MASt3R |
| E3 | Qwen3-VL-Reranker 2B and 8B |
| E4 | A vision model as judge on the top 10 pairs: Qwen3-VL 32B (open) against one closed model |
| E5 | Query expansion: average the query with its top 3 hits and search again (α-QE) |

Scored on: T1, T2, T3, T5, plus time added per search.

### F. Phone photos against the black-and-white archive

| ID | Option |
|---|---|
| F1 | Turn query and collection images grey before embedding |
| F2 | Add Wikimedia Commons photos as extra images of Mémoire buildings, keeping T2 query photos out |
| F3 | Fine-tune the best A model with LoRA and a contrastive loss on Commons–Mémoire pairs, kept apart from T2 |
| F4 | Colourise archive photos (DDColor) before embedding |
| F5 | Start from a checkpoint already tuned on old-vs-new pairs, if a usable one exists |

Scored on: T2. T1 must not drop by more than its error bar.

### G. Combining scores and deciding "not found"

| ID | Option |
|---|---|
| G1 | Weighted sum of min-max-normalised scores (OpenSearch hybrid pipeline) |
| G2 | Reciprocal-rank fusion |
| G3 | Logistic regression over image score, text score and E1 match count. It outputs a probability used as the "not found" score |

Scored on: T1–T5 and T7 for ranking, T8 for "not found" (wrong-but-confident rate at a fixed match rate).

### H. Different approaches altogether

| ID | Option |
|---|---|
| H1 | Ask a vision model "what is this monument or artwork, and where is it?" and look its answer up in POP with BM25. Qwen3-VL 32B against one closed model |
| H2 | Use the photo's GPS position to keep only records within 2 km. Tested on T2 photos that carry GPS. Needs coordinates for records (POP coordinates where present, else the commune centre) |
| H3 | H1 or H2 as one more signal into G3 |

## 3. Test sets

[technical-spec.md §7](technical-spec.md#7-evaluation) defines sets T1–T9. Every set except T9 is split 50/50 into a tuning half and a sealed half, **by building or object**, so no entity appears in both halves.

## 4. Measures

For every run:

- **Accuracy:** hit rate in the top 1, 5 and 10, and mean average precision where a query has several right answers
- **Error bars:** 95% bootstrap intervals (1,000 resamples). Differences between options use a paired bootstrap on the same queries
- **Time:** median and 95th-percentile search time, on CPU and on GPU
- **Size:** index size on disk and in memory
- **Money:** indexing cost, and cost per 1,000 searches for paid models

As a guide to what counts as a real difference, an interval on 231 queries spans about ±6 points, and on 1,000 queries about ±3 points.

## 5. Rounds

| Round | When | What | Output |
|---|---|---|---|
| R0 | Week 1 | Build the harness and test sets. Download images. Reproduce one published DINOv2 score on AmsterTime and Met (T9) to check the harness | Harness passes T9 within 2 points of the published figure |
| R1 | Week 1–2 | All A options alone on T1, T2, T3 (hand boxes), T5. All B options on T6 and T7 | Top 3 image models, top 2 text approaches |
| R2 | Week 2 | E and F on top of the R1 winners | Best re-ranker; best phone-gap fix |
| R3 | Week 2–3 | C, D, G and H on the R2 winners | The full stack for the website and the full POP index |
| R4 | Week 6 | The chosen stack and the two runners-up on the sealed halves, once | Final scores for the decision meeting |

Before each round, write in `docs/results.md` what we expect to see and why. Comparing that guess with the result is how we learn.
