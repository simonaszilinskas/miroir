# Search levels and the move to Scaleway

Plan set 2026-11-02. The ministry's Scaleway project ("Miroir") replaces Modal for compute: GPU machines, Object Storage and Generative APIs, all in Paris.

## Why

- The site searches 147k images. POP holds 2.14M distinct images in 2.1M records. The two targets the ministry has named (APMH047981, the Vétheuil altarpiece, and APMH00020986, the Virgin and Child of Thoisy-le-Désert) are not in the July export, but both are in our own POP download.
- Today's search compares a whole photo with whole photos. It cannot find a part inside a whole: a carved group from a sale catalogue against the 1896 photo of the altarpiece it came from, where the group is about 100 pixels tall among 15 others.
- The classic page (`/`) and the conversation (`/chat`) are two separate tools. A visitor should not have to choose between them up front.

## One search, three levels

One page, one photo or words box. The visitor picks a level, or starts at Instant and presses "Chercher plus loin", which keeps the photo, the box and the words.

| Level | What runs | Time | Cost per search |
|---|---|---|---|
| **Instant** | Today's search: whole-photo vectors (A7 or A3, by route), words, filters, position. The visitor can frame part of the photo or tap an outline. | under 2 s | nothing |
| **Medium** | Instant, plus: a vision model reads the photo (caption text, kind of object, period) and turns it into words and filters; the photo is compared with crops of index photos, so a part can find its whole; the first 50 are re-ranked by matching points between the photo and each candidate. | 10–20 s | under €0.01 |
| **Hard** | The agent (Mistral Large 4) uses Medium as its search tool, plus `read_record`, `more_like`, framing and side-by-side comparison, and asks the visitor questions when it is stuck. | 1–3 min | a few cents |

## Indexing everything

1. Download all 2.14M images to Object Storage, long side 1,024 px. Report every image that fails, by collection.
2. Whole-photo vectors with A7 (768 numbers) and A3 (8,448). A3 for 2.14M images is 36 GB, so test a shorter A3 (1,024 numbers) on the tune sets first, from the stored vectors on the Mac (no model).
3. Crop vectors for objects and interiors: outlines from the detector plus a fixed grid (whole, 2×2, 3×3). A7 and A3 make them, as for whole photos. DINOv3 (L, H+ and 7B) lost to both in R1 (results.md: T1, T3, T5), despite the paper's lead on Met and AmsterTime; it is not retried unless crop search on the part-to-whole tune half falls short. Patch features from the same pass are kept for re-ranking (AMES, below).
4. Words: the existing word index over all 2.1M records.
5. A coverage report: every record with an image has at least one vector, or the reason it does not.

## Test sets

The ministry has no test set to give. We build our own:

1. **Vétheuil:** the three images Camille sent (the altarpiece today, two groups in a sale catalogue) against APMH047981 and PM95000723.
2. **Part to whole:** POP records that hold both a detail photo and a whole view. Query with the detail, expect the record. Split into tune and sealed halves, as before.
3. **No regression:** T2, T2b, T8c and the H4 photos, so Instant does not get worse.

## Order of work

1. Move the jobs and the query search from Modal to Scaleway (Docker image on a GPU machine, Object Storage for files). No spend beyond a smoke test.
2. Build the part-to-whole test set and run today's search on it and on Vétheuil: the baseline.
3. Download and embed everything; build the full index on Scaleway.
4. Medium, in this order, each kept only if it helps on the tune halves:
   1. Query reading: a vision model writes the scene, material, period and kind ("Mise au tombeau, groupe sculpté, bois, retable"), which goes to the word search. POP texts often name the scenes inside an altarpiece, where a 3% group is nearly invisible to a whole-photo vector.
   2. Crop search over the crop vectors.
   3. AMES (ECCV 2024, open weights) re-ranks the first 100 from patch features: the largest single gain on ILIAS (CVPR 2025), the closest public benchmark to ours.
   4. Point matching (SuperPoint + LightGlue) on the first 20, to score and to show the visitor why two photos match. It will not save a carved group photographed from another angle; Hard handles those.
5. Hard on top of Medium; merge `/` and `/chat` into one page with the three levels; deploy on Scaleway.
