# Search levels and the move to Scaleway

Plan set 2026-11-02, cut down the same day. The ministry's Scaleway project ("Miroir") replaces Modal for compute: GPU machines and Object Storage in Paris. Spend cap: €100 without asking again.

## Why

- The site searches 147k images. POP holds 2.14M distinct images in 2.1M records. The two targets the ministry has named (APMH047981, the Vétheuil altarpiece, and APMH00020986, the Virgin and Child of Thoisy-le-Désert) are not in the July export, but both are in our own POP download. Other tools on the market (Arte-fact: 3M objects) search everything; so must we.
- Today's search compares a whole photo with whole photos. It cannot find a part inside a whole: a carved group from a sale catalogue against the 1896 photo of the altarpiece it came from, where the group covers about 3% of the frame.

## What changes

1. **Index every image of POP.** All 2.14M distinct images, every collection, publishable or not (our own models only), with A7 and A3 as today.
2. **Crop vectors.** Each photo also gets vectors for its parts (2×2 and 3×3 grids and the detector's outlines), so a visitor's photo of one part can meet the part of a POP photo where it stands. A record takes the score of its best crop. Same models, same ranking.
3. **Two levels on one page.**
   - **Instant** (under 2 s, free): today's search, crops included.
   - **Deeper** (1–3 min, a few cents): the agent (Mistral Large 4) already on `/chat`. It looks at the photo, writes the words a record would use ("Mise au tombeau, retable flamand"), searches, reads records with `read_record` and asks the visitor questions. "Chercher plus loin" starts it from the Instant search, keeping the photo, the framing and the words.

Nothing else is added unless the measurements below call for it. Held back: AMES re-ranking and point matching (SuperPoint + LightGlue), for the case where crops find the whole but rank it too low; DINOv3, which lost to A7 and A3 in R1.

## Indexing

1. Download all 2.14M images to Object Storage, at the largest size POP serves up to 1,024 px on the long side. Report every image that fails, by collection and reason.
2. Whole-photo vectors, A7 and A3. A3 for 2.14M images is 36 GB at full length, so test a shorter A3 (1,024 numbers) on the tune sets first, from the stored vectors on the Mac (no model).
3. Crop vectors. How many crops per photo, and with which model, is set from the measured speed of the GPU so that the whole run stays within €100: every image gets crops, the grid's fineness is what moves.
4. Words: the existing word index over all 2.1M records.
5. A coverage report: every record with an image has vectors, or the reason it does not.

## Measures

On VET (3 Vétheuil images) and PARTS (360 detail → whole pairs from POP, tune and sealed halves), plus T2, T2b, T8c and H4 so nothing gets worse:

1. Today's search on the full index.
2. The same with crops.
3. The agent.

## Order of work

1. Move the jobs and the query search from Modal to Scaleway. (Running.)
2. Build VET and PARTS. (Done: `miroir testsets vet`, `miroir testsets parts`.)
3. Download and embed everything; build the full index on Scaleway.
4. Crop vectors; measures 1 to 3.
5. One page with Instant and Deeper; deploy on Scaleway.
