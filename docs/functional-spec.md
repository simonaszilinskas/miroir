# MIROIR: functional specification

Status: draft for the proof of concept (PoC). Last updated 2026-10-12.

## 1. Purpose

MIROIR is Google Lens for French heritage. A person takes or uploads a photo, or describes what they see in words, and MIROIR tells them which heritage record it matches in [POP](https://pop.culture.gouv.fr/): this painting, this church portal, this vase. When it finds no match, it says so and shows the closest records instead.

The PoC has to prove two things in six weeks:

1. The search finds the right record often enough to be useful (measured score, section 3).
2. Members of the public want to use it (user tests, section 10).

## 2. Users

The PoC targets the **general public**: visitors in a museum, walkers in front of a building, people holding an old postcard or a photo of a painting.

They want to know **what they are looking at** and to reach the official record behind it, with no expert vocabulary and no knowledge of how POP is organised.

Heritage staff will use MIROIR too, but the PoC is not designed around their needs.

## 3. Pass bar

We fixed these numbers before seeing any score. The decision meeting at the end of week 6 says yes if MIROIR meets all four.

| Test | What counts as a hit | Target |
|---|---|---|
| Archive photo of a building | The right building is in the top 10 | ≥ 80% |
| Phone photo of a building | The right building is in the top 10 | ≥ 60% |
| One painting picked out of a photo of a room full of paintings | The right record is in the top 5 | ≥ 70% |
| User tests | Testers who found at least one thing they photographed and would use MIROIR again | ≥ 4 of 5 |

[technical-spec.md](technical-spec.md#7-evaluation) describes how each test set is built. [experiments.md](experiments.md) describes how we compare solutions on them. The final scores come from a sealed half of each set, opened once in week 6.

## 4. Features

All five are in scope. F5 comes last in the plan.

### F1. Same object, different view

A photo of a building, object or painting finds its record even when the photo was taken from another angle, at another time, in other light or in colour against a black-and-white archive print.

*Done when:* the archive-photo and phone-photo targets in section 3 are met.

### F2. Search by words

A person types what they see, in French or another language ("portail roman avec un tympan sculpté", "a still life with a dead hare"), and gets matching records. Words can also refine a photo search ("same, but in Toulouse").

*Done when:* we have scores on two sets: 50 French descriptions written by hand, and 1,000 descriptions written by an AI model for hidden images. We report these scores but set no pass bar for them.

### F3. Filters with image search

Any search can be narrowed by:

- collection (Joconde, Mémoire, Rose Valland)
- place (region, département, commune)
- domain (painting, architecture, ceramics…)
- period
- author

Filters apply before ranking, so a filtered search still returns a full page of results.

### F4. Several objects in one photo

When a photo contains several artworks (a wall of paintings, a shop window, a room), MIROIR outlines each one it finds. The person taps an outline to search that object alone.

*Done when:* the room-photo target in section 3 is met.

### F5. The whole from a detail

A photo of a detail (a carved capital, a signature, a stamp on the back of a canvas) finds the record of the whole object or building.

*Done when:* the Rose Valland detail images (section 9) find their own record. We report the score with no pass bar.

## 5. Screens and flow

MIROIR is a website built for phones first. It also works on desktop. There is no native app.

1. **Home.** A large "Take a photo" button, an "Upload" button and a text box.
2. **Photo view.** The photo with an outline around each object MIROIR found. The person can:
   - tap an outline to search that object
   - draw a box by hand to search any area, which also covers details (F5)
   - search the whole photo
3. **Results.** One card per match, best first, with filters in a panel. A text box refines the search.
4. **Record.** Larger images of the match, its key facts and a link to the full record on POP.

## 6. Results

### One card per thing, not per photo

Mémoire holds one record per photo, and Versailles alone has 589 of them. MIROIR groups photos by the building or object they show, using the Mérimée or Palissy ID that 76% of Mémoire records carry. Each group becomes one card.

Joconde and Rose Valland records already describe one object each, so each record is one card.

Each card shows:

- the image that best matches the query
- the title or building name
- the place
- the collection
- a link to the POP record

### When nothing matches

Below a set score, MIROIR shows **"Not found in the collections"**, then **"Looks similar to:"** with the closest cards. A wrong answer given with confidence does more harm than an honest "not found". We choose the cut-off from the test scores in week 3.

## 7. Rose Valland images

Rose Valland records have a median of 4 images and up to 52. Many are not the front of the work: details, backs, inscriptions, labels and stamps. MIROIR indexes all of them and tags each by type. Results always show the front view, and the matching image is marked ("matched on: back of canvas"). A stamp or label photographed on its own can then lead to its record.

## 8. Access

The PoC sits behind a login. Ministry staff get accounts, and so do the five public testers for their session. We can then show every image, including those not cleared for publication, and leave image rights to a later public launch.

## 9. Data

| Phase | Records | Images to index |
|---|---|---|
| Weeks 1–3 | The current exports: Mémoire 66,000, Joconde 33,000, Rose Valland 2,400 | About 138,000 |
| Weeks 4–6 | Every POP record with an image, from a full export | Not yet known |

What the current data holds:

- **Mémoire.** Heritage photos, 86% of them architecture. 88% are black-and-white, most are about 670×512 px, and 83% date from the 1940s to the 1980s.
- **Joconde.** Museum objects in colour. 12,000 records have two or more views.
- **Rose Valland.** Works looted under the Occupation and recovered after the war, with long provenance histories. It includes 15 photos of Jeu de Paume rooms from the 1940s. 231 painting records link back to the room photo they hang in.

The three collections do not link to each other.

## 10. User tests

- **Who:** five people recruited by the project lead. At least two must not work in heritage.
- **When:** week 5, on the full POP index.
- **Where:** wherever the testers choose. Each session lasts about one hour.
- **Task:** photograph what interests you, then find out what it is.

What we record per tester:

- photos searched
- matches found
- wrong answers taken as right
- "not found" answers that were right or wrong
- whether they would use MIROIR again, and why

Testers agree in writing that we keep their photos until the end of the PoC, then delete them.

## 11. Plan

| Week | Work | Demo at end of week |
|---|---|---|
| 1 | Load the exports, download images, build the test sets and the scoring harness, check it against published scores, compare image models (rounds R0–R1 in [experiments.md](experiments.md)) | First results table |
| 2 | Compare re-rankers and phone-gap fixes (R2). Start comparing detectors, detail search and score fusion (R3). Search API on the current winners | Search API with scores for every option tried |
| 3 | Finish R3. Mobile website, object outlines (F4), drawn boxes, "not found" cut-off | Website on a phone |
| 4 | Full POP export loaded and indexed with the winning stack, detail search (F5) | Full index live |
| 5 | User tests, fixes between sessions | Test notes |
| 6 | Final scores on the sealed halves (R4), report, decision meeting | Decision |

## 12. Out of scope for the PoC

- Public launch, image rights clearance, outside access without a login
- A native app
- Video
- Feeding matches back into POP records
- Heritage-staff features: duplicate finding across collections, bulk matching

## 13. Risks

| Risk | Effect | What we do |
|---|---|---|
| Phone photos match black-and-white archive prints poorly | The phone-photo target is missed | Measure in week 2. Then try, in order: searching in greyscale, adding modern Wikimedia Commons photos as extra views, fine-tuning the model on Commons–Mémoire pairs |
| The full POP export arrives late | The user tests run on 101k records, and most tester photos find nothing | The project lead starts the export in week 1. If it isn't loaded by the end of week 4, testers work in places the 101k cover well (Toulouse, Besançon, Rouen) |
| The full export holds POP bases we have not mapped yet (Mérimée, Palissy…) | Week 4 overruns | Map only the fields search needs: title, place, domain, period, author, images |
| Images sent to closed AI models (captions, experiments) | Images not cleared for publication leave the Ministry | Send only images cleared for publication, Commons photos and test queries. Testers' photos stay on our machines unless a closed model wins its place and testers agree |
| We tune until the scores look good on the sets we tuned on | The decision meeting sees inflated numbers | Final scores come only from the sealed halves, opened once |
| Testers photograph people | Personal data | Delete test photos at the end of the PoC. Keep no faces in reports |

## 14. Decisions

| Question | Decision |
|---|---|
| Users | General public |
| Proof of value | Measured score and five user tests |
| Features | F1–F5, with F5 last |
| Infrastructure | No limit for the PoC. Build so it can move to the Ministry's cloud |
| Models | Chosen by experiment, not reputation. Open preferred, closed allowed when clearly better |
| Time and people | 6 weeks, 1–2 people, decision meeting at the end |
| One result is… | A building or object, with its photos grouped |
| Access | Login only |
| Rose Valland images | All indexed, each tagged by type |
| Coverage | All of POP before the user tests |
| No match | "Not found" plus "looks similar to" |
| Interface | Website built for phones first |
| Testers | Five, recruited by the project lead |
| Spec | Markdown in this repository, in English |
