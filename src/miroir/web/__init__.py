"""The prototype website: photo or words in, heritage records out (functional spec §5, §6).

- `bundle`: builds an index bundle on the Mac (vectors, records, type guess, cut-offs).
- `catalog`, `facets`, `routing`, `ranking`, `engine`: the search itself, in NumPy. The same
  code runs in the Modal GPU container (with real models) and locally (with fake ones).
- `app`, `auth`: the FastAPI app, its login and its rate limit. `static/` holds the DSFR page.
- `modal_app`: the deployed Modal app. Model code stays in `miroir.remote.search`.
"""
