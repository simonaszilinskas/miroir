"""The prototype website: photo or words in, heritage records out (functional spec §5, §6).

- `bundle`: builds an index bundle on the Mac (vectors, records, type guess, cut-offs).
- `catalog`, `facets`, `routing`, `ranking`, `engine`: the search itself, in NumPy. The same
  code runs on the demo machine's GPU (with real models) and locally (with fake ones).
- `app`, `auth`: the FastAPI app, its login and its rate limit. The page is built from `web-ui/` into `ui/` (`pagebuild`).
- `ops`: what `miroir web` runs on the Mac. The demo machine serves the site
  (`miroir.remote.site`); model code stays in `miroir.remote.search`.
"""
