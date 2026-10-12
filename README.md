# MIROIR

Google Lens for French heritage: take a photo or describe what you see, and MIROIR finds the matching record in [POP](https://pop.culture.gouv.fr/).

This repository holds the proof of concept.

- [Functional spec](docs/functional-spec.md): who it is for, what it does, how we judge it
- [Technical spec](docs/technical-spec.md): data, models, indexes, API, evaluation

Data and images are not stored here. `MIROIR_EXPORTS_DIR` points at the POP exports; `MIROIR_DATA_DIR` is where builds go (see `.env.example`).

```sh
uv sync
make check
uv run miroir records build
```

Working on the code: read [CLAUDE.md](CLAUDE.md) and [CODING_STANDARDS.md](CODING_STANDARDS.md).
