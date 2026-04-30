# Project documentation

Short index of the documentation tree. Start with `architecture.md` for a
mental model of the pipeline, then dive into the specific area you need.

| Doc                                          | What you'll find                                                            |
|----------------------------------------------|-----------------------------------------------------------------------------|
| [architecture.md](architecture.md)           | Pipeline diagram, LangGraph flow, component boundaries, data flow           |
| [conventions.md](conventions.md)             | Service pattern, config, logging, naming, repo hygiene                      |
| [data-contracts.md](data-contracts.md)       | Stage-by-stage input/output schemas; final JSON shape                       |
| [decisions/](decisions/README.md)            | Architecture decision records (ADRs)                                        |
| [runbooks/](runbooks/README.md)              | Operational procedures: config, vLLM, model swap, taxonomy, thresholds, incidents |

Project-root docs:

- `README.md` — user-facing overview, setup, output schema.
- `CLAUDE.md` — guidance for AI assistants working on this codebase.
- `.env.example` — environment template; copy to `.env` and fill in.
