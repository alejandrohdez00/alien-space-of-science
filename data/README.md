# `data/`

This directory is populated from the OSF artifacts and is ignored by git except for this note.

Expected paths:

| Path | Purpose |
|---|---|
| `data/llm-papers.db` | SQLite paper/author metadata for availability datasets. |
| `data/vocab.json` | Token vocabulary used when rebuilding datasets from cluster metadata. |
| `data/coherence_dataset/` | Released coherence train/validation JSONL files and `token_mapping.json`. |
| `data/availability_dataset/` | Released two-tower availability train/validation JSONL files and `token_mapping.json`. |

See `DATA.md` for schemas and contracts.
