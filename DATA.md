# Data Contracts

## Released Artifact Layout

The main reproduction path assumes the OSF artifacts have been unpacked into the repository root:

```text
papers/
  clusters.json
  clusters_80.json
  {paper_id}/blog.md
  {paper_id}/ideas.json
  {paper_id}/refined_ideas.json
data/
  llm-papers.db
  coherence_dataset/
  availability_dataset/
models/
  coherence_model/
  availability_model/
```

`papers/clusters_80.json` is the recommended clustering artifact for paper reproduction. `papers/clusters.json` is the full clustering artifact.

`data/coherence_dataset/` and `data/availability_dataset/` contain pre-tokenized JSONL files and token mappings. Use these directly for retraining, or use the released `models/` checkpoints for the fast path.

Expected dataset files:

```text
data/coherence_dataset/coherence_train.jsonl
data/coherence_dataset/coherence_val.jsonl
data/coherence_dataset/token_mapping.json
data/availability_dataset/availability_train.jsonl
data/availability_dataset/availability_val.jsonl
data/availability_dataset/token_mapping.json
```

## Optional Paper List For Atomization

If rerunning atomization from PDFs, the input is a TSV file:

```text
paper_id<TAB>pdf_url
```

`paper_id` is used as the output directory name under `papers/`. Slashes are converted to `__` when loading the TSV so DBLP-style IDs can be used safely in paths.

## Atomization Outputs

Each processed paper writes:

```text
papers/{paper_id}/blog.md
papers/{paper_id}/ideas.json
papers/{paper_id}/refined_ideas.json
```

Clustering reads the `refined_ideas.json` files and writes a clusters JSON file.

## SQLite Metadata For Availability

The two-tower availability dataset uses author-paper metadata from SQLite. The released artifact is expected at `data/llm-papers.db`. Keep this schema:

```sql
CREATE TABLE papers (
    paper_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    keywords TEXT,
    pdf_url TEXT NOT NULL,
    conference TEXT,
    venue_year INTEGER NOT NULL,
    venue_track TEXT,
    openreview_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE authors (
    author_id TEXT PRIMARY KEY,
    display_name TEXT,
    email TEXT
);

CREATE TABLE paper_authors (
    paper_id TEXT NOT NULL,
    author_id TEXT NOT NULL,
    author_position INTEGER,
    PRIMARY KEY (paper_id, author_id)
);
```

The helper `crawlers/db/schema.py` initializes this schema. The dataset code uses:

- `papers.paper_id`
- `papers.venue_year`
- `authors.author_id`
- `authors.display_name`
- `paper_authors.paper_id`
- `paper_authors.author_id`
- `paper_authors.author_position`

If a cluster file contains filesystem-safe IDs such as `conf__nips__Paper19`, the availability loader also tries `conf/nips/Paper19` when looking up authors.

## Token Mapping

Dataset generation creates `token_mapping.json`, mapping cluster IDs to token IDs. Reuse the same mapping for coherence and availability datasets that will be trained and sampled together.

Full dataset rebuilds should use `data/vocab.json` or an equivalent token vocabulary with enough entries for all clusters.
