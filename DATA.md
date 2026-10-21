# Released data and models

Download and extract archives from the repository root. Download instructions are in [README.md](README.md).

| Download | Extracted contents | Purpose |
|---|---|---|
| `papers.tar.gz` | `papers/{paper_id}/blog.md`, `ideas.json`, `refined_ideas.json`; `clusters.json`, `clusters_80.json` | All 16,068 paper summaries and extracted atoms; original clustering and 80%-coverage assignments |
| `data.tar.gz` | `data/llm-papers.db`, `data/coherence_dataset/`, `data/vocab.json` | Paper/author metadata, coherence training splits, original token vocabulary |
| `availability-compact.tar.gz` | `data/availability_dataset/` | Availability training data |
| `contrastive_availability_{train,val}.jsonl.gz` | Standalone compressed JSONL | Optional exchange-format versions of the availability examples |
| `availability_metadata.json`, `availability_token_mapping.json` | Standalone JSON | Companion metadata/mapping for the JSONL downloads |
| `research-assets.tar.gz` | `research_assets/fixed_proposals/{alien,random,claude,gemini}/` | Fixed atom combinations and proposal texts |

Hugging Face model files go in:

```text
models/coherence_model/
  model.pt
  token_mapping.json
models/availability_model/
  model.pt
  token_mapping.json
  author_embeddings.pt
  author_id_to_index.json
```

The inference files total approximately 358 MB. The availability model includes a 39,791 × 512 author bank. Keep each checkpoint with its token mapping and the availability author bank.

## Dataset counts

The database contains 16,068 papers; 16,020 have mapped atoms. Coherence training uses the 15,203 papers with at least two atoms, each with ten serialized variants: 136,827 training and 15,203 validation records.

Availability starts with 46,329 author profiles with atoms, of which 44,809 have at least two atoms. Generation produced 47,887,640 records: 43,098,876 training and 4,788,764 validation records. Removing empty complements leaves **43,077,647 training and 4,786,404 validation records**. The training split includes 39,791 authors, whose full repertoires form the inference bank. Splitting is by example, not by author.

Fixed proposals can be downloaded separately:

```bash
hf download alejandrohdez/alien-data research-assets.tar.gz --repo-type dataset --local-dir downloads
tar -xzf downloads/research-assets.tar.gz
```

Each method folder contains Markdown proposal texts and a `combinations.json` index linking `cluster_ids` to `proposal_file`: Alien (299), Random (300), Claude Opus 4.7 (300), and Gemini 3.1 Pro (300).

## Formats

`clusters_80.json` contains `clusters` keyed by atom ID and `atom_assignments` containing `paper_id`, `text`, and `cluster`. Cluster `-1` means unassigned noise. Reconstruction uses the canonical atom text in each cluster's `name` field.

Coherence JSONL contains `input_ids` with BOS/atom/EOS tokens. Availability JSONL contains `author_id`, `display_name`, `query_ids`, `author_ids`, `seq_len`, and `sample_idx`; here `author_ids` is the complement's **token sequence**, not researcher identifiers. Query and complement are disjoint. Raw JSONL retains empty-complement rows; the compact training indices exclude them.

The compact availability directory contains:

- `rows.bin` and `complements.bin`: memory-mapped query/complement records.
- `train.npy` and `val.npy`: indices of usable records in each split.
- `profiles.json`, `author_ids.json`, and `training_authors.npy`: full profiles, their explicit row order, and training-bank membership.
- `token_mapping.json`, `metadata.json`, `atom_counts.npy`, and `READY`.

The binary layout is defined in `generation/availability_data.py`. Do not reorder profiles or infer row order by sorting the public IDs: `author_ids.json` preserves the original order after contact identifiers were replaced.

The SQLite tables are `papers`, `authors`, and `paper_authors`. Original paper identifiers and author links are retained. Email fields are omitted, and email-based researcher identifiers are consistently replaced across the database, profiles, training exports, and model-bank mapping. These replacements do not change atom sequences, split membership, numeric training indices, or embedding rows.

## Scores and provenance

Coherence is negative mean atom-token NLL over all permutations (six for triplets), excluding EOS. Availability is the median of the top ten author cosine similarities, averaging the fifth and sixth values. Unavailability is its negative. Fusion uses population-standardized scores over the chosen candidate pool.

CPU and CUDA floating-point results need not be identical.

Each upload directory contains a `manifest.json` with file sizes and SHA-256 checksums. Original input hashes are retained in training metadata; exported files have separate checksums because contact identifiers were replaced. Code, model weights, and the authors' contributed data are released under MIT. Original paper content and other third-party material retain their source terms; see the release cards for attribution.
