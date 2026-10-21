# Original pipeline tools

These are the tools used for our corpus. Adapt the prompts and source to your domain; retraining on another corpus requires rebuilding both datasets and both models.

## Atomize papers

Create a TSV with one `paper_id<TAB>pdf_url` per line. The pipeline writes `blog.md`, `ideas.json`, and `refined_ideas.json` for each paper and resumes existing outputs.

The original stages call Gemini through LiteLLM. Set `GEMINI_API_KEY` in your environment (never commit it), then run:

```bash
python alien.py atomize papers.tsv --output-dir papers --max-papers 10
```

Remove `--max-papers` for the full corpus. LLM calls incur provider costs. The extraction/refinement model constants remain in `atomization/stages/`; prompts are in `atomization/prompts.py`. The initial stage sends PDF content to the model, so changing providers may require adapting that request format.

## Cluster and reassign noise

```bash
python alien.py cluster --papers-dir papers --output papers/clusters.json
python alien.py reassign-noise --clusters papers/clusters.json \
  --embeddings-path papers/clusters.npy --checkpoints 0.80 --output-dir papers/reassigned
```

Clustering uses BGE embeddings, UMAP, HDBSCAN, and Gemini for canonical cluster text. Reassignment uses the saved embeddings and original centroids, assigning the most confident noise atoms first. Keep the embeddings paired with the clustering run. The second command writes `papers/reassigned/clusters_0.80.json`.

## Rebuild training data

For availability, supply the SQLite paper/author metadata described in [DATA.md](DATA.md); the PDF TSV alone does not supply author identities. These commands create new output directories:

```bash
python alien.py make-datasets --clusters papers/reassigned/clusters_0.80.json \
  --vocab data/vocab.json --split 0.9 --output-dir data/coherence_dataset_rebuilt
python alien.py make-availability --clusters papers/reassigned/clusters_0.80.json \
  --db data/llm-papers.db \
  --token-mapping data/coherence_dataset_rebuilt/token_mapping.json \
  --output-dir data/availability_dataset_rebuilt
```

Availability preparation streams compact records instead of retaining all examples in memory. Add `--export-jsonl` if you also want raw JSONL. It preserves the original subset/complement sampling policy and drops empty complements through its training/validation indices. Paper IDs must resolve to the supplied database; author display names are not identity keys.

Use the training commands from the README with these rebuilt paths, then pass both retrained model directories to `sample`. Models from a different atom catalogue cannot be mixed with these datasets.
