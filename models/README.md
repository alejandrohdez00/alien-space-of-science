# `models/`

This directory is populated from the OSF model artifact or by local training. It is ignored by git except for this note.

Expected released checkpoints:

| Path | Contents |
|---|---|
| `models/coherence_model/` | Coherence Transformer checkpoint: `model.pt`, `token_mapping.json`. |
| `models/availability_model/` | Two-tower availability checkpoint: `model.pt`, `author_embeddings.pt`, `author_id_to_index.json`, `token_mapping.json`. |

Suggested local retraining outputs:

| Path | Created by |
|---|---|
| `models/coherence_model_retrained/` | `python alien.py train-coherence ...` |
| `models/availability_model_retrained/` | `python alien.py train-availability ...` |
