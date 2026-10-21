# Models

Download the Hugging Face model repository into this directory, as shown in the main README.

- `coherence_model/`: `model.pt`, `token_mapping.json`.
- `availability_model/`: `model.pt`, `token_mapping.json`, `author_embeddings.pt`, `author_id_to_index.json`, and training provenance.

Keep each model with its token mapping and the availability model with its author bank and ID-to-row mapping.

Retraining defaults to `availability_model_retrained/`. Use `model.pt` and the accompanying bank for inference; `last.pt` stores training resume state.
