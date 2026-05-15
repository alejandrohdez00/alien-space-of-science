"""Train the two-tower availability model.

Uses a symmetric in-batch softmax loss to train an author-idea compatibility
function s(a, T). The author tower encodes an
author's atom repertoire; the set tower encodes a candidate atom set.

Consumes precomputed pairs in the ``subset_complement_v1`` schema: each record
carries a positive subset ``query_ids`` and the author's repertoire complement
``author_ids`` (sample-level holdout baked in at dataset generation time).

Usage:
    python alien.py train-availability \
        --train datasets/availability/availability_train.jsonl \
        --val   datasets/availability/availability_val.jsonl \
        --token-mapping datasets/availability/token_mapping.json \
        --output-dir models/availability \
        --epochs 30
"""

import argparse
import json
import math
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader

from generation.two_tower_model import TwoTowerAvailabilityModel
from generation.data import (
    TwoTowerDataset,
    two_tower_collate_fn,
    load_token_mapping,
)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Compute loss, in-batch recall@1, and in-batch recall@5 on a dataset."""
    model.eval()
    total_loss = 0.0
    total_correct_r1 = 0
    total_correct_r5 = 0
    total_samples = 0

    for batch in dataloader:
        author_ids = batch["author_input_ids"].to(device, non_blocking=True)
        author_mask = batch["author_attention_mask"].to(device, non_blocking=True)
        set_ids = batch["set_input_ids"].to(device, non_blocking=True)
        set_mask = batch["set_attention_mask"].to(device, non_blocking=True)
        aidx = batch["author_idx"].to(device, non_blocking=True)

        B = author_ids.size(0)
        same_author = aidx[:, None] == aidx[None, :]
        same_author.fill_diagonal_(False)

        logits = model(author_ids, author_mask, set_ids, set_mask)  # (B, B)
        logits = logits.masked_fill(same_author, float("-inf"))

        labels = torch.arange(B, dtype=torch.long, device=device)
        loss_a2s = F.cross_entropy(logits, labels, reduction="sum")
        loss_s2a = F.cross_entropy(logits.T, labels, reduction="sum")
        loss = (loss_a2s + loss_s2a) / 2

        total_loss += loss.item()

        # Recall@1: for each author, is the correct set the top-scored?
        preds = logits.argmax(dim=1)
        total_correct_r1 += (preds == labels).sum().item()

        # Recall@5
        if B >= 5:
            top5 = logits.topk(5, dim=1).indices
            total_correct_r5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        else:
            total_correct_r5 += B  # trivially correct if batch < 5

        total_samples += B

    model.train()
    avg_loss = total_loss / total_samples if total_samples > 0 else float("inf")
    recall_1 = total_correct_r1 / total_samples if total_samples > 0 else 0.0
    recall_5 = total_correct_r5 / total_samples if total_samples > 0 else 0.0
    return avg_loss, recall_1, recall_5


@torch.no_grad()
def evaluate_full_pool_retrieval(
    model,
    val_dataset,
    author_embeddings,
    author_ids_list,
    device,
    batch_size=256,
    indices=None,
):
    """Evaluate author retrieval against the full author pool.

    For each val sample, compute set embedding, dot with all author embeddings,
    check if the correct author is in the top-k.

    Args:
        indices: Optional sample indices to evaluate. If None, evaluates
            all samples.

    Returns:
        Dict with recall@1, recall@5, recall@10, MRR
    """
    model.eval()

    author_id_to_idx = {aid: i for i, aid in enumerate(author_ids_list)}

    total = 0
    correct_r1 = 0
    correct_r5 = 0
    correct_r10 = 0
    rr_sum = 0.0

    eval_indices = indices if indices is not None else range(len(val_dataset))

    for start in range(0, len(eval_indices), batch_size):
        end = min(start + batch_size, len(eval_indices))
        batch_items = [val_dataset[eval_indices[i]] for i in range(start, end)]

        # Only keep items whose author is in the training pool
        valid_items = [
            item for item in batch_items if item["author_id"] in author_id_to_idx
        ]
        if not valid_items:
            continue

        # Pad set atoms
        set_seqs = [item["set_atoms"] for item in valid_items]
        max_len = max(len(s) for s in set_seqs)
        set_ids = torch.zeros(len(valid_items), max_len, dtype=torch.long, device=device)
        set_mask = torch.zeros(len(valid_items), max_len, dtype=torch.long, device=device)
        for i, seq in enumerate(set_seqs):
            set_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            set_mask[i, :len(seq)] = 1

        set_embeds = model.encode_sets(set_ids, set_mask)  # (B, d)

        # Score against all authors
        scores = set_embeds @ author_embeddings.T  # (B, N_authors)

        # Vectorized rank computation. "Rank under ties = best possible rank";
        # for L2-normalized float embeddings ties are astronomically unlikely.
        true_idx = torch.tensor(
            [author_id_to_idx[item["author_id"]] for item in valid_items],
            device=device, dtype=torch.long,
        )
        true_scores = scores.gather(1, true_idx.unsqueeze(1))  # (B, 1)
        ranks = (scores > true_scores).sum(dim=1) + 1           # (B,)

        correct_r1 += (ranks <= 1).sum().item()
        correct_r5 += (ranks <= 5).sum().item()
        correct_r10 += (ranks <= 10).sum().item()
        rr_sum += (1.0 / ranks.float()).sum().item()
        total += ranks.size(0)

    model.train()

    if total == 0:
        return {"recall@1": 0, "recall@5": 0, "recall@10": 0, "mrr": 0}

    return {
        "recall@1": correct_r1 / total,
        "recall@5": correct_r5 / total,
        "recall@10": correct_r10 / total,
        "mrr": rr_sum / total,
    }


@torch.no_grad()
def precompute_author_embeddings(model, dataset, device, batch_size=256):
    """Pre-compute author embeddings for the full training pool.

    In the author-centric regime there is no paper-level holdout, so the full
    atom repertoire is always used; no ``include_val`` knob is needed.

    Returns:
        (author_embeddings tensor (N, d_model), list of author_ids)
    """
    model.eval()
    author_atoms_dict = dataset.get_all_author_atoms()
    author_ids_list = sorted(author_atoms_dict.keys())

    all_embeds = []
    for start in range(0, len(author_ids_list), batch_size):
        end = min(start + batch_size, len(author_ids_list))
        batch_aids = author_ids_list[start:end]
        batch_atoms = [author_atoms_dict[aid] for aid in batch_aids]

        max_len = max(len(a) for a in batch_atoms)
        input_ids = torch.zeros(len(batch_atoms), max_len, dtype=torch.long, device=device)
        mask = torch.zeros(len(batch_atoms), max_len, dtype=torch.long, device=device)
        for i, atoms in enumerate(batch_atoms):
            input_ids[i, :len(atoms)] = torch.tensor(atoms, dtype=torch.long)
            mask[i, :len(atoms)] = 1

        embeds = model.encode_authors(input_ids, mask)
        all_embeds.append(embeds.cpu())

    model.train()
    return torch.cat(all_embeds, dim=0), author_ids_list


@torch.no_grad()
def evaluate_qualitative_retrieval(
    model, author_embeddings, author_ids_list, author_profiles,
    fixed_combos, device, top_k=5,
):
    """Evaluate author retrieval quality on fixed diagnostic combinations.

    Returns dict with overlap_rate and mean_overlap metrics.
    """
    model.eval()
    author_embeddings_dev = author_embeddings.to(device)

    combos_tensor = torch.tensor(fixed_combos, dtype=torch.long, device=device)
    mask = torch.ones_like(combos_tensor)
    set_embeds = model.encode_sets(combos_tensor, mask)
    scores = set_embeds @ author_embeddings_dev.T  # (N_combos, N_authors)

    total_pairs = 0
    pairs_with_overlap = 0
    total_overlap = 0

    for i, combo in enumerate(fixed_combos):
        combo_set = set(combo)
        topk_indices = scores[i].topk(top_k).indices.tolist()
        for idx in topk_indices:
            aid = author_ids_list[idx]
            author_atoms = set(author_profiles.get(aid, []))
            overlap = len(combo_set & author_atoms)
            if overlap > 0:
                pairs_with_overlap += 1
            total_overlap += overlap
            total_pairs += 1

    model.train()
    return {
        "overlap_rate": pairs_with_overlap / total_pairs if total_pairs > 0 else 0,
        "mean_overlap": total_overlap / total_pairs if total_pairs > 0 else 0,
    }


def train(
    token_mapping_path: str,
    output_dir: str,
    train_path: str,
    val_path: str,
    run_name: str = None,
    num_epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 500,
    d_model: int = 768,
    num_layers: int = 6,
    nhead: int = 12,
    max_seq_len: int = 128,
    max_logit_scale: float = 100.0,
    author_dropout_prob: float = 0.0,
    diagnostic_combos_path: str | None = None,
    num_workers: int = 0,
    compile_model: bool = False,
):
    """Train the two-tower availability model."""
    # Load token mapping to get vocab size
    token_mapping = load_token_mapping(token_mapping_path)
    token_mapping, id_remap = token_mapping.compact()
    vocab_size = token_mapping.compute_vocab_size()

    sys.stderr.write(f"\n{'='*60}\n")
    sys.stderr.write("Two-Tower Availability Training\n")
    sys.stderr.write(f"{'='*60}\n")
    sys.stderr.write(f"Train file: {train_path}\n")
    sys.stderr.write(f"Val file: {val_path}\n")
    sys.stderr.write(f"Token mapping: {token_mapping_path}\n")
    sys.stderr.write(f"Output dir: {output_dir}\n")
    sys.stderr.write(f"Vocab size: {vocab_size}\n")
    sys.stderr.write(f"Max logit scale: {max_logit_scale}\n")
    sys.stderr.write(f"Author dropout prob: {author_dropout_prob}\n")
    if id_remap:
        sys.stderr.write(f"ID remap: {id_remap}\n")
    sys.stderr.write(f"{'='*60}\n\n")

    # Load precomputed datasets. Train builds the full-repertoire map for
    # retrieval diagnostics; val does not need it (retrieval scores val queries
    # against the train author pool).
    train_dataset = TwoTowerDataset(
        train_path, id_remap=id_remap, build_author_atoms=True,
    )
    val_dataset = TwoTowerDataset(
        val_path, id_remap=id_remap, build_author_atoms=False,
    )

    # Author-side dropout applies to train only (val stays at 0.0 default).
    train_dataset.author_dropout_prob = author_dropout_prob

    pad_token_id = token_mapping.eos_token_id
    collate = partial(two_tower_collate_fn, pad_token_id=pad_token_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 4,
        })

    # Plain random-shuffle loader: in-batch same-author collisions are handled
    # at the loss level via a (B, B) mask (see training loop).
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate,
        **loader_kwargs,
    )

    # Create model
    model = TwoTowerAvailabilityModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        max_logit_scale=max_logit_scale,
    )
    n_params = sum(p.numel() for p in model.parameters())
    sys.stderr.write(f"Created model with {n_params:,} parameters\n")
    sys.stderr.write(
        f"  vocab_size={vocab_size}, d_model={d_model}, "
        f"num_layers={num_layers}, nhead={nhead}\n"
    )

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    model.to(device)
    sys.stderr.write(f"Model on device: {device}\n")

    if compile_model:
        sys.stderr.write("Compiling model with torch.compile()...\n")
        model = torch.compile(model)

    # Create output directory and save token mapping
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "token_mapping.json", "w", encoding="utf-8") as f:
        json.dump(token_mapping.raw, f, indent=2)

    # Optimizer and scheduler; fused=True is CUDA-only.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        fused=(device.type == "cuda"),
    )

    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # bf16 autocast keeps fp32 dynamic range, so no gradient scaling is needed.

    # wandb logging (optional)
    use_wandb = os.environ.get("WANDB_PROJECT") is not None
    if use_wandb:
        wandb.init(
            project=os.environ["WANDB_PROJECT"],
            name=run_name,
            config={
                "model_type": "two_tower",
                "vocab_size": vocab_size,
                "d_model": d_model,
                "num_layers": num_layers,
                "nhead": nhead,
                "max_seq_len": max_seq_len,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "warmup_steps": warmup_steps,
                "num_epochs": num_epochs,
                "max_logit_scale": max_logit_scale,
                "author_dropout_prob": author_dropout_prob,
                "compile": compile_model,
            },
        )

    # Load diagnostic combinations for qualitative retrieval metric
    diagnostic_combos = None
    author_profiles_for_diag = None
    if diagnostic_combos_path and Path(diagnostic_combos_path).exists():
        with open(diagnostic_combos_path) as f:
            diagnostic_combos = json.load(f)
        # Build author profiles (full pre-augmentation atom sets) for overlap checking
        author_profiles_for_diag = {
            aid: sorted(atoms)
            for aid, atoms in train_dataset.get_all_author_atoms().items()
        }
        sys.stderr.write(
            f"Loaded {len(diagnostic_combos)} diagnostic combinations "
            f"from {diagnostic_combos_path}\n"
        )

    # Training loop
    sys.stderr.write("\nStarting training...\n")
    best_pool_mrr = 0.0
    global_step = 0
    nan_skip_count = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        pending_grads = False

        for batch_idx, batch in enumerate(train_loader):
            author_ids = batch["author_input_ids"].to(device, non_blocking=True)
            author_mask = batch["author_attention_mask"].to(device, non_blocking=True)
            set_ids = batch["set_input_ids"].to(device, non_blocking=True)
            set_mask = batch["set_attention_mask"].to(device, non_blocking=True)
            aidx = batch["author_idx"].to(device, non_blocking=True)

            B = author_ids.size(0)
            # Same-author off-diagonal entries are legitimate positives. Mask
            # them to -inf so they contribute weight 0 in the softmax; the mask
            # is symmetric, so it covers both logits and logits.T.
            same_author = aidx[:, None] == aidx[None, :]
            same_author.fill_diagonal_(False)

            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
            ):
                logits = model(author_ids, author_mask, set_ids, set_mask)  # (B, B)
                logits = logits.masked_fill(same_author, float("-inf"))

                labels = torch.arange(B, dtype=torch.long, device=device)
                loss_a2s = F.cross_entropy(logits, labels)
                loss_s2a = F.cross_entropy(logits.T, labels)
                loss = (loss_a2s + loss_s2a) / 2
                loss = loss / gradient_accumulation_steps

            # Loss-side NaN guard: without GradScaler's implicit skip, a NaN/Inf
            # loss here would propagate through backward and corrupt weights on
            # the next optimizer.step(). Drop the batch and continue.
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                pending_grads = False
                nan_skip_count += 1
                continue

            loss.backward()
            pending_grads = True

            # Track metrics
            epoch_loss += loss.item() * gradient_accumulation_steps * B
            preds = logits.detach().argmax(dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_samples += B

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                # Gradient-side NaN guard: clip_grad_norm returns NaN if any
                # per-param gradient contained NaN/Inf. Skip the step; the
                # optimizer state stays clean and we don't corrupt weights.
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    pending_grads = False
                    nan_skip_count += 1
                    continue
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                pending_grads = False
                global_step += 1

                if use_wandb and global_step % 100 == 0:
                    raw_model = (
                        model._orig_mod if hasattr(model, "_orig_mod") else model
                    )
                    wandb.log(
                        {
                            "train/loss": loss.item() * gradient_accumulation_steps,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/logit_scale": raw_model.logit_scale.item(),
                            "train/grad_norm": grad_norm.item(),
                            "train/nan_skips": nan_skip_count,
                            "train/step": global_step,
                        }
                    )

        # Handle leftover gradients only when the loop actually left some
        # (gated on pending_grads to avoid a spurious no-op step).
        if pending_grads:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            if torch.isfinite(grad_norm):
                optimizer.step()
                scheduler.step()
                global_step += 1
            else:
                nan_skip_count += 1
            optimizer.zero_grad(set_to_none=True)
            pending_grads = False

        avg_train_loss = (
            epoch_loss / epoch_samples if epoch_samples > 0 else float("inf")
        )
        train_recall = epoch_correct / epoch_samples if epoch_samples > 0 else 0.0

        # Validation: in-batch metrics
        val_loss, val_r1, val_r5 = evaluate(model, val_loader, device)

        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        logit_scale = raw_model.logit_scale.item()

        sys.stderr.write(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Train R@1: {train_recall:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val R@1: {val_r1:.4f} | "
            f"Val R@5: {val_r5:.4f} | "
            f"Scale: {logit_scale:.2f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}"
            + (f" | NaN skips: {nan_skip_count}" if nan_skip_count > 0 else "")
            + "\n"
        )

        # Full-pool author retrieval (per-epoch diagnostic)
        author_embeds, author_ids_list = precompute_author_embeddings(
            raw_model, train_dataset, device
        )
        author_embeds_device = author_embeds.to(device)
        retrieval = evaluate_full_pool_retrieval(
            raw_model,
            val_dataset,
            author_embeds_device,
            author_ids_list,
            device,
        )
        sys.stderr.write(
            f"  Full-pool retrieval: "
            f"R@1={retrieval['recall@1']:.4f} "
            f"R@5={retrieval['recall@5']:.4f} "
            f"R@10={retrieval['recall@10']:.4f} "
            f"MRR={retrieval['mrr']:.4f}\n"
        )


        # Qualitative retrieval diagnostic (every 10 epochs)
        qual_retrieval = None
        if diagnostic_combos is not None and epoch % 10 == 0:
            qual_retrieval = evaluate_qualitative_retrieval(
                raw_model, author_embeds, author_ids_list,
                author_profiles_for_diag, diagnostic_combos, device,
            )
            sys.stderr.write(
                f"  Qualitative retrieval: "
                f"overlap_rate={qual_retrieval['overlap_rate']:.4f} "
                f"mean_overlap={qual_retrieval['mean_overlap']:.4f}\n"
            )

        if use_wandb:
            log_dict = {
                "eval/loss": val_loss,
                "eval/recall@1": val_r1,
                "eval/recall@5": val_r5,
                "eval/logit_scale": logit_scale,
                "eval/pool_recall@1": retrieval["recall@1"],
                "eval/pool_recall@5": retrieval["recall@5"],
                "eval/pool_recall@10": retrieval["recall@10"],
                "eval/pool_mrr": retrieval["mrr"],
                "train/epoch_loss": avg_train_loss,
                "train/epoch_recall@1": train_recall,
                "epoch": epoch,
            }
            if qual_retrieval:
                log_dict.update({
                    "eval/retrieval_overlap_rate": qual_retrieval["overlap_rate"],
                    "eval/retrieval_mean_overlap": qual_retrieval["mean_overlap"],
                })
            wandb.log(log_dict)

        # Save best model by full-pool MRR, a ranking-quality metric.
        pool_mrr = retrieval["mrr"]
        if pool_mrr > best_pool_mrr:
            best_pool_mrr = pool_mrr
            raw_model.save(output_dir)
            sys.stderr.write(f"  Saved best model (pool_mrr={pool_mrr:.4f})\n")

    # Post-training: serialize author embeddings using full author repertoires
    sys.stderr.write("\nSerializing author embeddings...\n")
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    # Reload best checkpoint for embedding serialization
    raw_model = TwoTowerAvailabilityModel.load(output_dir, device=str(device))
    author_embeds, author_ids_list = precompute_author_embeddings(
        raw_model, train_dataset, device
    )
    torch.save(author_embeds, Path(output_dir) / "author_embeddings.pt")
    with open(Path(output_dir) / "author_id_to_index.json", "w", encoding="utf-8") as f:
        json.dump({aid: i for i, aid in enumerate(author_ids_list)}, f, indent=2)
    sys.stderr.write(
        f"  Saved {len(author_ids_list)} author embeddings to {output_dir}\n"
    )

    sys.stderr.write(f"\nTraining complete. Best pool MRR: {best_pool_mrr:.4f}\n")
    sys.stderr.write(f"Model saved to {output_dir}\n")

    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Train the two-tower availability model"
    )

    # Data arguments: precomputed subset_complement_v1 JSONLs
    parser.add_argument(
        "--train", type=str, required=True,
        help="Pre-split training JSONL (subset_complement_v1 schema)",
    )
    parser.add_argument(
        "--val", type=str, required=True,
        help="Pre-split validation JSONL (subset_complement_v1 schema)",
    )

    # Required arguments
    parser.add_argument(
        "--token-mapping", type=str, required=True, help="Path to token_mapping.json"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for model and checkpoints",
    )
    parser.add_argument(
        "--run-name", type=str, default=None, help="Weights & Biases run name"
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs (default: 15)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of warmup steps (default: 500)",
    )

    # Model architecture
    parser.add_argument(
        "--d-model", type=int, default=768, help="Embedding dimension (default: 768)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers in set tower (default: 6)",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=12,
        help="Number of attention heads (default: 12)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with torch.compile()",
    )

    # Two-tower specific
    parser.add_argument(
        "--max-logit-scale",
        type=float,
        default=100.0,
        help="Maximum logit scale (1/temperature) clamp (default: 100.0)",
    )
    parser.add_argument(
        "--author-dropout-prob",
        type=float,
        default=0.0,
        help="Per-atom dropout on author-tower input (default: 0.0, off). "
        "Opt-in regularization; no anti-leakage role under subset_complement_v1.",
    )
    parser.add_argument(
        "--diagnostic-combos",
        type=str,
        default=None,
        help="Path to JSON with fixed diagnostic combinations for retrieval metric",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers (default: 0 for portability)",
    )

    args = parser.parse_args()

    train(
        train_path=args.train,
        val_path=args.val,
        token_mapping_path=args.token_mapping,
        output_dir=args.output_dir,
        run_name=args.run_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        d_model=args.d_model,
        num_layers=args.num_layers,
        nhead=args.nhead,
        max_seq_len=args.max_seq_len,
        max_logit_scale=args.max_logit_scale,
        author_dropout_prob=args.author_dropout_prob,
        diagnostic_combos_path=args.diagnostic_combos,
        num_workers=args.num_workers,
        compile_model=args.compile,
    )


if __name__ == "__main__":
    main()
