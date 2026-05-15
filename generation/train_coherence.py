"""Train a causal Transformer on pre-tokenized atom-set datasets.

Usage:
    python alien.py train-coherence \
        --train datasets/coherence/coherence_train.jsonl \
        --val datasets/coherence/coherence_val.jsonl \
        --token-mapping datasets/coherence/token_mapping.json \
        --output-dir models/coherence \
        --epochs 15
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

from generation.model import CausalTransformerLM
from generation.data import PreTokenizedDataset, collate_fn, load_token_mapping


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Compute average per-token loss on a dataset.

    Returns:
        Average cross-entropy loss per valid token
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask=attention_mask)

        # Manual shift for causal LM: predict next token
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='sum',
        )
        valid_tokens = (shift_labels != -100).sum().item()

        total_loss += loss.item()
        total_tokens += valid_tokens

    model.train()
    return total_loss / total_tokens if total_tokens > 0 else float('inf')


def train(
    train_path: str,
    val_path: str,
    token_mapping_path: str,
    output_dir: str,
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
    dropout: float = 0.1,
    num_workers: int = 0,
    compile_model: bool = False,
):
    """Train Transformer on pre-tokenized datasets."""
    # Load token mapping to get vocab size
    token_mapping = load_token_mapping(token_mapping_path)
    token_mapping, id_remap = token_mapping.compact()
    vocab_size = token_mapping.compute_vocab_size()

    sys.stderr.write(f"\n{'='*60}\n")
    sys.stderr.write("Transformer Training on Pre-Tokenized Super-Atoms\n")
    sys.stderr.write(f"{'='*60}\n")
    sys.stderr.write(f"Train file: {train_path}\n")
    sys.stderr.write(f"Val file: {val_path}\n")
    sys.stderr.write(f"Token mapping: {token_mapping_path}\n")
    sys.stderr.write(f"Output dir: {output_dir}\n")
    sys.stderr.write(f"Vocab size: {vocab_size} (for {token_mapping.n_clusters} clusters)\n")
    if id_remap:
        sys.stderr.write(f"ID remap: {id_remap}\n")
    sys.stderr.write(f"{'='*60}\n\n")

    # Load datasets
    train_dataset = PreTokenizedDataset(train_path, id_remap=id_remap)
    val_dataset = PreTokenizedDataset(val_path, id_remap=id_remap)

    pad_token_id = token_mapping.eos_token_id
    collate = partial(collate_fn, pad_token_id=pad_token_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Create model
    model = CausalTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    sys.stderr.write(f"Created model with {n_params:,} parameters\n")
    sys.stderr.write(
        f"  vocab_size={vocab_size}, d_model={d_model}, "
        f"num_layers={num_layers}, nhead={nhead}\n"
    )

    model.to(device)
    sys.stderr.write(f"Model on device: {device}\n")

    if compile_model:
        sys.stderr.write("Compiling model with torch.compile()...\n")
        model = torch.compile(model)

    # Create output directory and save token mapping
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "token_mapping.json", 'w', encoding='utf-8') as f:
        json.dump(token_mapping.raw, f, indent=2)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # wandb logging (optional)
    use_wandb = os.environ.get("WANDB_PROJECT") is not None
    if use_wandb:
        wandb.init(
            project=os.environ["WANDB_PROJECT"],
            name=run_name,
            config={
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
                "dropout": dropout,
                "compile": compile_model,
            },
        )

    # Training loop
    sys.stderr.write("\nStarting training...\n")
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(input_ids, attention_mask=attention_mask)

                # Manual shift for causal LM
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            # Track epoch loss (unscaled)
            valid_tokens = (shift_labels != -100).sum().item()
            epoch_loss += loss.item() * gradient_accumulation_steps * valid_tokens
            epoch_tokens += valid_tokens

            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if use_wandb and global_step % 100 == 0:
                    wandb.log({
                        "train/loss": loss.item() * gradient_accumulation_steps,
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/step": global_step,
                    })

        # Handle leftover gradients
        if len(train_loader) % gradient_accumulation_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        avg_train_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else float('inf')

        # Validation
        val_loss = evaluate(model, val_loader, device)

        sys.stderr.write(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}\n"
        )

        if use_wandb:
            wandb.log({
                "eval/loss": val_loss,
                "train/epoch_loss": avg_train_loss,
                "epoch": epoch,
            })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Unwrap compiled model if needed
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            raw_model.save(output_dir)
            sys.stderr.write(f"  Saved best model (val_loss={val_loss:.4f})\n")

    sys.stderr.write(f"\nTraining complete. Best val loss: {best_val_loss:.4f}\n")
    sys.stderr.write(f"Model saved to {output_dir}\n")

    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Train Transformer on pre-tokenized super-atom datasets"
    )

    # Required arguments
    parser.add_argument(
        "--train",
        type=str,
        required=True,
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--val",
        type=str,
        required=True,
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--token-mapping",
        type=str,
        required=True,
        help="Path to token_mapping.json",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for model and checkpoints",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Weights & Biases run name (optional)",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Per-device batch size (default: 32)",
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
        "--d-model",
        type=int,
        default=768,
        help="Embedding dimension (default: 768)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=6,
        help="Number of transformer layers (default: 6)",
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
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with torch.compile() for faster training",
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
        dropout=args.dropout,
        num_workers=args.num_workers,
        compile_model=args.compile,
    )


if __name__ == "__main__":
    main()
