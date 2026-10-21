"""Train the dual encoder on the released compact subset/complement dataset.

Single device: python alien.py train-availability --dataset data/availability_dataset
Multiple GPUs: torchrun --standalone --nproc_per_node=4 -m generation.train_availability
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from generation.two_tower_model import TwoTowerAvailabilityModel
from generation.token_mapping import TokenMapping
from generation.availability_data import Corpus, BatchDataset
from generation.attention import zero_key_bias


def reduce_sum(value):
    if dist.is_initialized():
        dist.all_reduce(value)


class Encoders(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return (self.model.encode_authors(batch['author_input_ids'], batch['author_attention_mask']),
                self.model.encode_sets(batch['set_input_ids'], batch['set_attention_mask']),
                self.model.logit_scale)


def global_clip_loss(author, query, scale, author_idx):
    """Local rows against ALL columns, with differentiable collective backward.

    Collective backward sums contributions from every rank; DDP subsequently
    averages parameter gradients. Together these give the global-batch mean.
    """
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if world > 1:
        all_a = torch.cat(dist_nn.all_gather(author.contiguous()))
        all_q = torch.cat(dist_nn.all_gather(query.contiguous()))
        # NCCL requires contiguous collective inputs, including strided IDs.
        author_idx = author_idx.contiguous()
        ids = [torch.empty_like(author_idx) for _ in range(world)]
        dist.all_gather(ids, author_idx)
        all_ids = torch.cat(ids)
    else:
        all_a, all_q, all_ids = author, query, author_idx
    labels = torch.arange(len(author), device=author.device) + rank * len(author)
    mask = author_idx[:, None] == all_ids[None, :]
    mask.scatter_(1, labels[:, None], False)
    a_logits = (scale * (author @ all_q.T)).masked_fill(mask, -torch.inf)
    q_logits = (scale * (query @ all_a.T)).masked_fill(mask, -torch.inf)
    loss = (F.cross_entropy(a_logits, labels) + F.cross_entropy(q_logits, labels)) / 2
    return loss, (a_logits.argmax(1) == labels).sum()


def local_clip(model, batch):
    logits = model(batch['author_input_ids'], batch['author_attention_mask'],
                   batch['set_input_ids'], batch['set_attention_mask'])
    same = batch['author_idx'][:, None] == batch['author_idx'][None, :]
    same.fill_diagonal_(False)
    logits = logits.masked_fill(same, -torch.inf)
    labels = torch.arange(len(logits), device=logits.device)
    return ((F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2,
            (logits.argmax(1) == labels).sum())


def save_atomic(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def loader(dataset, workers):
    return DataLoader(dataset, batch_size=None, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0)


@torch.no_grad()
def author_bank(raw, corpus, device):
    chunks = []
    for start in range(0, len(corpus.training_authors), 512):
        ids, mask = corpus.author_batch(corpus.training_authors[start:start + 512])
        chunks.append(raw.encode_authors(ids.to(device), mask.to(device)))
    return torch.cat(chunks)


@torch.no_grad()
def validate(raw, corpus, device, rank, world, global_batch, workers):
    raw.eval()
    batches = BatchDataset(corpus, corpus.val, global_batch)
    positions = list(range(len(batches)))
    selected = torch.utils.data.Subset(batches, positions[rank::world])
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    for b in loader(selected, workers):
        b = to_device(b, device)
        # Original validation is fp32; retain it for checkpoint comparison.
        loss, correct = local_clip(raw, b)
        n = len(b['author_idx'])
        totals += torch.stack([loss.double() * n, correct.double(), loss.new_tensor(n).double()])
    reduce_sum(totals)
    result = dict(loss=(totals[0] / totals[2]).item(), accuracy=(totals[1] / totals[2]).item())
    bank = author_bank(raw, corpus, device)
    lookup = torch.full((len(corpus.author_ids),), -1, dtype=torch.long, device=device)
    lookup[torch.as_tensor(corpus.training_authors.astype(np.int64), device=device)] = torch.arange(len(bank), device=device)
    indices = corpus.val
    indices = indices[rank::world]
    retrieval = torch.zeros(5, dtype=torch.float64, device=device)
    for start in range(0, len(indices), 256):
        b = to_device(corpus.batch(indices[start:start + 256]), device)
        true = lookup[b['author_idx']]; valid = true >= 0
        if not valid.any():
            continue
        q = raw.encode_sets(b['set_input_ids'][valid], b['set_attention_mask'][valid])
        scores = q @ bank.T
        positive = scores.gather(1, true[valid, None])
        ranks = (scores > positive).sum(1) + 1
        retrieval += torch.stack([(1. / ranks.double()).sum(), (ranks <= 1).sum(),
                                  (ranks <= 5).sum(), (ranks <= 10).sum(), valid.sum()])
    reduce_sum(retrieval)
    if retrieval[4] == 0:
        raise ValueError('No validation authors in training bank')
    result.update(mrr=(retrieval[0] / retrieval[4]).item(),
                  recall1=(retrieval[1] / retrieval[4]).item(),
                  recall5=(retrieval[2] / retrieval[4]).item(),
                  recall10=(retrieval[3] / retrieval[4]).item())
    if not all(math.isfinite(x) for x in result.values()):
        raise FloatingPointError(result)
    raw.train()
    return result


def run(args):
    cuda = args.device != "cpu" and torch.cuda.is_available()
    if args.device == "cuda" and not cuda:
        raise ValueError("CUDA was requested but is not available")
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    device = torch.device("cuda", torch.cuda.current_device()) if cuda else torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    corpus = Corpus(args.dataset)
    if corpus.meta.get("kind") != "two_tower":
        raise ValueError("Expected a compact two_tower availability dataset")
    if args.batch_size % world or args.batch_size < world:
        raise ValueError("Global batch size must be a positive multiple of the GPU count")
    if len(corpus.train) < args.batch_size or not len(corpus.val):
        raise ValueError("Need at least one full training batch and nonempty validation data; reduce --batch-size")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mapping, _ = TokenMapping(corpus.mapping).compact()
    raw = TwoTowerAvailabilityModel(
        vocab_size=mapping.compute_vocab_size(), d_model=args.d_model,
        nhead=args.heads, num_layers=args.layers, dim_feedforward=args.ffn,
        max_seq_len=128, dropout=args.dropout, max_logit_scale=30.0,
    ).to(device)
    zero_key_bias(raw)
    encoders = Encoders(raw)
    wrapped = torch.compile(encoders, dynamic=True) if args.compile else encoders
    model = (DDP(wrapped, device_ids=[device.index] if cuda else None, broadcast_buffers=False)
             if world > 1 else wrapped)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, fused=cuda,
    )
    steps_per_epoch = len(corpus.train) // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    def schedule(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(total_steps - args.warmup, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    signature = dict(
        dataset_sha256=hashlib.sha256((corpus.path / "metadata.json").read_bytes()).hexdigest(),
        model=raw.config, batch=args.batch_size, epochs=args.epochs, warmup=args.warmup,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        seed=args.seed, world=world, compile=args.compile, zero_key_bias=True,
    )
    checkpoint = out / "last.pt"
    start_epoch, best, step = 1, -1.0, 0
    if args.resume:
        # Resume files are locally generated optimizer state, not inference weights.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["signature"] != signature:
            raise ValueError("Resume configuration/dataset differs from the saved run")
        raw.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best, step = state["epoch"] + 1, state["best"], state["step"]
        restore_rng(state["rng"][rank])
        zero_key_bias(raw)
    else:
        if checkpoint.exists() or (out / "model.pt").exists():
            raise FileExistsError("Output already contains a run; use --resume or a new directory")
        torch.manual_seed(args.seed + rank)
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
    if rank == 0:
        (out / "token_mapping.json").write_text(json.dumps(mapping.raw, indent=2))
        (out / "run_config.json").write_text(json.dumps(signature, indent=2))

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.monotonic()
        order = np.random.default_rng(args.seed + epoch).permutation(len(corpus.train))
        indices = corpus.train[order]
        dataset = BatchDataset(corpus, indices, args.batch_size, rank, world, drop_last=True)
        model.train()
        totals = torch.zeros(2, dtype=torch.float64, device=device)
        for bi, batch in enumerate(loader(dataset, args.workers)):
            b = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
                author, query, scale = model(b)
                loss, _ = global_clip_loss(author, query, scale, b["author_idx"])
            finite = torch.isfinite(loss).to(torch.int32)
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Nonfinite loss; stopping before updating weights")
            loss.backward()
            zero_key_bias(raw, gradients=True)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            zero_key_bias(raw)
            scheduler.step()
            step += 1
            n = len(b["author_idx"])
            totals += torch.stack([loss.detach().double() * n, loss.new_tensor(n).double()])
            if rank == 0 and (bi + 1) % 500 == 0:
                print(f"Epoch {epoch}: batch {bi + 1}/{steps_per_epoch}, loss {loss.item():.4f}", flush=True)
        del dataset, indices, order
        reduce_sum(totals)
        metrics = validate(raw, corpus, device, rank, world, args.batch_size, args.workers)
        if metrics["mrr"] > best:
            best = metrics["mrr"]
            if rank == 0:
                raw.eval()
                save_atomic(dict(config=raw.config, state_dict=raw.state_dict()), out / "model.pt")
                save_atomic(author_bank(raw, corpus, device).cpu(), out / "author_embeddings.pt")
                ids = [corpus.author_ids[int(i)] for i in corpus.training_authors]
                (out / "author_id_to_index.json").write_text(json.dumps(dict(zip(ids, range(len(ids)))), indent=2))
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, rng_state())
        else:
            states[0] = rng_state()
        if rank == 0:
            save_atomic(dict(signature=signature, model=raw.state_dict(), optimizer=optimizer.state_dict(),
                             scheduler=scheduler.state_dict(), epoch=epoch, best=best, step=step, rng=states), checkpoint)
            record = dict(epoch=epoch, train_loss=(totals[0] / totals[1]).item(),
                          seconds=time.monotonic() - started, **metrics)
            with (out / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    if rank == 0:
        (out / "COMPLETE").write_text(json.dumps(dict(best_mrr=best, epochs=args.epochs, step=step)))
        print(f"Saved selected model and matching author bank in {out}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("data/availability_dataset"))
    p.add_argument("--output-dir", "--output", default="models/availability_model_retrained")
    p.add_argument("--batch-size", type=int, default=4096, help="Global batch across all devices")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--warmup", "--warmup-steps", type=int, default=2000)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--layers", "--num-layers", type=int, default=6)
    p.add_argument("--heads", "--nhead", type=int, default=8)
    p.add_argument("--ffn", type=int, default=3072)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.warmup < 0:
        p.error("batch size and epochs must be positive; warmup must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
