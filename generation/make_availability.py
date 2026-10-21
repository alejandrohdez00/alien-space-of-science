"""Stream the original sampling policy into compact records and JSONL exports."""
import argparse
import array
import hashlib
import itertools
import json
import math
import random
import shutil
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from generation.availability_data import ROW, Corpus


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def profiles_from_sources(clusters, database, mapping):
    paper_atoms = defaultdict(set)
    for row in json.loads(Path(clusters).read_text())['atom_assignments']:
        if row['cluster'] != -1:
            paper_atoms[row['paper_id']].add(row['cluster'])
    conn = sqlite3.connect('file:' + str(Path(database).resolve()) + '?mode=ro', uri=True)
    authors = dict(conn.execute('SELECT author_id, display_name FROM authors'))
    paper_authors = defaultdict(list)
    for pid, aid in conn.execute('SELECT paper_id, author_id FROM paper_authors'):
        paper_authors[pid].append(aid)
    conn.close()
    aliases = {p.replace('/', '__'): p for p in paper_authors}
    result = {}
    for pid, atoms in paper_atoms.items():
        canonical = pid if pid in paper_authors else aliases.get(pid)
        if canonical is None:
            raise ValueError(f'Unresolved clustered paper: {pid}')
        for aid in paper_authors[canonical]:
            item = result.setdefault(aid, dict(atoms=set(), paper_count=0, display_name=authors[aid]))
            item['atoms'].update(atoms)
            item['paper_count'] += 1
    c2t = {int(k): int(v) for k, v in mapping['cluster_to_token_id'].items()}
    for item in result.values():
        item['atoms'] = sorted(item['atoms'])
        item['tokens'] = [c2t[a] for a in item['atoms']]
    return result, len(paper_atoms)


def samples(profiles, mapping, seed=42, cap=30, budget=20000, min_length=2, max_length=4):
    """Same RNG calls/order as the original generators, without retaining rows."""
    rng = random.Random(seed)
    c2t = {int(k): int(v) for k, v in mapping['cluster_to_token_id'].items()}
    for ai, aid in enumerate(sorted(profiles)):
        pool = profiles[aid]['atoms']
        pool_set = set(pool)
        if len(pool) < 2:
            continue
        lengths = list(range(min_length, min(max_length, len(pool)) + 1))
        if not lengths:
            continue
        seen = set()
        emitted, attempts = 0, 0
        per_length = defaultdict(int)
        if len(pool) <= cap:
            candidates = itertools.chain.from_iterable(itertools.combinations(pool, k) for k in lengths)
        else:
            weights = [math.comb(len(pool), k) for k in lengths]
            def draw():
                while attempts < budget * 8 and emitted < budget:
                    k = rng.choices(lengths, weights=weights, k=1)[0]
                    yield tuple(sorted(rng.sample(pool, k)))
            candidates = draw()
        for subset in candidates:
            if len(pool) > cap:
                attempts += 1
            # Encode a sorted set into a unique compact integer for dedup.
            key = len(subset)
            for atom in subset:
                key = (key << 16) | atom
            if len(pool) > cap:
                if key in seen:
                    continue
                seen.add(key)
            query = list(subset)
            rng.shuffle(query)
            complement = sorted(pool_set.difference(subset))
            rng.shuffle(complement)
            k = len(query)
            yield ai, [c2t[a] for a in query], [c2t[a] for a in complement], per_length[k]
            per_length[k] += 1
            emitted += 1


def build(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metadata.json').exists():
        raise FileExistsError(f'Completed dataset already exists: {out}')
    mapping = json.loads(Path(args.token_mapping).read_text())
    assert sorted(mapping['cluster_to_token_id'].values()) == list(range(mapping['n_clusters']))
    profiles, papers = profiles_from_sources(args.clusters, args.db, mapping)
    counts = dict(mapped_papers=papers, authors_with_atoms=len(profiles),
                  authors_min2=sum(len(p['atoms']) >= 2 for p in profiles.values()),
                  authors_min3=sum(len(p['atoms']) >= 3 for p in profiles.values()))
    print(counts, flush=True)
    shutil.copyfile(args.token_mapping, out / 'token_mapping.json')
    (out / 'profiles.json').write_text(json.dumps(profiles))
    (out / 'author_ids.json').write_text(json.dumps(sorted(profiles)))
    offset = n = 0
    started = time.monotonic()
    buffer = []
    complement_buffer = array.array('H')
    with (out / 'rows.bin').open('wb') as rows_f, (out / 'complements.bin').open('wb') as comp_f:
        for ai, q, comp, si in samples(profiles, mapping, args.seed, args.pool_cap, args.per_author_budget, args.min_length, args.max_length):
            buffer.append((ai, q + [mapping['n_clusters']] * (4 - len(q)), len(q), offset, len(comp), si))
            complement_buffer.extend(comp)
            offset += len(comp)
            n += 1
            if len(buffer) == 100000:
                np.asarray(buffer, dtype=ROW).tofile(rows_f)
                complement_buffer.tofile(comp_f)
                buffer.clear(); complement_buffer = array.array('H')
                if n % 1000000 == 0:
                    elapsed = time.monotonic() - started
                    print(f'Generated {n:,} records in {elapsed:.1f}s '
                          f'({n / elapsed:,.0f} records/s)', flush=True)
        np.asarray(buffer, dtype=ROW).tofile(rows_f)
        complement_buffer.tofile(comp_f)
    if n == 0:
        raise ValueError('No eligible author subsets found in the supplied corpus')
    rows = np.memmap(out / 'rows.bin', mode='r', dtype=ROW)
    # Reproduce the original random.Random(42).shuffle sample-level split.
    order = np.arange(n, dtype=np.int64)
    print(f'Generation complete; shuffling {n:,} records', flush=True)
    random.Random(args.seed).shuffle(order)
    cut = int(n * args.split)
    raw_train, raw_val = order[:cut], order[cut:]
    for split, indices in [('train', raw_train), ('val', raw_val)]:
        np.save(out / f'{split}_all.npy', indices)
        usable = indices[rows['count'][indices] > 0]
        np.save(out / f'{split}.npy', usable)
        counts[f'{split}_records'] = len(indices)
        counts[f'{split}_usable'] = len(usable)
        if split == 'train':
            np.save(out / 'training_authors.npy', np.unique(rows['author'][usable]))
            freq = np.zeros(mapping['n_clusters'], dtype=np.int64)
            for start in range(0, len(usable), 100000):
                q = rows['query'][usable[start:start + 100000]].ravel()
                freq += np.bincount(q[q < len(freq)], minlength=len(freq))
            np.save(out / 'atom_counts.npy', freq)
    metadata = dict(kind='two_tower', schema='subset_complement_v1',
                    seed=args.seed, split=args.split, pool_cap=args.pool_cap,
                    per_author_budget=args.per_author_budget, min_length=args.min_length,
                    max_length=args.max_length, min_atoms=2, counts=counts, total_records=n,
                    inputs={k: {'path': str(getattr(args, k)), 'sha256': sha256(getattr(args, k))}
                            for k in ['clusters', 'db', 'token_mapping']})
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    if args.export_jsonl:
        export(out)
    (out / 'READY').write_text('complete\n')


def export(directory):
    corpus = Corpus(directory)
    for split in ['train', 'val']:
        order = np.load(corpus.path / f'{split}_all.npy', mmap_mode='r')
        prefix = 'contrastive_availability'
        with (corpus.path / f'{prefix}_{split}.jsonl').open('w') as f:
            started = time.monotonic()
            for position, idx in enumerate(order, 1):
                r = corpus.rows[idx]; aid = corpus.author_ids[int(r['author'])]
                q = r['query'][:int(r['length'])].tolist()
                row = dict(author_id=aid, display_name=corpus.profiles[aid]['display_name'],
                           seq_len=len(q), sample_idx=int(r['sample']))
                start, count = int(r['offset']), int(r['count'])
                row.update(query_ids=q, author_ids=corpus.complements[start:start + count].tolist())
                f.write(json.dumps(row) + '\n')
                if position % 1000000 == 0:
                    elapsed = time.monotonic() - started
                    print(f'Exported {split}: {position:,}/{len(order):,} records '
                          f'({position / elapsed:,.0f} records/s)', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clusters', default='papers/clusters_80.json')
    p.add_argument('--db', default='data/llm-papers.db')
    p.add_argument('--token-mapping', default='data/coherence_dataset/token_mapping.json')
    p.add_argument('--output-dir', '--output', dest='output', default='data/availability_dataset_rebuilt')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split', type=float, default=0.9)
    p.add_argument('--pool-cap', type=int, default=30)
    p.add_argument('--per-author-budget', type=int, default=20000)
    p.add_argument('--min-length', type=int, default=2)
    p.add_argument('--max-length', type=int, default=4)
    p.add_argument('--export-jsonl', action='store_true')
    args = p.parse_args()
    if not 2 <= args.min_length <= args.max_length <= 4:
        p.error('compact query records support lengths 2 through 4')
    if not 0 < args.split < 1 or args.pool_cap < 2 or args.per_author_budget < 1:
        p.error('split must be between 0 and 1, pool cap >= 2, and budget positive')
    build(args)


if __name__ == '__main__':
    main()
