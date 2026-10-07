from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_obo(path: Path):
    parents: dict[str, set[str]] = collections.defaultdict(set)
    names: dict[str, str] = {}
    current: dict[str, object] = {}

    def commit():
        term = current.get("id")
        if term and not current.get("obsolete"):
            names[str(term)] = str(current.get("name", ""))
            parents[str(term)].update(current.get("parents", set()))

    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                commit(); current = {"parents": set()}
            elif line.startswith("["):
                commit(); current = {}
            elif current:
                if line.startswith("id: MONDO:"):
                    current["id"] = line[4:].strip()
                elif line.startswith("name: "):
                    current["name"] = line[6:].strip()
                elif line.startswith("is_a: MONDO:"):
                    current.setdefault("parents", set()).add(line[6:].split()[0])
                elif line == "is_obsolete: true":
                    current["obsolete"] = True
    commit()
    return parents, names


def mondo_id(raw: str) -> str | None:
    raw = raw.strip()
    if raw.startswith("MONDO:"):
        suffix = raw.split(":", 1)[1]
    elif raw.isdigit():
        suffix = raw
    else:
        return None
    return f"MONDO:{int(suffix):07d}"


def disease_ids(rows):
    result: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        disease = row["disease"]
        for value in row.get("ontology_id", "").split("|"):
            if normalized := mondo_id(value):
                result[disease].add(normalized)
        ids = row.get("kg_disease_ids", "").split("|")
        sources = row.get("kg_disease_sources", "").split("|")
        if "MONDO" in sources:
            for value in ids:
                if normalized := mondo_id(value):
                    result[disease].add(normalized)
    return result


def ancestors(term: str, parents: dict[str, set[str]], cache: dict[str, set[str]]):
    if term in cache:
        return cache[term]
    seen, stack = set(), list(parents.get(term, ()))
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node); stack.extend(parents.get(node, ()))
    cache[term] = seen
    return seen


def depth_from_root(term: str, parents: dict[str, set[str]], cache: dict[str, int], active=None):
    root = "MONDO:0000001"
    if term == root:
        return 0
    if term in cache:
        return cache[term]
    active = set() if active is None else active
    if term in active:
        return 10 ** 6
    active.add(term)
    candidates = [depth_from_root(parent, parents, cache, active) for parent in parents.get(term, ())]
    active.remove(term)
    value = 1 + min(candidates) if candidates else 10 ** 6
    cache[term] = value
    return value


def canonical_branch(terms, parents, ancestor_cache, depth_cache, branch_depth):
    candidates = set()
    for term in terms:
        for node in {term} | ancestors(term, parents, ancestor_cache):
            if depth_from_root(node, parents, depth_cache) == branch_depth:
                candidates.add(node)
    if candidates:
        return sorted(candidates)[0]
    return sorted(terms)[0] if terms else None


def allocate(groups, counts, test_fraction, valid_fraction, seed):
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda g: sum(counts[d] for d in g), reverse=True)
    total = sum(counts.values())
    targets = {"test": total * test_fraction, "valid": total * valid_fraction,
               "train": total * (1 - test_fraction - valid_fraction)}
    assigned = {key: [] for key in targets}
    loads = {key: 0 for key in targets}
    for group in shuffled:
        size = sum(counts[d] for d in group)
        split = min(targets, key=lambda key: loads[key] / targets[key])
        assigned[split].append(group); loads[split] += size
    return assigned, loads


def main():
    parser = argparse.ArgumentParser(description="Build ontology-aware disease-disjoint splits.")
    parser.add_argument("--input", type=Path, default=ROOT / "data/dataset_v2_ontology/drug_disease_drug_kg_mapped.csv")
    parser.add_argument("--ontology", type=Path, default=ROOT / "data/ontologies/mondo-2026-07-06.obo")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/dataset_v2_ontology_aware_ldo")
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--valid-fraction", type=float, default=0.10)
    parser.add_argument("--branch-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with args.input.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle); fields = list(reader.fieldnames or []); rows = list(reader)
    parents, names = parse_obo(args.ontology)
    diseases = sorted({row["disease"] for row in rows})
    ids = disease_ids(rows)
    ancestor_cache: dict[str, set[str]] = {}
    depth_cache: dict[str, int] = {}
    grouped: dict[str, set[str]] = collections.defaultdict(set)
    for disease in diseases:
        branch = canonical_branch(ids[disease], parents, ancestor_cache, depth_cache, args.branch_depth)
        key = branch if branch is not None else f"UNMAPPED:{disease}"
        grouped[key].add(disease)
    groups = list(grouped.values())
    counts = collections.Counter(row["disease"] for row in rows)
    assigned, loads = allocate(groups, counts, args.test_fraction, args.valid_fraction, args.seed)
    disease_split = {disease: split for split, split_groups in assigned.items() for group in split_groups for disease in group}
    out_rows = {split: [] for split in ("train", "valid", "test")}
    for row in rows:
        copied = dict(row); copied["split"] = disease_split[row["disease"]]
        out_rows[copied["split"]].append(copied)
    branch_splits = collections.defaultdict(set)
    for disease in diseases:
        branch = canonical_branch(ids[disease], parents, ancestor_cache, depth_cache, args.branch_depth)
        key = branch if branch is not None else f"UNMAPPED:{disease}"
        branch_splits[key].add(disease_split[disease])
    cross_branches = {key: value for key, value in branch_splits.items() if len(value) > 1}
    if cross_branches:
        raise RuntimeError(f"Canonical branch leakage: {list(cross_branches.items())[:5]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = args.output_dir / "drug_disease_drug_kg_mapped.csv"
    for path, selected in [(combined, sum((out_rows[x] for x in ("train", "valid", "test")), []))] + [
        (args.output_dir / f"{split}.csv", out_rows[split]) for split in ("train", "valid", "test")]:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(selected)
    group_rows = []
    branch_by_group = {frozenset(group): branch for branch, group in grouped.items()}
    for split, split_groups in assigned.items():
        for index, group in enumerate(split_groups, 1):
            group_rows.append({"split": split, "group": f"{split}_{index:04d}",
                               "canonical_branch": branch_by_group[frozenset(group)],
                               "canonical_branch_label": names.get(branch_by_group[frozenset(group)], "unmapped"),
                               "diseases": "|".join(sorted(group)),
                               "num_diseases": len(group),
                               "positive_triples": sum(counts[d] for d in group),
                               "mapped_diseases": sum(bool(ids[d]) for d in group)})
    with (args.output_dir / "ontology_groups.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=group_rows[0].keys()); writer.writeheader(); writer.writerows(group_rows)
    report = {
        "protocol": "ontology_aware_leave_disease_out",
        "grouping_rule": "Diseases sharing the same canonical MONDO ancestor at the configured root depth are indivisible.",
        "branch_depth": args.branch_depth,
        "unmapped_rule": "Diseases without a MONDO mapping are singleton groups.",
        "seed": args.seed, "source_sha256": sha256(args.input), "ontology_sha256": sha256(args.ontology),
        "counts": {s: {"positive_triples": len(out_rows[s]), "diseases": len({r['disease'] for r in out_rows[s]}),
                       "ontology_groups": len(assigned[s])} for s in ("train", "valid", "test")},
        "mapped_diseases": sum(bool(ids[d]) for d in diseases), "unmapped_diseases": sum(not ids[d] for d in diseases),
        "canonical_branches": len(groups), "cross_split_canonical_branches": len(cross_branches),
        "largest_group_diseases": max(map(len, groups)), "largest_group_triples": max(sum(counts[d] for d in g) for g in groups),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
