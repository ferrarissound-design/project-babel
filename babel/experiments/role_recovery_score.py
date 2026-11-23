"""Annotation-only scoring for the preregistered A–E role-recovery study."""
from __future__ import annotations

import math
from collections import Counter

ROLE_NAMES = ("subject", "predicate", "object")


def _pools(row, field):
    pools = row[field]
    if len(pools) != 3:
        raise ValueError(f"{field} must contain three role pools")
    return [set(map(int, pool)) for pool in pools]


def distinct_triple_count(pools):
    """Count injective role assignments without enumerating a large raw pool."""
    subject, predicate, obj = map(set, pools)
    return (len(subject) * len(predicate) * len(obj)
            - len(subject & predicate) * len(obj)
            - len(subject & obj) * len(predicate)
            - len(predicate & obj) * len(subject)
            + 2 * len(subject & predicate & obj))


def _contains(pools, triple):
    return len(set(triple)) == 3 and all(pos in pool for pos, pool in zip(triple, pools, strict=True))


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _sizes(values):
    """Nearest-rank p95; an empty eligible sample has no estimated size."""
    ordered = sorted(values)
    return {"total": sum(values), "mean": _ratio(sum(values), len(values)),
            "p95": ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None}


def score_predictions(rows, gold_by_key):
    """Score fixed predictions against same-triple annotations, never feeding gold back.

    The primary denominator includes every source-full row, even if no target
    triple exists. Candidate recall and pool sizes additionally require target
    gold. Candidate precision includes all source-full rows, including no-gold
    rows, so emitting more candidates cannot improve it by excluding failures.
    """
    counts = Counter({key: 0 for key in (
        "total", "eligible", "gold_eligible", "claimed", "correct", "wrong", "abstain",
        "high_confidence_wrong", "unsupported_claimed", "raw_retained", "budget_retained",
        "candidate_triples", "gold_candidate_triples", "raw_candidate_triples", "raw_gold_candidate_triples")})
    role_retained = {kind: Counter({role: 0 for role in ROLE_NAMES}) for kind in ("raw", "budget")}
    sizes = {kind: {role: [] for role in ROLE_NAMES} for kind in ("raw", "budget")}
    all_sizes = {kind: {role: [] for role in ROLE_NAMES} for kind in ("raw", "budget")}
    seen = set()
    for row in rows:
        key = (row["doc"], int(row["sent"]))
        if key in seen:
            raise ValueError(f"Duplicate prediction for {key}")
        seen.add(key)
        counts["total"] += 1
        selected = row["selected"]
        if selected is not None and len(selected) != 3:
            raise ValueError("selected must be a triple or None")
        if not row["source_full"]:
            counts["unsupported_claimed"] += selected is not None
            continue
        counts["eligible"] += 1
        gold = {tuple(map(int, triple)) for triple in gold_by_key.get(key, [])}
        if any(len(triple) != 3 or len(set(triple)) != 3 for triple in gold):
            raise ValueError(f"Invalid gold triple for {key}")
        pools = {"raw": _pools(row, "raw_candidates"), "budget": _pools(row, "candidates")}
        if any(len(pool) > 5 for pool in pools["budget"]):
            raise ValueError("Budget candidates exceed five positions per role")
        if any(not budget <= raw for raw, budget in zip(pools["raw"], pools["budget"], strict=True)):
            raise ValueError("Budget candidates must be a subset of raw candidates")
        if selected is None:
            counts["abstain"] += 1
        else:
            counts["claimed"] += 1
            correct = tuple(map(int, selected)) in gold
            counts["correct" if correct else "wrong"] += 1
            counts["high_confidence_wrong"] += not correct and row["confidence"] >= 0.8
        for kind, pool in pools.items():
            prefix = "raw_" if kind == "raw" else ""
            counts[prefix + "candidate_triples"] += distinct_triple_count(pool)
            counts[prefix + "gold_candidate_triples"] += sum(_contains(pool, triple) for triple in gold)
            for index, role in enumerate(ROLE_NAMES):
                all_sizes[kind][role].append(len(pool[index]))
        if gold:
            counts["gold_eligible"] += 1
            for kind, pool in pools.items():
                counts[kind + "_retained"] += any(_contains(pool, triple) for triple in gold)
                for index, role in enumerate(ROLE_NAMES):
                    role_retained[kind][role] += any(triple[index] in pool[index] for triple in gold)
                    sizes[kind][role].append(len(pool[index]))
    result = dict(counts)
    result.update(precision=_ratio(counts["correct"], counts["claimed"]),
                  wrong_rate=_ratio(counts["wrong"], counts["claimed"]),
                  abstain_rate=_ratio(counts["abstain"], counts["eligible"]),
                  accuracy=_ratio(counts["correct"], counts["eligible"]),
                  candidate_precision=_ratio(counts["gold_candidate_triples"], counts["candidate_triples"]),
                  raw_candidate_precision=_ratio(counts["raw_gold_candidate_triples"], counts["raw_candidate_triples"]))
    for kind in ("raw", "budget"):
        result[kind + "_recall"] = _ratio(counts[kind + "_retained"], counts["gold_eligible"])
        result[kind + "_role_retained"] = dict(role_retained[kind])
        result[kind + "_role_recall"] = {role: _ratio(value, counts["gold_eligible"])
                                         for role, value in role_retained[kind].items()}
        result[kind + "_pool_sizes"] = {role: _sizes(values) for role, values in sizes[kind].items()}
        result[kind + "_all_pool_sizes"] = {role: _sizes(values) for role, values in all_sizes[kind].items()}
    return result


def _binomial_upper(k, n):
    log_factorial = math.lgamma(n + 1)
    return min(1.0, math.fsum(math.exp(
        log_factorial - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        + i * math.log(1 / 3) + (n - i) * math.log(2 / 3)) for i in range(k, n + 1)))


def order_decision(rows):
    """Common secondary order rule; this is not the standard engine's full gate."""
    votes = Counter({key: 0 for key in ("initial", "medial", "final")})
    for row in rows:
        selected = row["selected"]
        if not row["source_full"] or selected is None:
            continue
        if len(selected) != 3:
            raise ValueError("An order vote requires three selected positions")
        subject, predicate, obj = map(int, selected)
        votes["initial" if predicate < min(subject, obj) else "final" if predicate > max(subject, obj) else "medial"] += 1
    n = sum(votes.values())
    best = max(votes, key=votes.get)
    p = _binomial_upper(votes[best], n) if n else None
    return {"value": best if n >= 5 and p <= 0.05 / 3 else None,
            "votes": dict(votes), "n": n, "p": p}
