"""Experimental public-input-only inference of subject, predicate and object.

The standard decoder is unchanged. Soft alignment uses t[source][target] times
the pair prior in each E step and in posterior scoring; the M step normalizes
expected counts within each source row. The reverse model uses the same pair
prior with its arguments swapped. The existing NULL factor 0.3 is separate.
These weights and the reported confidence are evidence proxies, not calibrated
probabilities of correct interpretation.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import product
from time import perf_counter

from .align import NULL, Aligner
from .engine import Decipherer

METHODS = ("A", "B", "C", "D", "E")
ROLES = ("head:subject", "pred", "head:object")
CONFIG = {
    "engine": {"v16_gates": False},
    "soft_conflict_weight": 0.2,
    "alignment_iterations": 12,
    "base_threshold": 0.25,
    "threshold_reduction": 0.20,
    "minimum_threshold": 0.05,
    "support_documents": 3,
    "positions_per_role": 5,
    "alternative_triples": 5,
    "minimum_joint_share": 0.8,
}


def weighted_ibm1(pairs, prior, n_iter=12):
    """IBM1 with a fixed nonnegative pair prior inside the E-step weights."""
    t = defaultdict(dict)
    for src, tgt in pairs:
        for source in [*src, NULL]:
            for target in tgt:
                if prior(source, target) > 0:
                    t[source][target] = 1.0
    for _ in range(n_iter):
        counts = defaultdict(lambda: defaultdict(float))
        for src, tgt in pairs:
            sources = [*src, NULL]
            for target in tgt:
                masses = [(source, t[source].get(target, 0.0) * prior(source, target)
                           * (0.3 if source == NULL else 1.0)) for source in sources]
                total = sum(mass for _, mass in masses)
                if total > 0:
                    for source, mass in masses:
                        if mass:
                            counts[source][target] += mass / total
        for source, row in counts.items():
            total = sum(row.values())
            t[source] = {target: value / total for target, value in row.items()}
    return t


def weighted_posterior(t, src, tgt, prior):
    """Posterior for non-NULL positions; NULL keeps its separate weight 0.3."""
    result = {}
    for target in tgt:
        total = sum(t[source].get(target, 0.0) * prior(source, target)
                    * (0.3 if source == NULL else 1.0) for source in [*src, NULL])
        result[target] = {source: t[source].get(target, 0.0) * prior(source, target) / total if total > 0 else 0.0
                          for source in src}
    return result


class WeightedAligner(Aligner):
    """Bidirectional EM with a soft POS prior and a hard affix/concept rule."""

    def __init__(self, items, constraint=None, conflict_weight=0.2):
        if not 0 < conflict_weight <= 1:
            raise ValueError("conflict_weight must be in (0, 1]")
        self.items = items
        self.mode, self.kappa = "em", 0.7
        self.n_tgt, self.n = Counter(), Counter()
        self.A = defaultdict(lambda: defaultdict(float))
        self.joint = defaultdict(Counter)
        self.links = []
        pairs = [(sorted({unit for unit, _ in item["units"]}),
                  sorted({"C:" + concept for concept, _ in item["concepts"]}
                         | {"F:" + feature for feature in item["feats"]})) for item in items]

        def prior(source, target):
            if source == NULL:
                return 1.0
            if source.startswith("A:") and target.startswith("C:"):
                return 0.0
            return 1.0 if constraint is None or constraint(source, target) else conflict_weight

        def reverse_prior(concept, unit):
            return 1.0 if concept == NULL else prior(unit, concept)

        self.prior = prior
        forward = weighted_ibm1(pairs, prior, CONFIG["alignment_iterations"])
        reverse = weighted_ibm1([(tgt, src) for src, tgt in pairs], reverse_prior, CONFIG["alignment_iterations"])
        for src, tgt in pairs:
            self.n_tgt.update(tgt)
            p1 = weighted_posterior(forward, src, tgt, prior)
            p2 = weighted_posterior(reverse, tgt, src, reverse_prior)
            links = {}
            for unit in src:
                self.n[unit] += 1
                for concept in tgt:
                    if prior(unit, concept) <= 0:
                        continue
                    self.joint[unit][concept] += 1
                    score = p1[concept][unit] * p2[unit][concept]
                    if score > 0:
                        self.A[unit][concept] += score
                        links[(unit, concept)] = score
            self.links.append(links)


class _AlignmentFinished(Exception):
    pass


class _PreparedDecipherer(Decipherer):
    """Run the identical first alignment and stop before unrelated later work."""

    def _fit_alignment(self):
        start = perf_counter()
        try:
            super()._fit_alignment()
        finally:
            self.hard_fit_seconds = perf_counter() - start

    def _count_orders(self, aligned):
        super()._count_orders(aligned)
        self.aligned_rows = aligned
        raise _AlignmentFinished


def evidence_threshold(q, documents):
    """Registered adaptive threshold from an uncalibrated evidence share q."""
    evidence = min(1.0, documents / CONFIG["support_documents"]) * q
    return max(CONFIG["minimum_threshold"], CONFIG["base_threshold"] - CONFIG["threshold_reduction"] * evidence)


def rank_triples(candidates, limit=5):
    """Rank distinct-position triples by strength product divided by span."""
    ranked = []
    for positions in product(*(sorted(pool) for pool in candidates)):
        if len(set(positions)) != 3:
            continue
        score = math.prod(pool[pos] for pool, pos in zip(candidates, positions, strict=True))
        score /= 1 + max(positions) - min(positions)
        ranked.append({"positions": list(positions), "score": score})
    ranked.sort(key=lambda row: (-row["score"], row["positions"]))
    return ranked[:limit]


def _supports(items, parallel):
    documents = defaultdict(set)
    for item, pair in zip(items, parallel, strict=True):
        concepts = {"C:" + concept for concept, _ in item["concepts"]}
        for unit in {unit for unit, _ in item["units"] if unit.startswith("S:")}:
            for concept in concepts:
                documents[(unit, concept)].add(pair["doc"])
    return {key: len(value) for key, value in documents.items()}


def _shares(aligner):
    shares = {}
    for unit, values in aligner.A.items():
        concepts = {concept: value for concept, value in values.items() if concept.startswith("C:")}
        total = sum(concepts.values())
        for concept, value in concepts.items():
            shares[(unit, concept)] = value / total if total > 0 else 0.0
    return shares


def _resolved_role(roles, cues):
    unique = set(roles)
    if len(unique) != 1:
        unique = {role for role in unique if role.split(":")[-1] in cues}
    return next(iter(unique)) if len(unique) == 1 else None


def _pools(item, links, affix_fn, support, shares, adaptive=False, multiple=False):
    concepts = defaultdict(set)
    for concept, role in item["concepts"]:
        concepts["C:" + concept].add(role)
    cues = defaultdict(set)
    for unit, pos in item["units"]:
        if unit.startswith("A:"):
            cues[pos].update(feature[2:] for feature in affix_fn.get(unit[2:], {}) if feature.startswith("R:"))
    by_unit = defaultdict(list)
    for (unit, concept), score in links.items():
        if concept.startswith("C:"):
            by_unit[unit].append((score, concept))
    pools = [{} for _ in ROLES]
    evidence = [{} for _ in ROLES]
    first = {}
    for unit, pos in item["units"]:
        if not unit.startswith("S:"):
            continue
        choices = by_unit.get(unit, [])
        if not multiple:
            best = max(choices, default=None)  # Literal legacy tuple tie-break.
            choices = [best] if best is not None else []
        for score, concept in choices:
            threshold = evidence_threshold(shares.get((unit, concept), 0.0), support.get((unit, concept), 0)) if adaptive else 0.25
            if score < threshold:
                continue
            role = _resolved_role(concepts[concept], cues[pos])
            if role not in ROLES:
                continue
            first.setdefault(role, pos)
            index = ROLES.index(role)
            # At a shared position, the strongest link supplies both score and support.
            # New evidence ties use concept and unit strings in descending order.
            key = (score, concept, unit)
            old = evidence[index].get(pos)
            if old is None or key > old[0]:
                pools[index][pos] = score
                evidence[index][pos] = (key, support.get((unit, concept), 0))
    return pools, evidence, first


def _prediction(method, pair, item, links, affix_fn, support, shares):
    raw, evidence, first = _pools(item, links, affix_fn, support, shares,
                                  adaptive=method in {"C", "E"}, multiple=method == "E")
    candidates = [dict(sorted(pool.items(), key=lambda entry: (-entry[1], entry[0]))[:CONFIG["positions_per_role"]])
                  for pool in raw]
    alternatives = rank_triples(candidates, CONFIG["alternative_triples"])
    source_full = {role for _, role in item["concepts"]}.issuperset(ROLES)
    selected = None
    if not source_full:
        reason = "incomplete_source"
    elif method in {"A", "B", "C"}:
        if all(role in first for role in ROLES):
            selected = [first[role] for role in ROLES]
            reason = "selected_first_position"
        else:
            reason = "missing_role"
    elif not alternatives:
        reason = "no_distinct_triple"
    elif method == "D":
        selected = alternatives[0]["positions"]
        reason = "selected_joint"
    else:
        top = alternatives[0]
        share = top["score"] / sum(row["score"] for row in alternatives)
        positions = top["positions"]
        supported = all(evidence[index][pos][1] >= CONFIG["support_documents"] for index, pos in enumerate(positions))
        if share < CONFIG["minimum_joint_share"]:
            reason = "insufficient_joint_share"
        elif not supported:
            reason = "insufficient_document_support"
        else:
            selected = positions
            reason = "selected_joint_with_evidence"
    total = sum(row["score"] for row in alternatives)
    selected_score = next((row["score"] for row in alternatives if row["positions"] == selected), 0.0)
    confidence = min(1.0, selected_score / total) if total > 0 else 0.0
    return {"doc": pair["doc"], "sent": pair["sent"], "source_full": source_full,
            "raw_candidates": raw, "candidates": candidates, "selected": selected,
            "alternatives": alternatives, "confidence": confidence, "reason": reason}


class RoleInference:
    """Prepare training documents once, then compare registered A–E inferences.

    Morphology, alignment, support counts and concept constraints use training
    documents only. For unbounded input, the underlying decoder's existing
    transductive segmentation behavior is preserved for A parity; the registered
    comparisons supply word boundaries. Timings separate the shared preparation,
    the hard first alignment, the soft re-fit and each method's selection. The
    hard alignment is also incurred when preparing the comparison's soft model.
    """

    def __init__(self, public):
        self.public = public
        self.timings = {"preprocess_seconds": 0.0, "hard_fit_seconds": 0.0, "soft_fit_seconds": 0.0,
                        "selection_seconds": {method: 0.0 for method in METHODS}}
        self.predictions = None

    def predict_all(self):
        if self.predictions is not None:
            return self.predictions
        start = perf_counter()
        engine = _PreparedDecipherer(self.public, CONFIG["engine"])
        train = set(engine.train_ids)
        # Filter before numeral/alignment learning as well as before concept constraints.
        parallel = [pair for pair in engine.pub.get("parallel", []) if pair["doc"] in train]
        engine.pub["parallel"] = parallel
        if not parallel:
            self.timings["preprocess_seconds"] = perf_counter() - start
            self.predictions = {method: [] for method in METHODS}
            return self.predictions
        try:
            engine.fit()
        except _AlignmentFinished:
            pass
        else:
            raise RuntimeError("First alignment did not reach its registered stopping point")
        self.timings["hard_fit_seconds"] = engine.hard_fit_seconds
        self.timings["preprocess_seconds"] = perf_counter() - start - engine.hard_fit_seconds
        hard = engine.aligner
        items = hard.items
        start = perf_counter()
        support = _supports(items, parallel)
        hard_shares = _shares(hard)
        constraint = engine._pos_constraint(items) if engine.cfg["pos_constraint"] else None
        self.timings["preprocess_seconds"] += perf_counter() - start
        start = perf_counter()
        soft = WeightedAligner(items, constraint, CONFIG["soft_conflict_weight"])
        soft_affixes = {unit[2:]: soft.functions(unit) for unit in sorted(soft.n) if unit.startswith("A:")}
        soft_shares = _shares(soft)
        self.timings["soft_fit_seconds"] = perf_counter() - start
        predictions = {}
        for method in METHODS:
            start = perf_counter()
            aligner, affix_fn, shares = ((soft, soft_affixes, soft_shares) if method in {"B", "E"}
                                        else (hard, engine.affix_fn, hard_shares))
            predictions[method] = [_prediction(method, pair, item, links, affix_fn, support, shares)
                                   for pair, item, links in zip(parallel, items, aligner.links, strict=True)]
            self.timings["selection_seconds"][method] = perf_counter() - start
        # The comparison must retain the standard engine's actual A selections.
        start = perf_counter()
        for predicted, (item, row) in zip(predictions["A"], engine.aligned_rows, strict=True):
            cues = defaultdict(set)
            for unit, pos in item["units"]:
                if unit.startswith("A:"):
                    cues[pos].update(feature[2:] for feature in engine.affix_fn.get(unit[2:], {}) if feature.startswith("R:"))
            positions = {}
            for pos, _, _, roles in row:
                role = _resolved_role(roles, cues[pos])
                if role is not None:
                    positions.setdefault(role, pos)
            expected = [positions[role] for role in ROLES] if all(role in positions for role in ROLES) else None
            if predicted["selected"] != expected:
                raise RuntimeError("A differs from the standard decoder's first-position selections")
        self.timings["selection_seconds"]["A"] += perf_counter() - start
        self.predictions = predictions
        return predictions
