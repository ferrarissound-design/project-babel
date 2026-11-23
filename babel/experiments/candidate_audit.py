"""Audit where structurally correct PUD role candidates disappear, without changing predictions."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from babel.decipher.engine import VERSION, Decipherer
from babel.experiments import real_parallel as rp
from babel.experiments.clause_probe import ROLES, role_candidates, target_view

STAGES = ("units", "pos_allowed", "positive_link", "threshold", "best_concept", "resolved_role", "first_position")
LABELS = ("subject", "predicate", "object", "triple")
EXAMPLE_IDS = ("n01052006", "n01087035", "w02011029")


class AuditDecipherer(Decipherer):
    """Capture actual first-pass rows before any permutation realignments."""

    audit_rows = None

    def _count_orders(self, aligned):
        super()._count_orders(aligned)
        if self.audit_rows is None:
            self.audit_rows = aligned


def stage_candidates(item, links, affix_fn, constraint=None):
    """Replay the fitted alignment's candidate selection, using no target annotations."""
    stages = {stage: [set() for _ in ROLES] for stage in STAGES}
    concepts = defaultdict(set)
    for concept, role in item["concepts"]:
        concepts["C:" + concept].add(role)
    cues = defaultdict(set)
    for unit, pos in item["units"]:
        if unit.startswith("A:"):
            cues[pos].update(feature[2:] for feature in affix_fn.get(unit[2:], {}) if feature.startswith("R:"))
    first = {}
    for unit, pos in item["units"]:
        if not unit.startswith("S:"):
            continue
        for index, role in enumerate(ROLES):
            stages["units"][index].add(pos)
            for concept, roles in concepts.items():
                if role not in roles or (constraint is not None and not constraint(unit, concept)):
                    continue
                stages["pos_allowed"][index].add(pos)
                score = links.get((unit, concept), 0)
                if score > 0:
                    stages["positive_link"][index].add(pos)
                if score >= 0.25:
                    stages["threshold"][index].add(pos)
        # Match _fit_alignment's tuple tie-break, including concepts outside ROLES.
        best = max(((score, concept) for (src, concept), score in links.items()
                    if src == unit and concept.startswith("C:")), default=None)
        if best is None or best[0] < 0.25:
            continue
        roles = concepts[best[1]]
        for index, role in enumerate(ROLES):
            if role in roles:
                stages["best_concept"][index].add(pos)
        resolved = roles if len(roles) == 1 else {role for role in roles if role.split(":")[-1] in cues[pos]}
        if len(resolved) == 1:
            role = next(iter(resolved))
            first.setdefault(role, pos)
            if role in ROLES:
                stages["resolved_role"][ROLES.index(role)].add(pos)
    for index, role in enumerate(ROLES):
        if role in first:
            stages["first_position"][index].add(first[role])
    for earlier, later in zip(STAGES, STAGES[1:]):
        if any(not b <= a for a, b in zip(stages[earlier], stages[later], strict=True)):
            raise RuntimeError(f"Non-nested candidates: {earlier} -> {later}")
    return stages


def direct_triples(tokens):
    """Annotated direct nsubj–VERB–obj triples, expressed as positions in this view."""
    children = defaultdict(lambda: defaultdict(list))
    for pos, token in enumerate(tokens):
        children[token["head"]][token["deprel"]].append(pos)
    return [(s, v, o) for v, token in enumerate(tokens) if token["upos"] == "VERB"
            for s in children[token["id"]]["nsubj"] for o in children[token["id"]]["obj"]]


def score_stages(stages, triples):
    """Sentence-level role recall and joint recall of at least one SAME gold triple."""
    scores = {}
    for stage, choices in stages.items():
        row = {label: int(any(t[i] in choices[i] for t in triples)) for i, label in enumerate(LABELS[:3])}
        row["triple"] = int(any(all(pos in pool for pos, pool in zip(t, choices, strict=True)) for t in triples))
        scores[stage] = row
    return scores


def first_losses(scores):
    return {label: next((stage for stage in STAGES if not scores[stage][label]), "retained") for label in LABELS}


def selected_positions(row, item, affix_fn):
    """Replay _count_orders on captured engine rows, to verify the independent audit."""
    cues = defaultdict(set)
    for unit, pos in item["units"]:
        if unit.startswith("A:"):
            cues[pos].update(feature[2:] for feature in affix_fn.get(unit[2:], {}) if feature.startswith("R:"))
    result = {}
    for pos, _, _, roles in row:
        roles = set(roles)
        if len(roles) != 1:
            roles = {role for role in roles if role.split(":")[-1] in cues[pos]}
        if len(roles) == 1:
            result.setdefault(next(iter(roles)), pos)
    return result


def run(job):
    (repo, code), mismatch = job
    source, source_sha = rp.fetch(*rp.SOURCE)
    translations = {s["sent_id"]: tr for s in rp.parse(source) if (tr := rp.translation(s))}
    path, target_sha = rp.fetch(repo, code)
    parsed = rp.parse(path)
    raw = {s["sent_id"]: s for s in parsed}
    sents = rp.target_sentences(parsed)
    ids = [sid for sid, _, _ in sents]
    words, _ = rp.rl.cipher([tokens for _, tokens, _ in sents], allow_extended=True)
    public, _ = rp.build_public(code, words, ids, translations, 1.0, mismatch)
    engine = AuditDecipherer(public, {"v16_gates": False}).fit()
    constraint = engine._pos_constraint(engine.aligner.items) if engine.cfg["pos_constraint"] else None
    totals = Counter()
    stage_totals = {stage: Counter({label: 0 for label in LABELS}) for stage in STAGES}
    losses = {label: Counter({stage: 0 for stage in (*STAGES, "retained")}) for label in LABELS}
    pool_totals = Counter({label: 0 for label in LABELS})
    vote_counts = Counter()
    sentences = []
    for pair, (item, row), links in zip(engine.pub["parallel"], engine.audit_rows, engine.aligner.links, strict=True):
        idx = (int(pair["doc"][1:]) - 1) * 10 + pair["sent"]
        sid = ids[idx]
        stages = stage_candidates(item, links, engine.affix_fn, constraint)
        positions = selected_positions(row, item, engine.affix_fn)
        actual = [{positions[role]} if role in positions else set() for role in ROLES]
        if actual != stages["first_position"]:
            raise RuntimeError(f"Audit differs from decoder rows: {code}/{sid}")
        if all(role in positions for role in ROLES):
            s, v, o = (positions[role] for role in ROLES)
            vote_counts["initial" if v < min(s, o) else "final" if v > max(s, o) else "medial"] += 1
        # Target annotation is consulted only after candidates and predictions are fixed.
        tokens, _ = target_view(raw[sid])
        if [rp.rl._clean(t["form"]) for t in tokens] != [w[0] for w in sents[idx][1]]:
            raise RuntimeError(f"Target positions differ: {code}/{sid}")
        original = direct_triples(raw[sid]["tokens"])
        triples = direct_triples(tokens)
        full_source = {role for _, role in item["concepts"]}.issuperset(ROLES)
        eligible = bool(full_source and triples)
        totals["parallel"] += 1
        totals["source_full"] += full_source
        totals["source_full_target_raw"] += bool(full_source and original)
        totals["eligible"] += eligible
        scores = score_stages(stages, triples)
        pool_score = score_stages({"pool": role_candidates(item, links)}, triples)["pool"]
        loss = first_losses(scores)
        if eligible:
            for stage in STAGES:
                stage_totals[stage].update(scores[stage])
            pool_totals.update(pool_score)
            for label in LABELS:
                losses[label][loss[label]] += 1
        sentences.append({"sent_id": sid, "source_full": full_source, "raw_triples": len(original),
                          "kept_triples": len(triples), "eligible": eligible, "scores": scores,
                          "first_loss": loss if eligible else None, "clause_pool": pool_score,
                          "selected": {role: {"position": pos, "form": tokens[pos]["form"], "upos": tokens[pos]["upos"]}
                                       for role, pos in positions.items() if role in ROLES}})
    if vote_counts != engine.order_counts["verb_rel"]:
        raise RuntimeError(f"Audit votes differ from decoder: {code}")
    return {"language": code, "mismatch": mismatch, "engine_version": VERSION, "ud_tag": rp.rl.UD_TAG,
            "source_sha256": source_sha, "target_sha256": target_sha, "config": engine.cfg,
            "totals": totals, "stages": stage_totals, "first_loss": losses, "clause_pool": pool_totals,
            "votes": vote_counts, "sentences": sentences}


def summary_row(row):
    summary = {k: v for k, v in row.items() if k != "sentences"}
    summary["examples"] = {s["sent_id"]: s for s in row["sentences"]
                           if row["language"] == "ja_pud" and s["sent_id"] in EXAMPLE_IDS}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--mismatch", action="store_true")
    parser.add_argument("--code", choices=[code for _, code in rp.TARGETS] + ["gl_pud"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--details-out", type=Path, help="Optional full per-sentence trace (large; not committed)")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.details_out and args.details_out.resolve() == args.out.resolve():
        parser.error("--out and --details-out must differ")
    targets = rp.TARGETS + [("UD_Galician-PUD", "gl_pud")]
    if args.code:
        targets = [t for t in targets if t[1] == args.code]
    # Fetch each cache file once in the parent; avoid concurrent writes of the shared English input.
    for target in [rp.SOURCE, *targets]:
        rp.fetch(*target)
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(run, [(target, args.mismatch) for target in targets]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(summary_row(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    if args.details_out:
        args.details_out.parent.mkdir(parents=True, exist_ok=True)
        args.details_out.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    for row in rows:
        print(json.dumps(summary_row(row), ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
