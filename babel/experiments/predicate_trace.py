"""Trace first-pass predicate links and relative-order votes on an aligned PUD corpus.

Example: python -m babel.experiments.predicate_trace ja_pud
"""

import argparse
import json
from collections import Counter

from babel.decipher.engine import Decipherer
from babel.experiments import real_parallel as rp


class TracedDecipherer(Decipherer):
    """Keep the first alignment only. Subsequent alignments are permutation controls."""

    observations = None

    def _count_orders(self, aligned):
        super()._count_orders(aligned)
        if self.observations is not None:
            return
        self.observations = []
        for pair, (item, row) in zip(self.pub["parallel"], aligned, strict=True):
            positions = {}
            for index, _, _, roles in row:
                roles = set(roles)
                if len(roles) > 1:
                    cues = {feature[2:] for unit, pos in item["units"] if pos == index and unit.startswith("A:")
                            for feature in self.affix_fn.get(unit[2:], {}) if feature.startswith("R:")}
                    roles = {role for role in roles if role.split(":")[-1] in cues}
                if len(roles) == 1:
                    positions.setdefault(next(iter(roles)), index)
            if "pred" in positions:
                self.observations.append((pair, positions))


def trace(repo, code, mismatch=False, pos_constraint=True):
    source, en_sha = rp.fetch(*rp.SOURCE)
    translations = {s["sent_id"]: tr for s in rp.parse(source) if (tr := rp.translation(s))}
    target, target_sha = rp.fetch(repo, code)
    parsed = rp.parse(target)
    sents = rp.target_sentences(parsed)
    ids = [sid for sid, _, _ in sents]
    words, _ = rp.rl.cipher([tokens for _, tokens, _ in sents], allow_extended=True)
    public, count = rp.build_public(f"real-{code}", words, ids, translations, 1.0, mismatch)
    engine = TracedDecipherer(public, dict(rp.VARIANTS["v1.8"], pos_constraint=pos_constraint)).fit()
    # The target annotation is read for the audit only after prediction.
    structure = engine._structure()
    observations = []
    votes = Counter()
    for pair, positions in engine.observations or []:
        doc = int(pair["doc"][1:]) - 1
        sid, tokens, _ = sents[doc * 10 + pair["sent"]]
        index = positions["pred"]
        item = {"sent_id": sid, "concept": pair["translation"]["pred"],
                "form": tokens[index][0], "upos": tokens[index][1]}
        if "head:subject" in positions and "head:object" in positions:
            subject, obj = positions["head:subject"], positions["head:object"]
            vote = "initial" if index < min(subject, obj) else "final" if index > max(subject, obj) else "medial"
            votes[vote] += 1
            item.update(vote=vote, subject=tokens[subject][0], object=tokens[obj][0])
        observations.append(item)
    if votes != engine.order_counts["verb_rel"]:
        raise RuntimeError("Diagnostic votes differ from the decoder tally")
    return {"code": code, "translations": count, "en_sha256": en_sha, "target_sha256": target_sha,
            "mismatch": mismatch, "pos_constraint": pos_constraint, "votes": votes,
            "gold": rp.gold_verb_rel(parsed), "decision": structure["verb_position"],
            "so_order": structure["so_order"], "gate": engine.align_null["verb_rel"], "observations": observations}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code", choices=[code for _, code in rp.TARGETS] + ["gl_pud"])
    parser.add_argument("--mismatch", action="store_true")
    parser.add_argument("--no-pos-constraint", action="store_true")
    args = parser.parse_args()
    repos = {code: repo for repo, code in rp.TARGETS}
    repos["gl_pud"] = "UD_Galician-PUD"
    print(json.dumps(trace(repos[args.code], args.code, args.mismatch, not args.no_pos_constraint), ensure_ascii=False))
