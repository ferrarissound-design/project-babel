"""Evaluate punctuation-bounded subject, predicate and object links on PUD."""
import argparse
import json
import math
from collections import Counter
from itertools import product

from babel.experiments import real_parallel as rp
from babel.experiments.predicate_trace import TracedDecipherer

ROLES = ("head:subject", "pred", "head:object")
BREAKS = set(",;:!?，、；：！？。「」『』“”")


def target_view(sentence):
    segment = 0
    kept, segments = [], []
    for token in sentence["tokens"]:
        if any(c in BREAKS for c in token["form"]) and not rp.rl._clean(token["form"]):
            segment += 1
        if token["upos"] not in rp.rl.DROP_UPOS and rp.rl._clean(token["form"]):
            kept.append(token)
            segments.append(segment)
    return kept, segments


def role_candidates(item, links):
    concepts = {}
    for concept, role in item["concepts"]:
        concepts.setdefault(concept, set()).add(role)
    choices = []
    for role in ROLES:
        scores = {}
        for unit, pos in item["units"]:
            if not unit.startswith("S:"):
                continue
            for concept, roles in concepts.items():
                if roles == {role}:
                    score = links.get((unit, "C:" + concept), 0)
                    if score >= 0.25:
                        scores[pos] = max(scores.get(pos, 0), score)
        choices.append(scores)
    return choices


def joint(choices, segments):
    best, result = 0, None
    for triple in product(*choices):
        if len(set(triple)) != 3 or len({segments[i] for i in triple}) != 1:
            continue
        strength = math.prod(scores[i] for scores, i in zip(choices, triple))
        strength /= 1 + max(triple) - min(triple)
        if strength > best:
            best, result = strength, triple
    return result


def clause_heads(sentence):
    by_id = {token["id"]: token for token in sentence["tokens"]}
    anchors = {}
    for token in by_id.values():
        head = token
        while head["head"] in by_id:
            relation = head["deprel"].split(":")[0]
            if relation in {"advcl", "acl", "ccomp", "xcomp", "csubj", "parataxis"}:
                break
            if relation == "conj" and head["upos"] in {"VERB", "AUX"}:
                break
            head = by_id[head["head"]]
        anchors[token["id"]] = head["id"]
    return anchors


def grade(triple, tokens, anchors, counts):
    if triple is None:
        return
    subject, verb, obj = (tokens[i] for i in triple)
    counts["votes"] += 1
    counts["same_clause"] += len({anchors[t["id"]] for t in (subject, verb, obj)}) == 1
    counts["verb"] += verb["upos"] == "VERB"
    counts["exact"] += (verb["upos"] == "VERB" and subject["head"] == obj["head"] == verb["id"]
                        and subject["deprel"] == "nsubj" and obj["deprel"] == "obj")


def run(job):
    (repo, code), mismatch = job
    en, _ = rp.fetch(*rp.SOURCE)
    trs = {s["sent_id"]: t for s in rp.parse(en) if (t := rp.translation(s))}
    path, sha = rp.fetch(repo, code)
    parsed = rp.parse(path)
    raw = {s["sent_id"]: s for s in parsed}
    sents = rp.target_sentences(parsed)
    ids = [sid for sid, _, _ in sents]
    words, _ = rp.rl.cipher([w for _, w, _ in sents], allow_extended=True)
    public, _ = rp.build_public(code, words, ids, trs, 1, mismatch)
    # Only the initial alignment is scored. Permutation controls do not change these links.
    d = TracedDecipherer(public, {"v16_gates": False}).fit()
    by_id = {(p["doc"], p["sent"]): pos for p, pos in d.observations}
    counts = {k: Counter() for k in ("baseline", "filter", "joint")}
    examples = {}
    oracle = Counter()
    for p, item, links in zip(public["parallel"], d.aligner.items, d.aligner.links, strict=True):
        idx = (int(p["doc"][1:]) - 1) * 10 + p["sent"]
        sid = ids[idx]
        tokens, segments = target_view(raw[sid])
        assert [rp.rl._clean(t["form"]) for t in tokens] == [w[0] for w in sents[idx][1]]
        pos = by_id.get((p["doc"], p["sent"]), {})
        base = tuple(pos[r] for r in ROLES) if all(r in pos for r in ROLES) else None
        same = base if base and len({segments[i] for i in base}) == 1 else None
        choices = role_candidates(item, links)
        picks = {"baseline": base, "filter": same, "joint": joint(choices, segments)}
        # Post-prediction upper bound: could any selection from these candidates
        # recover a directly attached subject and object of the same annotated verb?
        by_head = {}
        for i, token in enumerate(tokens):
            by_head.setdefault(token["head"], {}).setdefault(token["deprel"], []).append(i)
        possible = []
        for v, token in enumerate(tokens):
            deps = by_head.get(token["id"], {})
            if token["upos"] == "VERB":
                possible.extend((s, v, o) for s in deps.get("nsubj", []) for o in deps.get("obj", []))
        full_source = {role for _, role in item["concepts"]}.issuperset(ROLES)
        oracle["eligible"] += bool(possible and full_source)
        if full_source and possible:
            for role_index, label in enumerate(("subject", "predicate", "object")):
                oracle[label] += any(t[role_index] in choices[role_index] for t in possible)
        available = [t for t in possible if all(i in scores for i, scores in zip(t, choices))]
        oracle["available"] += bool(available)
        oracle["within_segment"] += any(len({segments[i] for i in t}) == 1 for t in available)
        anchors = clause_heads(raw[sid])
        for name, triple in picks.items():
            grade(triple, tokens, anchors, counts[name])
        if code == "ja_pud" and sid in ("n01052006", "n01087035", "w02011029"):
            examples[sid] = {k: [tokens[i]["form"] for i in v] if v else None for k, v in picks.items()}
    return {"language": code, "sha256": sha, "mismatch": mismatch, "scores": counts, "examples": examples, "oracle": oracle}


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--mismatch", action="store_true")
    args = parser.parse_args()
    with ProcessPoolExecutor(args.jobs) as pool:
        targets = rp.TARGETS + [("UD_Galician-PUD", "gl_pud")]
        for result in pool.map(run, [(target, args.mismatch) for target in targets]):
            print(json.dumps(result, ensure_ascii=False), flush=True)
