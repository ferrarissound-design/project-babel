"""Decoding engine v2: correctness of the inference, separation from answers, end-to-end behaviour."""
import ast
import json
import math
import random
import subprocess
import sys
from itertools import product
from pathlib import Path

import pytest

from babel.decipher.morph import Morphology
from babel.decode import calibration
from babel.decode.anchors import NameAnchors
from babel.decode.engine import Decoder, decode
from babel.decode.model import DEFAULTS, Corpus, RoleModel
from babel.decode.report import render
from babel.experiments.decode_eval import to_rows
from babel.experiments.role_recovery_score import score_predictions

ROOT = Path(__file__).resolve().parents[1]


def toy_language(n_sent=240, seed=0, mismatch=False, order="SOV"):
    """A small invented language: case particles, verb suffix, noun phrases with a determiner and fillers.

    Returns (public, gold) where gold maps (doc, sent) -> (s, v, o) positions. Built here so that the
    test does not depend on the project's generator.
    """
    rng = random.Random(seed)
    syll = ["ka", "to", "mi", "re", "su", "na", "lo", "pe", "di", "gu", "ze", "ho"]

    def word(k):
        return "".join(rng.choice(syll) for _ in range(k))

    nouns = {f"n{i}": word(rng.choice((2, 3))) for i in range(60)}
    verbs = {f"v{i}": word(2) for i in range(40)}
    fillers = [word(3) for _ in range(30)]
    det, subj_p, obj_p, vsuf = "la", "ga", "wo", "ru"
    docs, parallel, gold = [], [], {}
    for i in range(n_sent):
        d, j = f"d{i // 10:03d}", i % 10
        if j == 0:
            docs.append({"id": d, "sentences": []})
        s, o = rng.sample(sorted(nouns), 2)
        v = rng.choice(sorted(verbs))
        pre = [rng.choice(fillers) for _ in range(rng.randint(0, 3))]
        sp = [det, nouns[s], subj_p]
        op = [det, nouns[o], obj_p]
        vp = [verbs[v] + vsuf]
        parts = {"SOV": (sp, op, vp), "SVO": (sp, vp, op)}[order]
        toks = pre + [t for p in parts for t in p] + [rng.choice(fillers)]
        sidx = toks.index(nouns[s], len(pre))
        oidx = len(pre) + toks[len(pre):].index(obj_p) - 1
        vidx = toks.index(verbs[v] + vsuf, len(pre))
        gold[(d, j)] = (sidx, vidx, oidx)
        docs[-1]["sentences"].append(" ".join(toks))
        parallel.append({"doc": d, "sent": j, "translation": {"kind": "clause", "pred": v,
                         "args": [{"role": "subject", "head": s}, {"role": "object", "head": o}]}})
    if mismatch:
        trs = [p["translation"] for p in parallel]
        trs = trs[1:] + trs[:1]
        parallel = [dict(p, translation=t) for p, t in zip(parallel, trs, strict=True)]
    return {"language_id": "toy", "has_word_boundaries": True, "documents": docs, "parallel": parallel}, gold


def _score(result, gold):
    rows = to_rows(result["sentences"])
    return score_predictions(rows, {k: [v] for k, v in gold.items()})


def test_decode_package_is_isolated_from_answers():
    bad = []
    for p in sorted((ROOT / "babel/decode").glob("*.py")):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            bad += [f"{p.name}: {n}" for n in names if any(x in n for x in ("generator", "evaluate", "experiments"))]
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and \
                    any(x in node.value.lower() for x in ("secret", "answer.json", "conllu", "deprel")):
                bad.append(f"{p.name}: {node.value!r}")
    assert bad == []


@pytest.mark.parametrize("order", ["SOV", "SVO"])
def test_end_to_end_recovers_roles_in_an_invented_language(order):
    public, gold = toy_language(order=order)
    res = decode(json.loads(json.dumps(public)), {"calibration": "off"})
    sc = _score(res, gold)
    assert sc["claimed"] >= 0.5 * sc["eligible"]
    assert sc["precision"] >= 0.9
    assert res["word_order"]["verb_position"] == ("final" if order == "SOV" else "medial")


def test_mismatched_translations_reduce_grounded_lexicon_and_claims():
    public, gold = toy_language()
    bad, _ = toy_language(mismatch=True)
    good = decode(public, {"calibration": "off"}, hypotheses=False)
    shifted = decode(bad, {"calibration": "off"}, hypotheses=False)
    assert len(shifted["lexicon"]) < 0.5 * len(good["lexicon"])
    # structure can still be found (the sentences are the same), but meanings are not
    good_lex = {r["concept"]: r["stem"] for r in good["lexicon"]}
    agree = sum(good_lex.get(r["concept"]) == r["stem"] for r in shifted["lexicon"])
    assert agree <= 0.2 * max(1, len(shifted["lexicon"]))


def test_without_translations_no_meaning_and_no_roles_are_claimed():
    public, _ = toy_language()
    public = dict(public, parallel=[])
    res = decode(public)
    assert res["input"]["role_model_trained"] is False
    assert res["sentences"] == [] and res["lexicon"] == [] and res["structural_hypotheses"] == []
    assert res["structure"]["boundaries"]["tokens"] > 0


def test_too_few_translations_are_reported_not_guessed():
    public, _ = toy_language(n_sent=10)
    res = decode(public)
    assert res["summary"]["claimed"] == 0
    assert all(r["status"] == "undetermined" for r in res["sentences"])
    assert any("translated sentences" in w for w in res["warnings"])


def _brute_force(model, item):
    n = len(model.c.sentences[item["sent"]])
    u = model.unary(item)
    feats = model.c.features[item["sent"]]
    from babel.decode.model import BUCKETS, bucket

    def factor(role, v, i):
        if i is None:
            return model.placement(role, "NULL") * model.null_ratio(role, item)
        width = sum(1 for k in range(n) if k != v and bucket(k - v) == bucket(i - v))
        cross = math.prod(model.crossing(role, feats[t]) for t in range(min(i, v) + 1, max(i, v)))
        assert bucket(i - v) in BUCKETS
        return u[role][i] * model.placement(role, bucket(i - v)) / width * cross

    joint = {}
    for v, s, o in product(range(n), [None, *range(n)], [None, *range(n)]):
        if v in (s, o) or (s is not None and s == o):
            continue
        joint[(s, v, o)] = u["V"][v] * factor("S", v, s) * factor("O", v, o)
    z = sum(joint.values())
    return {k: x / z for k, x in joint.items()}


def test_posterior_is_exact_against_brute_force_enumeration():
    public, _ = toy_language(n_sent=80)
    d = Decoder(public, {"restarts": (1,), "passes": 4}).fit()
    m = d.models[0]
    for item in m.items[:5]:
        post = m.posterior(item)
        brute = _brute_force(m, item)
        n = len(m.c.sentences[item["sent"]])
        for v in range(n):
            assert post["V"][v] == pytest.approx(sum(p for (s, vv, o), p in brute.items() if vv == v), abs=1e-9)
        for i in range(n):
            assert post["S"][i] == pytest.approx(sum(p for (s, v, o), p in brute.items() if s == i), abs=1e-9)
            assert post["O"][i] == pytest.approx(sum(p for (s, v, o), p in brute.items() if o == i), abs=1e-9)
        assert post["S"][n] == pytest.approx(sum(p for (s, v, o), p in brute.items() if s is None), abs=1e-9)
        best = max(((k, p) for k, p in brute.items() if None not in k), key=lambda kv: kv[1])
        assert post["best"] == best[0] and post["best_p"] == pytest.approx(best[1])


def test_counts_equal_the_sum_of_stored_contributions_so_leave_one_out_is_exact():
    public, _ = toy_language(n_sent=60)
    d = Decoder(public, {"restarts": (1,), "passes": 3}).fit()
    m = d.models[0]
    total = {}
    for c in m.contrib:
        for k, v in (c or {}).items():
            total[k] = total.get(k, 0.0) + v
    for k, v in m.G.items():
        assert v == pytest.approx(total.get(k, 0.0), abs=1e-6)


def test_hapax_concept_is_neutral_under_leave_one_out():
    sents = [["aa", "bb", "cc"], ["dd", "ee", "ff"]]
    corpus = Corpus(sents, Morphology({w: 1 for s in sents for w in s}, enable=False), DEFAULTS)
    m = RoleModel(corpus, [{"sent": 0, "V": "once", "S": None, "O": None}])
    assert m.lexical("once", "bb") == pytest.approx(1.0)


def test_anchor_gate_finds_shared_names_and_stays_silent_on_unrelated_strings():
    rng = random.Random(3)
    letters = "abcdefghijklmnopqrstuvwxyz"
    table = dict(zip(letters, rng.sample("αβγδεζηθικλμνξοπρστυφχψωϊϋ", 26), strict=True))
    names = ["obama", "trump", "london", "germany", "merkel", "putin", "macron", "brussels", "nairobi", "tokyo",
             "kenya", "canada", "mexico", "brazil", "toronto", "chicago", "madrid", "berlin", "warsaw", "vienna"]

    def noise():
        return "".join(rng.choice(list(table.values())) for _ in range(rng.randint(3, 8)))

    sents, occ, gold = [], [], set()
    for i in range(200):
        name = names[i % len(names)]
        toks = [noise() for _ in range(6)]
        k = rng.randrange(6)
        toks[k] = "".join(table[c] for c in name) + rng.choice(["", "ος", "ι"])
        sents.append(toks)
        occ.append((i, "S", name, i))
        gold.add((i, "S", k))
    a = NameAnchors(sents, occ).fit()
    found = {key for key, p in a.posterior.items() if p >= 0.5}
    assert len(found & gold) >= 0.7 * len(gold) and len(found - gold) <= 0.05 * len(gold)
    # unrelated strings: concepts that never appear in the sentence
    occ2 = [(i, "S", names[(i + 7) % len(names)] + "x", i) for i in range(200)]
    sents2 = [[noise() for _ in range(6)] for _ in range(200)]
    b = NameAnchors(sents2, occ2).fit()
    assert sum(p >= 0.5 for p in b.posterior.values()) <= 4


def test_rows_for_the_shared_scorer_and_markdown_report():
    public, gold = toy_language(n_sent=120)
    res = decode(public, {"calibration": "off"})
    rows = to_rows(res["sentences"])
    assert all(len(r["candidates"]) == 3 for r in rows)
    sc = score_predictions(rows, {k: [v] for k, v in gold.items()})
    assert sc["eligible"] == 120 and sc["claimed"] + sc["abstain"] == 120
    md = render(res)
    assert md.startswith("# 解読レポート") and "判定不能" in md and "構造" in md
    for rec in res["sentences"]:
        assert rec["status"] in ("claimed", "undetermined") and rec["reason"]
        if rec["status"] == "claimed":
            assert rec["basis"] == "translation" and set(rec["triple"]) == {"subject", "predicate", "object"}
    for rec in res["structural_hypotheses"]:
        assert rec["basis"] == "structure" and "concepts" not in rec


def test_decode_is_deterministic():
    public, _ = toy_language(n_sent=80)
    a = decode(public, {"calibration": "off"})
    b = decode(json.loads(json.dumps(public)), {"calibration": "off"})
    a.pop("fit_seconds"), b.pop("fit_seconds")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_calibration_round_trip_and_feature_list_guard(tmp_path):
    rows = [({n: random.random() for n in calibration.NAMES}, i % 2) for i in range(50)]
    for f, y in rows:
        f["log_p"] = y + random.random() * 0.1
    model = calibration.fit(rows, iters=200)
    hi = calibration.predict(model, dict(rows[1][0], log_p=1.05))
    lo = calibration.predict(model, dict(rows[0][0], log_p=0.05))
    assert 0 < lo < hi < 1
    p = tmp_path / "c.json"
    p.write_text(json.dumps(dict(model, names=["x"])))
    with pytest.raises(ValueError):
        calibration.load(p)
    assert calibration.load(tmp_path / "missing.json") is None


def test_cli_decode_writes_json_and_markdown(tmp_path):
    public, _ = toy_language(n_sent=60)
    src = tmp_path / "corpus.json"
    src.write_text(json.dumps(public), encoding="utf-8")
    out = tmp_path / "out" / "result.json"
    r = subprocess.run([sys.executable, "-m", "babel", "decode", str(src), "--out", str(out), "--restarts", "1"],
                       cwd=ROOT, capture_output=True, text=True, check=True)
    summary = json.loads(r.stdout)
    assert summary["translated"] == 60
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["schema"] == "babel-decode-v2"
    assert out.with_suffix(".md").exists()
