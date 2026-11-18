import math
from collections import Counter
from itertools import product

import pytest

from babel.decipher.engine import Decipherer, decipher
from babel.experiments.real_parallel import VERSIONS16


def corpus(n):
    sentences = [f"ka{s} ti{v} mo{o}" for s, v, o in product(range(2), repeat=3)] * 20
    parallel = []
    for i, (s, o, v) in enumerate([(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 1)][:n]):
        sentences[i] = f"ka{s} ti{v} mo{o}"
        parallel.append({"doc": "d", "sent": i, "translation": {"pred": f"verb{v}", "args": [
            {"role": "subject", "head": f"subj{s}"}, {"role": "object", "head": f"obj{o}"}]}})
    return {"documents": [{"id": "d", "sentences": sentences}], "parallel": parallel, "level": "A"}


def test_evaluation_only_split():
    with pytest.raises(ValueError, match="学習用"):
        decipher(dict(corpus(3), split={"heldout": ["d"]}))


def test_duplicate_translations_cannot_strengthen_claim():
    p = corpus(3)
    original = decipher(p)
    repeated = decipher(dict(p, parallel=p["parallel"] * 4))
    assert original["structure"] == repeated["structure"]
    assert repeated["verdict"]["alignment_null"]["verb_rel"]["real"] == 3
    with pytest.raises(ValueError, match="異なる対訳"):
        decipher(dict(p, parallel=p["parallel"] + [dict(p["parallel"][0], translation={})]))


def test_v16_runner_preserves_old_decision():
    p = corpus(5)
    assert decipher(p)["structure"]["verb_position"]["value"] == "medial"
    assert decipher(p, VERSIONS16["v1.6"])["structure"]["verb_position"]["value"] is None


def test_large_vote_binomial_tail(monkeypatch):
    d = Decipherer(corpus(0))
    d.order_counts = {"so": Counter(), "adj": Counter(), "verb_rel": Counter(initial=450, medial=400, final=350)}
    monkeypatch.setattr(d, "_verb_class_support", lambda par: None)
    monkeypatch.setattr(d, "_realign", lambda par: ({key: Counter() for key in d.order_counts}, None))
    result = d._alignment_null()["verb_rel"]
    exact = sum(math.comb(1200, i) * 2 ** (1200 - i) for i in range(450, 1201)) / 3 ** 1200
    assert result["binomial_p"] == round(exact, 5)
    assert result["pass"]


def test_external_score_is_separate_from_internal_total():
    import json
    from pathlib import Path

    from babel.experiments.scorecard import build

    results = json.loads(Path("results/latest.json").read_text())
    internal = build(results, [])
    with_external = build(results, [], {"external_independent": {"points": 6, "detail": []}})
    assert internal["external_points"] is None and with_external["external_points"] == 6
    for key in ("internal_points", "evaluated_max", "unevaluated_max"):
        assert with_external[key] == internal[key]
