from fractions import Fraction

import pytest

from babel.decipher import numerals
from babel.decipher.baselines import CONFIGS
from babel.decipher.engine import decipher
from babel.decipher.morph import Morphology
from babel.decipher.variants import Normalizer
from babel.evaluate.score import score
from babel.generator.corpus import build_corpus, null_corpus


def test_linear_system_solves_relative_values():
    s = numerals.LinearSystem({"a", "b", "c"}, has_unit=False)
    assert s.add({"a": 2, "b": -1})          # 2a = b
    assert s.add({"a": 1, "b": 1, "c": -1})  # a + b = c
    vals, rel = s.solve()
    out = numerals._validate({**vals, "d": Fraction(0)} | {"e": vals["a"]}, rel)
    assert rel and out is None or out["b"] == 2 * out["a"]


def test_linear_system_rejects_contradiction_with_unit():
    s = numerals.LinearSystem({"a"}, has_unit=True)
    assert s.add({"a": 1, numerals.UNIT: -3})
    assert not s.add({"a": 1, numerals.UNIT: -4})
    assert s.rejected == 1


def test_infer_base():
    assert numerals.infer_base([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 100])[0] == 10
    assert numerals.infer_base(list(range(1, 20)) + [20, 40, 60, 80])[0] == 20
    assert numerals.infer_base(list(range(1, 12)) + [12, 24, 36, 144])[0] == 12
    assert numerals.infer_base([1, 2, 3])[0] is None


def test_morphology_finds_suffix_paradigm():
    stems = ["kamir", "tozan", "belud", "sorak", "minet", "paluk", "derim", "gosan", "vitor", "nakel"]
    vocab = {s + suf: 3 for s in stems for suf in ("", "ta", "ko", "lar", "larta", "larko")}
    m = Morphology(vocab)
    assert m.position == "suffix"
    assert m.analyze("kamirlarta") == ([], "kamir", ["lar", "ta"])
    assert m.analyze("tozan") == ([], "tozan", [])


def test_normalizer_merges_document_conditioned_spelling():
    words = ["kamir", "tokan", "sukol", "makit", "kelud", "rakon", "bokas", "nikat"]
    docs = []
    for i in range(40):
        ws = [w.replace("k", "q") if i % 2 else w for w in words]
        docs.append(ws * 3 + ["lumen", "padis", "torel"])
    nz = Normalizer(docs)
    assert any({o["a"], o["b"]} == {"k", "q"} for o in nz.ops)
    assert nz.norm("qamir")[0] == nz.norm("kamir")[0]


def test_normalizer_keeps_morphological_alternation_apart():
    stems = ["kamir", "tozan", "belud", "sorak", "minet", "paluk", "derim", "gosan"]
    docs = [[s + suf for s in stems for suf in ("ta", "to")] for _ in range(30)]
    nz = Normalizer(docs)
    assert nz.norm("kamirta")[0] != nz.norm("kamirto")[0]


@pytest.mark.parametrize("family", ["KES", "UMB", "SOT"])
@pytest.mark.parametrize("level", ["A", "B", "C"])
def test_engine_end_to_end(family, level):
    public, secret = build_corpus(family, 0, level, n_docs=60)
    pred = decipher(public)
    r = score(pred, secret, public)
    assert set(pred["tokens"]) == {d["id"] for d in public["documents"]}
    assert r["overclaim"]["overclaims"] == 0
    assert 0 <= r["segmentation"]["f1"] <= 1
    assert all(h["status"] for h in pred["hypotheses"])


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_every_config_runs(name):
    public, secret = build_corpus("KES", 1, "A", n_docs=40)
    score(decipher(public, CONFIGS[name]), secret, public)


def test_structureless_corpus_is_reported_as_undecipherable():
    for kind in ("random", "shuffled"):
        public, secret = null_corpus(0, kind, n_docs=80)
        pred = decipher(public)
        assert pred["verdict"]["syntax"]["present"] is False
        assert pred["summary"]["claimed"] == 0
    public, _ = build_corpus("KES", 0, "C", n_docs=80)
    assert decipher(public)["verdict"]["syntax"]["present"] is True


def test_engine_is_deterministic():
    public, _ = build_corpus("SOT", 2, "A", n_docs=40)
    a, b = decipher(public), decipher(public)
    a.pop("timing"), b.pop("timing")
    assert a == b


def test_oov_count_does_not_depend_on_document_order():
    public, secret = build_corpus("KES", 0, "A", n_docs=40)
    pred = decipher(public)
    a = score(pred, secret, public)["oov"]
    rev = dict(secret, tokens=dict(reversed(list(secret["tokens"].items()))))
    b = score(pred, rev, public)["oov"]
    assert a == b


def test_numerals_from_parallel_when_no_total_lines():
    """合計行のない会計記録（独立初見テストで外れた仮定）でも、対訳の数量から一部の数詞を正しく解く。"""
    from babel.evaluate.score import score as _score
    ok = wrong = 0
    for fam, seed in (("KES", 1), ("UMB", 4), ("SOT", 2)):
        public, secret = build_corpus(fam, seed, "A", totals=False)
        r = _score(decipher(public), secret, public)["numerals"]
        ok, wrong = ok + r["correct"], wrong + r["wrong"]
    assert ok >= 10 and wrong <= 1


def test_base_is_not_claimed_from_few_values():
    assert numerals.infer_base([1, 2, 4, 6])[0] is None
