"""v1.3 帰無対照の関門（docs/PREREG_v1.3_NULL_GATES.md）。"""
from pathlib import Path

from babel.decipher import nulltest
from babel.decipher.engine import decipher
from babel.external import load_corpus
from babel.generator.corpus import build_corpus, null_corpus

ROOT = Path(__file__).resolve().parents[1]


def _claims(pred):
    return {k: v["value"] for k, v in pred["structure"].items() if k != "sum_marker" and v.get("value") is not None}


def test_syllable_null_has_no_structure_claims():
    pred = decipher(null_corpus(3, "syllable")[0])
    assert _claims(pred) == {}
    assert pred["verdict"]["morphology"]["present"] is False


def test_external_random_control_no_affix_claim():
    """外部の無作為対照で、v1.2.0 は接辞の位置を both と主張した。"""
    path = ROOT / "external/inbox/gpt6-random-control-20261114"
    public = load_corpus(path)
    assert decipher(public, {"null_gates": False})["structure"]["affix_position"]["value"] == "both"
    pred = decipher(public)
    assert pred["structure"]["affix_position"]["value"] is None
    assert "G1'" in pred["structure"]["affix_position"]["null_gate"]


def test_gates_keep_structure_on_real_corpus():
    public, _ = build_corpus("KES", 2, "A")
    assert _claims(decipher(public)) == _claims(decipher(public, {"null_gates": False}))


def test_syntax_perm_on_shuffled_words():
    public, _ = null_corpus(1, "shuffled")
    sents = [s.split() for d in public["documents"] for s in d["sentences"]]
    assert nulltest.syntax_test_perm(sents)["present"] is False
