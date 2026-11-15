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


def test_unspaced_abstains_on_verb_and_affix():
    public, _ = build_corpus("UMB", 4, "C", boundaries=False)
    st = decipher(public)["structure"]
    assert st["verb_position"]["value"] is None and st["affix_position"]["value"] is None
    assert "G4'" in st["affix_position"].get("null_gate", "") or decipher(public, {"null_gates": False})[
        "structure"]["affix_position"]["value"] is None


def test_affix_alpha_is_per_side_bonferroni():
    """v1.4: 片側ごとの水準。p 値そのものは水準に依存しない。"""
    pred = decipher(null_corpus(5, "syllable")[0], {"affix_null_alpha": 0.05})
    mn = pred["verdict"]["morphology_null"]
    pred14 = decipher(null_corpus(5, "syllable")[0])
    mn14 = pred14["verdict"]["morphology_null"]
    for side in ("suffix", "prefix"):
        assert mn[side]["p"] == mn14[side]["p"]
        assert mn[side]["significant"] == (mn[side]["p"] <= 0.05)
        assert mn14[side]["significant"] == (mn14[side]["p"] <= 0.025)


def test_v15_verb_position_needs_alignment_evidence():
    """v1.5（V1）: 対訳のない LEVEL C では動詞の位置を主張せず、候補クラスの統計だけを出す。"""
    public, _ = build_corpus("KES", 2, "C")
    st15 = decipher(public)["structure"]
    st14 = decipher(public, {"v15_gates": False})["structure"]
    assert st14["verb_position"]["value"] is not None       # v1.4.0 は主張する
    assert st15["verb_position"]["value"] is None
    assert st15["verb_candidate_position"]["top"] == st14["verb_position"]["value"]
    assert "value" not in st15["verb_candidate_position"]   # 主張として数えられない


def test_v15_keeps_verb_position_with_alignment():
    public, _ = build_corpus("KES", 2, "A")
    assert decipher(public)["structure"]["verb_position"]["value"] == \
        decipher(public, {"v15_gates": False})["structure"]["verb_position"]["value"]


def test_v15_open_class_rejects_concentrated_class():
    from collections import Counter

    from babel.decipher import classes
    # 語幹 a に偏った群（実効語幹数 ≈ 1.1）と、均等な群
    infos, sents = {}, []
    for i, st in enumerate(["a"] * 40 + ["b", "c", "d", "e"]):
        w = f"{st}{i}"
        infos[w] = ([], st, ["x"])
        sents.append([w])
    stem_group = {s: "G" for s in "abcde"}
    pos, info = classes.label_classes(sents, infos, stem_group, open_class_min=8.0)
    assert info["verb_group"] is None and info["groups"]["G"]["effective_stems"] < 2
    assert isinstance(Counter(), Counter)


def test_v16_alignment_null_runs_on_level_a():
    """v1.6（A1・A2）: 対訳があるときだけ帰無対照を作り、結果を verdict に残す。"""
    public, _ = build_corpus("KES", 2, "A")
    pred = decipher(public)
    an = pred["verdict"]["alignment_null"]
    assert an["k"] == 19 and set(an) >= {"verb_support", "so", "adj"}
    assert decipher(build_corpus("KES", 2, "C")[0])["verdict"]["alignment_null"] is None


def test_v16_mismatched_translations_block_order_claims():
    from babel.decipher.engine import Decipherer
    public, _ = build_corpus("UMB", 3, "A")
    bad = dict(public, parallel=Decipherer._permuted(public["parallel"], "test"))
    st = decipher(bad)["structure"]
    assert st["so_order"]["value"] is None and st["adj_noun"]["value"] is None


def test_permuted_is_derangement():
    from babel.decipher.engine import Decipherer
    par = [{"doc": "d", "sent": i, "translation": {"pred": str(i)}} for i in range(7)]
    perm = Decipherer._permuted(par, "x")
    assert all(p["translation"]["pred"] != str(i) for i, p in enumerate(perm))
    assert sorted(p["translation"]["pred"] for p in perm) == [str(i) for i in range(7)]


def test_v17_verb_rel_counts_and_flag_off_keeps_v16():
    """v1.7（R1）: 述語・主語・目的語の位置がそろう文を数える。v17_gates=False では主張が v1.6.0 と同じ。"""
    from babel.decipher.engine import Decipherer
    public, _ = build_corpus("UMB", 0, "A")
    d = Decipherer(public).fit()
    assert set(d.order_counts) == {"so", "adj", "verb", "verb_rel"}
    assert d.align_null["verb_rel"]["pass"] in (True, False) and "real" in d.align_null["verb_rel"]
    st_off = decipher(public, {"v17_gates": False})["structure"]
    assert "alignment-relative" != st_off["verb_position"].get("source")
