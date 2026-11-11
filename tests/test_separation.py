"""研究上の不正を防ぐための検査: 解読器が正解に触れられないこと。"""
import json

from babel.decipher.engine import decipher
from babel.experiments import scorecard
from babel.experiments.run import DEV_SEEDS, TEST_SEEDS
from babel.generator.corpus import build_corpus


def test_decipherer_does_not_import_generator_or_evaluator():
    assert scorecard.check_import_isolation() == []


def test_public_corpus_contains_no_answers():
    assert scorecard.check_public_only() == []


def test_heldout_is_disjoint_and_has_no_translations():
    assert scorecard.check_heldout() == []


def test_dev_and_evaluation_seeds_are_disjoint():
    assert not set(DEV_SEEDS) & set(TEST_SEEDS)
    assert len(TEST_SEEDS) >= 20


def test_decipherer_works_from_serialised_public_only():
    public, _ = build_corpus("KES", 0, "A", n_docs=30)
    pred = decipher(json.loads(json.dumps(public)))     # JSON 往復 = 生成器のオブジェクトを一切共有しない
    assert pred["tokens"] and pred["structure"]


def test_level_c_public_has_no_semantic_clues_and_engine_claims_no_meaning():
    public, _ = build_corpus("SOT", 1, "C", n_docs=60)
    assert not public["parallel"] and not public["known"]
    assert all(not d["meta"] for d in public["documents"])
    pred = decipher(public)
    assert pred["summary"]["claimed"] == 0
    for sents in pred["tokens"].values():
        for s in sents:
            for t in s:
                c = t.get("concept")
                assert c is None or c.startswith("#")       # 数詞の相対値（内部構造）だけが許される
