import json

import pytest

from babel.experiments.role_recovery_internal import (
    build_study_corpus,
    canonical,
    internal_tasks,
    lock_predictions,
    synthetic_gold,
)
from babel.generator import corpus


@pytest.mark.parametrize("family", ("KES", "UMB", "SOT"))
def test_role_annotation_preserves_public_and_restores_factory(family):
    factory = corpus.make_language
    expected, _ = corpus.build_corpus(family, 0, "A", n_docs=8, parallel_frac=0.25)
    public, annotation = build_study_corpus(family, 0, "A", n_docs=8, parallel_frac=0.25)
    assert canonical(public) == canonical(expected)
    assert corpus.make_language is factory
    assert b"STUDY_ROLE:" not in canonical(public)
    assert any(feature.startswith("STUDY_ROLE:") for document in annotation["tokens"].values()
               for sentence in document for token in sentence for feature in token["feats"])


def test_sot_count_case_keeps_original_object_role_and_scores_it():
    _, annotation = build_study_corpus("SOT", 0, "A", n_docs=8, parallel_frac=0.25)
    assert any("R:possessor" in token["feats"] and "STUDY_ROLE:object" in token["feats"]
               for document in annotation["tokens"].values() for sentence in document for token in sentence)
    public = {"parallel": [{"doc": "d", "sent": 0, "translation": {
        "pred": "give", "args": [{"role": "subject", "head": "king"}, {"role": "object", "head": "grain"}]}}]}
    gold = {"tokens": {"d": [[
        {"pos": "N", "concept": "king", "feats": ["STUDY_ROLE:subject"]},
        {"pos": "V", "concept": "give", "feats": []},
        {"pos": "N", "concept": "grain", "feats": ["R:possessor", "STUDY_ROLE:object"]},
        {"pos": "N", "concept": "grain", "feats": ["R:possessor", "STUDY_ROLE:possessor"]},
    ]]}}
    assert synthetic_gold(public, gold) == {("d", 0): [(0, 1, 2)]}


def test_prediction_lock_is_idempotent_but_rejects_replacement(tmp_path):
    path = tmp_path / "locked.json"
    prediction = {"A": [{"selected": [0, 1, 2]}]}
    fingerprint = lock_predictions(path, prediction)
    assert lock_predictions(path, prediction) == fingerprint
    with pytest.raises(ValueError, match="already fixed"):
        lock_predictions(path, {"A": [{"selected": None}]})
    assert json.loads(path.read_text()) == prediction


def test_registered_internal_cases_have_disjoint_seeds_and_all_conditions():
    dev, final = internal_tasks("dev"), internal_tasks("final")
    assert len(dev) == len(final) == 45
    assert {task[2] for task in dev}.isdisjoint({task[2] for task in final})
    assert {task[3] for task in dev} == {"A05-matched", "A05-shift1", "A25-matched", "A25-shift1", "C-no-evidence"}
