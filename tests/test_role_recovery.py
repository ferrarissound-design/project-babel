import json
import math
import random
from collections import Counter
from itertools import product

import pytest

from babel.experiments.role_recovery_score import distinct_triple_count, order_decision, score_predictions


def prediction(sent=0, selected=(0, 1, 2), pools=None, source_full=True, confidence=0.9):
    pools = pools if pools is not None else [{0: 0.8}, {1: 0.8}, {2: 0.8}]
    return {"doc": "d", "sent": sent, "source_full": source_full,
            "raw_candidates": pools, "candidates": pools, "selected": list(selected) if selected is not None else None,
            "alternatives": [], "confidence": confidence}


def test_no_target_gold_emission_is_an_error_and_abstention_denominator_is_source_full():
    rows = [prediction(0), prediction(1), prediction(2, None), prediction(3, None, source_full=False)]
    gold = {("d", 0): [(0, 1, 2)], ("d", 2): [(0, 1, 2)]}
    scores = score_predictions(json.loads(json.dumps(rows)), gold)
    assert {key: scores[key] for key in ("total", "eligible", "gold_eligible", "claimed", "correct", "wrong", "abstain")} == {
        "total": 4, "eligible": 3, "gold_eligible": 2, "claimed": 2, "correct": 1, "wrong": 1, "abstain": 1}
    assert scores["high_confidence_wrong"] == 1
    assert scores["precision"] == scores["wrong_rate"] == 0.5
    assert scores["accuracy"] == scores["abstain_rate"] == 1 / 3
    assert scores["candidate_triples"] == 3 and scores["gold_candidate_triples"] == 2
    assert scores["candidate_precision"] == 2 / 3
    assert scores["raw_retained"] == scores["budget_retained"] == 2
    assert scores["budget_pool_sizes"]["subject"]["total"] == 2
    assert scores["budget_all_pool_sizes"]["subject"]["total"] == 3


def test_separate_gold_roles_do_not_count_as_a_retained_gold_triple():
    row = prediction(pools=[{0: 1}, {4: 1}, {2: 1}], selected=(0, 4, 2))
    scores = score_predictions([row], {("d", 0): [(0, 1, 2), (3, 4, 5)]})
    assert scores["budget_role_retained"] == {"subject": 1, "predicate": 1, "object": 1}
    assert scores["budget_retained"] == scores["raw_retained"] == scores["correct"] == 0
    assert scores["candidate_triples"] == 1 and scores["candidate_precision"] == 0
    assert scores["wrong"] == 1


def test_raw_retention_and_common_budget_are_separate_and_duplicate_gold_not_double_counted():
    row = prediction(pools=[{0: 1, 3: 0.5}, {1: 1}, {2: 1}], selected=(0, 1, 2))
    row["candidates"] = [{0: 1}, {1: 1}, {2: 1}]
    scores = score_predictions([row], {("d", 0): [(3, 1, 2), (3, 1, 2)]})
    assert scores["raw_retained"] == 1 and scores["budget_retained"] == 0
    assert scores["raw_gold_candidate_triples"] == 1 and scores["gold_candidate_triples"] == 0
    assert scores["raw_candidate_triples"] == 2 and scores["candidate_triples"] == 1
    assert scores["raw_pool_sizes"]["subject"] == {"total": 2, "mean": 2, "p95": 2}


def test_inclusion_exclusion_matches_independent_enumeration_and_never_counts_overlap():
    rng = random.Random(17)
    for _ in range(30):
        pools = [{i for i in range(6) if rng.random() < 0.6} for _ in range(3)]
        expected = sum(len(set(triple)) == 3 for triple in product(*pools))
        assert distinct_triple_count(pools) == expected
    assert distinct_triple_count([{0}, {0}, {0}]) == 0
    assert distinct_triple_count([set(range(10000))] * 3) == 10000 * 9999 * 9998


def test_empty_denominators_are_unevaluated_and_unsupported_outputs_are_visible():
    scores = score_predictions([], {})
    for key in ("accuracy", "precision", "wrong_rate", "abstain_rate", "raw_recall", "budget_recall", "candidate_precision"):
        assert scores[key] is None
    assert scores["budget_pool_sizes"]["predicate"] == {"total": 0, "mean": None, "p95": None}
    unsupported = score_predictions([prediction(source_full=False)], {})
    assert unsupported["eligible"] == unsupported["claimed"] == 0
    assert unsupported["unsupported_claimed"] == 1


def test_duplicate_rows_and_budget_inflation_are_rejected():
    with pytest.raises(ValueError, match="Duplicate prediction"):
        score_predictions([prediction(), prediction()], {})
    row = prediction()
    row["candidates"] = [{i: 1 for i in range(6)}, {1: 1}, {2: 1}]
    with pytest.raises(ValueError, match="exceed five"):
        score_predictions([row], {})
    row["candidates"] = [{99: 1}, {1: 1}, {2: 1}]
    with pytest.raises(ValueError, match="subset"):
        score_predictions([row], {})


def test_secondary_order_rule_requires_both_five_votes_and_corrected_binomial_evidence():
    assert order_decision([prediction(i) for i in range(4)])["value"] is None
    result = order_decision([prediction(i) for i in range(5)])
    assert result["value"] == "medial" and result["n"] == 5
    assert result["p"] == pytest.approx((1 / 3) ** 5)
    scattered = [prediction(i, selected=(1, 0, 2) if i < 2 else (0, 1, 2) if i < 4 else (0, 2, 1)) for i in range(6)]
    result = order_decision(scattered)
    exact = sum(math.comb(6, i) * 2 ** (6 - i) for i in range(2, 7)) / 3 ** 6
    assert result["value"] is None and result["p"] == pytest.approx(exact)
    assert order_decision([])["p"] is None


def test_overlapping_baseline_positions_vote_like_engine_but_are_not_structural_successes():
    rows = [prediction(i, selected=(0, 0, 2)) for i in range(5)]
    assert order_decision(rows)["value"] == "medial"
    scores = score_predictions(rows, {("d", i): [(0, 1, 2)] for i in range(5)})
    assert scores["wrong"] == 5 and scores["correct"] == 0


def toy_public():
    combinations = list(product(range(2), repeat=3))
    sentences = [f"kas{s} tiv{v} moo{o}" for s, v, o in combinations] * 10
    parallel = [{"doc": "d", "sent": i, "translation": {"pred": f"verb{v}", "args": [
        {"role": "subject", "head": f"subj{s}"}, {"role": "object", "head": f"obj{o}"}]}}
        for i, (s, v, o) in enumerate(combinations)]
    return {"documents": [{"id": "d", "sentences": sentences}], "parallel": parallel, "level": "A"}


def test_a_matches_real_decoder_rows_and_votes_from_serialized_public():
    from babel.decipher.engine import Decipherer
    from babel.decipher.role_inference import ROLES, RoleInference

    class Capture(Decipherer):
        def _count_orders(self, aligned):
            super()._count_orders(aligned)
            self.rows = aligned

    public = json.loads(json.dumps(toy_public()))
    decoder = Capture(public, {"v16_gates": False}).fit()
    expected = []
    for item, row in decoder.rows:
        positions = {}
        for pos, _, _, roles in row:
            unique = set(roles)
            if len(unique) != 1:
                cues = {feature[2:] for unit, index in item["units"] if index == pos and unit.startswith("A:")
                        for feature in decoder.affix_fn.get(unit[2:], {}) if feature.startswith("R:")}
                unique = {role for role in unique if role.split(":")[-1] in cues}
            if len(unique) == 1:
                positions.setdefault(next(iter(unique)), pos)
        expected.append([positions[role] for role in ROLES] if all(role in positions for role in ROLES) else None)
    predicted = RoleInference(public).predict_all()
    assert [row["selected"] for row in predicted["A"]] == expected
    assert any(value is not None for value in expected)
    assert Counter({key: value for key, value in order_decision(predicted["A"])["votes"].items() if value}) == decoder.order_counts["verb_rel"]
    assert predicted == RoleInference(json.loads(json.dumps(public))).predict_all()


def test_soft_prior_of_one_matches_unconstrained_alignment_and_relaxes_only_pos_rejection():
    from babel.decipher.align import Aligner
    from babel.decipher.role_inference import WeightedAligner

    items = [{"units": [("S:king", 0), ("S:take", 1), ("A:S:past", 1)],
              "concepts": [("king", "head:subject"), ("take", "pred")], "feats": ["T:PAST"]}]
    def constraint(unit, concept):
        return not (unit == "S:king" and concept == "C:take")

    hard = Aligner(items, constraint=constraint)
    soft = WeightedAligner(items, constraint=constraint)
    assert ("S:king", "C:take") not in hard.links[0]
    assert soft.links[0][("S:king", "C:take")] > 0
    assert ("A:S:past", "C:take") not in soft.links[0]
    assert WeightedAligner(items, constraint=constraint, conflict_weight=1).links == Aligner(items).links


def test_adaptive_threshold_uses_distinct_documents_and_evidence_share():
    from babel.decipher.role_inference import _supports, evidence_threshold

    item = {"units": [("S:king", 0)], "concepts": [("king", "head:subject")]}
    support = _supports([item] * 4, [{"doc": "same"}] * 3 + [{"doc": "other"}])
    assert support[("S:king", "C:king")] == 2
    assert evidence_threshold(1, 0) == evidence_threshold(0, 3) == 0.25
    assert evidence_threshold(0.9, 1) == pytest.approx(0.19)
    assert evidence_threshold(0.9, 3) == pytest.approx(0.07)
    assert evidence_threshold(1, 30) == 0.05


def role_item(extra_subject=False):
    units = [("S:s", 0), ("S:v", 1), ("S:o", 2)]
    if extra_subject:
        units.append(("S:s", 3))
    item = {"units": units, "concepts": [("s", "head:subject"), ("v", "pred"), ("o", "head:object")]}
    links = {(unit, "C:" + unit[2:]): 0.9 for unit, _ in units}
    support = {key: 3 for key in links}
    shares = {key: 1.0 for key in links}
    return item, links, support, shares


def test_joint_selection_requires_distinct_positions_and_resolves_score_ties_by_positions():
    from babel.decipher.role_inference import rank_triples

    candidates = [{0: 1}, {0: 2, 1: 1, 2: 1}, {1: 1, 2: 1}]
    ranked = rank_triples(candidates)
    assert [row["positions"] for row in ranked] == [[0, 1, 2], [0, 2, 1]]
    assert rank_triples([{0: 1}, {0: 1}, {1: 1}]) == []


def test_e_abstains_for_insufficient_documents_and_competing_joint_hypotheses():
    from babel.decipher.role_inference import _prediction

    pair = {"doc": "d", "sent": 0}
    item, links, support, shares = role_item()
    assert _prediction("E", pair, item, links, {}, support, shares)["selected"] == [0, 1, 2]
    weak_support = {key: 2 for key in support}
    low = _prediction("E", pair, item, links, {}, weak_support, shares)
    assert low["selected"] is None and low["reason"] == "insufficient_document_support"
    item, links, support, shares = role_item(extra_subject=True)
    ambiguous = _prediction("E", pair, item, links, {}, support, shares)
    assert ambiguous["selected"] is None and ambiguous["reason"] == "insufficient_joint_share"
    assert len(ambiguous["alternatives"]) == 2


def test_common_budget_does_not_change_a_first_position_and_d_can_use_a_distinct_triple():
    from babel.decipher.role_inference import _prediction

    item, links, support, shares = role_item()
    item["units"] = [(f"S:s{i}", i) for i in range(7)] + [("S:v", 7), ("S:o", 8)]
    links.update({(f"S:s{i}", "C:s"): 0.3 if i == 0 else 0.9 for i in range(7)})
    baseline = _prediction("A", {"doc": "d", "sent": 0}, item, links, {}, support, shares)
    assert baseline["selected"] == [0, 7, 8]
    assert len(baseline["raw_candidates"][0]) == 7 and len(baseline["candidates"][0]) == 5
    assert 0 not in baseline["candidates"][0]
    item["units"] = [("S:s", 0), ("S:v", 0), ("S:v", 1), ("S:o", 2)]
    assert _prediction("A", {"doc": "d", "sent": 0}, item, links, {}, support, shares)["selected"] == [0, 0, 2]
    assert _prediction("D", {"doc": "d", "sent": 0}, item, links, {}, support, shares)["selected"] == [0, 1, 2]


def test_e_keeps_role_compatible_runner_up_concepts_but_requires_ambiguity_resolution():
    from babel.decipher.role_inference import _prediction

    item, links, support, shares = role_item()
    links[("S:s", "C:v")] = 0.95
    support[("S:s", "C:v")] = 3
    shares[("S:s", "C:v")] = 0.5
    pair = {"doc": "d", "sent": 0}
    assert _prediction("A", pair, item, links, {}, support, shares)["selected"] is None
    assert _prediction("E", pair, item, links, {}, support, shares)["selected"] == [0, 1, 2]
    item["concepts"].append(("s", "head:object"))
    assert _prediction("E", pair, item, links, {}, support, shares)["selected"] is None


def test_public_only_isolation_no_parallel_no_claims_and_inherited_input_defenses():
    from babel.decipher.role_inference import METHODS, RoleInference
    from babel.experiments.scorecard import check_import_isolation

    assert check_import_isolation() == []
    empty = {"documents": [{"id": "d", "sentences": ["ka ta mo"]}], "level": "C"}
    assert RoleInference(empty).predict_all() == {method: [] for method in METHODS}
    public = toy_public()
    baseline = RoleInference(public).predict_all()
    public["parallel"] *= 2
    assert RoleInference(public).predict_all() == baseline
    public["parallel"].append(dict(public["parallel"][0], translation={"pred": "different"}))
    with pytest.raises(ValueError, match="異なる対訳"):
        RoleInference(public).predict_all()
    heldout = dict(toy_public(), split={"heldout": ["d"]})
    with pytest.raises(ValueError, match="学習用"):
        RoleInference(heldout).predict_all()


def test_heldout_translation_cannot_change_learning_or_document_support():
    from babel.decipher.role_inference import RoleInference

    public = toy_public()
    public["documents"].append({"id": "h", "sentences": ["kas0 tiv0 moo0"]})
    public["split"] = {"train": ["d"], "heldout": ["h"]}
    expected = RoleInference(json.loads(json.dumps(public))).predict_all()
    public["parallel"].append({"doc": "h", "sent": 0, "translation": {
        "pred": "subj0", "args": [{"role": "subject", "head": "verb0"}, {"role": "object", "head": "new"}]}})
    assert RoleInference(public).predict_all() == expected
