import copy
import hashlib
import json
from functools import lru_cache

import pytest

from babel.experiments import role_recovery as protocol
from babel.experiments import role_recovery_report as report
from babel.experiments.role_recovery_score import order_decision, score_predictions


@lru_cache(maxsize=None)
def _metric_template(correct=1, wrong=4, retained=2, empty=False, scattered=False):
    rows, gold = [], {}
    for i in range(0 if empty else 20):
        if i < correct:
            selected = [0, 1, 2]
        elif i < correct + wrong:
            selected = ([1, 0, 2], [0, 1, 3], [0, 2, 1])[i % 3] if scattered else [0, 1, 3]
        else:
            selected = None
        positions = [0, 1, 2] if i < retained else selected or [0, 1, 3]
        pools = [{position: 0.9} for position in positions]
        rows.append({"doc": "d", "sent": i, "source_full": True, "selected": selected,
                     "raw_candidates": pools, "candidates": pools, "confidence": 0.1})
        gold["d", i] = [(0, 1, 2)]
    metrics = score_predictions(rows, gold)
    decision = order_decision(rows)
    metrics["order"] = dict(decision, gold="medial", judge=(
        "abstain" if decision["value"] is None else "correct" if decision["value"] == "medial" else "wrong"))
    return metrics


def metric(**kwargs):
    return copy.deepcopy(_metric_template(**kwargs))


def _timings():
    return {"preprocess_seconds": 0.1, "hard_fit_seconds": 0.2, "soft_fit_seconds": 0.3,
            "selection_seconds": {method: 0.01 for method in report.METHODS}}


def synthetic_inputs(phase="dev", improve=True):
    manifest = {"membership_sha256": "1" * 64, "source_sha256": "2" * 64,
                "article_phase": {"n-dev": "dev", "n-final": "final"},
                "sentence_article": {"dev-sentence": "n-dev", "final-sentence": "n-final"}}
    real = {"phase": phase, "seeds": [0, 1, 2], "split_sha256": manifest["membership_sha256"], "rows": []}
    for code in (*report.KNOWN, "gl_pud"):
        for seed in range(3):
            for mismatch in (False, True):
                methods = {method: metric(correct=0, wrong=0, retained=0) if mismatch else metric()
                           for method in report.METHODS}
                if improve and not mismatch:
                    methods["B"] = metric(correct=2, wrong=4, retained=3)
                real["rows"].append({"code": code, "seed": seed, "phase": phase, "mismatch": mismatch,
                                     "split_sha256": manifest["membership_sha256"], "source_sha256": manifest["source_sha256"],
                                     "target_sha256": hashlib.sha256(code.encode()).hexdigest(), "public_sha256": "3" * 64,
                                     "predictions_sha256": "4" * 64, "sentence_ids": [phase + "-sentence"],
                                     "methods": methods, "timings": _timings()})
    seeds = [0, 1, 2] if phase == "dev" else [600, 601, 602]
    internal = {"phase": phase, "seeds": seeds, "n_docs": 65, "rows": []}
    for family in ("KES", "UMB", "SOT"):
        for seed in seeds:
            for condition in report.CONDITIONS:
                no_evidence = condition == "C-no-evidence"
                mismatch = condition.endswith("shift1")
                methods = {}
                for method in report.METHODS:
                    value = (metric(empty=True) if no_evidence else metric(correct=0, wrong=0, retained=0) if mismatch
                             else metric(correct=2, wrong=1, retained=3) if method == "B" and improve
                             else metric(correct=1, wrong=1, retained=2))
                    order = value.pop("order")
                    methods[method] = {"scores": value, "order": order, "order_judge": order["judge"]}
                internal["rows"].append({"family": family, "seed": seed, "condition": condition, "phase": phase,
                                         "n_docs": 65, "no_evidence": no_evidence, "mismatch": mismatch,
                                         "input_sha256": "5" * 64, "prediction_sha256": "6" * 64,
                                         "methods": methods, "timings": _timings()})
    report.validate_real(real, phase, manifest)
    report.validate_internal(internal, phase)
    return real, internal, manifest


def test_valid_full_grid_selects_only_method_passing_all_fixed_criteria():
    real, internal, _ = synthetic_inputs()
    selection = report.evaluate_methods(real, internal)
    assert selection["candidate"] == "B"
    assert selection["methods"]["B"]["passed"]
    assert all(not selection["methods"][method]["passed"] for method in "CDE")
    assert report.evaluate_methods(dict(real, rows=list(reversed(real["rows"]))), internal) == selection


@pytest.mark.parametrize("collapse", [False, True])
def test_candidate_inflation_or_lower_errors_with_lost_coverage_do_not_pass(collapse):
    real, internal, _ = synthetic_inputs()
    for row in real["rows"]:
        if not row["mismatch"]:
            row["methods"]["B"] = metric(correct=0, wrong=0, retained=5) if collapse else metric(retained=5)
    selection = report.evaluate_methods(real, internal)
    assert selection["candidate"] is None
    assert all(not value["passed"] for value in selection["methods"].values())
    assert not selection["methods"]["B"]["seeds"]["0"]["checks"]["correct_plus_three"]


def test_one_bad_cipher_seed_cannot_be_hidden_by_pooling_better_seeds():
    real, internal, _ = synthetic_inputs()
    for row in real["rows"]:
        if not row["mismatch"]:
            row["methods"]["B"] = metric(correct=2, wrong=5, retained=3) if row["seed"] == 2 else metric(correct=10, wrong=0, retained=11)
    selection = report.evaluate_methods(real, internal)
    assert selection["candidate"] is None
    assert selection["methods"]["B"]["seeds"]["0"]["passed"]
    assert not selection["methods"]["B"]["seeds"]["2"]["checks"]["wrong_not_higher"]


def test_mismatch_order_claim_vetoes_a_method_even_with_equal_mismatch_output_counts():
    real, internal, _ = synthetic_inputs()
    for row in real["rows"]:
        if row["mismatch"]:
            row["methods"]["A"] = metric(correct=0, wrong=5, retained=0, scattered=True)
            row["methods"]["B"] = metric(correct=0, wrong=5, retained=0)
    selection = report.evaluate_methods(real, internal)
    checks = selection["methods"]["B"]["seeds"]["0"]["checks"]
    assert checks["mismatch_claims_not_higher"] and not checks["mismatch_order_claims_zero"]
    assert selection["candidate"] is None


def test_internal_regression_or_no_evidence_claims_veto_real_data_improvement():
    real, internal, _ = synthetic_inputs()
    internal["rows"][0]["methods"]["B"]["scores"] = metric(correct=0, wrong=20, retained=0)
    assert report.evaluate_methods(real, internal)["candidate"] is None
    real, internal, _ = synthetic_inputs()
    row = next(row for row in internal["rows"] if row["no_evidence"])
    row["methods"]["B"]["scores"]["unsupported_claimed"] = 1
    assert report.evaluate_methods(real, internal)["candidate"] is None


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_duplicate_real_and_internal_cases_are_rejected(duplicate):
    real, internal, manifest = synthetic_inputs()
    if duplicate:
        real["rows"][-1] = copy.deepcopy(real["rows"][0])
        internal["rows"][-1] = copy.deepcopy(internal["rows"][0])
    else:
        real["rows"].pop()
        internal["rows"].pop()
    with pytest.raises(ValueError):
        report.validate_real(real, "dev", manifest)
    with pytest.raises(ValueError):
        report.validate_internal(internal, "dev")


def test_article_boundary_and_nonconserving_metrics_are_rejected():
    real, internal, manifest = synthetic_inputs()
    real["rows"][0]["sentence_ids"] = ["final-sentence"]
    with pytest.raises(ValueError, match="boundary"):
        report.validate_real(real, "dev", manifest)
    real, _, manifest = synthetic_inputs()
    real["rows"][0]["methods"]["A"]["correct"] += 1
    with pytest.raises(ValueError, match="conserve"):
        report.validate_real(real, "dev", manifest)
    internal["rows"][0]["methods"]["B"]["scores"]["eligible"] = True
    with pytest.raises(ValueError, match="integer"):
        report.validate_internal(internal, "dev")


def test_a_final_winner_cannot_replace_an_absent_frozen_development_candidate():
    dev, internal_dev, manifest = synthetic_inputs(improve=False)
    final, internal_final, _ = synthetic_inputs("final", improve=True)
    selection = report.evaluate_methods(dev, internal_dev)
    assert selection["candidate"] is None and report.evaluate_methods(final, internal_final)["candidate"] == "B"
    frozen = {"selection": selection, "freeze_sha256": "7" * 64}
    regression = {"cases": 27, "all_predictions_equal": True, "all_scores_equal": True}
    text = report.render_report(dev, final, internal_dev, internal_final, regression, frozen, manifest)
    assert "標準方式Aを維持する" in text
    assert "候補は **なし**" in text
    assert "**未実施**" in text


def _temporary_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "OUT", tmp_path)
    monkeypatch.setattr(protocol, "source_fingerprint", lambda: {"algorithm.py": "a" * 64})
    monkeypatch.setattr(protocol.subprocess, "check_output", lambda *args, **kwargs: "test-commit\n")
    for name in ("dev.json", "internal-dev.json"):
        (tmp_path / name).write_text("{}\n", encoding="utf-8")
    return protocol.freeze_selection({"candidate": None})


def test_freeze_rejects_changed_selection_code_or_development_results(tmp_path, monkeypatch):
    payload = _temporary_freeze(tmp_path, monkeypatch)
    assert protocol.check_freeze() == payload
    with pytest.raises(ValueError, match="already exists"):
        protocol.freeze_selection({"candidate": "E"})
    monkeypatch.setattr(protocol, "source_fingerprint", lambda: {"algorithm.py": "b" * 64})
    with pytest.raises(ValueError, match="changed"):
        protocol.check_freeze()
    monkeypatch.setattr(protocol, "source_fingerprint", lambda: {"algorithm.py": "a" * 64})
    (tmp_path / "dev.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        protocol.check_freeze()


def test_editing_only_the_frozen_selection_is_detected_by_its_self_hash(tmp_path, monkeypatch):
    payload = _temporary_freeze(tmp_path, monkeypatch)
    payload["selection"]["candidate"] = "E"
    (tmp_path / "development-freeze.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        protocol.check_freeze()
