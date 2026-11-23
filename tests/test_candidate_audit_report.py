import copy
import hashlib
import json
from pathlib import Path

import pytest

from babel.experiments import candidate_audit_report as report
from babel.experiments.candidate_audit import LABELS, STAGES
from babel.experiments.real_parallel import TARGETS


def measurements():
    rows = []
    counts = (2, 2, 2, 1, 1, 1, 1)
    losses = {stage: 0 for stage in (*STAGES, "retained")}
    losses.update(threshold=1, retained=1)
    for _, code in [*TARGETS, ("UD_Galician-PUD", "gl_pud")]:
        example = {"source_full": True, "kept_triples": 1, "eligible": True,
                   "first_loss": {label: "retained" for label in LABELS}, "selected": {}}
        rows.append({
            "language": code, "mismatch": False, "engine_version": "1.8.0", "ud_tag": "r2.15",
            "source_sha256": "a" * 64, "target_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "config": {"v16_gates": False},
            "totals": {"parallel": 4, "source_full": 3, "source_full_target_raw": 2, "eligible": 2},
            "stages": {stage: {label: count for label in LABELS} for stage, count in zip(STAGES, counts, strict=True)},
            "first_loss": {label: dict(losses) for label in LABELS},
            "clause_pool": {label: 1 for label in LABELS}, "votes": {"medial": 1},
            "examples": {sid: dict(example, sent_id=sid) for sid in report.EXAMPLE_IDS} if code == "ja_pud" else {},
        })
    shifted = copy.deepcopy(rows)
    for row in shifted:
        row["mismatch"] = True
    return rows, shifted


def test_complete_measurements_generate_deterministic_report_with_separate_galician_results():
    normal, shifted = measurements()
    correct, control = report.validate_inputs(normal, shifted)
    assert len(correct) == len(control) == 20
    text = report.render_report(normal, shifted)
    assert text == report.render_report(list(reversed(normal)), list(reversed(shifted)))
    assert "19/38（50.0%）" in text
    assert "ガリシア語（別集計）" in text
    assert all(sid in text for sid in report.EXAMPLE_IDS)
    assert "英語との訳義の一致や翻訳精度ではない" in text


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_duplicate_language_cannot_silently_change_the_denominator(duplicate):
    normal, shifted = measurements()
    if duplicate:
        shifted[-1] = copy.deepcopy(shifted[0])
    else:
        shifted.pop()
    with pytest.raises(ValueError, match="20 target languages"):
        report.render_report(normal, shifted)


@pytest.mark.parametrize("field,value", [
    ("source_sha256", "b" * 64),
    ("target_sha256", "b" * 64),
    ("config", {"v16_gates": False, "pos_constraint": False}),
    ("ud_tag", "r2.16"),
    ("mismatch", False),
])
def test_different_inputs_or_conditions_cannot_be_combined(field, value):
    normal, shifted = measurements()
    shifted[0][field] = value
    with pytest.raises(ValueError):
        report.render_report(normal, shifted)


def test_nonconserving_first_loss_and_increasing_candidate_counts_are_rejected():
    normal, shifted = measurements()
    normal[0]["first_loss"]["subject"]["threshold"] = 0
    with pytest.raises(ValueError, match="first losses"):
        report.validate_inputs(normal, shifted)
    normal, shifted = measurements()
    normal[0]["stages"]["best_concept"]["subject"] = 2
    with pytest.raises(ValueError, match="non-nested"):
        report.validate_inputs(normal, shifted)


def test_malformed_counts_hashes_and_missing_japanese_examples_are_rejected():
    normal, shifted = measurements()
    normal[0]["totals"]["eligible"] = True
    with pytest.raises(ValueError, match="nonnegative integer"):
        report.validate_inputs(normal, shifted)
    normal, shifted = measurements()
    normal[0]["target_sha256"] = "not-a-sha256"
    with pytest.raises(ValueError, match="invalid target_sha256"):
        report.validate_inputs(normal, shifted)
    normal, shifted = measurements()
    japanese = next(row for row in normal if row["language"] == "ja_pud")
    japanese["examples"].pop(report.EXAMPLE_IDS[0])
    with pytest.raises(ValueError, match="Missing or malformed"):
        report.validate_inputs(normal, shifted)


ROOT = Path(__file__).resolve().parents[1]


def _committed_rows(name):
    return [json.loads(line) for line in (ROOT / "results/real_parallel" / name).read_text(encoding="utf-8").splitlines()]


def test_committed_report_is_reproduced_from_all_40_conditions():
    normal = _committed_rows("candidate-audit.jsonl")
    shifted = _committed_rows("candidate-audit-mismatch.jsonl")
    assert report.render_report(normal, shifted) == (ROOT / "docs/CANDIDATE_AUDIT.md").read_text(encoding="utf-8")


def test_all_committed_conditions_agree_with_previous_clause_probe():
    previous = {(r["language"], r["mismatch"]): r for r in _committed_rows("clause-probe.jsonl")}
    rows = _committed_rows("candidate-audit.jsonl") + _committed_rows("candidate-audit-mismatch.jsonl")
    for row in rows:
        prior = previous[row["language"], row["mismatch"]]
        assert row["target_sha256"] == prior["sha256"]
        assert row["totals"]["eligible"] == prior["oracle"]["eligible"]
        for label in LABELS[:3]:
            assert row["clause_pool"][label] == prior["oracle"].get(label, 0)
        assert row["clause_pool"]["triple"] == prior["oracle"].get("available", 0)
        assert row["stages"]["first_position"]["triple"] == prior["scores"]["baseline"].get("exact", 0)
        assert sum(row["votes"].values()) == prior["scores"]["baseline"].get("votes", 0)
