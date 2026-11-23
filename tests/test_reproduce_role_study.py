import pytest

from babel.experiments import reproduce_role_study as verification


def test_verifier_accepts_timing_differences_but_requires_all_other_fields(monkeypatch):
    original = {"predictions_sha256": "a" * 64, "methods": {"A": {"wrong": 3}}, "timings": {"fit": 1.0}}
    monkeypatch.setattr(verification, "run_job", lambda job: dict(original, timings={"fit": 99.0}))
    result = verification.compare_case((("ja_pud", "dev", 0, False), original))
    assert result["all_non_timing_fields_equal"]
    assert result["predictions_sha256"] == original["predictions_sha256"]


@pytest.mark.parametrize("changed", [{"predictions_sha256": "b" * 64}, {"methods": {"A": {"wrong": 4}}}])
def test_verifier_rejects_changed_predictions_or_scoring(monkeypatch, changed):
    original = {"predictions_sha256": "a" * 64, "methods": {"A": {"wrong": 3}}}
    monkeypatch.setattr(verification, "run_job", lambda job: dict(original, **changed))
    with pytest.raises(ValueError, match="reproduction failed"):
        verification.compare_case((("ja_pud", "final", 0, False), original))
