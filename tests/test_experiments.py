from babel.experiments.run import conditions, ece, run_boundary, run_null, run_one


def test_run_one_reports_metrics_and_failures_honestly():
    ok = run_one(("E1_main", "levelA", "A", "engine", {"n_docs": 30}, "KES", 0))
    assert ok["ok"] and "seg_f1" in ok["metrics"] and ok["digest"]
    bad = run_one(("E1_main", "levelA", "A", "engine", {"n_docs": 30}, "NOPE", 0))
    assert bad["ok"] is False and "error" in bad      # 失敗は成功として記録されない


def test_conditions_cover_required_experiments():
    exps = {c[0] for c in conditions()}
    assert {"E1_main", "E2_baseline", "E3_data_size", "E4_parallel", "E5_noise", "E6_unseen", "E8_ablation"} <= exps


def test_ece():
    assert ece([(0.9, 1), (0.9, 1), (0.9, 0), (0.1, 0)])[0] is not None
    assert ece([])[0] is None


def test_null_and_boundary_runs():
    assert run_null(("random", 0))["ok"]
    r = run_boundary(("KES", 0))
    assert r["ok"] and 0 < r["metrics"]["word_boundary_f1_engine"] <= 1
