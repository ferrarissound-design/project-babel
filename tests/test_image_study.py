"""実画像の評価パイプライン（babel.vision.study）の回帰テスト。実画像はないので合成画像で動作を確かめる。"""
import json

import pytest

cv2 = pytest.importorskip("cv2")

from babel.vision import groundtruth as G  # noqa: E402
from babel.vision import study as S  # noqa: E402

FAST = {"permutations": 199, "subsamples": 2, "perturbations": 1}


def _page(tmp_path, style="font", n=8):
    from babel.vision_eval.synth import render, text_for
    lines, alpha, _ = text_for(3, n)
    img, gt = render(lines, style, "clean", 3, n_classes=len(alpha))
    p = tmp_path / "page.png"
    cv2.imwrite(str(p), img)
    return p, G.validate(gt)


def test_prepare_refuses_missing_image(tmp_path):
    with pytest.raises(FileNotFoundError):
        S.prepare(tmp_path / "none.png", tmp_path / "study")


def test_prepare_makes_unverified_draft_and_never_overwrites_it(tmp_path):
    p, _ = _page(tmp_path)
    res = S.prepare(p, tmp_path / "study")
    gt = G.load(tmp_path / "study" / "gt.json")
    assert res["gt_created"] and all(not i["verified"] and i["label"] is None for i in gt["items"])
    assert all("improved" in i["proposed"] and "baseline" in i["proposed"] for i in gt["items"])
    for m in S.METHODS:
        assert (tmp_path / "study" / m / "glyph-report.json").exists()
    assert (tmp_path / "study" / "comparison.png").exists()
    gt["items"][0].update(label="X", verified=True)            # 人の作業
    G.save(gt, tmp_path / "study" / "gt.json")
    again = S.prepare(p, tmp_path / "study")
    assert not again["gt_created"] and G.load(tmp_path / "study" / "gt.json")["items"][0]["label"] == "X"


def test_without_verified_ground_truth_results_are_pending(tmp_path):
    p, _ = _page(tmp_path)
    S.prepare(p, tmp_path / "study")
    assert S.status(tmp_path / "study")["state"] == "not_started"
    ev = S.evaluate_study(tmp_path / "study")
    assert all(v["status"] == "測定不能" for v in ev["methods"].values())
    st = S.structure_study(tmp_path / "study", FAST)
    assert st["status"] == S.PENDING and "comparison" not in st


def test_complete_ground_truth_gives_measured_comparison(tmp_path):
    p, gt = _page(tmp_path)
    S.prepare(p, tmp_path / "study")
    G.save(gt, tmp_path / "study" / "gt.json")                 # 人がすべて確認した状態の代わり
    assert S.status(tmp_path / "study")["state"] == "complete"
    ev = S.evaluate_study(tmp_path / "study")
    assert ev["methods"]["improved"]["detection"]["f1"] > 0.9
    res = S.structure_study(tmp_path / "study", FAST)
    assert res["status"] == "measured"
    for k in ("B", "C", "C2"):
        c = res["comparison"][k]
        assert len(c["reproduced_from_A"]) + len(c["missed_from_A"]) == c["A_hypotheses"]
        assert c["line_split"]["verified_gt_lines"] == len(gt["lines"])
        for h in c["only_in_auto"]:
            assert len(h["symbols_as_gt"]) == len(h["symbols"])
    assert (tmp_path / "study" / "structure-comparison.md").exists()


def test_gt_report_uses_only_verified_labelled_glyphs(tmp_path):
    p, gt = _page(tmp_path)
    gt["items"][0]["verified"] = False
    gt["items"][1]["uncertain"] = True
    rep = S.gt_report(G.validate(gt))
    ids = {g["gt_id"] for g in rep["glyphs"]}
    assert gt["items"][0]["id"] not in ids and gt["items"][1]["id"] not in ids


def test_cli_image_study(tmp_path, capsys):
    from babel.__main__ import main
    p, _ = _page(tmp_path)
    assert main(["image-study", "prepare", str(tmp_path / "s"), "--image", str(p)]) == 0
    assert main(["image-study", "status", str(tmp_path / "s")]) == 0
    assert json.loads(capsys.readouterr().out.split("\n}\n")[-2] + "\n}")["state"] == "not_started"
