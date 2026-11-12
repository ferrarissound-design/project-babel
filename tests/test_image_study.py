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


def test_line_split_ignores_glyphs_on_unverified_gt_lines():
    """確認済みの記号でも、所属行が未確認なら行の分け方の診断に入れない（Codex レビュー、PR #11）。"""
    gt = G.validate({"schema": G.SCHEMA, "image_size": [200, 100], "lines": [
        {"id": "L1", "box": [0, 0, 200, 30], "verified": True},
        {"id": "L2", "box": [0, 50, 200, 80], "verified": False}],
        "items": [
        {"id": "a", "box": [10, 5, 30, 25], "line": "L1", "label": "A", "verified": True},
        {"id": "b", "box": [40, 5, 60, 25], "line": "L1", "label": "B", "verified": True},
        {"id": "c", "box": [10, 55, 30, 75], "line": "L2", "label": "A", "verified": True},
        {"id": "d", "box": [40, 55, 60, 75], "line": "L2", "label": "B", "verified": True}]})
    boxes = [[10, 5, 30, 25], [40, 5, 60, 25], [10, 55, 30, 75], [40, 55, 60, 75]]
    rep = {"glyphs": [{"box": b, "class_id": k % 2} for k, b in enumerate(boxes)],
           "rows": [{"box": [0, 0, 200, 30], "glyph_indices": [0, 1]},
                    {"box": [0, 50, 100, 80], "glyph_indices": [2]},          # 未確認の行が2つの予測行に分かれている
                    {"box": [35, 50, 200, 80], "glyph_indices": [3]}]}
    ls = S._line_split(rep, gt)
    assert ls["gt_lines_split_into_several_rows"] == 0 and ls["rows_spanning_several_gt_lines"] == 0
    gt["lines"][1]["verified"] = True                           # 確認済みにすれば分割として数える
    assert S._line_split(rep, gt)["gt_lines_split_into_several_rows"] == 1


def _snapshot(d):
    return {str(f.relative_to(d)): f.read_bytes() for f in sorted(d.rglob("*")) if f.is_file()}


def test_prepare_refuses_a_different_image_without_touching_the_study(tmp_path):
    """同じ study に別の画像を指定したら、何も上書きせずに止める（Codex レビュー、PR #11）。"""
    p, _ = _page(tmp_path)
    S.prepare(p, tmp_path / "study")
    other = tmp_path / "other"
    other.mkdir()
    q, _ = _page(other, style="stroke")
    before = _snapshot(tmp_path / "study")
    with pytest.raises(S.StudyImageMismatch):
        S.prepare(q, tmp_path / "study")
    assert _snapshot(tmp_path / "study") == before
    S.prepare(p, tmp_path / "study")                             # 同じ画像なら続けられる（gt.json は上書きしない）
    assert G.load(tmp_path / "study" / "gt.json")["image_sha256"] == S._study(tmp_path / "study")["image_sha256"]


def test_evaluate_and_structure_refuse_ground_truth_of_another_image(tmp_path):
    p, _ = _page(tmp_path)
    S.prepare(p, tmp_path / "study")
    g = json.loads((tmp_path / "study" / "gt.json").read_text(encoding="utf-8"))
    g["image_sha256"] = "0" * 64
    (tmp_path / "study" / "gt.json").write_text(json.dumps(g), encoding="utf-8")
    with pytest.raises(S.StudyImageMismatch):
        S.evaluate_study(tmp_path / "study")
    with pytest.raises(S.StudyImageMismatch):
        S.structure_study(tmp_path / "study", FAST)
    assert not (tmp_path / "study" / "evaluation.json").exists()
