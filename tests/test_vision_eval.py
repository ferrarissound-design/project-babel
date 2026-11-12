"""正解転写・画像認識の評価器・認識方式・アノテーション UI の回帰テスト（外部APIなし、画像は合成）。"""
import ast
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from babel.vision import groundtruth as G
from babel.vision.evaluate import UNMEASURABLE, adjusted_rand, evaluate, match, nmi

ROOT = Path(__file__).parents[1]


def _gt(items, lines=None):
    return G.validate({"schema": G.SCHEMA, "image": "x.png", "lines": lines or [], "items": items})


def _it(i, box, label="A", verified=True, kind="glyph", line=None, **kw):
    return {"id": f"g{i}", "box": box, "label": label, "verified": verified, "kind": kind, "line": line, **kw}


def _report(boxes_classes, rows=None):
    glyphs = [{"box": b, "class_id": c} for b, c in boxes_classes]
    rows = rows or [{"box": [0, 0, 1000, 1000], "classes": [c for _, c in boxes_classes],
                     "glyph_indices": list(range(len(glyphs)))}]
    return {"rows": rows, "glyphs": glyphs}


# ------------------------------------------------------------------------------------------------ 分離

def test_vision_package_does_not_import_the_evaluation_data():
    bad = []
    for p in sorted((ROOT / "babel" / "vision").glob("*.py")):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            bad += [f"{p.name}: {m}" for m in mods if "vision_eval" in m or "structure_eval" in m or "generator" in m]
    assert bad == []


# ------------------------------------------------------------------------------------------------ 正解の形式

def test_groundtruth_validation_and_draft_is_unverified():
    with pytest.raises(G.GroundTruthError):
        _gt([_it(1, [5, 5, 3, 9])])                               # 右下が左上より小さい
    with pytest.raises(G.GroundTruthError):
        _gt([_it(1, [0, 0, 5, 5], kind="ink")])
    with pytest.raises(G.GroundTruthError):
        _gt([_it(1, [0, 0, 5, 5], line="L9")])
    with pytest.raises(G.GroundTruthError):
        _gt([_it(1, [0, 0, 5, 5], same_as="Z")])                  # 統合先の label がない
    rep = {"source_image": "p.png", "settings": {"similarity": 0.7},
           "rows": [{"box": [0, 0, 30, 10], "classes": [3, 4], "glyph_indices": [0, 1]}],
           "glyphs": [{"box": [0, 0, 8, 10], "class_id": 3, "match_iou": 1.0}, {"box": [10, 0, 18, 10], "class_id": 4, "match_iou": 0.8}]}
    d = G.draft_from_report(rep)
    assert all(not i["verified"] and i["label"] is None for i in d["items"])      # AI の分類を正解にしない
    assert [i["proposed"]["class"] for i in d["items"]] == [3, 4]
    assert G.status(d)["verified_glyphs"] == 0


# ------------------------------------------------------------------------------------------------ 評価器

def test_unmeasurable_without_verified_glyphs():
    gt = _gt([_it(1, [0, 0, 10, 10], verified=False)])
    ev = evaluate(_report([([0, 0, 10, 10], 1)]), gt)
    assert ev["status"] == UNMEASURABLE and "detection" not in ev


def test_perfect_prediction_and_class_numbers_are_arbitrary():
    gt = _gt([_it(1, [0, 0, 10, 10], "A"), _it(2, [20, 0, 30, 10], "B"), _it(3, [40, 0, 50, 10], "A")])
    ev = evaluate(_report([([0, 0, 10, 10], 7), ([20, 0, 30, 10], 2), ([40, 0, 50, 10], 7)]), gt)
    assert ev["detection"]["f1"] == 1.0
    assert ev["classes"]["ari"] == 1.0 and ev["classes"]["nmi"] == 1.0


def test_one_ground_truth_is_matched_by_at_most_one_prediction():
    gt = _gt([_it(1, [0, 0, 10, 10])])
    ev = evaluate(_report([([0, 0, 10, 10], 1), ([0, 0, 10, 9], 1), ([1, 0, 10, 10], 1)]), gt)
    assert ev["detection"]["tp"] == 1 and ev["detection"]["fp"] == 2
    assert len(match([[0, 0, 10, 10]], [[0, 0, 10, 10], [0, 0, 10, 10]])) == 1


def test_split_merge_oversplit_and_mismerge_are_counted():
    gt = _gt([_it(1, [0, 0, 10, 10], "A"), _it(2, [20, 0, 30, 10], "B"), _it(3, [32, 0, 42, 10], "C"),
              _it(4, [60, 0, 70, 10], "A"), _it(5, [80, 0, 90, 10], "B")])
    rep = _report([([0, 0, 5, 10], 1), ([5, 0, 10, 10], 2),           # A が2つに分割
                   ([20, 0, 42, 10], 3),                              # B と C が結合
                   ([60, 0, 70, 10], 4),                              # A が別のクラス（過剰分割）
                   ([80, 0, 90, 10], 4)])                             # B が A と同じクラス（誤統合）
    ev = evaluate(rep, gt)
    assert ev["segmentation_errors"]["split_glyphs"] == 1
    assert ev["segmentation_errors"]["merged_predictions"] == 1
    assert ev["classes"]["mismerged_predicted_classes"] == 1


def test_non_glyph_items_and_lines():
    lines = [{"id": "L1", "box": [0, 0, 50, 10], "verified": True, "continues_sentence": True}]
    gt = _gt([_it(1, [0, 0, 10, 10], line="L1"), {"id": "r", "box": [0, 20, 100, 21], "kind": "rule", "verified": True}], lines)
    rep = {"rows": [{"box": [0, 0, 50, 10], "classes": [1], "glyph_indices": [0]},
                    {"box": [0, 20, 100, 21], "classes": [2], "glyph_indices": [1]}],
           "glyphs": [{"box": [0, 0, 10, 10], "class_id": 1}, {"box": [0, 20, 100, 21], "class_id": 2}]}
    ev = evaluate(rep, gt)
    assert ev["detection"]["non_glyph_detected_as_glyph"] == 1
    assert ev["lines"]["recall"] == 1.0 and ev["lines"]["precision"] == 0.5
    assert ev["lines"]["line_breaks_not_sentence_ends"] == 1
    assert ev["lines"]["glyph_line_assignment_accuracy"] == 1.0


def test_ari_and_nmi_known_values():
    assert adjusted_rand([0, 0, 1, 1], [5, 5, 9, 9]) == 1.0
    assert abs(adjusted_rand([0, 0, 1, 1], [0, 1, 0, 1]) - (-0.5)) < 1e-9
    assert nmi([0, 0, 1, 1], [0, 0, 0, 0]) == 0.0


def test_same_as_merges_aliases_in_class_evaluation():
    gt = _gt([_it(1, [0, 0, 10, 10], "A"), _it(2, [20, 0, 30, 10], "A2", same_as="A"), _it(3, [40, 0, 50, 10], "B")])
    ev = evaluate(_report([([0, 0, 10, 10], 1), ([20, 0, 30, 10], 1), ([40, 0, 50, 10], 2)]), gt)
    assert ev["classes"]["true_classes"] == 2 and ev["classes"]["ari"] == 1.0


# ------------------------------------------------------------------------------------------------ 人工画像と認識方式

cv2 = pytest.importorskip("cv2")


def _case(tmp_path, style, cond, seed=0):
    from babel.vision_eval.synth import render, text_for
    lines, alpha, _ = text_for(seed, 6)
    img, gt = render(lines, style, cond, seed, n_classes=len(alpha))
    p = tmp_path / f"{style}-{cond}.png"
    cv2.imwrite(str(p), img)
    return p, G.validate(gt)


def test_synthetic_ground_truth_is_complete_and_separate(tmp_path):
    p, gt = _case(tmp_path, "stroke", "rules")
    st = G.status(gt)
    assert st["glyphs"] == st["verified_glyphs"] > 0 and st["non_glyph"] >= 2
    assert not list(tmp_path.glob("*.json"))                      # 画像の隣に正解を書かない


def test_baseline_collapses_on_noise_but_improved_does_not(tmp_path):
    """開発中に見つかった既存方式の弱点（雑音で全体が1つの塊になる）の回帰テスト。"""
    from babel.vision.recognizers import recognize
    p, gt = _case(tmp_path, "font", "noise")
    base = evaluate(recognize(p, "baseline"), gt)["detection"]["f1"]
    imp = evaluate(recognize(p, "improved"), gt)["detection"]["f1"]
    assert base < 0.2 and imp > 0.9


@pytest.mark.parametrize("method", ["cc+iou", "cc+template", "cc+shape", "cc+combined", "proj+combined", "improved"])
def test_every_method_returns_a_valid_report(tmp_path, method):
    from babel.structure.corpus import from_glyph_report
    from babel.vision.recognizers import recognize
    p, gt = _case(tmp_path, "font", "clean")
    rep = recognize(p, method)
    ev = evaluate(rep, gt)
    assert ev["detection"]["f1"] > 0.9
    c = from_glyph_report(rep)                                     # 構造解析にそのまま渡せる
    assert c.n_symbols == len(rep["glyphs"])


def test_cli_vision_method_and_eval(tmp_path, capsys):
    from babel.__main__ import main
    p, gt = _case(tmp_path, "font", "clean")
    G.save(gt, tmp_path / "gt" / "page.json")
    assert main(["vision", str(p), "--out-dir", str(tmp_path / "v"), "--method", "improved"]) == 0
    assert (tmp_path / "v" / "corpus-continuous.json").exists()
    assert main(["vision-eval", str(tmp_path / "v" / "glyph-report.json"), str(tmp_path / "gt" / "page.json")]) == 0
    assert '"f1": 1.0' in capsys.readouterr().out


# ------------------------------------------------------------------------------------------------ アノテーション UI

def test_annotation_server_validates_before_saving(tmp_path):
    from http.server import ThreadingHTTPServer

    from babel.vision.annotate import make_handler, prepare
    p, _ = _case(tmp_path, "font", "clean")
    gtp = tmp_path / "ann.json"
    prepare(p, gtp)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(p, gtp))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        assert b"babel-glyph-gt-v1" in urllib.request.urlopen(base + "/").read()
        assert urllib.request.urlopen(base + "/image").read()[:4] == b"\x89PNG"
        bad = json.dumps({"schema": "x"}).encode()
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(base + "/gt", data=bad, method="POST"))
        assert e.value.code == 400
        good = {"schema": G.SCHEMA, "image": p.name, "lines": [],
                "items": [{"id": "h1", "box": [1, 1, 5, 5], "label": "A", "verified": True}]}
        r = urllib.request.urlopen(urllib.request.Request(base + "/gt", data=json.dumps(good).encode(), method="POST"))
        assert "保存した" in r.read().decode()
        assert G.load(gtp)["items"][0]["label"] == "A"
    finally:
        srv.shutdown()


# ------------------------------------------------------------------------------------------------ 構造解析への影響の実験

def test_impact_corruption_keeps_the_mapping_to_true_symbols():
    import random

    from babel.vision_eval.impact import _corrupt
    lines = [list("SabcE"), list("SdefE")]
    out, rename, origin = _corrupt(lines, "split", random.Random(0))
    assert all(rename[s] == s.rstrip("'") for ln in out for s in ln)
    out, rename, origin = _corrupt(lines, "seg", random.Random(3))
    assert all(len(a) == len(b) for a, b in zip(out, origin))
    assert all(rename[s] is None for ln in out for s in ln if "~" in s or "+" in s)


# --- Codex レビュー（PR #10）の指摘の回帰テスト ---

def test_predictions_on_unverified_items_are_not_false_positives():
    gt = _gt([_it(1, [0, 0, 10, 10], "A"), _it(2, [20, 0, 30, 10], "B", verified=False)])
    ev = evaluate(_report([([0, 0, 10, 10], 1), ([20, 0, 30, 10], 2)]), gt)
    d = ev["detection"]
    assert d["fp"] == 0 and d["precision"] == 1.0 and d["f1"] == 1.0
    assert d["ignored_predictions_on_unverified"] == 1


def test_hollow_frame_does_not_absorb_glyph_errors_inside():
    gt = _gt([_it(1, [10, 10, 20, 20], "A"),
              {"id": "f", "box": [1, 1, 99, 99], "kind": "decoration", "verified": True, "outline": True}])
    ev = evaluate(_report([([10, 10, 20, 20], 1), ([40, 40, 45, 45], 2), ([1, 30, 3, 40], 3)]), gt)
    assert ev["detection"]["non_glyph_detected_as_glyph"] == 1     # 縁の上の1つだけ。枠の内部の破片は数えない


def test_substitution_collapsed_by_renaming_is_not_a_discovery():
    """クラスの分けすぎ（v と v'）を読み替えた v, v を、接尾辞の置換可能性の発見に数えない（Codex レビュー、PR #10）。"""
    from babel.structure_eval.score import match_gold
    rule = {"family": "substitution", "symbols": ["x", "q", "v"], "label": "x/q/v"}
    assert not match_gold(rule, {"family": "substitution", "symbols": ["v", "v"]})
    assert match_gold(rule, {"family": "substitution", "symbols": ["x", "v"]})


def test_unverified_hollow_frame_only_ignores_its_border():
    """未確認の中空の枠でも、除外するのは縁の上の予測だけ（Codex レビュー、PR #10）。"""
    for verified in (False, True):
        gt = _gt([_it(1, [10, 10, 20, 20], "A"),
                  {"id": "f", "box": [1, 1, 99, 99], "kind": "decoration", "verified": verified, "outline": True}])
        d = evaluate(_report([([10, 10, 20, 20], 1), ([40, 40, 45, 45], 2)]), gt)["detection"]
        assert d["fp"] == 1 and d["f1"] == 0.6667, verified       # 確認の有無で結果が変わらない


# --- Codex レビュー（PR #10 マージ後）の指摘2件の回帰テスト ---

def _abab(extra=()):
    items = [_it(i, [20 * i, 0, 20 * i + 10, 10], lab) for i, lab in enumerate("ABAB", 1)] + list(extra)
    rep = _report([([20 * i, 0, 20 * i + 10, 10], c) for i, c in zip(range(1, 5), (1, 2, 1, 2))])
    return _gt(items), rep


def test_unverified_same_as_does_not_merge_verified_classes():
    gt, rep = _abab()
    base = evaluate(rep, gt)["classes"]
    gt2, _ = _abab([_it(9, [200, 0, 210, 10], "B", verified=False, same_as="A")])
    ev = evaluate(rep, gt2)
    assert ev["classes"]["ari"] == base["ari"] == 1.0 and ev["classes"]["nmi"] == base["nmi"] == 1.0
    assert ev["evaluation_scope"]["unverified_same_as_ignored"] == 1
    gt3, _ = _abab([_it(9, [200, 0, 210, 10], "B", same_as="A")])     # 確認済みの統合は使う
    assert evaluate(rep, gt3)["classes"]["true_classes"] == 1


def test_alias_cycles_and_conflicts_are_resolved_deterministically():
    items = [_it(1, [0, 0, 10, 10], "A"), _it(2, [20, 0, 30, 10], "B", same_as="A"), _it(3, [40, 0, 50, 10], "A", same_as="B"),
             _it(4, [60, 0, 70, 10], "C"), _it(5, [80, 0, 90, 10], "D", same_as="C"), _it(6, [100, 0, 110, 10], "D", same_as="A")]
    ar = G.alias_resolution(_gt(items))
    assert ar["map"]["B"] == "A" and ar["map"]["C"] == "A" and ar["map"]["D"] == "A"   # 循環も矛盾も1つにまとまる
    assert "A" in ar["cycles"] and ar["conflicts"] == [{"label": "D", "same_as": ["A", "C"]}]


def test_unverified_line_does_not_lower_line_scores():
    lines = [{"id": "L1", "box": [0, 0, 30, 10], "verified": True}, {"id": "L2", "box": [0, 30, 30, 40], "verified": False}]
    gt = _gt([_it(1, [0, 0, 10, 10], "A", line="L1"), _it(2, [0, 30, 10, 40], "B", line="L2")], lines)
    rep = {"rows": [{"box": [0, 0, 30, 10], "classes": [1], "glyph_indices": [0]},
                    {"box": [0, 30, 30, 40], "classes": [2], "glyph_indices": [1]}],
           "glyphs": [{"box": [0, 0, 10, 10], "class_id": 1}, {"box": [0, 30, 10, 40], "class_id": 2}]}
    ln = evaluate(rep, gt)["lines"]
    assert ln["precision"] == 1.0 and ln["f1"] == 1.0 and ln["glyph_line_assignment_accuracy"] == 1.0
    assert ln["ignored_rows_on_unverified_lines"] == 1
    # 未確認の領域の外にある余分な行は、偽陽性のまま
    rep["rows"].append({"box": [0, 60, 30, 70], "classes": [3], "glyph_indices": [2]})
    rep["glyphs"].append({"box": [0, 60, 10, 70], "class_id": 3})
    assert evaluate(rep, gt)["lines"]["precision"] == 0.5


def test_line_precision_unmeasurable_when_all_rows_fall_on_unverified_lines():
    lines = [{"id": "L1", "box": [0, 0, 30, 10], "verified": True}, {"id": "L2", "box": [0, 30, 30, 40], "verified": False}]
    gt = _gt([_it(1, [0, 0, 10, 10], "A", line="L1")], lines)
    rep = {"rows": [{"box": [0, 30, 30, 40], "classes": [2], "glyph_indices": [0]}], "glyphs": [{"box": [0, 30, 10, 40], "class_id": 2}]}
    ln = evaluate(rep, gt)["lines"]
    assert ln["precision"] == UNMEASURABLE and ln["recall"] == 0.0
