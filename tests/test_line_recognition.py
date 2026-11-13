"""行認識（babel/vision/lines.py、improved-v3）と、その評価の回帰テスト（外部APIなし、画像は合成）。"""
import hashlib
import json
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from babel.vision import lines as L  # noqa: E402
from babel.vision.evaluate import evaluate, line_errors  # noqa: E402
from babel.vision.groundtruth import validate  # noqa: E402
from babel.vision.recognizers import METHODS, binarize, recognize, segment_components  # noqa: E402
from babel.vision_eval import layout, lines_bench, synth  # noqa: E402

ROOT = Path(__file__).parents[1]


def _case(tmp_path, family, condition, seed, n_lines=None):
    lines, alpha, _ = synth.text_for(seed, n_lines or layout.n_lines_for(seed))
    img, gt = layout.render(lines, family, condition, seed, n_classes=len(alpha))
    p = tmp_path / f"{family}-{condition}-{seed}.png"
    cv2.imwrite(str(p), img)
    return p, validate(gt)


def _errors(path, gt, method):
    rep = recognize(path, method, regions=gt.get("regions"))
    return rep, evaluate(rep, gt)["lines"]["errors"]


# ------------------------------------------------------------------------------------------------ 既存方式を変えていない

def test_existing_methods_are_unchanged():
    """baseline・improved・improved-v2 の出力が、main（PR #11 の後）の出力と同じ。"""
    fx = json.loads((ROOT / "tests" / "data" / "legacy_recognizer_fingerprints.json").read_text(encoding="utf-8"))
    if fx["opencv"] != cv2.__version__:
        pytest.skip(f"指紋は OpenCV {fx['opencv']} のもの（この環境は {cv2.__version__}）。下の写しの一致のテストで確かめる")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for key, want in fx["cases"].items():
            style, cond, seed, m = key.split("/")
            lines, alpha, _ = synth.text_for(int(seed), 8)
            img, _ = synth.render(lines, style, cond, int(seed), n_classes=len(alpha))
            p = Path(td) / "x.png"
            cv2.imwrite(str(p), img)
            rep = recognize(p, m)
            payload = json.dumps([[r["glyph_indices"] for r in rep["rows"]], [[g["box"], g["class_id"]] for g in rep["glyphs"]]])
            assert hashlib.sha256(payload.encode()).hexdigest() == want["sha256"], key


def test_existing_method_definitions_are_kept():
    assert METHODS["improved"] == {"preprocess": "median", "segment": "components", "classify": "template"}
    assert METHODS["improved-v2"]["seg"] == {"join_width": 0.95, "join_width_gap": 0.5}
    assert METHODS["improved-v3"]["seg"]["line_method"] == "hybrid"
    assert {k: v for k, v in METHODS["improved-v3"]["seg"].items() if k.startswith("join")} == METHODS["improved-v2"]["seg"]


@pytest.mark.parametrize("family,condition,seed", [("stroke", "plain", 200), ("stack", "mixed", 201), ("font", "tight", 202)])
def test_center_grouping_is_a_faithful_copy_of_the_legacy_code(tmp_path, family, condition, seed):
    """lines.group_center（原因の切り分けに使う写し）が、segment_components の元の行のまとめ方と同じ行を返す。"""
    p, _ = _case(tmp_path, family, condition, seed)
    mask = binarize(cv2.imread(str(p), cv2.IMREAD_COLOR), "median")
    comps, _ = L.components(mask)
    legacy = segment_components(mask)
    copy = L.group_center(comps)
    assert len(copy) == len(legacy)
    for a, b in zip(copy, legacy):                           # 行ごとの成分の外接矩形が同じ
        assert [min(c[0] for c in a), min(c[1] for c in a), max(c[2] for c in a), max(c[3] for c in a)] == \
            [min(g[0] for g in b), min(g[1] for g in b), max(g[2] for g in b), max(g[3] for g in b)]


# ------------------------------------------------------------------------------------------------ 過剰分割の再現と修正

def test_oversplit_is_reproduced_by_v2_and_fixed_by_v3(tmp_path):
    """上下に離れた部品を持つ字: improved-v2 は上の部品と下の部品を別の行にし、improved-v3 は1つの行にまとめる。"""
    p, gt = _case(tmp_path, "stack", "plain", 200)
    rep2, e2 = _errors(p, gt, "improved-v2")
    rep3, e3 = _errors(p, gt, "improved-v3")
    assert len(rep2["rows"]) > len(gt["lines"]) and e2["oversplit_extra_rows"] >= len(gt["lines"]) // 2
    assert len(rep3["rows"]) == len(gt["lines"])
    assert e3["oversplit_extra_rows"] == 0 and e3["merged_rows"] == 0


def test_scale_is_less_pulled_down_by_small_parts(tmp_path):
    """原因: 現行方式の尺度（成分の高さの中央値）は、部品の多い字で字の高さよりずっと小さくなる。
    S（インク量で重みを付けた成分の大きさの中央値）は、それより字の大きさに近い（上下に分かれた字では字の高さの半分程度）。"""
    p, gt = _case(tmp_path, "stack", "diacritic", 201)
    comps, _ = L.components(binarize(cv2.imread(str(p), cv2.IMREAD_COLOR), "median"))
    hs = sorted(g["box"][3] - g["box"][1] for g in gt["items"] if g["kind"] == "glyph")
    h_true = hs[len(hs) // 2]
    assert L.median_height(comps) < 0.4 * h_true
    assert L.glyph_scale(comps) >= 1.4 * L.median_height(comps)
    p, gt = _case(tmp_path, "font", "plain", 203)                       # 部品に分かれない字では S は字の高さに近い
    comps, _ = L.components(binarize(cv2.imread(str(p), cv2.IMREAD_COLOR), "median"))
    hs = sorted(g["box"][3] - g["box"][1] for g in gt["items"] if g["kind"] == "glyph")
    assert 0.8 * hs[len(hs) // 2] <= L.glyph_scale(comps) <= 1.3 * hs[len(hs) // 2]


@pytest.mark.parametrize("family,condition,seed", [("font", "plain", 203), ("stroke", "varheight", 204), ("font", "diacritic", 205),
                                                     ("stroke", "jitter", 206), ("stroke", "uneven", 207), ("font", "rule_touch", 200),
                                                     ("stack", "two_column", 203)])
def test_v3_groups_rows_without_split_or_merge(tmp_path, family, condition, seed):
    p, gt = _case(tmp_path, family, condition, seed)
    rep, e = _errors(p, gt, "improved-v3")
    assert e["oversplit_extra_rows"] == 0 and e["merged_rows"] == 0
    assert len(rep["rows"]) == len(gt["lines"])


def test_adjacent_lines_with_overlapping_extents_are_not_merged(tmp_path):
    """副作用の確認: 上に伸びる字・下に伸びる字で隣の行と範囲が重なっても、別の行のままにする。"""
    p, gt = _case(tmp_path, "font", "tight", 203)
    boxes = sorted(ln["box"] for ln in gt["lines"])
    assert any(a[3] > b[1] for a, b in zip(sorted(boxes, key=lambda b: b[1]), sorted(boxes, key=lambda b: b[1])[1:]))  # 範囲が重なる
    _, e = _errors(p, gt, "improved-v3")
    assert e["merged_rows"] == 0


def test_two_separate_lines_of_identical_text_stay_separate(tmp_path):
    """同じ文が2行続いて字の列が完全にそろっていても、行の間が空いていれば別の行。"""
    img = np.full((120, 300), 255, np.uint8)
    for y in (40, 90):
        cv2.putText(img, "abcabc dab", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
    p = tmp_path / "same.png"
    cv2.imwrite(str(p), img)
    assert len(recognize(p, "improved-v3")["rows"]) == 2


def test_marks_between_lines_go_to_the_line_of_their_glyph(tmp_path):
    """字の上の点は、上の行ではなく、真下にある字の行に入る。"""
    img = np.full((140, 260), 255, np.uint8)
    for k in range(6):
        cv2.rectangle(img, (20 + 38 * k, 20), (44 + 38 * k, 50), 0, 2)        # 上の行
        cv2.rectangle(img, (20 + 38 * k, 92), (44 + 38 * k, 122), 0, 2)       # 下の行
    for k in (1, 3):
        cv2.circle(img, (32 + 38 * k, 82), 2, 0, -1)                           # 下の行の字の上の点（上の行の下端から 30、字から 8）
    p = tmp_path / "marks.png"
    cv2.imwrite(str(p), img)
    rep = recognize(p, "improved-v3")
    assert len(rep["rows"]) == 2
    assert rep["rows"][0]["box"][3] <= 55 and rep["rows"][1]["box"][1] < 88    # 点（上端 80 付近）は下の行の枠に含まれる


def test_empty_and_single_component_images():
    blank = np.zeros((40, 80), np.uint8)
    assert segment_components(blank, line_method="hybrid") == []
    one = blank.copy()
    one[10:30, 10:30] = 255
    assert len(segment_components(one, line_method="hybrid")) == 1
    for m in L.LINE_METHODS:
        assert len(segment_components(one, line_method=m)) == 1


# ------------------------------------------------------------------------------------------------ 行の部品

def test_crossing_profile_has_a_valley_between_lines():
    comps = [[10 * k, 0, 10 * k + 8, 20, 100] for k in range(5)] + [[10 * k, 30, 10 * k + 8, 50, 100] for k in range(5)]
    prof = L.crossing_profile(comps, 50)
    assert prof[10] == prof.max() and prof[25] == 0
    assert L._valley_cuts(prof, 0, 50, 0.4) == [25]


# ------------------------------------------------------------------------------------------------ 評価器

def _gt(lines, items):
    return validate({"schema": "babel-glyph-gt-v1", "image": "x.png",
                     "lines": [{"id": i, "box": b, "verified": True} for i, b in lines],
                     "items": [{"id": f"g{k}", "box": b, "line": ln, "label": "A", "verified": True} for k, (b, ln) in enumerate(items)]})


def _rep(rows):
    glyphs, out = [], []
    for boxes in rows:
        idx = list(range(len(glyphs), len(glyphs) + len(boxes)))
        glyphs += [{"box": b, "class_id": 0} for b in boxes]
        out.append({"box": [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)],
                    "classes": [0] * len(boxes), "glyph_indices": idx})
    return {"rows": out, "glyphs": glyphs}


def test_line_errors_count_split_and_merge_separately():
    gt = _gt([("L1", [0, 0, 50, 20]), ("L2", [0, 30, 50, 50])],
             [([0, 0, 20, 20], "L1"), ([30, 0, 50, 20], "L1"), ([0, 30, 20, 50], "L2"), ([30, 30, 50, 50], "L2")])
    ok = _rep([[[0, 0, 20, 20], [30, 0, 50, 20]], [[0, 30, 20, 50], [30, 30, 50, 50]]])
    assert line_errors(ok, gt) == {"oversplit_lines": 0, "oversplit_extra_rows": 0, "merged_rows": 0, "merged_extra_lines": 0,
                                   "rows_without_gt_glyph": 0, "lines_without_prediction": 0}
    # L1 の字の上半分と下半分が別の行になった（過剰分割）。L2 は正しい
    split = _rep([[[0, 0, 20, 9], [30, 0, 50, 9]], [[0, 11, 20, 20], [30, 11, 50, 20]], [[0, 30, 20, 50], [30, 30, 50, 50]]])
    e = line_errors(split, gt)
    assert (e["oversplit_lines"], e["oversplit_extra_rows"], e["merged_rows"]) == (1, 1, 0)
    merged = _rep([[[0, 0, 20, 20], [30, 0, 50, 20], [0, 30, 20, 50], [30, 30, 50, 50]]])
    e = line_errors(merged, gt)
    assert (e["oversplit_extra_rows"], e["merged_rows"], e["merged_extra_lines"]) == (0, 1, 1)
    assert evaluate(split, gt)["lines"]["errors"] == line_errors(split, gt)


def test_line_errors_count_a_prediction_covering_glyphs_of_two_lines():
    """上下の行の字が1つの記号に結合されると、予測の枠が2つの行を覆う。その行は誤結合として数える（Codex レビュー、PR #12）。"""
    gt = _gt([("L1", [0, 0, 50, 20]), ("L2", [0, 30, 50, 50])],
             [([0, 0, 20, 20], "L1"), ([30, 0, 50, 20], "L1"), ([0, 30, 20, 50], "L2"), ([30, 30, 50, 50], "L2")])
    e = line_errors(_rep([[[0, 0, 20, 50], [30, 0, 50, 50]]]), gt)             # 各列で上下を覆う記号からなる1行
    assert (e["merged_rows"], e["merged_extra_lines"], e["lines_without_prediction"]) == (1, 1, 0)
    # 隣の行の字に少し掛かるだけ（面積の半分未満）なら、誤結合にしない
    e = line_errors(_rep([[[0, 0, 20, 36], [30, 0, 50, 20]], [[0, 38, 20, 50], [30, 30, 50, 50]]]), gt)
    assert e["merged_rows"] == 0


def test_line_errors_ignore_unverified_lines():
    gt = _gt([("L1", [0, 0, 50, 20]), ("L2", [0, 30, 50, 50])], [([0, 0, 20, 20], "L1"), ([0, 30, 20, 50], "L2")])
    gt["lines"][1]["verified"] = False
    e = line_errors(_rep([[[0, 0, 20, 20]], [[0, 30, 20, 39]], [[0, 41, 20, 50]]]), gt)
    assert e["oversplit_extra_rows"] == 0 and e["lines_without_prediction"] == 0


# ------------------------------------------------------------------------------------------------ 評価データの分離

def test_layout_splits_do_not_share_seeds_or_holdout_families():
    seeds = [set(v) for v in layout.SPLITS.values()]
    assert all(not (a & b) for i, a in enumerate(seeds) for b in seeds[i + 1:])
    used = {s for v in synth.SPLITS.values() for s in v}                       # v1・v2 の評価で使ったシード
    assert not (used & {s for v in layout.SPLITS.values() for s in v})
    assert not (set(layout.FAMILIES["holdout"]) & (set(layout.FAMILIES["dev"]) | set(layout.FAMILIES["val"])))
    assert "photo" in layout.conditions_for("holdout") and "photo" not in layout.conditions_for("dev")
    from babel.vision_eval import impact, impact_v3
    assert not (set(impact_v3.SEEDS) & ({s for v in layout.SPLITS.values() for s in v} | used | set(impact.SEEDS)))


def test_holdout_cannot_be_used_for_selection():
    with pytest.raises(ValueError):
        lines_bench.run("holdout")                                             # --final なし
    with pytest.raises(ValueError):
        lines_bench.run("holdout", final=True, variants={"x": ("improved-v2", {"line_method": "hybrid"})})
    with pytest.raises(ValueError):
        lines_bench.sweep("holdout")
    with pytest.raises(ValueError):
        lines_bench.run("holdout", final=True, workdir="anywhere")             # holdout の画像をキャッシュに残さない
    from babel.vision_eval import lines_diagnose, lines_figure
    with pytest.raises(ValueError):
        lines_diagnose.diagnose("holdout")
    with pytest.raises(ValueError):
        lines_figure.build("holdout")


def test_layout_ground_truth_is_valid_and_varies_in_line_count(tmp_path):
    counts = set()
    for seed in layout.SPLITS["dev"][:4]:
        _, gt = _case(tmp_path, "stack", "mixed", seed)
        counts.add(len(gt["lines"]))
        assert all(i["line"] for i in gt["items"] if i["kind"] == "glyph")
        for ln in gt["lines"]:                                                 # 行の枠は、その行の記号の枠を囲む
            its = [i["box"] for i in gt["items"] if i["line"] == ln["id"]]
            assert ln["box"] == [min(b[0] for b in its), min(b[1] for b in its), max(b[2] for b in its), max(b[3] for b in its)]
    assert len(counts) > 1                                                     # 行数は画像ごとに違う（特定の行数に合わせない）
    _, gt = _case(tmp_path, "curve", "two_column", 7000)
    assert [r["id"] for r in gt["regions"]] == ["left", "right"]


def test_line_recognition_code_has_no_image_specific_constants():
    """行認識のコードに、実画像の行数（13・9・4）や画像名を埋め込まない。"""
    src = (ROOT / "babel" / "vision" / "lines.py").read_text(encoding="utf-8")
    assert "4494" not in src and "IMG_" not in src
    import ast
    nums = {n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, int)
            and not isinstance(n.value, bool)}
    assert not (nums & {13, 25, 17, 9})


def test_image_cache_is_not_reused_for_another_split(tmp_path):
    """キャッシュした画像を、別の分割・別の条件として使わない（Codex レビュー、PR #12）。"""
    kw = dict(families=("font",), conditions=("plain",), measure_memory=False, methods=["improved-v3"])
    lines_bench.run("dev", seeds=[200], workdir=tmp_path / "c", **kw)
    r = lines_bench.run("dev", seeds=[200], workdir=tmp_path / "c", **kw)      # 同じ条件なら再利用できる
    assert r["meta"]["images"] == 1 and all(x["split"] == "dev" for x in r["runs"])
    with pytest.raises(ValueError):
        lines_bench.run("val", seeds=[200], workdir=tmp_path / "c", **kw)      # 別の分割
    with pytest.raises(ValueError):
        lines_bench.run("dev", seeds=[201], workdir=tmp_path / "c", **kw)      # 別のシード
    with pytest.raises(ValueError):
        lines_bench.sweep("dev", workdir=tmp_path / "c")                       # 条件（字の作りなど）が違う
    # holdout の画像と、生成条件の記録のない cases.json を置いたディレクトリは、dev として使えない
    cases = layout.build(tmp_path / "h", "holdout", ("radical",), ("plain",), [7000])
    (tmp_path / "h" / "cases.json").write_text(json.dumps(cases), encoding="utf-8")
    with pytest.raises(ValueError):
        lines_bench.run("dev", workdir=tmp_path / "h", variants={"x": ("improved-v2", {"line_method": "hybrid"})}, methods=[])
    (tmp_path / "h" / "cases.json").unlink()
    with pytest.raises(ValueError):
        lines_bench.sweep("dev", workdir=tmp_path / "h")
