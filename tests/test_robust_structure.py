"""認識の誤りに強い構造検定（babel/robust）と、その評価（babel/robust_eval）のテスト。外部APIなし。"""
import ast
import json
import random
from pathlib import Path

import pytest

from babel.robust import compare, crossalign, rowsim
from babel.robust_eval import summary, texts

ROOT = Path(__file__).parents[1]
REAL = ROOT / "results" / "robust" / "real-img4494"


# ------------------------------------------------------------------------------------------------ 行の組の検定

def test_lcs_score_and_alignment():
    def w(x, y):
        return 1.0 if x == y else 0.0
    assert rowsim.lcs_score([1, 2, 3, 4], [1, 9, 3, 4], w) == 3
    assert rowsim.lcs_score([1, 2, 3], [4, 5, 6], w) == 0
    al = rowsim.lcs_alignment([1, 2, 3, 4], [1, 9, 9, 3, 4], w)
    assert [(a, b) for a, b, _ in al] == [(0, 0), (2, 3), (3, 4)]


def test_numpy_and_python_paths_agree():
    pytest.importorskip("numpy")
    rows, _, _ = texts.planted_text(3, vocab=40)
    a = rowsim.row_similarity(rows, cfg={"permutations": 199}, use_numpy=True)
    b = rowsim.row_similarity(rows, cfg={"permutations": 199}, use_numpy=False)
    assert a["pairs"] == b["pairs"]
    # 重みの行列（字形の類似度）でも同じ
    n = sum(len(r) for r in rows)
    flat = [x for r in rows for x in r]
    W = [[1.0 if flat[i] == flat[j] else 0.4 if (flat[i] + flat[j]) % 7 == 0 else 0.0 for j in range(n)] for i in range(n)]
    items, k = [], 0
    for r in rows:
        items.append(list(range(k, k + len(r))))
        k += len(r)
    a = rowsim.row_similarity(items, weight_matrix=W, cfg={"permutations": 99}, use_numpy=True)
    b = rowsim.row_similarity(items, weight_matrix=W, cfg={"permutations": 99}, use_numpy=False)
    assert a["pairs"] == b["pairs"]


def test_planted_pairs_are_found_and_reported_with_alignment():
    rows, planted, _ = texts.planted_text(11, vocab=80, n_pairs=2)
    res = rowsim.row_similarity(rows, cfg={"permutations": 499})
    sig = set(rowsim.significant_pairs(res))
    assert set(planted) <= sig
    for p in res["pairs"]:
        if p["significant"]:
            assert len(p["alignment"]) == p["score"] >= 3


def test_family_wise_error_is_controlled_on_structureless_text():
    """仕込みのない文で、1つでも有意な組が出る割合が有意水準の近くに収まる（60 件、上限は二項分布の 99.9% 点より上）。"""
    hits = 0
    for seed in range(60):
        rows, planted, _ = texts.planted_text(1000 + seed, vocab=(30, 80, 250)[seed % 3], n_pairs=0)
        assert planted == []
        hits += bool(rowsim.significant_pairs(rowsim.row_similarity(rows, cfg={"permutations": 199, "seed": seed})))
    assert hits <= 10


def test_one_misread_symbol_breaks_ngram_sharing_but_not_lcs():
    """IMG_4494 で H01 が消えた仕組みの最小の再現: 共有する記号が1つおきに取り違えられると、共有する 2 記号列は 0 になる。"""
    from babel.structure.stats import LinePairFamily
    rng = random.Random(0)
    rows = [[rng.randrange(100, 400) for _ in range(8)] for _ in range(11)]
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    b = [1, 2, 3, 4, 5, 6, 7, 9]
    b_misread = [1, 50, 3, 51, 5, 52, 7, 9]              # 2・4・6 番目を別のクラスに取り違えた転写
    clean = [a, b] + rows
    noisy = [a, b_misread] + rows
    fam = LinePairFamily.__new__(LinePairFamily)
    assert (0, 1) in fam.compute([tuple(r) for r in clean])
    assert (0, 1) not in fam.compute([tuple(r) for r in noisy])          # 既存の統計量は 0 になる
    for text in (clean, noisy):
        assert (0, 1) in rowsim.significant_pairs(rowsim.row_similarity(text, cfg={"permutations": 499}))


def test_min_score_keeps_short_fragment_rows_out():
    rows = [[1, 2], [1, 2]] + [[k, k + 1, k + 2, k + 3, k + 4, k + 5] for k in range(10, 100, 10)]
    res = rowsim.row_similarity(rows, cfg={"permutations": 199})
    top = res["pairs"][0]
    assert top["rows"] == [0, 1] and top["p_adj"] <= 0.05 and not top["significant"]     # 一致 2 記号は報告しない


# ------------------------------------------------------------------------------------------------ 転写の突き合わせ

def _relabel(rows, seed):
    ids = sorted({x for r in rows for x in r})
    perm = ids[:]
    random.Random(seed).shuffle(perm)
    m = dict(zip(ids, (p + 1000 for p in perm)))
    return [[m[x] for x in r] for r in rows]


def test_alignment_ignores_class_numbers():
    rows, _, _ = texts.planted_text(5, vocab=60)
    other = _relabel(rows, 1)
    al = crossalign.align(rows, other)
    assert al["anchor_rows"] == list(range(len(rows)))
    ag = crossalign.pair_agreement(rows, other, al)
    assert ag["adjusted_rand"] == 1.0 and ag["same_only_in_a"] == 0 and ag["same_only_in_b"] == 0
    br = crossalign.bracket(rows, other, al)
    assert br["fine_classes"] == br["coarse_classes"] == len({x for r in rows for x in r})


def test_alignment_recovers_a_split_symbol():
    rows, _, _ = texts.planted_text(6, vocab=30)
    other = _relabel(rows, 2)
    k = 4
    other[k] = other[k][:3] + [9001, 9002] + other[k][4:]            # 行 4 の 4 番目の記号を2つに分けた転写
    al = crossalign.align(rows, other)
    assert k not in al["anchor_rows"]
    cols = al["columns"][k]
    assert ((3,), (3, 4)) in cols
    assert sum(1 for pa, pb in cols if len(pa) == 1 and len(pb) == 1) == len(rows[k]) - 1


def test_bracket_separates_fine_and_coarse():
    a = [[1, 2, 3, 1], [4, 5, 6, 4]]
    b = [[7, 8, 9, 10], [11, 12, 13, 11]]                 # b は行 0 の最初と最後を別のクラスとみなした
    al = crossalign.align(a, b)
    ag = crossalign.pair_agreement(a, b, al)
    assert (ag["same_in_both"], ag["same_only_in_a"], ag["same_only_in_b"]) == (1, 1, 0)
    br = crossalign.bracket(a, b, al)
    assert br["fine"][0][0] != br["fine"][0][3] and br["fine"][1][0] == br["fine"][1][3]
    assert br["coarse"][0][0] == br["coarse"][0][3]


def test_different_row_counts_are_refused():
    with pytest.raises(ValueError):
        crossalign.align([[1, 2]], [[1, 2], [3]])


# ------------------------------------------------------------------------------------------------ 実画像の転写（記録の再現）

def test_real_transcriptions_are_the_recorded_ones():
    """比較に使った転写が、記録した出どころ（PROVENANCE.json）のものと同じ。"""
    import hashlib
    prov = json.loads((REAL / "PROVENANCE.json").read_text(encoding="utf-8"))
    for item in prov["files"]:
        data = (ROOT / item["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == item["sha256"], item["path"]


def test_img4494_row_pairs_survive_both_transcriptions():
    res = compare.compare(ROOT / "results/vision/real-img4494/corpus-continuous.json", REAL / "inputs/v3-continuous.json",
                          extra=[REAL / "inputs/v2-continuous.json"], cfg={"permutations": 999})
    verdict = {tuple(v["rows"]): v["class"] for v in res["verdicts"]}
    assert verdict[(0, 1)] == "どの転写でも有意" and verdict[(3, 4)] == "どの転写でも有意"
    assert verdict.get((4, 6)) != "どの転写でも有意"
    assert res["stats"]["a"]["classes"] == 74 and res["stats"]["b"]["classes"] == 79 and res["stats"]["b"]["hapax"] == 58
    md = compare.compare_markdown(res, "baseline", "improved-v3")
    assert "画像は使っていない" in md and "どちらの転写が正しいかは分からない" in md
    v2 = res["extra"][str(REAL / "inputs/v2-continuous.json")]
    assert {tuple(p["rows"]) for p in v2["pairs"]} == {(0, 2), (6, 8)}      # 過剰分割された 25 行の転写でも、同じ2組の本体の行


def test_existing_analyzer_reproduces_the_reported_change():
    """既存の linepair 族は、baseline の転写では行 (3, 4) だけ、improved-v3 の転写では行 (0, 1) だけを有意とする（記録の再現）。"""
    def pairs(path):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return {tuple(p) for h in d["hypotheses"] if h["family"] == "linepair" for p in h["key"]["pairs"]}
    assert pairs(ROOT / "results/vision/real-img4494/analysis.json") == {(3, 4)}
    assert pairs(REAL / "analyze-v3.json") == {(0, 1)}


# ------------------------------------------------------------------------------------------------ 評価の作り

def test_planted_text_is_deterministic_and_pairs_share_content():
    a = texts.planted_text(7, vocab=80)
    b = texts.planted_text(7, vocab=80)
    assert a == b
    rows, planted, detail = a
    assert len(rows) == 13 and len(planted) == 2 and all(6 <= len(r) <= 13 for r in rows)
    for i, j in planted:
        assert rowsim.lcs_score(rows[i], rows[j], lambda x, y: float(x == y)) >= 2


def test_summary_counts_false_discoveries_by_kind():
    runs = [{"n_pairs": 0, "planted": [], "methods": {"m": [{"gt": [0, 1], "kind": "unplanted", "p_adj": 0.01}]},
             "votes": [{"gt": [0, 1], "kind": "unplanted", "votes": 1, "members": ["x"]}]},
            {"n_pairs": 0, "planted": [], "methods": {"m": []}, "votes": []},
            {"n_pairs": 2, "planted": [[0, 1], [2, 3]],
             "methods": {"m": [{"gt": [0, 1], "kind": "planted", "p_adj": 0.01}, {"gt": None, "kind": "same_line", "p_adj": 0.02}]},
             "votes": [{"gt": [0, 1], "kind": "planted", "votes": 8, "members": []}, {"gt": [4, 5], "kind": "unplanted", "votes": 2, "members": []}]}]
    s = summary.score(runs, "m")
    assert s["fwer"] == 0.5 and s["recall"] == 0.5 and s["precision"] == round(1 / 3, 4) and s["false_kinds"]["same_line"] == 1
    v = summary.score(runs, "vote>=3")
    assert v["fwer"] == 0.0 and v["recall"] == 0.5 and v["precision"] == 1.0
    d = summary.vote_discrimination(runs)
    assert d["auc"] == 1.0 and d["by_votes"]["8"]["p_true"] == 1.0


def test_dev_and_holdout_cases_do_not_overlap():
    from babel.robust_eval import bench
    keys = {}
    for split in bench.SPLITS:
        keys[split] = {(c["renderer"], c["family"], c["seed"]) for c in bench.cases(split)}
    assert not keys["dev"] & keys["holdout-synth"] and not keys["dev"] & keys["holdout-font"]
    fam = {s: set(bench.SPLITS[s]["families"]) for s in bench.SPLITS}
    assert not fam["dev"] & fam["holdout-synth"] and not fam["dev"] & fam["holdout-font"]


def test_robust_package_does_not_import_answers_or_evaluators():
    """babel/robust は生成器・採点・評価用のパッケージを import しない（正解に触れない）。"""
    banned = ("babel.generator", "babel.evaluate", "babel.vision_eval", "babel.structure_eval", "babel.robust_eval", "babel.vision.evaluate",
              "babel.vision.groundtruth")
    for path in (ROOT / "babel" / "robust").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = "babel." + ".".join(path.relative_to(ROOT / "babel").parts[:-1]) if node.level else ""
                if node.level == 1:
                    mods = [f"babel.robust.{node.module}" if node.module else "babel.robust"]
                elif node.level == 2:
                    mods = [f"babel.{node.module}" if node.module else "babel"] + [f"babel.{a.name}" for a in node.names if not node.module]
                else:
                    mods = [node.module or base]
            for m in mods:
                assert not any(m == b or m.startswith(b + ".") for b in banned), f"{path.name}: {m}"


# ------------------------------------------------------------------------------------------------ 画像（OpenCV が要る）

def _image_case(tmp_path, seed=100, family="stroke", condition="plain", vocab=80):
    cv2 = pytest.importorskip("cv2")
    from babel.robust_eval import bench
    case = {"split": "dev", "renderer": "layout", "family": family, "condition": condition, "seed": seed, "vocab": vocab, "n_pairs": 2}
    rows, planted, _, img, gt, _ = bench.render_case(case)
    p = tmp_path / "page.png"
    cv2.imwrite(str(p), img)
    return case, p, gt, planted


def test_ensemble_votes_for_planted_pairs(tmp_path):
    pytest.importorskip("cv2")
    from babel.robust import ensemble
    from babel.robust_eval import bench
    case, path, gt, planted = _image_case(tmp_path)
    res = ensemble.analyze_image(path, regions=gt.get("regions"), cfg={"permutations": 299})
    assert res["members"] == [m["name"] for m in ensemble.MEMBERS] and len(res["members"]) == summary.N_MEMBERS
    ref_line = bench._row_lines({"rows": [{"glyph_indices": list(range(k, k + 1))} for k in range(0)], "glyphs": []}, gt)
    assert ref_line == []
    reports = ensemble.transcribe_all(path, gt.get("regions"))
    line = bench._row_lines(reports["v3"], gt)
    strong = {tuple(sorted((line[v["rows"][0]], line[v["rows"][1]]))) for v in res["votes"] if v["votes"] >= 5}
    assert strong == set(planted)
    json.dumps(res)                                           # JSON にできる
    assert "方式をまたいで再現する行の組" in compare.image_markdown(res)


def test_reclassify_keeps_segmentation_and_scaled_member_returns_original_coordinates(tmp_path):
    pytest.importorskip("cv2")
    from babel.robust import ensemble
    _, path, gt, _ = _image_case(tmp_path, seed=101)
    reps = ensemble.transcribe_all(path, gt.get("regions"))
    base = reps["v3"]
    for name in ("v3-fine", "v3-coarse", "v3-shape", "v3-iou"):
        assert [g["box"] for g in reps[name]["glyphs"]] == [g["box"] for g in base["glyphs"]]
        assert [r["glyph_indices"] for r in reps[name]["rows"]] == [r["glyph_indices"] for r in base["rows"]]
    assert len({g["class_id"] for g in reps["v3-fine"]["glyphs"]}) >= len({g["class_id"] for g in reps["v3-coarse"]["glyphs"]})
    assert ensemble.map_rows(base, base) == list(range(len(base["rows"])))
    sc = reps["v3-scaled"]
    assert sc["image_size"] == base["image_size"]
    mapped = ensemble.map_rows(base, sc)
    assert sum(1 for k, m in enumerate(mapped) if m is not None) >= len(sc["rows"]) - 1


def test_existing_recognizers_are_not_changed_by_the_ensemble(tmp_path):
    """方式の集まりは既存の METHODS を呼ぶだけで、書き換えない。"""
    pytest.importorskip("cv2")
    from babel.robust import ensemble
    from babel.vision.recognizers import METHODS, recognize
    before = json.dumps(METHODS, sort_keys=True)
    _, path, gt, _ = _image_case(tmp_path, seed=102)
    reps = ensemble.transcribe_all(path, gt.get("regions"))
    assert json.dumps(METHODS, sort_keys=True) == before
    for name, method in (("v3", "improved-v3"), ("v2", "improved-v2"), ("baseline", "baseline")):
        direct = recognize(path, method, regions=gt.get("regions"))
        assert [r["classes"] for r in direct["rows"]] == [r["classes"] for r in reps[name]["rows"]]


def test_soft_weights_are_similarities(tmp_path):
    np = pytest.importorskip("numpy")
    from babel.robust import ensemble
    _, path, gt, _ = _image_case(tmp_path, seed=103)
    rep = ensemble.transcribe_all(path, gt.get("regions"), members=[ensemble.MEMBERS[0]])["v3"]
    W = ensemble.soft_weights(path, rep, gt.get("regions"))
    n = len(rep["glyphs"])
    assert W.shape == (n, n) and np.allclose(W, W.T) and W.min() >= 0 and W.max() <= 1 and np.allclose(np.diag(W), 1)
    cls = [g["class_id"] for g in rep["glyphs"]]
    same = [W[i, j] for i in range(n) for j in range(i + 1, n) if cls[i] == cls[j]]
    diff = [W[i, j] for i in range(n) for j in range(i + 1, n) if cls[i] != cls[j]]
    assert sum(same) / len(same) > sum(diff) / len(diff) + 0.3


def test_real_font_renderer_ground_truth(tmp_path):
    pytest.importorskip("cv2")
    from babel.robust_eval import realfont
    if not realfont.available("georgian"):
        pytest.skip("Pillow か DejaVu Sans がない")
    from babel.vision.groundtruth import validate
    rows, _, _ = texts.planted_text(1, vocab=30)
    img, gt = realfont.render(rows, "georgian", 1, 30)
    gt = validate(gt)
    assert len(gt["lines"]) == 13 and len([i for i in gt["items"] if i["kind"] == "glyph"]) == sum(len(r) for r in rows)
    for it in gt["items"]:
        x1, y1, x2, y2 = it["box"]
        assert (img[y1:y2, x1:x2] < 128).any()                # 枠の中にインクがある
    img2, gt2 = realfont.render(rows, "georgian", 1, 30)
    assert (img == img2).all() and [i["box"] for i in gt2["items"]] == [i["box"] for i in gt["items"]]


def test_cli_transcription_compare(tmp_path):
    from babel.__main__ import main
    out = tmp_path / "cmp.json"
    main(["transcription-compare", str(ROOT / "results/vision/real-img4494/corpus-continuous.json"), str(REAL / "inputs/v3-continuous.json"),
          "--out", str(out), "--permutations", "199", "--name-a", "baseline", "--name-b", "improved-v3"])
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["schema"] == "babel-transcription-compare-v1" and out.with_suffix(".md").exists()
