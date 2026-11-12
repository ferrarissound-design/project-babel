"""構造発見モジュール（babel.structure）の回帰テスト。外部APIは使わない。"""
import ast
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path

import pytest

from babel.structure import boundaries as bd
from babel.structure import corpus as sc
from babel.structure import stats
from babel.structure.analyze import analyze, perturb
from babel.structure.report import markdown
from babel.structure.transcription import compare, same_hypothesis

ROOT = Path(__file__).parents[1]
FAST = {"permutations": 199, "subsamples": 3, "subsample_permutations": 49, "perturbations": 2,
        "perturbation_permutations": 99, "compression_reps": 9, "affix_permutations": 49}


def _rand_lines(n, seed, V=20, lo=6, hi=14):
    r = random.Random(seed)
    alpha = "abcdefghijklmnopqrstuvwxyz"[:V]
    return [[r.choice(alpha) for _ in range(r.randint(lo, hi))] for _ in range(n)]


def _marked_lines(n, seed):
    """行頭に S、行末に E がある規則的な行（内容は無作為）。"""
    r = random.Random(seed)
    return [["S"] + [r.choice("abcdefghij") for _ in range(r.randint(5, 9))] + ["E"] for _ in range(n)]


# ------------------------------------------------------------------------------------------------ 分離

def test_structure_package_does_not_import_answers_or_evaluation():
    bad = []
    for p in sorted((ROOT / "babel" / "structure").glob("*.py")):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for n in names:
                if any(x in n for x in ("generator", "evaluate", "experiments", "structure_eval", "external")):
                    bad.append(f"{p.name}: {n}")
    assert bad == []


# ------------------------------------------------------------------------------------------------ 統計の部品

def test_poisson_and_hypergeometric_tails():
    mu, k = 1.3, 4
    exact = 1 - sum(math.exp(-mu) * mu ** i / math.factorial(i) for i in range(k))
    assert math.isclose(math.exp(stats.poisson_logsf(k, mu)), exact, rel_tol=1e-9)
    assert stats.poisson_logsf(0, 2.0) == 0.0
    N, K, n, k = 15, 6, 5, 3
    tot = math.comb(N, n)
    exact = sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(k, min(K, n) + 1)) / tot
    assert math.isclose(math.exp(stats.hypergeom_logsf(k, N, K, n)), exact, rel_tol=1e-9)


def test_expectation_matches_exhaustive_permutation():
    line = (0, 0, 1, 2, 1)
    exp = stats.Expectation([line])
    perms = set(itertools.permutations(line))
    for seq in [(0, 1), (0, 0), (1, 2, 1), (0,)]:
        brute = sum(sum(1 for i in range(len(p) - len(seq) + 1) if p[i:i + len(seq)] == seq) for p in perms) / len(perms)
        assert math.isclose(exp.ngram(seq), brute, rel_tol=1e-9), seq
        edge = sum(p[:len(seq)] == seq for p in perms) / len(perms)
        assert math.isclose(exp.edge(seq), edge, rel_tol=1e-9)
    gap = sum(sum(1 for i in range(len(p) - 2) if p[i] == 0 and p[i + 2] == 0) for p in perms) / len(perms)
    assert math.isclose(exp.gap(0, 2), gap, rel_tol=1e-9)


def test_repeated_ngrams_equals_brute_force():
    r = random.Random(3)
    for _ in range(40):
        lines = [tuple(r.randrange(r.choice([2, 4, 8])) for _ in range(r.randrange(1, 20))) for _ in range(r.randrange(1, 12))]
        c = Counter(g for ln in lines for n in range(2, 9) for g in stats.ngrams(ln, n))
        assert {g: k for g, k in c.items() if k >= 2} == dict(stats.repeated_ngrams(lines, 2, 8))


def test_shuffles_preserve_what_they_claim():
    lines = [tuple(x) for x in _rand_lines(10, 1)]
    rng = random.Random(0)
    w = stats.shuffle_within(lines, rng)
    assert [sorted(a) for a in w] == [sorted(a) for a in lines]
    g = stats.shuffle_global(lines, rng)
    assert [len(a) for a in g] == [len(a) for a in lines]
    assert Counter(s for ln in g for s in ln) == Counter(s for ln in lines for s in ln)
    p = stats.shuffle_presence(lines, rng)
    assert Counter(x for ln in p for x in ln) == Counter(x for ln in lines for x in set(ln))


# ------------------------------------------------------------------------------------------------ 誤検出と検出

@pytest.mark.parametrize("n,seed", [(13, 0), (13, 1), (30, 2)])
def test_random_symbols_give_no_hypotheses(n, seed):
    a = analyze(sc.from_lines(_rand_lines(n, seed)), FAST)
    assert a["hypotheses"] == []
    assert a["verdict"]["code"] == "no_structure_detected"
    assert "証明ではない" in a["verdict"]["text"]


def test_zipf_frequency_skew_alone_is_not_structure():
    r = random.Random(5)
    alpha = "abcdefghijklmnopqrst"
    w = [1 / (i + 1) ** 1.2 for i in range(len(alpha))]
    lines = [r.choices(alpha, w, k=r.randint(6, 14)) for _ in range(30)]
    assert analyze(sc.from_lines(lines), FAST)["hypotheses"] == []


def test_line_markers_found_with_evidence_and_counterexamples():
    lines = _marked_lines(13, 0)
    lines[3] = lines[3][1:]                    # 反例: 行頭に S のない行
    lines[5].insert(3, "S")                   # 反例: 行の途中の S
    a = analyze(sc.from_lines(lines), FAST)
    pos = [h for h in a["hypotheses"] if h["family"] == "position" and h["key"]["symbols"] == ["S"]]
    assert pos and pos[0]["key"]["side"] == "^"
    h = pos[0]
    assert h["support"] == 12
    assert h["counterexamples"]["count"] == 1 and h["counterexamples"]["lines_without"] == 1
    assert all(e["start"] == 0 for e in h["evidence"])
    assert h["statistics"]["p_family_corrected"] <= h["statistics"]["threshold"]
    assert h["interpretation"]["status"].startswith("未検証")
    assert "文頭" in h["interpretation"]["candidates"][0]          # 候補であって主張ではない
    assert any(x["key"]["side"] == "$" and x["key"]["symbols"] == ["E"] for x in a["hypotheses"] if x["family"] == "position")


def test_embedded_repeat_is_found_and_not_split_into_fragments():
    r = random.Random(7)
    alpha = "abcdefghijklmnopqrst"
    motif = list("kpqrsx")
    lines = []
    for i in range(30):
        s = [r.choice(alpha) for _ in range(r.randint(8, 14))]
        if i % 3 == 0:
            j = r.randint(0, len(s) - 6)
            s[j:j + 6] = motif
        lines.append(s)
    a = analyze(sc.from_lines(lines), FAST)
    ng = [h for h in a["hypotheses"] if h["family"] == "ngram"]
    assert any(h["key"]["symbols"] == motif for h in ng)
    # 6記号列の部分列（2〜5記号）は別の仮説として並べない
    assert not any(len(h["key"]["symbols"]) < 6 and "".join(h["key"]["symbols"]) in "kpqrsx" for h in ng)


def test_periodic_symbol_found():
    r = random.Random(2)
    lines = [["P" if i % 4 == 0 else r.choice("abcdefghijklmnop") for i in range(r.randint(9, 15))] for _ in range(20)]
    a = analyze(sc.from_lines(lines), FAST)
    assert any(h["family"] == "gap" and h["key"]["symbols"] == ["P"] and h["key"]["distance"] % 4 == 0 for h in a["hypotheses"])


def test_obligatory_symbol_is_dispersion_not_cooccurrence():
    """どの行にも1回ずつ出る2記号を「共起」と報告しない（開発中に見つかった誤り）。"""
    r = random.Random(4)
    lines = []
    for _ in range(40):
        s = [r.choice("abcdefghijklmnop") for _ in range(r.randint(6, 10))]
        s.insert(r.randint(0, len(s)), "x")
        s.insert(r.randint(0, len(s)), "q")
        lines.append(s)
    a = analyze(sc.from_lines(lines), FAST)
    assert not any(h["family"] == "cooccurrence" and set(h["key"]["symbols"]) == {"x", "q"} for h in a["hypotheses"])
    assert any(h["family"] == "dispersion" and h["key"]["side"] == "even" and h["key"]["symbols"] == ["x"] for h in a["hypotheses"])


# ------------------------------------------------------------------------------------------------ 短いコーパス

def test_tengu_short_corpus_is_descriptive_only():
    c = sc.load(ROOT / "external/inbox/tengu-pilot-v0-real/corpus.json")
    a = analyze(c, FAST)
    assert a["mode"] == "short-corpus"
    assert a["hypotheses"] == []
    assert a["statistics"]["symbol_types"] == 97 and a["statistics"]["symbols_total"] == 145
    assert any("短いコーパス" in x for x in a["data_limits"])
    assert a["detectability"]["bigram_occurrences_needed"]
    assert a["boundaries"]["compression_test"]["p"] > 0.05
    md = markdown(a)
    assert "証明ではない" in md and "人間が次に確認すべき箇所" in md


def test_same_size_positive_control_is_detected():
    c = sc.load(ROOT / "external/inbox/tengu-positive-control-unspaced/corpus.json")
    a = analyze(c, FAST)
    assert c.n_symbols == 145
    assert a["verdict"]["code"] in ("structure_observed", "structure_unstable")
    assert any(h["family"] == "position" for h in a["hypotheses"])


def test_five_lines_do_not_crash_and_skip_subsampling():
    a = analyze(sc.from_lines(_marked_lines(4, 1)), FAST)
    assert any("部分標本" in x for x in a["data_limits"])
    for h in a["hypotheses"]:
        assert h["stability"]["sampling"] is None


# ------------------------------------------------------------------------------------------------ 単語境界

def test_boundary_methods_return_valid_cuts_and_find_repeated_words():
    r = random.Random(0)
    words = ["abc", "de", "fgh", "ij", "klm"]
    rows = [[r.choice(words) for _ in range(5)] for _ in range(40)]
    lines = []
    sym = {}
    for ws in rows:
        lines.append(tuple(sym.setdefault(ch, len(sym)) for w in ws for ch in w))
    gold = []
    for ws in rows:
        k, cs = 0, set()
        for w in ws[:-1]:
            k += len(w)
            cs.add(k)
        gold.append(cs)
    res = bd.candidates(lines, n_subsamples=2, compression_reps=9)
    for c in res["candidates"]:
        for ln, cs in zip(lines, c["cuts"]):
            assert all(0 < x < len(ln) for x in cs)
    f1 = {c["method"]: bd.boundary_f1([set(x) for x in c["cuts"]], gold) for c in res["candidates"]}
    assert max(f1.values()) > 0.6
    assert res["compression_test"]["p"] <= 0.1
    assert res["top"] and len(res["candidates"]) == 6


def test_unigram_candidate_is_the_existing_babel_segmenter():
    from babel.decipher import segment as legacy
    lines = sc.from_lines(_rand_lines(10, 4)).lines
    strs = bd._as_str(lines)
    assert bd.seg_unigram(lines) == [bd._cuts_from_words(w) for w in legacy.unigram_segment(strs)]


def test_given_spaces_are_not_used_by_the_analysis():
    pub = {"has_word_boundaries": True, "documents": [{"id": "a", "sentences": ["ab cd ef", "cd ab", "ef ef ab"]}]}
    c = sc.from_public(pub)
    assert c.given_boundaries == [{2, 4}, {2}, {2, 4}]
    c2 = sc.from_public(dict(pub, has_word_boundaries=False, documents=[{"id": "a", "sentences": ["abcdef", "cdab", "efefab"]}]))
    assert c.lines == c2.lines


# ------------------------------------------------------------------------------------------------ 転写の訂正・誤差

def test_human_corrections_apply_with_original_positions():
    c = sc.from_lines([list("abcde"), list("xyz")])
    corr = {"substitute": [{"line": 0, "pos": 1, "symbol": "B"}],
            "split": [{"line": 0, "pos": 3, "into": ["d1", "d2"]}],
            "join": [{"line": 1, "pos": 0, "length": 2, "symbol": "XY"}],
            "delete": [{"line": 0, "pos": 4}],
            "insert": [{"line": 0, "pos": 0, "symbol": "_"}],
            "merge": [["c", "z"]],
            "uncertain": [{"line": 1, "pos": 2}]}
    out = sc.apply_corrections(c, corr)
    assert out.names(out.lines[0]) == ["_", "a", "B", "c", "d1", "d2"]
    assert out.names(out.lines[1]) == ["XY", "c"]
    assert out.origins[0] == [0, 0, 1, 2, 3, 3]
    assert out.origins[1] == [0, 2]
    assert (1, 2) in out.uncertain
    assert {x["kind"] for x in out.corrections_applied} >= {"substitute", "split", "join", "delete", "insert", "rename", "uncertain"}
    with pytest.raises(ValueError):
        sc.apply_corrections(c, {"delete": [{"line": 5, "pos": 0}]})


def test_perturbation_keeps_origin_mapping():
    c = sc.from_lines(_rand_lines(20, 3))
    p = perturb(c, 0.2, random.Random(0))
    for ln, org in zip(p.lines, p.origins):
        assert len(ln) == len(org)
    assert p.lines != c.lines


def test_glyph_report_positions_reach_evidence():
    rows, glyphs = [], []
    r = random.Random(1)
    for li in range(12):
        classes = [0] + [r.randrange(1, 12) for _ in range(7)]
        idx = []
        for k, cl in enumerate(classes):
            idx.append(len(glyphs))
            glyphs.append({"box": [10 * k, 20 * li, 10 * k + 8, 20 * li + 15], "class_id": cl, "match_iou": 0.6 if k == 3 else 1.0, "pixels": 5})
        rows.append({"region": "page", "box": [0, 20 * li, 100, 20 * li + 15], "classes": classes, "glyph_indices": idx})
    rep = {"schema": "babel-glyph-pilot-v1", "source_image": "x.png", "rows": rows, "glyphs": glyphs, "settings": {}}
    c = sc.from_glyph_report(rep)
    assert len(c.source["low_confidence_positions"]) == 12
    a = analyze(c, FAST)
    h = [h for h in a["hypotheses"] if h["family"] == "position" and h["key"]["symbols"] == ["g0"]]
    assert h and h[0]["evidence"][0]["image_box"] == [0, 0, 8, 15]
    assert any("一致度が低い" in r["what"] for r in a["review_points"])
    assert same_hypothesis(h[0], h[0])
    cmp = compare(a, [("same", a)])
    assert all(all(x["reproduced_in"].values()) for x in cmp["by_hypothesis"])


# ------------------------------------------------------------------------------------------------ 出力と CLI

def test_output_contains_required_sections(tmp_path):
    a = analyze(sc.from_lines(_marked_lines(13, 2)), FAST)
    for k in ("statistics", "global_tests", "hypotheses", "descriptive_observations", "boundaries", "detectability",
              "data_limits", "review_points", "multiple_testing", "verdict"):
        assert k in a
    assert math.isclose(a["multiple_testing"]["per_family_threshold"], 0.05 / len(a["multiple_testing"]["families"]), rel_tol=1e-3)
    for h in a["hypotheses"]:
        for k in ("observation", "evidence", "support", "counterexamples", "statistics", "stability", "limits", "needed_data",
                  "interpretation"):
            assert k in h
    json.dumps(a, ensure_ascii=False)


def test_cli_analyze_writes_json_and_markdown(tmp_path, capsys):
    from babel.__main__ import main
    corpus = tmp_path / "c.json"
    corpus.write_text(json.dumps({"documents": [{"id": "d", "sentences": ["".join(x) for x in _marked_lines(13, 3)]}]}),
                      encoding="utf-8")
    out = tmp_path / "out" / "analysis.json"
    corr = tmp_path / "corr.json"
    corr.write_text(json.dumps({"uncertain": [{"line": 0, "pos": 0}]}), encoding="utf-8")
    assert main(["analyze", str(corpus), "--out", str(out), "--permutations", "199", "--subsamples", "2",
                 "--perturbations", "1", "--corrections", str(corr)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["hypotheses"] and data["input"]["uncertain_marked"] == 1
    assert out.with_suffix(".md").read_text(encoding="utf-8").startswith("# 構造解析レポート")
    assert "structure_observed" in capsys.readouterr().out or data["verdict"]["code"] != "no_structure_detected"


def test_cli_image_input_compares_vision_settings(tmp_path):
    pytest.importorskip("cv2")
    from babel.__main__ import main
    from babel.structure_eval.generators import Regular
    from babel.structure_eval.transcription_experiment import render
    rows = Regular(7).sample(13)
    img = tmp_path / "page.png"
    truth = render(rows, img)
    assert len(truth) == sum(len("".join(w)) for w, _ in rows)
    out = tmp_path / "a" / "analysis.json"
    assert main(["analyze", str(img), "--out", str(out), "--similarity", "0.8,0.6", "--permutations", "199",
                 "--subsamples", "2", "--perturbations", "1"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["input"]["format"] == "glyph-report"
    assert data["transcription_comparison"]["alternatives"] == ["similarity=0.6"]
    assert out.with_suffix(".png").exists()
    pos = [h for h in data["hypotheses"] if h["family"] == "position"]
    assert pos and all("image_box" in e for e in pos[0]["evidence"])
    assert "across_transcriptions" in pos[0]["stability"]


# --- Codex レビュー（PR #7）の指摘の回帰テスト（修正前の 289c904 では失敗する） ---

def test_cooccurrence_null_accounts_for_line_length():
    """独立・同分布の記号でも、長さ2の行と長さ100の行が混ざると、長い行どうしで全組が共起して見えていた。"""
    r = random.Random(0)
    lines = [tuple(r.randrange(20) for _ in range(2)) for _ in range(50)] + \
            [tuple(r.randrange(20) for _ in range(100)) for _ in range(50)]
    res = stats.run_tests(lines, {"max_n": 8, "max_gap": 8, "permutations": 199, "families": ["cooccurrence"]},
                          with_global=False)
    assert sum(v["p_family"] <= 0.05 / 8 for v in res["cooccurrence"]["keys"].values()) == 0


def test_oracle_linepair_truth_keeps_word_boundaries():
    from babel.structure_eval.generators import RealEnglish
    from babel.structure_eval.score import Oracle
    o = Oracle(RealEnglish(1000))
    a = (["complex", "is", "better", "than", "complicated"], None)
    b = (["and", "god", "said", "let", "there", "be", "light"], None)
    assert not o._pair_true(a, b)                      # 「be」は「better」の中にあるだけ
    assert o._pair_true((["let", "there", "be"], None), b)


def test_transcription_matching_uses_family_keys():
    def hyp(fam, key, boxes=()):
        return {"family": fam, "key": dict(key, family=fam), "evidence": [{"image_box": b} for b in boxes]}
    assert not same_hypothesis(hyp("gap", {"symbols": ["P"], "distance": 4}), hyp("gap", {"symbols": ["P"], "distance": 7}))
    assert same_hypothesis(hyp("gap", {"symbols": ["P"], "distance": 4}), hyp("gap", {"symbols": ["P"], "distance": 4}))
    l1 = hyp("linepair", {"lines": [0, 1], "pairs": [[0, 1]]})
    l2 = hyp("linepair", {"lines": [8, 9], "pairs": [[8, 9]]})
    assert not same_hypothesis(l1, l2) and same_hypothesis(l1, l1)
    box = [0, 0, 10, 10]
    a = hyp("position", {"side": "^", "symbols": ["g1"]}, [box])
    b = hyp("position", {"side": "$", "symbols": ["g7"]}, [box])
    assert not same_hypothesis(a, b)                   # 位置が重なっても行頭と行末は別の仮説
    lp = {"family": "linepair", "key": {"family": "linepair", "lines": [0, 1]},
          "evidence": [{"line_a": {"image_box": box}, "line_b": {"image_box": [0, 20, 10, 30]}}]}
    from babel.structure.transcription import _boxes
    assert _boxes(lp) == [box, [0, 20, 10, 30]]


# --- Codex レビュー（PR #7、マージ後の2回目）の指摘の回帰テスト ---

def _glyph_report(offset, n=25, seed=3):
    """同じ画像（同じ枠）で、視覚クラスの番号だけをずらした glyph-report。記号 99 が各行に1回ずつ入る。"""
    rows, glyphs = [], []
    r = random.Random(seed)
    for li in range(n):
        classes = [r.randrange(2, 14) for _ in range(r.randint(6, 10))]
        classes.insert(r.randint(0, len(classes)), 99)
        idx, cls = [], []
        for k, cl in enumerate(classes):
            idx.append(len(glyphs))
            cls.append(cl + offset)
            glyphs.append({"box": [10 * k, 20 * li, 10 * k + 8, 20 * li + 15], "class_id": cl + offset, "match_iou": 1.0, "pixels": 5})
        rows.append({"region": "p", "box": [0, 0, 1, 1], "classes": cls, "glyph_indices": idx})
    return {"schema": "babel-glyph-pilot-v1", "source_image": "x.png", "rows": rows, "glyphs": glyphs, "settings": {}}


def test_dispersion_cooccurrence_substitution_evidence_keeps_image_boxes():
    a = analyze(sc.from_glyph_report(_glyph_report(0)), FAST)
    b = analyze(sc.from_glyph_report(_glyph_report(100)), FAST)         # クラス番号だけが違う同じ画像
    disp = [h for h in a["hypotheses"] if h["family"] == "dispersion"]
    assert disp and all("image_box" in e for e in disp[0]["evidence"])
    cmp = compare(a, [("shifted-classes", b)])
    rows = {x["id"]: x for x in cmp["by_hypothesis"]}
    assert all(rows[h["id"]]["reproduced_in"]["shifted-classes"] for h in disp)
    from babel.structure.hypotheses import Builder
    c = sc.from_glyph_report(_glyph_report(0))
    bld = Builder(c)
    k = (c.symbols.index("g2"), c.symbols.index("g3"))
    for rec in (bld.cooccurrence(k), bld.substitution(k)):
        boxes = [e for e in rec["evidence"] if "image_box" in e]
        assert boxes, rec["observation"]


def test_uncertain_marks_inside_linepair_evidence_are_reported():
    lines = [list("abcdefghij") for _ in range(10)]
    c = sc.apply_corrections(sc.from_lines(lines), {"uncertain": [{"line": i, "pos": 0} for i in range(10)]})
    a = analyze(c, FAST)
    lp = [h for h in a["hypotheses"] if h["family"] == "linepair"]
    assert lp
    assert all(any("不確か" in x for x in h["limits"]) for h in lp)


# --- Codex レビュー（PR #8、マージ後）の指摘の回帰テスト ---

def _abc_report(n=8, classes=(0, 1, 2)):
    rows, glyphs = [], []
    for li in range(n):
        idx = []
        for k, cl in enumerate(classes):
            idx.append(len(glyphs))
            glyphs.append({"box": [10 * k, 20 * li, 10 * k + 8, 20 * li + 15], "class_id": cl, "match_iou": 1.0})
        rows.append({"region": "p", "box": [0, 0, 1, 1], "classes": list(classes), "glyph_indices": idx})
    return sc.from_glyph_report({"schema": "babel-glyph-pilot-v1", "rows": rows, "glyphs": glyphs})


def test_pair_hypotheses_need_both_members_to_match_in_the_image():
    from babel.structure.hypotheses import Builder
    c = _abc_report()
    shifted = _abc_report(classes=(50, 51, 52))           # 同じ画像、クラス番号だけ違う
    b, bs = Builder(c), Builder(shifted)
    A, B, C = (c.symbols.index(x) for x in ("g0", "g1", "g2"))
    A2, B2 = shifted.symbols.index("g50"), shifted.symbols.index("g51")
    for fam in ("cooccurrence", "substitution"):
        ab = {"family": fam, "key": {"family": fam, "symbols": ["g0", "g1"]}, **getattr(b, fam)((A, B))}
        ac = {"family": fam, "key": {"family": fam, "symbols": ["g0", "g2"]}, **getattr(b, fam)((A, C))}
        ab2 = {"family": fam, "key": {"family": fam, "symbols": ["g50", "g51"]}, **getattr(bs, fam)((A2, B2))}
        assert not same_hypothesis(ab, ac), fam           # 片方の記号（A）だけが共通な別の組
        assert same_hypothesis(ab, ab2), fam              # クラス番号が変わっても同じ組は一致


def test_linepair_needs_both_lines_to_match_in_the_image():
    box = lambda li: [0, 20 * li, 80, 20 * li + 15]      # noqa: E731
    def lp(i, j):
        return {"family": "linepair", "key": {"family": "linepair", "lines": [i, j]},
                "evidence": [{"line_a": {"image_box": box(i)}, "line_b": {"image_box": box(j)}}]}
    assert not same_hypothesis(lp(0, 1), lp(0, 5))
    assert same_hypothesis(lp(0, 1), lp(0, 1)) and same_hypothesis(lp(0, 1), lp(1, 0))


def test_role_overlap_threshold_rounds_up_for_odd_groups():
    box = lambda i: [10 * i, 0, 10 * i + 8, 15]          # noqa: E731
    def h(xs, ys):
        return {"family": "cooccurrence", "key": {"family": "cooccurrence", "symbols": ["x", "y"]},
                "evidence": [{"image_box": box(i), "role": "x"} for i in xs] + [{"image_box": box(i), "role": "y"} for i in ys]}
    assert not same_hypothesis(h([0, 1, 2], [10, 11, 12]), h([0, 5, 6], [10, 15, 16]))   # 各記号 1/3 だけ重なる
    assert same_hypothesis(h([0, 1, 2], [10, 11, 12]), h([0, 1, 6], [10, 11, 16]))       # 各記号 2/3 重なる
