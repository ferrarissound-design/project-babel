"""docs/VISION_EVAL_V2.md を results/vision/*-v2.json などから機械的に作る: python -m babel.vision_eval.report_v2

v1（docs/VISION_EVAL.md、PR #10）の記録は上書きしない。この版は、評価器の修正（#66・#67）、離れた部品の結合を加えた
improved-v2、実画像の評価パイプラインの後の記録。
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from .report import _c, paired

R = Path("results/vision")


def _load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _m(a, d=3):
    return "—" if not a else f"{a['mean']:.{d}f}"


def build(out="docs/VISION_EVAL_V2.md"):
    test, val = _load("methods-test-v2.json"), _load("methods-val-v2.json")
    jd, jv = _load("join-sweep-dev.json"), _load("join-sweep-val.json")
    diag, imp = _load("multipart-diagnosis-dev.json"), _load("impact-v2.json")
    L = ["# 画像認識の評価 v2: 評価器の最終修正・離れた部品の結合・実画像の評価の準備", "",
         "このファイルは `python -m babel.vision_eval.report_v2` が results/vision/ の JSON から生成する。"
         "v1 の記録（docs/VISION_EVAL.md、results/vision/methods-*.json、impact.json）は上書きしていない。", ""]

    L += ["## 1. 評価器の修正（Codex の指摘2件）", "",
          "- 評価用の字形の統合（same_as）は、人が確認した項目のものだけを使う。循環・矛盾は1つにまとめて警告する（docs/ISSUES.md #66）",
          "- 行の precision は、未確認の行に対応した予測を分母から外す。行割り当ては確認済みの行の記号だけで数える（#67）",
          "- 人工画像の正解はすべて確認済みで same_as を使わないので、v1 の評価の数値は変わらない", ""]

    # ------------------------------------------------------------ 2. 離れた部品
    L += ["## 2. 離れた部品を持つ字の分割（dev で調べ、val で確かめた。test は選択に使っていない）", ""]
    if diag:
        L += ["### 分割と結合の幾何（dev、improved、字の高さ H に対する比の十分位）", "",
              "| 量 | 描画方式 | 件数 | 10% | 50% | 90% |", "|---|---|---:|---:|---:|---:|"]
        for key, name in (("within_glyph_part_gap", "分割された字の部品の隙間"), ("between_adjacent_glyphs_gap", "隣り合う字の隙間"),
                          ("split_glyph_union_width", "分割された字の部品をまとめた幅"), ("glyph_width", "字の幅")):
            for style, v in sorted(diag[key].items()):
                d = v.get("deciles")
                L.append(f"| {name} | {style} | {v['n']} | {d[0] if d else '—'} | {d[4] if d else '—'} | {d[8] if d else '—'} |")
        L += ["", "部品の隙間と字どうしの隙間は重なっていて、隙間だけでは区別できない。一方、分割された字の部品をまとめた幅は"
              "ほぼ1字の幅（H 程度）に収まり、隣り合う2字を合わせた幅（2字分＋隙間）より狭い。"
              "そこで「まとめた幅が α·H 以下、隙間が β·H 以下なら結合する」を試した。", ""]
    for name, js in (("dev", jd), ("val", jv)):
        if not js:
            continue
        L += [f"### 結合の条件の試行（{name}）", "",
              "| α（幅） | β（隙間） | 全体 F1 | 全体 ARI | font F1 | font 分割 | font 結合 | stroke F1 | stroke 分割 | stroke 結合 |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in js["rows"]:
            L.append(f"| {r['join_width'] or 'なし'} | {r['join_width_gap'] or '—'} | {_m(r['all_f1'])} | {_m(r['all_ari'])} | "
                     f"{_m(r['font_f1'])} | {_m(r['font_split'], 2)} | {_m(r['font_merge'], 2)} | {_m(r['stroke_f1'])} | "
                     f"{_m(r['stroke_split'], 2)} | {_m(r['stroke_merge'], 2)} |")
        L.append("")
    L += ["dev で全体 F1 が最大の α=0.95・β=0.5 を選び、val で確かめてから `improved-v2` として固定した（`improved` は変えずに残す）。", ""]

    # ------------------------------------------------------------ 3. test
    for res in (test, val):
        if not res:
            continue
        m = res["meta"]
        L += [f"## 3{'a' if res is test else 'b'}. 方式の比較（{m['split']}、シード {m['seeds'][0]}〜{m['seeds'][-1]}、画像 {m['images']} 枚、OpenCV {m.get('opencv')}）", "",  # noqa: E501
              "| 方式 | 検出 F1 | ARI | 分割 | 結合 | 行 F1 | 文字以外の誤検出 | 秒/枚 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        dropped = {}
        for meth, s in res["summary"].items():
            a = s["all"]
            ari = _c(a["ari"])
            if a["ari"] and a["ari"]["n"] < m["images"]:     # 測れなかった画像を平均から除いたことを隠さない
                dropped[meth] = m["images"] - a["ari"]["n"]
                ari += f" （{a['ari']['n']} 枚）"
            L.append(f"| {meth} | {_c(a['f1'])} | {ari} | {_c(a['split_glyphs'], 1)} | {_c(a['merged_predictions'], 1)} | "
                     f"{_c(a['line_f1'])} | {_c(a['non_glyph'], 1)} | {_c(a['seconds'], 3)} |")
        L.append("")
        for meth, k in dropped.items():
            L += [f"**注意:** {meth} の ARI は {m['images'] - k} 枚の平均。対応した記号が1つもない（検出 F1 0）画像 {k} 枚では "
                  "ARI を定義できず、平均から除いている。そのため他の方式と同じ画像の平均ではなく、"
                  f"{meth} の ARI は過大に見える。同じ画像どうしの比較は 3a の「同じ画像での比較」を見ること（docs/ISSUES.md #71）", ""]
        if res is test:
            L += ["### 同じ画像での比較（test）", "",
                  "| 比較 | 指標 | 画像 | 差の平均 | 良くなった | 悪くなった | 同じ |", "|---|---|---:|---:|---:|---:|---:|"]
            for a_, b_ in (("improved", "improved-v2"), ("baseline", "improved-v2")):
                for key, name, sign in (("f1", "検出 F1", 1), ("ari", "ARI", 1), ("split_glyphs", "分割（少ないほど良い）", -1),
                                        ("merged_predictions", "結合（少ないほど良い）", -1)):
                    p = paired(test["runs"], a_, b_, key)
                    if p:
                        better, worse = (p["b_better"], p["a_better"]) if sign > 0 else (p["a_better"], p["b_better"])
                        L.append(f"| {b_} − {a_} | {name} | {p['n']} | {p['mean_diff']:+.4f} | {better} | {worse} | {p['ties']} |")
            L += ["", "### 条件ごとの検出 F1（test）", "", "| 条件 | baseline | improved | improved-v2 |", "|---|---:|---:|---:|"]
            for k in test["summary"]["baseline"]:
                if k != "all":
                    L.append(f"| {k} | " + " | ".join(_m(test["summary"][x][k]["f1"], 2) for x in ("baseline", "improved", "improved-v2")) + " |")
            L += ["", "罫線と文字が接触する条件（rules）と、文字の間隔が不均一な条件（spacing）も、この表の行で読める。", ""]

    # ------------------------------------------------------------ 4. impact
    if imp:
        m = imp["meta"]
        L += ["## 4. 認識の誤差が構造解析に与える影響（v2）", "",
              f"v1 と同じ実験（regular、シード {m['seeds'][0]}〜{m['seeds'][-1]}、13・30行、2描画方式 × 4条件、構造解析 v0.1.4 の同じ設定）に"
              "改善方式 v2（C2）を加えた。発見した規則は設計した8件のうち何件か。", "",
              "| 行数 | 入力 | 実行 | 発見した規則（平均） | 偽の仮説（平均） | A との差（規則） |", "|---:|---|---:|---:|---:|---:|"]
        by = defaultdict(list)
        for r in imp["runs"]:
            by[(r["lines"], r["input"])].append(r)
        for n in sorted({k[0] for k in by}):
            a_mean = statistics.mean(r["gold_found"] for r in by[(n, "A")])
            for inp in ("A", "B", "C", "C2", "D-sub", "D-split", "D-seg"):
                rs = by.get((n, inp))
                if rs:
                    g = statistics.mean(r["gold_found"] for r in rs)
                    L.append(f"| {n} | {inp} | {len(rs)} | {g:.2f} | {statistics.mean(r['false'] for r in rs):.2f} | {g - a_mean:+.2f} |")
        pairs = defaultdict(dict)
        for r in imp["runs"]:
            if r["input"] in ("B", "C", "C2"):
                pairs[(r["seed"], r["lines"], r["style"], r["condition"])][r["input"]] = r
        L += ["", "### 同じ画像での比較（発見した規則の差）", ""]
        for x, y in (("B", "C2"), ("C", "C2")):
            d = [v[y]["gold_found"] - v[x]["gold_found"] for v in pairs.values() if x in v and y in v]
            if d:
                L.append(f"- {y} − {x}: 画像 {len(d)} 枚、平均 {statistics.mean(d):+.2f}（標準偏差 {statistics.pstdev(d):.2f}）、"
                         f"{y} が多い {sum(v > 0 for v in d)}・{x} が多い {sum(v < 0 for v in d)}・同じ {sum(v == 0 for v in d)}")
        L.append("")

    # ------------------------------------------------------------ 5. 実画像
    L += ["## 5. 実画像（IMG_4494）", "",
          "画像本体はまだない。実画像の評価パイプライン（`python -m babel image-study`、手順は docs/REAL_IMAGE_STUDY.md）を用意し、"
          "合成画像で動作を確かめた（tests/test_image_study.py）。画像を受け取ったら prepare → annotate → evaluate → structure の順に進める。",
          "**実画像の認識精度と構造解析の比較は、正解転写ができるまで「評価待ち」。** 過去の 114候補・74クラスは正解として扱わない。", ""]

    # ------------------------------------------------------------ 6. 問い
    L += ["## 6. 研究上の問いへの答え", "",
          "「BABEL は、正しく転写された未知言語なら構造を発見できるのか。画像認識の誤りは、その能力をどれほど下げているのか」", "",
          "### 人工データで示せたこと", ""]
    if imp:
        by = defaultdict(list)
        for r in imp["runs"]:
            by[(r["lines"], r["input"])].append(r["gold_found"])
        a13, a30 = statistics.mean(by[(13, "A")]), statistics.mean(by[(30, "A")])
        L.append(f"- 正しい転写なら、設計した8規則のうち 13行で平均 {a13:.2f}、30行で {a30:.2f} を見つける（偽の仮説は少ない）。"
                 "見つからない規則は、データ量（13行）と構造解析の限界（接辞・接尾辞の置換可能性: #41・#45）による")
        for inp, name in (("B", "既存の認識"), ("C", "改善した認識 v1"), ("C2", "改善した認識 v2")):
            if by.get((13, inp)):
                L.append(f"- {name}の転写では、13行で {statistics.mean(by[(13, inp)]):.2f}、30行で {statistics.mean(by[(30, inp)]):.2f}"
                         f"（正しい転写との差 {statistics.mean(by[(13, inp)]) - a13:+.2f} / {statistics.mean(by[(30, inp)]) - a30:+.2f}）")
    L += ["- 誤りの種類ごとの影響（D）: クラスの分けすぎが最も大きく発見を減らし、偽の仮説を作る。記号の誤分類10%や分割・結合10%の影響はそれより小さい", "",
          "### 実画像で確認できたこと", "",
          "- 残っている転写（114記号・74クラス・1回だけ47種）から、現在の構造解析で有意な仮説は1件（行3・4の類似）",
          "- その統計は、真の字形数 20〜400 のどれとも両立し、過剰な分割か字形の多さかを転写だけでは区別できない（docs/VISION_EVAL.md 第4節）", "",
          "### 現段階では判断できないこと", "",
          "- 実画像の認識がどれだけ正しいか（正解転写がない）",
          "- 実画像の文字に、人工言語のような構造があるか（正しい転写がないので、データ不足・認識の誤り・構造の欠如を分けられない）",
          "- 人工データで測った認識の誤りの影響が、実画像の字形・劣化でも同じ大きさか（人工画像は2つの描画方式だけ）", ""]
    L += ["## 7. 限界", "",
          "- improved-v2 の結合の条件は「字の幅はほぼ高さ以下」という仮定に依存する。横に長い字（字の幅 > 0.95H）を持つ文字体系では、"
          "1字の中の部品を結合しきれない。逆に、細い字が密に並ぶ文字体系では、隣の字を誤って結合しうる",
          "- 人工画像の字形・劣化は2つの生成方式で作ったもので、実際の筆写・写真を代表しない", ""]
    Path(out).write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
