"""docs/VISION_EVAL.md を results/vision/*.json から機械的に作る: python -m babel.vision_eval.report"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

R = Path("results/vision")


def _load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _c(a, d=3):
    return "—" if not a else f"{a['mean']:.{d}f} ± {a['sd']:.{d}f}"


def paired(runs, a, b, key):
    """同じ画像での方式 a と b の差（b − a）。"""
    by = defaultdict(dict)
    for r in runs:
        by[r["name"]][r["method"]] = r.get(key)
    diffs = [v[b] - v[a] for v in by.values() if v.get(a) is not None and v.get(b) is not None]
    if not diffs:
        return None
    return {"n": len(diffs), "mean_diff": round(statistics.mean(diffs), 4), "sd": round(statistics.pstdev(diffs), 4),
            "b_better": sum(d > 1e-9 for d in diffs), "a_better": sum(d < -1e-9 for d in diffs),
            "ties": sum(abs(d) <= 1e-9 for d in diffs)}


def _methods_section(res, L):
    m = res["meta"]
    L += [f"### {m['split']}（シード {m['seeds'][0]}〜{m['seeds'][-1]}、画像 {m['images']} 枚）", "",
          "| 方式 | 検出 F1 | ARI | NMI | 分割 | 結合 | クラス数比 | 行 F1 | 文字以外の誤検出 | 秒/枚 | 最大メモリ KiB |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for meth, s in res["summary"].items():
        a = s["all"]
        L.append(f"| {meth} | {_c(a['f1'])} | {_c(a['ari'])} | {_c(a['nmi'])} | {_c(a['split_glyphs'], 1)} | "
                 f"{_c(a['merged_predictions'], 1)} | {_c(a['class_ratio'], 2)} | {_c(a['line_f1'])} | {_c(a['non_glyph'], 1)} | "
                 f"{_c(a['seconds'], 3)} | {_c(a['peak_kib'], 0)} |")
    L.append("")


def build(out="docs/VISION_EVAL.md"):
    dev, val, test = _load("methods-dev.json"), _load("methods-val.json"), _load("methods-test.json")
    imp, inv, real = _load("impact.json"), _load("real-img4494/inventory.json"), _load("real-img4494/analysis.json")
    L = ["# 画像認識の定量評価と、未知言語の構造解析への影響", "",
         "このファイルは `python -m babel.vision_eval.report` が results/vision/*.json から生成する。手で数値を書き写していない。", "",
         "正解のある人工画像だけで精度を測った。実画像（IMG_4494）は画像がリポジトリにないため、精度は**測定不能**（第4節）。", ""]

    # ------------------------------------------------------------ 1. 認識方式
    L += ["## 1. 認識方式の比較", "",
          "人工画像: 2つの描画方式（font = OpenCV のフォント、stroke = 線・弧・点で作った字形）× 8条件"
          "（clean, noise, broken, similar, size_thickness, rules, spacing, wrapped）。各画像13行、規則のある人工言語の文。",
          "dev（シード 0〜9）で方式と閾値を選び、val（100〜109）で確かめ、test（1000〜1019）は最終評価にだけ使った。"
          "ただし test は2回実行している: 1回目は OpenCV 5.0（宣言した範囲外）と不具合のある評価器での実行で無効（docs/ISSUES.md #61〜#64）。"
          "方式と閾値は変えずに、OpenCV 4.14 で評価器を直して再実行した。以下はすべて2回目の値。", "",
          ]
    sweep = _load("baseline-threshold-dev.json")
    if sweep:
        best = max(sweep["rows"], key=lambda r: r["ari"]["mean"])
        f1s = sorted({round(r["f1"]["mean"], 3) for r in sweep["rows"]})
        L += [f"既存方式（baseline）の類似度の閾値を dev で {sweep['rows'][0]['similarity']}〜{sweep['rows'][-1]['similarity']} に変えると、"
              f"検出 F1 は {'〜'.join(map(str, f1s))}、ARI は最大 {best['ari']['mean']:.3f}（{best['similarity']}）"
              "（results/vision/baseline-threshold-dev.json）。閾値の違いだけでは、以下の差は説明できない。", ""]
    for res in (test, val, dev):
        if res:
            _methods_section(res, L)
    if test:
        L += ["### 同じ画像での既存方式と改善候補の比較（test、b = improved − a = baseline）", "",
              "| 指標 | 画像 | 差の平均 | 差の標準偏差 | improved が良い | baseline が良い | 同じ |", "|---|---:|---:|---:|---:|---:|---:|"]
        for key, name, sign in (("f1", "検出 F1", 1), ("ari", "ARI", 1), ("split_glyphs", "分割の誤り（少ないほど良い）", -1),
                                ("merged_predictions", "結合の誤り（少ないほど良い）", -1), ("line_f1", "行 F1", 1)):
            p = paired(test["runs"], "baseline", "improved", key)
            if p:
                better, worse = (p["b_better"], p["a_better"]) if sign > 0 else (p["a_better"], p["b_better"])
                L.append(f"| {name} | {p['n']} | {p['mean_diff']:+.4f} | {p['sd']:.4f} | {better} | {worse} | {p['ties']} |")
        L += ["", "### 条件ごと（test、検出 F1 / ARI / 分割 / 結合）", "",
              "| 条件 | baseline | improved |", "|---|---|---|"]
        for k in test["summary"]["baseline"]:
            if k == "all":
                continue
            row = []
            for meth in ("baseline", "improved"):
                s = test["summary"][meth][k]
                f = lambda n, d=2: "—" if not s[n] else f"{s[n]['mean']:.{d}f}"   # noqa: E731
                row.append(f"{f('f1')} / {f('ari')} / {f('split_glyphs', 1)} / {f('merged_predictions', 1)}")
            L.append(f"| {k} | {row[0]} | {row[1]} |")
        L += ["", "**改善候補（improved）** = 中央値フィルタ → 連結成分による切り出し（点などの小さな成分を近い記号に付け、"
              "横に重なる成分をまとめる）→ 縦横比を保ったテンプレートの正規化相互相関による平均連結クラスタリング（閾値 0.20、dev で選択）。",
              ""]
        better, worse = [], []
        for k in test["summary"]["baseline"]:
            if k == "all":
                continue
            b0, b1 = test["summary"]["baseline"][k], test["summary"]["improved"][k]
            df = (b1["f1"]["mean"] if b1["f1"] else 0) - (b0["f1"]["mean"] if b0["f1"] else 0)
            if df >= 0.02:
                better.append(f"{k}（F1 {b0['f1']['mean']:.2f}→{b1['f1']['mean']:.2f}）")
            elif df <= -0.02:
                worse.append(f"{k}（F1 {b0['f1']['mean']:.2f}→{b1['f1']['mean']:.2f}）")
        L += [f"検出 F1 が 0.02 以上良くなった条件: {'、'.join(better) or 'なし'}。",
              f"**悪くなった条件: {'、'.join(worse) or 'なし'}。** 悪化は離れた部品を持つ字形（stroke）に集中する"
              "（字の中の部品の隙間が字と字の隙間より広く、位置だけでは区別できない。docs/ISSUES.md #57）。"
              "クラスの構造（ARI）は多くの条件で良くなった。処理時間は約2倍。", ""]

    # ------------------------------------------------------------ 2. 構造解析への影響
    if imp:
        m = imp["meta"]
        L += ["## 2. 認識の誤差が構造解析に与える影響", "",
              f"同じ文章（regular、シード {m['seeds'][0]}〜{m['seeds'][-1]}）を、同じ設定の babel analyze（v0.1.4、置換 {m['config']['permutations']} 回）に、"
              "入力の転写だけを変えて渡した。設計した規則は8件（文頭標識・文末標識・v→E・x/q/v の置換可能性・格接尾辞 x/q・x と v の必須性）。",
              "B/C の予測クラスは、採点のときだけ、位置で対応した正解の記号に読み替えた。対応しない記号（分割の破片・余分な検出）を含む仮説は偽と数えた。", "",
              "| 行数 | 入力 | 条件 | 実行 | 発見した規則（/8） | 仮説 | 偽の仮説 | うち対応しない記号を含む | 記号の種類 | 境界 F1（合議） | 標本の安定性 | 転写の安定性 | 認識の F1 | 認識の ARI |",  # noqa: E501
              "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for s in imp["summary"]:
            cond = f"{s['style']}/{s['condition']}" if s["style"] else "—"
            L.append(f"| {s['lines']} | {s['input']} | {cond} | {s['runs']} | {_c(s['gold_found'], 2)} | {_c(s['hypotheses'], 1)} | "
                     f"{_c(s['false'], 1)} | {_c(s.get('false_unmapped'), 1)} | {_c(s['symbol_types'], 1)} | {_c(s['seg_f1_consensus'])} | "
                     f"{_c(s['sampling_stability'], 2)} | {_c(s['transcription_stability'], 2)} | {_c(s['vision_f1'])} | {_c(s['vision_ari'])} |")
        # 入力ごとのまとめ
        L += ["", "### 入力ごとのまとめ（全条件・全描画方式を通して）", "",
              "| 行数 | 入力 | 実行 | 発見した規則（平均） | 偽の仮説（平均） | A との差（規則） |", "|---:|---|---:|---:|---:|---:|"]
        by = defaultdict(list)
        for r in imp["runs"]:
            by[(r["lines"], r["input"])].append(r)
        for n in sorted({k[0] for k in by}):
            a_mean = statistics.mean(r["gold_found"] for r in by[(n, "A")]) if by.get((n, "A")) else None
            for inp in ("A", "B", "C", "D-sub", "D-split", "D-seg"):
                rs = by.get((n, inp))
                if not rs:
                    continue
                g = statistics.mean(r["gold_found"] for r in rs)
                f = statistics.mean(r["false"] for r in rs)
                L.append(f"| {n} | {inp} | {len(rs)} | {g:.2f} | {f:.2f} | {g - a_mean:+.2f} |")
        # B と C の対応のある比較
        pairs = defaultdict(dict)
        for r in imp["runs"]:
            if r["input"] in ("B", "C"):
                pairs[(r["seed"], r["lines"], r["style"], r["condition"])][r["input"]] = r
        diffs = [(v["C"]["gold_found"] - v["B"]["gold_found"], v["C"]["false"] - v["B"]["false"]) for v in pairs.values()
                 if "B" in v and "C" in v]
        def frate(r):
            return r["false"] / (r["true"] + r["false"]) if r["true"] + r["false"] else None
        both = [(frate(v["B"]), frate(v["C"])) for v in pairs.values() if "B" in v and "C" in v
                and frate(v["B"]) is not None and frate(v["C"]) is not None]
        collapsed = sum(1 for v in pairs.values() if "B" in v and v["B"]["hypotheses"] == 0)
        collapsed_c = sum(1 for v in pairs.values() if "C" in v and v["C"]["hypotheses"] == 0)
        if diffs:
            dg = [d[0] for d in diffs]
            df = [d[1] for d in diffs]
            L += ["", "### 改善した認識器（C）と既存（B）の、同じ画像での比較", "",
                  f"- 画像 {len(diffs)} 枚。発見した規則の差（C − B）: 平均 {statistics.mean(dg):+.2f}（標準偏差 {statistics.pstdev(dg):.2f}）、"
                  f"C が多い {sum(x > 0 for x in dg)}・B が多い {sum(x < 0 for x in dg)}・同じ {sum(x == 0 for x in dg)}",
                  f"- 偽の仮説の差（C − B）: 平均 {statistics.mean(df):+.2f}（標準偏差 {statistics.pstdev(df):.2f}）、"
                  f"C が少ない {sum(x < 0 for x in df)}・B が少ない {sum(x > 0 for x in df)}・同じ {sum(x == 0 for x in df)}",
                  f"- ただし B は {collapsed} 枚で仮説が0件だった（認識が崩れて規則も偽も出ない。C は {collapsed_c} 枚）。偽の数の平均はこれに引っ張られる。"
                  f"両方が仮説を出した {len(both)} 枚での偽の割合（偽 / 検証できた仮説）: B 平均 {statistics.mean(b for b, _ in both):.3f}、"
                  f"C 平均 {statistics.mean(c for _, c in both):.3f}、C が低い {sum(c < b for b, c in both)}・B が低い {sum(c > b for b, c in both)}"
                  f"・同じ {sum(c == b for b, c in both)}" if both else "",
                  "- 偽の仮説の出どころは描画方式で違う: font では別の字形を同じクラスにまとめた誤り（対応しない記号を含まない偽）、"
                  "stroke では分割の破片を含む仮説（「うち対応しない記号を含む」の列）", ""]

    # ------------------------------------------------------------ 3. 要因の分離
    L += ["## 3. 解読できない原因の分離（人工データでの答え）", ""]
    if imp:
        by = defaultdict(list)
        for r in imp["runs"]:
            by[(r["lines"], r["input"])].append(r["gold_found"])
        a13, a30 = by.get((13, "A")), by.get((30, "A"))
        b13, c13 = by.get((13, "B")), by.get((13, "C"))
        if a13 and a30:
            sb = json.loads(Path("results/structure/benchmark-eval.json").read_text(encoding="utf-8")) \
                if Path("results/structure/benchmark-eval.json").exists() else None
            r100 = [r for r in (sb or {}).get("runs", []) if r["generator"] == "regular" and r["size_requested"] == 100]
            extra = (f"。構造解析の評価（results/structure/benchmark-eval.json、文字列で入力、シード 1000〜1009）では regular の100行で平均 "
                     f"{statistics.mean(len(r['score']['gold_found']) for r in r100):.2f}/8") if r100 else ""
            L.append(f"- **データ不足**: 正解の転写でも、13行で発見した規則は平均 {statistics.mean(a13):.2f}/8、30行で "
                     f"{statistics.mean(a30):.2f}/8{extra}。行を増やすと増えるが、30行と100行の差は小さい（残りは構造解析の限界）")
        if a13:
            L.append("- **構造解析の限界**: 正解の転写・30行でも見つからない規則がある（格接尾辞・接尾辞の置換可能性。docs/ISSUES.md #41, #45）。"
                     "これは画像認識と無関係")
        if b13 and c13 and a13:
            L.append(f"- **画像認識の誤り**: 13行で、正解の転写 {statistics.mean(a13):.2f} → 既存の認識 {statistics.mean(b13):.2f} → "
                     f"改善した認識 {statistics.mean(c13):.2f}。認識の誤りは発見を減らすだけでなく偽の仮説を作る"
                     "（font では別の字形の誤統合、stroke では分割の破片から。第2節）")
    L += ["", "人工データでは3つの要因を分けて測れる。**実画像では、正解の転写がないので分けられない**（第4節）。", ""]

    # ------------------------------------------------------------ 4. 実画像
    L += ["## 4. 実画像（IMG_4494）", "",
          "画像本体はリポジトリにも実験ブランチにもない。残っているのは、PR #6 の版の babel vision が出した未検証の転写"
          "（`results/vision/real-img4494/corpus-continuous.json`、実験ブランチ experiment/babel-structure-img4494-20261112 から複写）だけで、"
          "記号の画像上の位置もない。したがって、**検出・分類の精度は測定不能**。114候補・74クラスという数値は正解として扱わない。", ""]
    if real:
        st = real["statistics"]
        L += [f"- 転写: {st['lines']} 行、記号 {st['symbols_total']}、クラス {st['symbol_types']}、1回だけのクラス {st['hapax_types']}",
              f"- 構造解析（v0.1.4、置換 {real['config']['permutations']} 回）: {real['verdict']['text']}"]
        for h in real["hypotheses"]:
            L.append(f"  - {h['id']}（段階 {h['tier']}）{h['observation']}、補正後 p = {h['statistics']['p_family_corrected']}、"
                     f"標本の安定性 {h['stability']['sampling']}、転写の安定性 {h['stability']['transcription']}")
        sig = [f"{g['description']}（Holm p = {g['p_holm']:.3f}）" for g in real["global_tests"].values() if g["p_holm"] <= 0.05]
        L.append(f"- コーパス全体で有意: {'、'.join(sig) or 'なし'}")
        L.append("")
    if inv:
        o = inv["observed"]
        cons = inv["consistent"]
        Ks = sorted({c["true_glyphs"] for c in cons})
        L += ["### 74クラスは過剰な分割か、字形が多いのか（転写の統計だけでの検討）", "",
              f"真の字形 K 種（Zipf 分布）から {o['symbols']} 記号を引き、出現ごとに確率 p でクラスを分けすぎた（1字形あたり最大6変種）"
              f"シミュレーション（各400回）で、観測（クラス {o['classes']}、1回だけ {o['hapax_classes']}）が中央90%の範囲に入る組み合わせ:", "",
              "| 真の字形数 | Zipf 指数 | 分けすぎの確率 | クラス数の90%範囲 | 1回だけの90%範囲 |", "|---:|---:|---:|---|---|"]
        for c in cons:
            L.append(f"| {c['true_glyphs']} | {c['zipf_s']} | {c['p_oversplit']} | {c['types_90']} | {c['hapax_90']} |")
        te = json.loads(Path("results/structure/transcription-experiment.json").read_text(encoding="utf-8")) \
            if Path("results/structure/transcription-experiment.json").exists() else {}
        deg = [(v["transcription_errors"]["visual_classes"], v["transcription_errors"]["true_characters"])
               for x in te.get("T2_synthetic_image", []) if x.get("degraded") for v in x["vision"].values()]
        degtxt = (f"（合成の劣化画像では、{min(t for _, t in deg)}〜{max(t for _, t in deg)}字が {min(c for c, _ in deg)}〜{max(c for c, _ in deg)} の"
                  "視覚クラスになった: docs/STRUCTURE_RESULTS.md 表7）") if deg else ""
        p0 = sorted({c["true_glyphs"] for c in cons if c["p_oversplit"] == 0})
        L += ["", f"両立する真の字形数は {Ks[0]}〜{Ks[-1]} と幅が広い。分けすぎなし（p=0）でも K={'・'.join(map(str, p0)) or '—'} なら両立し、"
              f"K={Ks[0]} でも分けすぎが強ければ両立する{degtxt}。"
              "**転写の統計だけでは、過剰な分割と字形の多さを区別できない。**", ""]
    if real:
        L += ["### 人が次に確認すべき箇所（優先順）", "",
              "記号は babel vision の視覚クラスの番号（#7 = クラス7）。行の番号は0から数える。", ""] + review_priorities(real) + [""]

    # ------------------------------------------------------------ 5. 結論と追加実験
    L += ["## 5. 研究上の問いへの答え", "",
          "「BABEL が未知文字を解読できない主な原因は、画像認識の誤りか、構造解析の限界か、データ不足か」", "",
          "- **人工データ**では3つとも効いており、分けて測れた（第3節）。13行では、正解の転写でもデータ不足で規則の半分程度しか見つからない。"
          "画像認識の誤りはそこからさらに発見を減らし、存在しない記号についての偽の仮説を増やす",
          "- **実画像では、現時点で識別できない**。正解の転写がなく、転写の統計（74クラス・1回だけ47種）は過剰な分割とも字形の多さとも両立するため",
          "- 識別に必要な追加実験:",
          "  1. 実画像を提供してもらい、`python -m babel vision IMG.png --out-dir v` → `python -m babel annotate IMG.png --gt gt.json --report v/glyph-report.json` で、"  # noqa: E501
          "人が13行すべての記号の枠と字形名を確認する（verified=true）。これで認識の精度（F1・ARI・分割・結合）が測れる",
          "  2. 確認した正解の転写を `babel analyze` にかけ、認識器の転写の結果と比べる（本報告の A と B に当たる）。ここで画像認識の誤りの寄与が分かる",
          "  3. 正解の転写でも規則が出なければ、同じ長さの正の対照（docs/STRUCTURE_RESULTS.md 表8）と比べて、データ不足か構造の欠如かを判断する。"
          "13行で検出できる規則の大きさ（detectability）を併記する",
          "  4. 同じ文字体系の別の資料（行数を増やす）があれば、データ不足の寄与を直接測れる", ""]
    L += ["## 6. 限界", "",
          "- 人工画像の字形・劣化は2つの生成方式で作ったもので、実際の筆写・写真の多様さ（傾き、照明、紙の模様、手書きの揺れ）を代表しない",
          "- 改善候補は dev の人工画像で選んだ。実画像で同じ改善が得られるかは未検証",
          "- 構造解析への影響の実験は regular という1つの人工言語で、6シード × 2行数 × 2描画方式 × 4条件。他の言語では差が変わりうる",
          "- 離れた部品を持つ字形の分割は、改善候補で悪化した（未解決）",
          ""]
    Path(out).write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def _nm(sym):
    """私用領域の文字（babel vision の視覚クラス）を、読める番号に直す。"""
    return "".join(f"#{ord(ch) - 0xE000}" if 0xE000 <= ord(ch) <= 0xF8FF else ch for ch in str(sym))


def review_priorities(real):
    st = real["statistics"]
    out = []
    k = 1
    for h in real["hypotheses"]:
        if h["family"] == "linepair":
            ev = h["evidence"][0]
            out.append(f"{k}. **行 {', '.join(map(str, h['key']['lines']))}**（有意な仮説 {h['id']} の根拠）: 共通部分 "
                       f"{' / '.join(' '.join(_nm(y) for y in x) for x in ev.get('fixed_parts', [])[:4])}。同じ句の反復か、画面の表示・切り取りの重複か、"
                       "似た別の字形を同じクラスにまとめた誤りかを画像で確認する")
            k += 1
    pre = st.get("shared_line_prefixes") or []
    if pre:
        heads = "、".join(" ".join(_nm(x) for x in p["symbols"]) + "（" + str(p["lines"]) + "行）" for p in pre[:3])
        out.append(f"{k}. **行頭**: {heads} が複数の行の頭にある"
                   "（行頭・行末の偏りが有意）。行頭の記号が同じ字形か、行番号・記号の枠などの UI 要素でないかを確認する")
        k += 1
    top = st["top_symbols"][:5]
    tops = "、".join(_nm(t["symbol"]) + "（" + str(t["count"]) + "回）" for t in top)
    out.append(f"{k}. **頻度の高いクラス** {tops}: すべての出現が同じ字形か"
               "（違う字形をまとめた誤り）を確認する。規則の根拠になるのはこれらのクラス")
    k += 1
    out.append(f"{k}. **1回だけのクラス {st['hapax_types']} 種**: それぞれが、頻度の高いクラスの変種（大きさ・太さ・欠けの違い）でないかを確認する。"
               "変種なら訂正ファイルの merge で統合する。アノテーション UI では、認識器の一致度（proposed.match_iou）の低い記号から見る")
    k += 1
    out.append(f"{k}. **改行**: {st['lines']} 行（左の領域9行・右の領域4行）が文の区切りか、折り返しか（continues_sentence）を確認する")
    k += 1
    out.append(f"{k}. **文字以外の要素**: 罫線・枠・アイコンなどの UI 要素が記号候補に混ざっていないか（kind = rule / decoration）")
    return out


if __name__ == "__main__":
    print(build())
