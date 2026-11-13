"""docs/VISION_EVAL_V3.md を results/vision/v3/*.json から機械的に作る: python -m babel.vision_eval.report_v3

行の過剰分割の原因分析、行認識の方式の比較（dev・val）、holdout の最終評価、既存の評価データでの退行の確認、
構造解析への影響。v1・v2 の記録（docs/VISION_EVAL.md・VISION_EVAL_V2.md、results/vision/*.json）は上書きしない。
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

R = Path("results/vision/v3")


def _load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _m(a, d=2):
    return "—" if not a else f"{a['mean']:.{d}f}"


def _ms(a, d=2):
    return "—" if not a else f"{a['mean']:.{d}f} ± {a['sd']:.{d}f}"


def _paired(runs, a, b, key, lower_is_better=False):
    ix = defaultdict(dict)
    for r in runs:
        ix[r["name"]][r["method"]] = r.get(key)
    d = [v[a] - v[b] for v in ix.values() if isinstance(v.get(a), (int, float)) and isinstance(v.get(b), (int, float))]
    if not d:
        return None
    pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
    return {"n": len(d), "mean": statistics.mean(d), "better": neg if lower_is_better else pos,
            "worse": pos if lower_is_better else neg, "same": sum(x == 0 for x in d)}


def _lines_table(res, key="all", extra=False):
    L = ["| 方式 | 行 F1 | 行数比（予測/正解） | 過剰分割（余分な行/枚） | 誤結合（行/枚） | 行の誤りのない画像 | 検出 F1 | 分割/枚 | 結合/枚 | ARI | NMI |"
         + (" 秒/枚 | 最大メモリ KiB |" if extra else ""),
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|---:|" if extra else "")]
    for name, s in res["summary"].items():
        a, t = s[key], s["totals"]
        row = (f"| {name} | {_m(a['line_f1'], 3)} | {_m(a['row_ratio'])} | {_m(a['oversplit_extra_rows'])} | {_m(a['merged_rows'])} | "
               f"{t['images_without_line_error'] if key == 'all' else '—'}{'/' + str(res['meta']['images']) if key == 'all' else ''} | "
               f"{_m(a['f1'], 3)} | {_m(a['split_glyphs'], 1)} | {_m(a['merged_predictions'], 1)} | {_m(a['ari'], 3)} | {_m(a['nmi'], 3)} |")
        if extra:
            row += f" {_m(a['seconds'], 3)} | {_m(a['peak_kib'], 0)} |"
        L.append(row)
    return L


def _breakdown(res, metric, prefix, d=2, methods=None):
    first = next(iter(res["summary"].values()))
    keys = [k for k in first if k.startswith(prefix)]
    L = ["| 方式 | " + " | ".join(k.split(":", 1)[1] for k in keys) + " |", "|---|" + "---:|" * len(keys)]
    for name, s in res["summary"].items():
        if methods and name not in methods:
            continue
        L.append(f"| {name} | " + " | ".join(_m(s[k][metric], d) for k in keys) + " |")
    return L


def build(out="docs/VISION_EVAL_V3.md"):
    diag = _load("oversplit-diagnosis-dev.json")
    sel = _load("selected-params.json")
    sw1, sw2, swv = _load("hybrid-sweep-dev-1-alignment.json"), _load("hybrid-sweep-dev-2.json"), _load("hybrid-sweep-val.json")
    mdev, mval, hold = _load("lines-methods-dev.json"), _load("lines-methods-val.json"), _load("lines-holdout.json")
    old = {k: _load(f"methods-{k}-v3.json") for k in ("dev", "val", "test")}
    imp, imp_old = _load("impact-v3.json"), _load("impact-old-seeds-v3.json")
    figs = {k: _load(f"figures/index-{k}.json") for k in ("dev", "holdout")}
    L = ["# 画像認識の評価 v3: 行の過剰分割の原因と、新しい行認識（improved-v3）", "",
         "このファイルは `python -m babel.vision_eval.report_v3` が results/vision/v3/ の JSON から生成する。"
         "v1・v2 の記録（docs/VISION_EVAL.md、docs/VISION_EVAL_V2.md、results/vision/*.json）は上書きしていない。", "",
         "問い: **BABEL が実画像で行を過剰に分割する原因は何か。その誤りを直すと、未知文字の構造解析は本当に改善するのか。**", ""]

    # ------------------------------------------------------------ 0. 実画像
    L += ["## 0. 実画像（IMG_4494）について: 画像がないので、再現も再評価もしていない", "",
          "- 画像本体は、リポジトリにも、この作業環境にもない（docs/ISSUES.md #60）。残っているのは baseline の未検証の転写"
          "（results/vision/real-img4494/corpus-continuous.json、13行・114記号）だけで、記号の位置はない",
          "- 「improved・improved-v2 で 25行（左17・右8）、baseline で 13行」という数値は依頼で伝えられたもので、"
          "その認識結果（glyph-report）もリポジトリにない。**この報告では再現できていない**",
          "- したがって、実画像での原因は**確定していない**。以下の原因分析は、コードの読解と、行の作りを変えた人工画像での再現による",
          "- 実画像での新方式の行数・記号数・クラス数、仮説 H01 が残るかどうかは、**測っていない**。過去の転写からの推測もしていない",
          "- 画像が届いたら `python -m babel image-study prepare`（docs/REAL_IMAGE_STUDY.md）が、4方式の認識結果と、"
          "行の分け方を並べた比較画像 rows-comparison.png を作る。行数が 13 に近いかどうかは精度の根拠にしない（正解転写で測る）", ""]

    # ------------------------------------------------------------ 1. 原因
    L += ["## 1. 過剰分割の原因", "",
          "### コードから分かること（babel/vision/recognizers.py の segment_components、improved・improved-v2 が使う）", "",
          "1. 尺度は「すべての連結成分の高さの中央値」`med_h`。点・短い線・字の部品・雑音の破片も1成分として数える",
          "2. 高さが `0.35 × med_h` 以上の成分は「大きな成分」として、縦の中心の順に並べ、行の中心（それまでの成分の平均）から "
          "`0.6 × med_h` より離れると**新しい行を始める**",
          "3. 「小さな成分」（高さ `0.35 × med_h` 未満）だけが、近い行に付けられる。小さな成分は行を作らない", "",
          "この作りでは、1つの字が上下に離れた部品からなるとき、(a) 部品が多いほど `med_h` が字の高さより小さくなり、"
          "(b) 上の部品と下の部品は縦の中心が違うので、`0.6 × med_h` を超えて別の行になる。"
          "字の高さが `med_h` の2倍あれば、同じ行の上半分と下半分が2行に分かれる（行数がほぼ2倍になる）。", ""]
    if diag:
        g = diag["groups"]
        a = g["all"]
        tot = a["torn_rows"]["mean"] + a["partly_torn_rows"]["mean"] + a["whole_glyph_rows"]["mean"]
        L += ["### 再現実験（dev の人工画像 " + str(a["A_current/extra_rows"]["n"]) + " 枚、前処理は improved と同じ）", "",
              "「正解の字の高さ」を尺度に入れた行は診断のためのもので、認識器としては使わない。", "",
              "| 画像の群 | 現行の尺度 ÷ 正解の字の高さ | S ÷ 正解の字の高さ | 余分な行/枚（現行） | 尺度だけ正解の字の高さに替える | 尺度だけ S に替える | "
              "誤結合/枚（現行） |", "|---|---:|---:|---:|---:|---:|---:|"]
        for k, d in g.items():
            if k.startswith("cell"):
                continue
            L.append(f"| {k} | {_m(d['median_height_over_true'])} | {_m(d['scale_S_over_true'])} | {_m(d['A_current/extra_rows'])} | "
                     f"{_m(d['A_true_height/extra_rows'])} | {_m(d['A_scale_S/extra_rows'])} | {_m(d['A_current/merged_rows'])} |")
        cur, tru, ss = a["A_current/extra_rows"]["mean"], a["A_true_height/extra_rows"]["mean"], a["A_scale_S/extra_rows"]["mean"]
        L += ["", f"- 現行の尺度は、正解の字の高さの平均 {_m(a['median_height_over_true'])} 倍（上下に分かれた字 stack では "
              f"{_m(g['family:stack']['median_height_over_true'])} 倍）しかない",
              f"- 行のまとめ方を変えず、尺度だけを正解の字の高さに替えると、余分な行は 1 枚あたり {cur:.2f} → {tru:.2f}"
              f"（{100 * (1 - tru / cur):.0f}% 減）。**過剰分割の大部分は尺度の過小評価による**",
              f"- 残り（{tru:.2f}/枚）は、尺度が正しくても縦の中心だけでは同じ行にできない部品（字の上下の記号、"
              "上に伸びる字・下に伸びる字の部品）による。diacritic・tight・mixed に集中する",
              f"- 余分な行 {tot:.2f}/枚 の内訳: 本体の行にも部品がある字の部品だけでできた行（字が上下に裂けた）"
              f"{a['torn_rows']['mean']:.2f}（{100 * a['torn_rows']['mean'] / tot:.0f}%）、一部がそうである行 "
              f"{a['partly_torn_rows']['mean']:.2f}（{100 * a['partly_torn_rows']['mean'] / tot:.0f}%）、"
              f"字が丸ごと別の行になったもの {a['whole_glyph_rows']['mean']:.2f}（{100 * a['whole_glyph_rows']['mean'] / tot:.0f}%）",
              f"- 尺度を字の大きさ S（成分の max(幅, 高さ) の、インク量で重みを付けた中央値）に替えるだけでは {ss:.2f}/枚 が残る"
              "（S も、上下に分かれた字では字の高さより小さい）。閾値・尺度の調整だけでは直りきらない（第3節の A'）", "",
              "**結論（人工画像での再現の範囲で）**: 原因は「縦の中心だけで行を決めること」単独ではなく、"
              "**小さな部品に引きずられた尺度（成分の高さの中央値）と、縦の中心による行の開始の組み合わせ**。"
              "依頼にあった仮説（縦の中心だけを重視する処理）は、半分だけ当たっている。", "",
              "実画像の 13 → 25 行（左 9 → 17、右 4 → 8）は「ほぼすべての行が2つに分かれた」数で、この仕組み（上の部品の行と下の部品の行）と"
              "矛盾しない。ただし画像で確かめていないので、**実画像の原因としては仮説のまま**。", ""]

    # ------------------------------------------------------------ 2. 評価データ
    L += ["## 2. 評価データ（babel/vision_eval/layout.py）", "",
          "v1・v2 の人工画像（synth.py）は行の間隔が一定で字の高さがほぼそろっており、improved の行 F1 は 0.98 だった。"
          "過剰分割はこのデータでは起きないので、行の作りを変える生成器を新しく作った。正解は行の番号と字の枠（生成時の座標）。", "",
          "| 分割 | シード | 字の作り | 行の作り | 用途 |", "|---|---|---|---|---|",
          "| dev | 200〜207 | font・stroke・stack（上下に分かれた字） | 9種 | 原因の診断、方式と閾値の選択 |",
          "| val | 300〜307 | 同上 | 9種 | 選んだ方式と閾値の確認 |",
          "| holdout | 7000〜7009 | radical（部品の組み合わせ）・curve（曲線＋上下の点） | 9種 + photo | 最終評価に1回だけ |",
          "| 構造解析 | 8000〜8005 | radical・curve | 6種 | 構造解析への影響（第6節） |", "",
          "- 行の作り: plain（対照）、varheight（字ごとに高さが違う）、diacritic（字の上下に小さな記号）、jitter（字が上下にずれる）、"
          "tight（隣の行と上下の範囲が重なる）、uneven（行の間隔が不均一）、rule_touch（罫線が字に接する）、"
          "two_column（2つの文字領域）、mixed（複合）、photo（holdout だけ: 傾き・ぼかし・明るさのむら・雑音・JPEG）",
          "- holdout の字の作り（radical・curve）と劣化（photo）は、dev・val に一度も出てこない。字の描き方の原理も違う",
          "- 行数は画像ごとに 5〜14、字の高さは 16〜34 画素で変える（特定の行数・画素数に合わせた方式を見分けるため）",
          "- v1・v2 の test（シード 1000〜1019）は improved・improved-v2 の評価に使用済みなので、未見データとしては扱わず、"
          "退行の確認にだけ使う（第5節）",
          "- 生成器は、方式を作る途中で2回直した: tight の行の詰め方（上下の字のインクが入り組んで人にも行が読めなかった）と、"
          "diacritic の行の間隔（上下の記号の分だけ行を離す）。どちらも dev の画像を見て決めたもので、holdout は見ていない", ""]

    # ------------------------------------------------------------ 3. 方式の比較
    L += ["## 3. 行認識の方式の比較（dev で選び、val で確かめた）", "",
          "どの方式も、前処理・行の中の部品の結合・分類は improved-v2 と同じで、**行のまとめ方だけ**を差し替えた（babel/vision/lines.py）。", "",
          "| 記号 | 方式 | 概要 |", "|---|---|---|",
          "| A | 縦の中心（現行） | 大きな成分を縦の中心でまとめる。尺度は成分の高さの中央値 |",
          "| A' | 縦の中心・尺度 S | A の尺度だけを字の大きさ S に替える（閾値・尺度の調整だけの候補） |",
          "| B | 行の投影 | インクのある高さの連なりを行とする（隙間が bridge × S 以下ならつなぐ） |",
          "| C | 上下の範囲の重なり | 成分の縦の範囲が行の範囲と tau 以上重なれば同じ行 |",
          "| D | 下端と字の高さ | 大きな成分を下端（基準線）でまとめ、行の範囲を推定し、残りを最も近い行に入れる |",
          "| E | 複合 | 成分の位置から作った投影の谷で帯に分け、隙間の小さい帯をまとめ、小さな部品は縦の近さと横のそろい方で行を決める |", "",
          "選択の基準は、どの方式も同じ: **過剰分割（余分な行）と誤結合（行）の 1 枚あたりの合計が最小**（同点は行 F1）。", ""]
    if sel:
        L += ["dev で選んだ閾値: " + "、".join(f"{k} {json.dumps(v, ensure_ascii=False)}" for k, v in sel["params"].items()), ""]
    if mdev:
        L += [f"### dev（画像 {mdev['meta']['images']} 枚）", ""] + _lines_table(mdev) + [""]
        L += ["過剰分割（余分な行/枚）、行の作りごと:", ""] + _breakdown(mdev, "oversplit_extra_rows", "condition:") + [""]
        L += ["誤結合（行/枚）、行の作りごと:", ""] + _breakdown(mdev, "merged_rows", "condition:") + [""]
        L += ["過剰分割（余分な行/枚）、字の作りごと:", ""] + _breakdown(mdev, "oversplit_extra_rows", "family:") + [""]
    if mval:
        L += [f"### val（画像 {mval['meta']['images']} 枚、閾値は dev のまま）", ""] + _lines_table(mval) + [""]
    if sw2:
        L += ["### E（複合）の閾値の試行（dev）", "",
              "| depth | join_gap | small | 余分な行/枚 | 誤結合/枚 | 合計 | 行 F1 | 検出 F1 | 行の誤りのない画像 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in sw2["rows"][:8] + sw2["rows"][-2:]:
            p = r["params"]
            L.append(f"| {p['depth']} | {p['join_gap']} | {p['small']} | {r['oversplit_extra_rows']:.2f} | {r['merged_rows']:.2f} | "
                     f"{r['line_errors']:.2f} | {r['line_f1']:.3f} | {r['f1']:.3f} | {r['images_without_line_error']} |")
        L += ["", f"（{len(sw2['rows'])} 通りのうち上位 8 と下位 2。全体は results/vision/v3/hybrid-sweep-dev-2.json）", ""]
    if swv:
        best = swv["rows"][0]
        chosen = next((r for r in swv["rows"] if sel and r["params"] == sel["params"]["hybrid"]), None)
        L += ["### E の閾値の確認（val）", "",
              "| depth | join_gap | small | 余分な行/枚 | 誤結合/枚 | 合計 | 行 F1 | 検出 F1 |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in swv["rows"]:
            p = r["params"]
            mark = " ← dev で選んだ値" if chosen is r else ""
            L.append(f"| {p['depth']} | {p['join_gap']} | {p['small']}{mark} | {r['oversplit_extra_rows']:.2f} | {r['merged_rows']:.2f} | "
                     f"{r['line_errors']:.2f} | {r['line_f1']:.3f} | {r['f1']:.3f} |")
        if chosen:
            rank = swv["rows"].index(chosen) + 1
            L += ["", f"dev で選んだ値は、val では {len(swv['rows'])} 通り中 {rank} 位（合計 {chosen['line_errors']:.2f}、"
                  f"val で最良の組は {best['line_errors']:.2f}）。val の結果で値を選び直してはいない。", ""]
    if sw1:
        L += ["### 試して採用しなかったもの", "",
              "- **字の列がそろっているかで帯をまとめる条件**: 上下に分かれた字の上の部品と下の部品は同じ列にあるので、"
              "帯どうしの列のそろい方を条件に加えた。dev では、条件を外しても（align 0.0）結果が同じだった"
              "（results/vision/v3/hybrid-sweep-dev-1-alignment.json。この試行のときの閾値の名前は gap_aligned・align）。"
              "同じ書き出しの行が続くと別々の行でも列がそろうので、区別に使えなかった。条件は入れていない",
              "- **上下の範囲の重なりで帯を作る（C を E の第1段に使う）**: 字が上下にずれる画像で帯が細かく割れ、"
              "別の行の破片どうしがまとまった。投影の谷で分ける方法に替えた", ""]

    # ------------------------------------------------------------ 4. holdout
    if hold:
        m = hold["meta"]
        L += [f"## 4. 最終評価（holdout、シード {m['seeds'][0]}〜{m['seeds'][-1]}、画像 {m['images']} 枚、OpenCV {m['opencv']}）", "",
              "方式と閾値を固定してから1回だけ実行した。字の作り（radical・curve）も photo の劣化も、開発では使っていない。", ""]
        L += _lines_table(hold, extra=True) + [""]
        t = {k: v["totals"] for k, v in hold["summary"].items()}
        L += ["行数の合計（正解 " + str(next(iter(t.values()))["gt_lines"]) + " 行）: "
              + "、".join(f"{k} {v['predicted_rows']} 行" for k, v in t.items())
              + "。行数が正解と同じ画像: " + "、".join(f"{k} {v['images_with_exact_row_count']}/{m['images']}" for k, v in t.items()) + "。",
              "行数が合うことは行の分け方が正しいことを意味しない（分割と結合が打ち消し合う）ので、過剰分割と誤結合を別々に数えている。", ""]
        L += ["### 同じ画像での比較（improved-v3 − improved-v2）", "",
              "| 指標 | 画像 | 差の平均 | v3 が良い | v2 が良い | 同じ |", "|---|---:|---:|---:|---:|---:|"]
        for key, name, low in (("oversplit_extra_rows", "過剰分割（余分な行、少ないほど良い）", True), ("merged_rows", "誤結合（行、少ないほど良い）", True),
                               ("line_f1", "行 F1", False), ("f1", "記号の検出 F1", False), ("split_glyphs", "字の分割（少ないほど良い）", True),
                               ("merged_predictions", "字の結合（少ないほど良い）", True), ("ari", "ARI", False), ("nmi", "NMI", False)):
            p = _paired(hold["runs"], "improved-v3", "improved-v2", key, low)
            if p:
                L.append(f"| {name} | {p['n']} | {p['mean']:+.3f} | {p['better']} | {p['worse']} | {p['same']} |")
        L += ["", "### 行の作りごと（holdout）", "", "過剰分割（余分な行/枚）:", ""] + _breakdown(hold, "oversplit_extra_rows", "condition:")
        L += ["", "誤結合（行/枚）:", ""] + _breakdown(hold, "merged_rows", "condition:")
        L += ["", "行 F1:", ""] + _breakdown(hold, "line_f1", "condition:")
        L += ["", "記号の検出 F1:", ""] + _breakdown(hold, "f1", "condition:")
        L += ["", "ARI:", ""] + _breakdown(hold, "ari", "condition:")
        L += ["", "### 字の作りごと（holdout）", "", "過剰分割 / 誤結合 / 行 F1 / 検出 F1 / ARI:", "",
              "| 方式 | " + " | ".join(m["families"]) + " |", "|---|" + "---|" * len(m["families"])]
        for name, s in hold["summary"].items():
            L.append(f"| {name} | " + " | ".join(
                " / ".join(_m(s[f"family:{f}"][k], d) for k, d in (("oversplit_extra_rows", 2), ("merged_rows", 2), ("line_f1", 3), ("f1", 3), ("ari", 3)))
                for f in m["families"]) + " |")
        L.append("")

    # ------------------------------------------------------------ 5. 退行
    if any(old.values()):
        L += ["## 5. 既存の評価データでの退行の確認（synth.py の dev・val・test）", "",
              "v1・v2 の評価画像（行の間隔が一定）で、improved-v3 が improved-v2 より悪くならないかを確かめた。"
              "**test（シード 1000〜1019）は v1・v2 の評価に使用済みで、未見データではない。** ここでは退行の確認にだけ使う。", "",
              "| 分割 | 方式 | 検出 F1 | ARI | NMI | 分割/枚 | 結合/枚 | 行 F1 | 余分な行/枚 | 誤結合の行/枚 | 秒/枚 | 最大メモリ KiB |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k, res in old.items():
            if not res:
                continue
            for meth, s in res["summary"].items():
                a = s["all"]
                L.append(f"| {k}（{res['meta']['images']} 枚） | {meth} | {_ms(a['f1'], 3)} | {_ms(a['ari'], 3)} | {_m(a['nmi'], 3)} | "
                         f"{_m(a['split_glyphs'], 1)} | {_m(a['merged_predictions'], 1)} | {_ms(a['line_f1'], 3)} | "
                         f"{_m(a.get('line_oversplit_extra_rows'))} | {_m(a.get('line_merged_rows'))} | {_m(a['seconds'], 3)} | {_m(a['peak_kib'], 0)} |")
        L.append("")
        for k, res in old.items():
            if not res:
                continue
            L += [f"同じ画像での比較（{k}、improved-v3 − improved-v2）:", "",
                  "| 指標 | 画像 | 差の平均 | v3 が良い | v2 が良い | 同じ |", "|---|---:|---:|---:|---:|---:|"]
            for key, name, low in (("f1", "検出 F1", False), ("ari", "ARI", False), ("line_f1", "行 F1", False),
                                   ("split_glyphs", "字の分割（少ないほど良い）", True), ("merged_predictions", "字の結合（少ないほど良い）", True)):
                p = _paired(res["runs"], "improved-v3", "improved-v2", key, low)
                if p:
                    L.append(f"| {name} | {p['n']} | {p['mean']:+.4f} | {p['better']} | {p['worse']} | {p['same']} |")
            L.append("")
            keys = [c for c in res["summary"]["improved-v2"] if c != "all"]
            worse = []
            for c in keys:
                a2, a3 = res["summary"]["improved-v2"][c], res["summary"]["improved-v3"][c]
                for metric, nm in (("f1", "検出 F1"), ("line_f1", "行 F1"), ("ari", "ARI")):
                    if a2[metric] and a3[metric] and a3[metric]["mean"] - a2[metric]["mean"] <= -0.02:
                        worse.append(f"{c} の{nm}（{a2[metric]['mean']:.2f} → {a3[metric]['mean']:.2f}）")
            L += [f"{k} で improved-v2 より 0.02 以上悪くなった条件: {'、'.join(worse) or 'なし'}。", ""]

    # ------------------------------------------------------------ 6. 構造解析
    if imp:
        m = imp["meta"]
        L += ["## 6. 構造解析への影響", "",
              f"同じ文章（regular、シード {m['seeds'][0]}〜{m['seeds'][-1]}、{'・'.join(map(str, m['sizes']))} 行）を、行の作りを変えた画像"
              f"（字の作り {'・'.join(m['families'])}、行の作り {'・'.join(m['conditions'])}）に描き、同じ設定の babel analyze"
              f"（置換 {m['config']['permutations']} 回）にかけた。字の作りとシードは、行認識の開発（dev・val）に使っていない。",
              "設計した規則は8件。うち4件（文頭標識 S・文末標識 E・x が各行に1回・v が各行に1回）は行境界に依存する。"
              "採点は v1・v2 と同じ（babel/vision_eval/impact.py）: 予測クラスは位置で対応した正解の記号の多数決で読み替え、"
              "対応しない記号を含む仮説は偽と数える。", "",
              "- A: 正解の転写 / B: baseline / C2: improved-v2 / D: improved-v3",
              "- L-split: 正解の転写の各行を中央で2行に分けたもの / L-merge: 隣り合う2行を1行にまとめたもの"
              "（記号は正しいまま、行境界の誤りだけを加える）", "",
              "| 行数 | 入力 | 実行 | 発見した規則（/8） | うち行境界に依存（/4） | それ以外（/4） | 仮説の数 | 正しい仮説 | 偽の仮説 | "
              "標本の安定性 | 転写の安定性 | 単語境界 F1（合議） | 解析できない | 余分な行/枚 | 誤結合/枚 | 検出 F1 | ARI |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        order = {"A": 0, "L-split": 1, "L-merge": 2, "B": 3, "C2": 4, "D": 5}
        for s in sorted(imp["summary"], key=lambda s: (s["lines"], order.get(s["input"], 9))):
            L.append(f"| {s['lines']} | {s['input']} | {s['runs']} | {_m(s['gold_found'])} | {_m(s['gold_found_line_rules'])} | "
                     f"{_m(s['gold_found_other_rules'])} | {_m(s['hypotheses'], 1)} | {_m(s['true'], 1)} | {_m(s['false'], 1)} | "
                     f"{_m(s['sampling_stability'])} | {_m(s['transcription_stability'])} | {_m(s['seg_f1_consensus'], 3)} | "
                     f"{s['analysis_impossible']} | {_m(s['oversplit_extra_rows'])} | {_m(s['merged_rows'])} | {_m(s['vision_f1'], 3)} | "
                     f"{_m(s['vision_ari'], 3)} |")
        L += ["", "### 同じ画像での比較", "", "| 比較 | 指標 | 画像 | 差の平均 | 前者が多い | 後者が多い | 同じ |", "|---|---|---:|---:|---:|---:|---:|"]
        names = {"gold_found": "発見した規則", "gold_found_line_rules": "行境界に依存する規則", "gold_found_other_rules": "それ以外の規則",
                 "false": "偽の仮説", "hypotheses": "仮説の数"}
        for k, p in imp["paired"].items():
            if p:
                pair, key = k.split("/")
                L.append(f"| {pair.replace('-', ' − ', 1) if pair.count('-') == 1 else pair} | {names[key]} | {p['n']} | {p['mean']:+.2f} | "
                         f"{p['better']} | {p['worse']} | {p['same']} |")
        L += ["", "### 行の作りごと（発見した規則 / 偽の仮説 / 余分な行）", ""]
        conds = m["conditions"]
        L += ["| 行数 | 入力 | " + " | ".join(conds) + " |", "|---:|---|" + "---|" * len(conds)]
        bc = {(s["lines"], s["input"], s["condition"]): s for s in imp["by_condition"]}
        for n in m["sizes"]:
            for inp in ("B", "C2", "D"):
                L.append(f"| {n} | {inp} | " + " | ".join(
                    f"{_m(bc[(n, inp, c)]['gold_found'])} / {_m(bc[(n, inp, c)]['false'], 1)} / {_m(bc[(n, inp, c)]['oversplit_extra_rows'], 1)}"
                    for c in conds) + " |")
        L.append("")
    if imp_old:
        L += ["### v1・v2 と同じ画像（synth.py、シード 1000〜1005、使用済み）での確認", "",
              "| 行数 | 入力 | 実行 | 発見した規則（平均） | 偽の仮説（平均） |", "|---:|---|---:|---:|---:|"]
        by = defaultdict(list)
        for r in imp_old["runs"]:
            by[(r["lines"], r["input"])].append(r)
        for (n, inp), rs in sorted(by.items()):
            L.append(f"| {n} | {inp} | {len(rs)} | {statistics.mean(r['gold_found'] for r in rs):.2f} | {statistics.mean(r['false'] for r in rs):.2f} |")
        L += ["", "（C2 = improved-v2、D = improved-v3。行の間隔が一定の画像なので、行認識の違いはほとんど出ない想定）", ""]

    # ------------------------------------------------------------ 7. 図
    if any(figs.values()):
        L += ["## 7. 旧方式と新方式の比較画像（人工画像）", "",
              "results/vision/v3/figures/ に、baseline・improved-v2・improved-v3 の行の分け方を横に並べた画像を置いた"
              "（同じ行の記号は同じ色の枠、左に行の番号）。holdout の画像は最終評価の後に描いた。実画像の比較画像はない（第0節）。", "",
              "| 画像 | 正解の行 | " + " | ".join(f"{k}（行 / 余分な行 / 誤結合）" for k in ("baseline", "improved-v2", "improved-v3")) + " |",
              "|---|---:|---|---|---|"]
        for idx in figs.values():
            for r in idx or []:
                L.append(f"| [{r['image']}](../results/vision/v3/figures/{r['image']}) | {r['gt_lines']} | " + " | ".join(
                    f"{r['methods'][k]['rows']} / {r['methods'][k]['oversplit_extra_rows']} / {r['methods'][k]['merged_rows']}"
                    for k in ("baseline", "improved-v2", "improved-v3")) + " |")
        L.append("")

    extra = Path("docs/VISION_EVAL_V3_conclusions.md")
    if extra.exists():
        L += [extra.read_text(encoding="utf-8").rstrip(), ""]
    Path(out).write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
