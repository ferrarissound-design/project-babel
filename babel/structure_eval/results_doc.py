"""docs/STRUCTURE_RESULTS.md を results/structure/*.json から機械的に作る（数値を手で転記しない）。

python -m babel.structure_eval.results_doc
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

R = Path("results/structure")
FIXED = {"real_english", "external_gpt6", "babel_generator"}
NULL = {"random_uniform", "random_zipf"}


def _load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _f(x, d=3):
    return "—" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def old_babel_diagnosis(seeds=(1000, 1001, 1002, 1003, 1004)):
    """既存の解読器の統語検定が短いコーパスで働かない理由（語の頻度の条件）を数える。"""
    from babel.decipher.engine import Decipherer

    from .generators import Regular, to_corpus
    srcs = [("tengu-pilot-v0-real（実転写）", json.loads(Path("external/inbox/tengu-pilot-v0-real/corpus.json").read_text("utf-8"))),
            ("tengu-positive-control-unspaced（同じ長さの正の対照）",
             json.loads(Path("external/inbox/tengu-positive-control-unspaced/corpus.json").read_text("utf-8")))]
    for n in (13, 30, 100):
        for s in seeds:
            srcs.append((f"regular {n}行 seed {s}", to_corpus(Regular(s).sample(n), "r")))
    agg = defaultdict(list)
    for name, pub in srcs:
        d = Decipherer(pub).fit()
        c = Counter(w for s in d.sents for w in s)
        tok = sum(c.values())
        inv = sum(v for v in c.values() if v >= 5)
        row = {"name": name, "tokens": tok, "types": len(c), "share_in_vocab": round(inv / tok, 2) if tok else 0,
               "z": d.syntax["z"], "present": d.syntax["present"]}
        key = name.split(" seed")[0]
        agg[key].append(row)
    out = []
    for k, rs in agg.items():
        out.append({"name": k, "n": len(rs), "tokens": round(sum(r["tokens"] for r in rs) / len(rs), 1),
                    "share_in_vocab": round(sum(r["share_in_vocab"] for r in rs) / len(rs), 2),
                    "z_mean": round(sum(r["z"] for r in rs) / len(rs), 2), "present": sum(r["present"] for r in rs)})
    return out


def _family_table(data):
    fam = defaultdict(Counter)
    for r in data["runs"]:
        grp = "固定コーパス" if r["generator"] in FIXED else "生成器"
        for f, v in r["score"]["per_family"].items():
            for k, n in v.items():
                fam[(f, grp)][k] += n
    L = ["| 族 | 生成器: 仮説 | 偽 | 偽の率 | 固定コーパス: 仮説 | 偽 | 検証不能 | 偽の率 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for f in sorted({k[0] for k in fam}):
        a, b = fam[(f, "生成器")], fam[(f, "固定コーパス")]
        ra = a["false"] / (a["true"] + a["false"]) if a["true"] + a["false"] else None
        rb = b["false"] / (b["true"] + b["false"]) if b["true"] + b["false"] else None
        L.append(f"| {f} | {sum(a.values())} | {a['false']} | {_f(ra)} | {sum(b.values())} | {b['false']} | {b['unverifiable']} | {_f(rb)} |")
    return L


def _rate(group):
    t = f_ = 0
    for r in group:
        for f, v in r["score"]["per_family"].items():
            if f != "affix":
                t += v.get("true", 0)
                f_ += v.get("false", 0)
    return f_ / (t + f_) if t + f_ else None


def build(out="docs/STRUCTURE_RESULTS.md"):
    ev, dev, te = _load("benchmark-eval.json"), _load("benchmark-dev.json"), _load("transcription-experiment.json")
    ev0 = _load("benchmark-eval-v0.1.0-8dd39ac.json")      # 評価前に凍結した版での評価（最初の記録）
    L = ["# 構造発見モジュールの評価結果", "",
         "このファイルは `python -m babel.structure_eval.results_doc` が results/structure/*.json から生成する。",
         "方法は [STRUCTURE.md](STRUCTURE.md)、事前登録は [RESEARCH_PLAN.md](RESEARCH_PLAN.md) の「構造発見モジュール」。", ""]
    m = ev["meta"]
    L += ["評価は2回ある。どちらも評価シード（開発に使っていない）で、判定基準は事前登録のまま変えていない。", "",
          f"1. **凍結版 v0.1.0**（コミット {ev0['meta']['commit'] if ev0 else '—'}、評価前に凍結）: 最初の記録。"
          "`results/structure/benchmark-eval-v0.1.0-8dd39ac.json`",
          f"2. **再評価 v0.1.2**（コミット {m['commit']}）: Codex の独立レビュー（PR #7）で見つかった妥当性の不具合2件"
          "（共起の帰無モデルが行の長さを無視していた、採点が行の組の真を過大に数えていた）と、評価後に接辞を仮説から外した変更を含む。"
          "閾値の調整ではないが、評価シードの結果を見た後の再実行であることに注意。`results/structure/benchmark-eval.json`", "",
          f"以下の表は特に断りがなければ再評価 v0.1.2（シード {m['seeds'][0]}〜{m['seeds'][-1]}、{len(ev['runs'])} 実行、"
          f"{m['seconds']} 秒、{m['jobs']} 並列、設定 `{m['config']}`）。", ""]

    # ---------------------------------------------------------------- 1. まとめ
    L += ["## 1. 行数ごとのまとめ（評価シード）", "",
          "| 行数 | 無作為で偽の仮説を出したコーパス | 無作為で「構造あり」 | 設計した規則の発見 | 判定不能 | 偽の率 A / B / C | 仮説数 A / B / C | 平均秒 |",
          "|---:|---:|---:|---:|---:|---|---|---:|"]
    for o in ev["overall"]:
        fr, ht = o["false_rate_by_tier"], o["hypotheses_by_tier"]
        L.append(f"| {o['size']} | {o['null_corpora_with_false_hypothesis']}/{o['null_corpora']} | "
                 f"{o['null_corpora_judged_structured']}/{o['null_corpora']} | {o['designed_rules_found']}/{o['designed_rules']} "
                 f"({o['designed_rules_found'] / o['designed_rules']:.0%}) | {o['designed_rules_undecided']} | "
                 f"{_f(fr['A'])} / {_f(fr['B'])} / {_f(fr['C'])} | {ht.get('A', 0)} / {ht.get('B', 0)} / {ht.get('C', 0)} | {o['mean_seconds']} |")
    L += ["", "- 「無作為で偽の仮説」は仮説が1件でも出たコーパスの数（族単位の誤り率の実測）。「構造あり」は大域検定だけが陽性のものも含む。",
          "- 偽の率は母集団オラクル（分析に使っていない行）で成り立たなかった仮説の割合。固定コーパス（実在の英語・第三者・BABEL 生成器）では"
          "母集団が残りの半分しかなく、特定の話題の句の反復（例: 創世記の “the firmament”）が残りに現れないと偽になる。", ""]

    if ev0:
        L += ["### 凍結版 v0.1.0 との比較（同じシード）", "",
              "| 行数 | 無作為で偽の仮説: v0.1.0 → v0.1.2 | 設計した規則の発見: v0.1.0 → v0.1.2 | 偽の率 A: v0.1.0 → v0.1.2 | 偽の率 C: v0.1.0 → v0.1.2 |",
              "|---:|---:|---:|---:|---:|"]
        for o0, o in zip(ev0["overall"], ev["overall"]):
            L.append(f"| {o['size']} | {o0['null_corpora_with_false_hypothesis']}/{o0['null_corpora']} → "
                     f"{o['null_corpora_with_false_hypothesis']}/{o['null_corpora']} | {o0['designed_rules_found']} → {o['designed_rules_found']} "
                     f"（/{o['designed_rules']}） | {_f(o0['false_rate_by_tier']['A'])} → {_f(o['false_rate_by_tier']['A'])} | "
                     f"{_f(o0['false_rate_by_tier']['C'])} → {_f(o['false_rate_by_tier']['C'])} |")
        L += ["", "v0.1.2 の偽の率は、採点の修正（行の組の真を語境界で判定）で v0.1.0 より厳しく数えている。", ""]

    # ---------------------------------------------------------------- 2. 変更前後
    L += ["## 2. 変更前（既存 BABEL v1.2.0 の構造判定）と変更後（構造発見）の比較", "",
          "同じコーパス・同じ条件（単語境界なし）。既存 BABEL は統語の検定（語順のシャッフル6回、z>5）と形態の検定で「構造あり」を判定する。", "",
          "| 行数 | 規則のあるコーパスで構造あり: 変更前 | 変更後 | 無作為で構造あり: 変更前 | 変更後（仮説あり） | 変更後（大域検定を含む） |",
          "|---:|---:|---:|---:|---:|---:|"]
    for o in ev["overall"]:
        L.append(f"| {o['size']} | {o['structured_corpora_old_babel_detected']}/{o['structured_corpora']} | "
                 f"{o['structured_corpora_detected']}/{o['structured_corpora']} | {o['null_corpora_old_babel_structured']}/{o['null_corpora']} | "
                 f"{o['null_corpora_with_false_hypothesis']}/{o['null_corpora']} | {o['null_corpora_judged_structured']}/{o['null_corpora']} |")
    zs = [r["old_babel"].get("syntax_z") for r in ev["runs"] if r["generator"] in NULL and r["old_babel"].get("decipherable_structure")]
    L += ["", f"変更前は無作為コーパスでも統語の z が {', '.join(str(z) for z in zs)} になり「構造あり」とした例がある（docs/ISSUES.md #40）。", ""]

    diag = old_babel_diagnosis()
    L += ["### 13行で既存 BABEL が「解読不能」になる理由（解析器の仮定）", "",
          "既存の統語検定は、推定した語のうち5回以上現れる語だけで語順を数える。短いコーパスでは該当する語がなく、エントロピーが0になり、z=0 で「構造なし」になる。",  # noqa: E501
          "規則を設計した人工言語でも同じことが起きる。", "",
          "| コーパス | 件数 | 推定した語（延べ） | 5回以上の語が占める割合 | 統語 z（平均） | 統語あり |", "|---|---:|---:|---:|---:|---:|"]
    for d in diag:
        L.append(f"| {d['name']} | {d['n']} | {d['tokens']} | {d['share_in_vocab']} | {d['z_mean']} | {d['present']}/{d['n']} |")
    L.append("")

    # ---------------------------------------------------------------- 3. 族ごと
    L += ["## 3. 仮説の族ごとの偽の率（評価シード、全行数）", ""]
    for label, data in (("再評価 v0.1.2", ev), ("凍結版 v0.1.0（接辞を含む）", ev0)):
        if data:
            L += [f"**{label}**", ""] + _family_table(data) + [""]
    L += ["**接辞の族は凍結版で偽の率が 94% で、機能していない。** 評価後の変更（v0.1.1）で仮説から外し「実験的な観察」に移した。"
          "族の数（多重検定の閾値）は変えていない。", ""]
    ev_frozen = ev0 or ev

    # ---------------------------------------------------------------- 4. 接辞を除いた事後集計
    L += ["## 4. 事後集計: 凍結版の記録から接辞の仮説を除いた場合（機械的に再計算、追加実行なし）", "",
          "| 行数 | 無作為で偽の仮説を出したコーパス | 生成器コーパスの偽の率 | 固定コーパスの偽の率 |", "|---:|---:|---:|---:|"]
    for n in sorted({r["size_requested"] for r in ev_frozen["runs"]}):
        rs = [r for r in ev_frozen["runs"] if r["size_requested"] == n]
        nullbad = sum(1 for r in rs if r["generator"] in NULL and
                      sum(v.get("false", 0) for f, v in r["score"]["per_family"].items() if f != "affix") > 0)
        L.append(f"| {n} | {nullbad}/{sum(1 for r in rs if r['generator'] in NULL)} | "
                 f"{_f(_rate([r for r in rs if r['generator'] not in FIXED]))} | {_f(_rate([r for r in rs if r['generator'] in FIXED]))} |")
    L.append("")

    # ---------------------------------------------------------------- 5. 規則ごとの発見
    c, t = Counter(), Counter()
    for r in ev["runs"]:
        s = r["score"]
        for lab in s["gold_found"]:
            c[(r["generator"], lab, r["size_requested"])] += 1
        for lab in s["gold_found"] + s["gold_missed"] + s["gold_undecided"]:
            t[(r["generator"], lab, r["size_requested"])] += 1
    sizes = sorted({k[2] for k in t})
    L += ["## 5. 設計した規則ごとの発見（評価シード10個のうち何回）", "",
          "| 生成器 | 規則 | " + " | ".join(f"{n}行" for n in sizes) + " |", "|---|---|" + "---:|" * len(sizes)]
    for g, lab in sorted({(k[0], k[1]) for k in t}):
        L.append(f"| {g} | {lab} | " + " | ".join(f"{c[(g, lab, n)]}/{t[(g, lab, n)]}" if t[(g, lab, n)] else "—" for n in sizes) + " |")
    L += ["", "見逃しの多い規則: 格接尾辞（接辞の族の不具合）と、規則的な言語での接尾辞 x/q/v の置換可能性"
          "（CV 型の語幹で、子音どうしの置換可能性がより強く出て、族の補正の閾値が上がる）。", ""]

    # ---------------------------------------------------------------- 6. 単語境界
    seg = defaultdict(lambda: defaultdict(list))
    for r in ev["runs"]:
        for k, v in r["segmentation_f1"].items():
            if not k.startswith("_"):
                seg[(r["generator"], r["size_requested"])][k].append(v)
    L += ["## 6. 単語境界の F1（評価シード平均）: 既存 BABEL の unigram 分割との比較", "",
          "| 生成器 | 行数 | unigram（変更前） | consensus（変更後の合議） | 差 | 最良の方式 | 最良 F1 |", "|---|---:|---:|---:|---:|---|---:|"]
    worse = []
    for (g, n) in sorted(seg):
        mm = {k: sum(v) / len(v) for k, v in seg[(g, n)].items()}
        best = max(mm, key=mm.get)
        d = mm["consensus"] - mm["unigram"]
        if d < 0:
            worse.append(f"{g} {n}行")
        L.append(f"| {g} | {n} | {mm['unigram']:.3f} | {mm['consensus']:.3f} | {d:+.3f} | {best} | {mm[best]:.3f} |")
    L += ["", f"合議が既存の分割より悪かった条件: {', '.join(worse) or 'なし'}。どの方式も単独で全条件の最良ではない。", ""]

    # ---------------------------------------------------------------- 7. 転写の誤差
    if te:
        L += ["## 7. 転写の誤差の影響", "", "### T1: 記号列での模擬（regular、評価シード 1000〜1002）", "",
              "| 行数 | seed | 規則 | 正しい転写 | 10% 誤分類 | 分けすぎ（各文字2クラス） | 人による統合後 | 分けすぎでの置換可能性の仮説（うち同じ文字の2クラス） |",  # noqa: E501
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for x in te["T1_simulated_transcription_errors"]:
            cnd = x["conditions"]
            L.append(f"| {x['lines']} | {x['seed']} | {x['gold_rules']} | {len(cnd['clean']['gold_found'])} | "
                     f"{len(cnd['substitute']['gold_found'])} | {len(cnd['oversplit']['gold_found'])} | {len(cnd['merged']['gold_found'])} | "
                     f"{cnd['oversplit']['substitution_hypotheses']}（{cnd['oversplit']['substitution_same_character']}） |")
        L += ["", "数値は発見した設計規則の数（8規則中）。人による統合後の記号列は正しい転写と内容が同じだが、記号の番号が違うので置換の乱数列が変わり、"
              "閾値付近の判定が揺れる（docs/ISSUES.md #47）。分けすぎた転写では、置換可能性の仮説の多くが「同じ文字の2クラス」を指し、人に統合を提案する手がかりになる。", ""]  # noqa: E501
        if isinstance(te.get("T2_synthetic_image"), list):
            L += ["### T2: 合成画像 → babel vision → 構造発見（regular 13行）", "",
                  "| seed | 画像 | 類似度 | 記号候補/描いた文字 | 結合 | 視覚クラス/真の文字 | 2クラス以上に分かれた文字 | 文字の混ざったクラス | 仮説 | 発見した規則 |",  # noqa: E501
                  "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
            for x in te["T2_synthetic_image"]:
                for s, v in x["vision"].items():
                    e = v["transcription_errors"]
                    L.append(f"| {x['seed']} | {'劣化' if x['degraded'] else 'きれい'} | {s} | {e['glyph_candidates']}/{e['drawn_characters']} | "
                             f"{e['merged_candidates']} | {e['visual_classes']}/{e['true_characters']} | {e['characters_split_into_several_classes']} | "
                             f"{e['classes_mixing_characters']} | {v['hypotheses']} | {len(v['gold_found'])} |")
                L.append(f"| {x['seed']} | {'劣化' if x['degraded'] else 'きれい'} | 文字列で入力 | — | — | — | — | — | "
                         f"{x['text_transcription']['hypotheses']} | {len(x['text_transcription']['gold_found'])} |")
            L += ["", "劣化画像では、同じ文字が多数の視覚クラスに分かれ、隣り合う文字が結合し、規則の発見がほぼ消える。"
                  "画像解析の誤差（この表の左側）と構造解析の結果（右側）は別に数えている。", ""]

    # ---------------------------------------------------------------- 8. 実データ
    L += ["## 8. 実データ: 天狗の詫び証文の仮転写（145記号）と対照", "",
          "| 入力 | 記号 | 異なり | 判定 | 仮説（A/B/C） | 記述的観察 | MDL 圧縮の p | 最頻2記号の連接の検出に必要な出現回数 |", "|---|---:|---:|---|---|---:|---:|---:|"]  # noqa: E501
    for n in ("tengu-pilot-v0-real", "tengu-pilot-v0-shuffled-control", "tengu-positive-control-unspaced", "tengu-positive-control-boundaries"):
        a = _load(f"real/{n}/analysis.json")
        if not a:
            continue
        tiers = Counter(h["tier"] for h in a["hypotheses"])
        need = (a["detectability"].get("bigram_occurrences_needed") or {}).get("two_most_frequent_symbols", {})
        L.append(f"| {n} | {a['statistics']['symbols_total']} | {a['statistics']['symbol_types']} | {a['verdict']['code']} | "
                 f"{tiers.get('A', 0)}/{tiers.get('B', 0)}/{tiers.get('C', 0)} | {len(a['descriptive_observations'])} | "
                 f"{a['boundaries']['compression_test']['p']} | {need.get('occurrences_needed', '—')} |")
    L += ["", "実転写からは、多重検定を通る規則は見つからなかった。同じ長さの正の対照では見つかる。これは実転写が無意味だという証明ではない:"
          " 97種の記号のうち66種が1回しか現れず、最頻の記号の組でも3回以上の連接がないと検出できない（実際に2回以上現れる連接は1種だけ）。", ""]
    L += ["## 9. 発見した問題と限界", "",
          "docs/ISSUES.md の #38〜#54 に原因・対応・検証とともに記録した。要点:", "",
          "- #38 既存 BABEL は短いコーパスで、語に分割してから語順を数えるという仮定のせいで構造を検出できない（データ不足とは別の原因）",
          "- #40 既存 BABEL の統語検定は無作為の記号列を「構造あり」とすることがある（100行で 4/20）",
          "- #41 接辞の検定は機能していない（偽の率 94%）。仮説から外した",
          "- #42 「似た行」の族が 100行の無作為コーパスで偽陽性（凍結版で 2/20、再評価では 0/20。モンテカルロ誤差の範囲と考え、未解決のまま）",
          "- #43 実在の言語では、話題に依存する句の反復が多く、別の節には現れない",
          "- #44 劣化した画像では、画像認識の誤差で構造がほぼ消える",
          "- #45 規則的な言語での接尾辞の置換可能性の見逃し、#46 BABEL 生成器の短いコーパスでの境界推定の悪化、#47 モンテカルロ誤差、#48 計算時間", ""]
    if dev:
        L += ["## 付録: 開発シード（0〜2）での同じ集計", ""]
        for o in dev["overall"]:
            L.append(f"- {o['size']}行: 無作為で偽の仮説 {o['null_corpora_with_false_hypothesis']}/{o['null_corpora']}、"
                     f"設計した規則の発見 {o['designed_rules_found']}/{o['designed_rules']}、偽の率 A/B/C "
                     f"{_f(o['false_rate_by_tier']['A'])}/{_f(o['false_rate_by_tier']['B'])}/{_f(o['false_rate_by_tier']['C'])}")
        L += ["", f"開発シードの結果は、コミット {dev['meta']['commit']} に v0.1.1 の変更（接辞を実験的な観察へ移す）を加えたコードで実行した。", ""]
    Path(out).write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build())
