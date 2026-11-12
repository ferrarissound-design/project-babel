"""研究ダッシュボード用の JSON を生成する（docs/data/）。表示は静的な HTML/CSS/JS が行う。"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from babel import external
from babel.decipher.engine import Decipherer
from babel.evaluate.score import _same_concept, score
from babel.experiments import scorecard
from babel.generator.corpus import build_corpus
from babel.generator.languages import FAMILIES

FAMILY_INFO = {
    "KES": "膠着語・接尾辞・母音調和・SOV・10進法",
    "UMB": "接頭辞型・VSO・20進法",
    "SOT": "屈折語（融合語尾・2変化類・一致）・SVO・12進法・不規則動詞",
}


def _dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def showcase(seed=0):
    out = []
    for fam in FAMILIES:
        for lv in "ABC":
            public, secret = build_corpus(fam, seed, lv)
            d = Decipherer(public).fit()
            pred = d.predict()
            sc = score(pred, secret, public)
            aliases = secret.get("aliases", {})
            gold_of = defaultdict(Counter)
            for did, sents in secret["tokens"].items():
                for gs, ps in zip(sents, pred["tokens"][did]):
                    for g, p in zip(gs, ps):
                        if not g.get("lacuna"):
                            gold_of[p["stem"]][g["concept"]] += 1
            lex = []
            for e in sorted(pred["lexicon"], key=lambda e: -e["count"])[:40]:
                g = gold_of[e["stem"]].most_common(1)[0][0] if gold_of[e["stem"]] else None
                lex.append({"stem": e["stem"], "forms": e["forms"][:3], "count": e["count"], "pos": e["pos"],
                            "meaning": e["meaning"], "confidence": e["confidence"], "status": e["status"],
                            "reason": e["reason"], "gold": g,
                            "correct": None if e["meaning"] is None else _same_concept(e["meaning"], g, aliases)})
            examples = []
            for did in public["split"]["heldout"][:3]:
                for si in range(min(3, len(secret["tokens"][did]))):
                    toks = []
                    for g, p in zip(secret["tokens"][did][si], pred["tokens"][did][si]):
                        if g.get("lacuna"):
                            toks.append({"s": "[…]", "lacuna": True})
                            continue
                        toks.append({"s": g["s"], "parts": [p["s"][a:b] for a, b in
                                                            zip([0] + p["cuts"], p["cuts"] + [len(p["s"])])],
                                     "pos": p["pos"], "concept": p["concept"], "feats": p["feats"],
                                     "gold": g["concept"], "gold_feats": g["feats"],
                                     "ok": None if p["concept"] is None else _same_concept(p["concept"], g["concept"], aliases)})
                    examples.append({"doc": did, "tokens": toks})
            gold_num = {}
            for sents, psents in ((secret["tokens"][k], pred["tokens"][k]) for k in secret["tokens"]):
                for gs, ps in zip(sents, psents):
                    for g, p in zip(gs, ps):
                        if not g.get("lacuna") and g["pos"] == "NUM":
                            gold_num[p["norm"]] = int(g["concept"][1:])
            nums = [{"form": f, "value": v, "gold": gold_num.get(f)} for f, v in
                    sorted(pred["numerals"]["values"].items(), key=lambda kv: kv[1])]
            history = []
            for frac in (0.25, 0.5, 0.75):
                sub = dict(public, split={"train": public["split"]["train"][: int(len(public["split"]["train"]) * frac)],
                                          "heldout": []})
                sub["parallel"] = [p for p in public["parallel"] if p["doc"] in set(sub["split"]["train"])]
                history.append(_snapshot(Decipherer(sub).fit(), frac))
            history.append(_snapshot(d, 1.0))
            out.append({
                "family": fam, "typology": FAMILY_INFO[fam], "level": lv, "seed": seed,
                "language_id": public["language_id"],
                "samples": [{"id": doc["id"], "meta": doc["meta"], "sentences": doc["sentences"][:6]}
                            for doc in public["documents"][:3]],
                "frequency": pred["frequency"], "verdict": pred["verdict"], "summary": pred["summary"],
                "structure": {k: {"value": v.get("value"), "confidence": v.get("confidence"),
                                  "gold": secret["structure"].get(k), "reason": v.get("reason")}
                              for k, v in pred["structure"].items() if k != "sum_marker"},
                "sum_marker": pred["structure"]["sum_marker"]["value"],
                "hypotheses": pred["hypotheses"], "variants": pred["variants"]["rules"],
                "affixes": [{"affix": a["affix"], "types": a["types"], "tokens": a["tokens"],
                             "function": a["function"]} for a in pred["morphology"]["affixes"][:14]],
                "numerals": {"status": pred["numerals"]["status"], "reason": pred["numerals"]["reason"],
                             "relative": pred["numerals"]["relative"], "equations": pred["numerals"]["equations"],
                             "confidence": pred["numerals"]["confidence"], "values": nums,
                             "affix_hypotheses": pred["numerals"].get("affix_hypotheses", [])},
                "lexicon": lex, "examples": examples, "history": history,
                "score": {"segmentation_f1": sc["segmentation"]["f1"], "pos_accuracy": sc["pos"]["accuracy"],
                          "lexicon": sc["lexicon"], "gloss_heldout": sc["gloss_heldout"],
                          "numerals": sc["numerals"], "structure": sc["structure"]},
                "errors": sc["errors"][:6],
            })
    return out


def _snapshot(d, frac):
    st = d._structure()
    claimed = sum(1 for s in d.lexicon if d._claim(s)[0])
    return {"fraction": frac, "documents": len(d.train_ids),
            "verb_position": [st["verb_position"]["value"], st["verb_position"]["confidence"]],
            "affix_position": [st["affix_position"]["value"], st["affix_position"]["confidence"]],
            "numeral_base": [st["numeral_base"]["value"], st["numeral_base"]["confidence"]],
            "so_order": [st["so_order"]["value"], st["so_order"]["confidence"]],
            "numerals_solved": len(d.num_values), "meanings_claimed": claimed}


def issues_markdown(issues):
    lines = ["# 発見された問題と対応の記録", "",
             "開発中に見つかった問題を、原因・対応・検証とともに残す。未解決のものも隠さず載せる。", ""]
    for i in issues:
        lines += [f"## {i['id']}. {i['title']}（{i['status']}）", "", f"- 発見: {i['found']}", f"- 原因: {i['cause']}",
                  f"- 対応: {i['fix']}", f"- 検証: {i['verified']}", ""]
    return "\n".join(lines)


def report_markdown(results, card, ext_scores, limits):
    """研究報告書（Markdown）。数値はすべて results から機械的に埋める（手書きの転記をしない）。"""
    conds = {c["label"]: c for c in results["conditions"]}
    m = results["meta"]

    def v(label, key, d=3):
        x = conds.get(label, {}).get("metrics", {}).get(key, {})
        return "–" if x.get("mean") is None else f"{x['mean']:.{d}f} ± {x['std']:.{d}f}"

    def tab(labels, keys, names):
        out = ["| 条件 | " + " | ".join(names) + " |", "|---|" + "---|" * len(keys)]
        for lab, name in labels:
            if lab in conds:
                out.append(f"| {name} | " + " | ".join(v(lab, k) for k in keys) + " |")
        return out

    L = ["# PROJECT BABEL 研究報告書", "",
         "この文書は `python -m babel report` が実験結果から自動生成する。数値は手で書き換えていない。", "",
         "## 1. 実行条件", "",
         f"- 実行日時 (UTC): {m['started_utc']}", f"- Git コミット: `{m['commit']}`", f"- 実行環境: {m['environment']}",
         f"- シード: {m['seeds'][0]}〜{m['seeds'][-1]}（{len(m['seeds'])} 個、役割: {'評価用' if m['seed_role'] == 'evaluation' else '開発用'}）",
         f"- 総実行数: {m['total_runs']}（失敗 {m['failed_runs']}）、所要 {m['seconds']} 秒",
         f"- 1言語あたり約 {m['sentences_per_language']} 文（3系統で約 {3 * (m['sentences_per_language'] or 0)} 文）",
         f"- 同一条件の再実行で出力ハッシュ一致: {results['reproducibility']['identical']}", "",
         "## 2. 主結果（3系統 × 評価シードの平均 ± 標準偏差）", ""]
    keys = ["seg_f1", "affix_f1", "pos_accuracy", "verb_position_correct", "affix_position_correct", "numeral_base_correct",
            "numeral_accuracy", "numeral_wrong_rate"]
    names = ["形態素境界F1", "接辞目録F1", "品詞", "動詞位置", "接辞側", "基数", "数詞値 正解", "数詞値 誤答"]
    main_rows = [("levelA", "LEVEL A"), ("levelB", "LEVEL B"), ("levelC", "LEVEL C"),
                 ("levelA:no_totals", "LEVEL A 合計行なし"), ("levelC:no_totals", "LEVEL C 合計行なし")]
    L += tab(main_rows, keys, names) + [""]
    keys2 = ["lexicon_precision", "lexicon_claimed", "gloss_correct", "gloss_wrong", "gloss_abstain", "overclaims"]
    names2 = ["語彙主張の適合率", "主張語数", "語釈 正解", "語釈 誤答", "語釈 判定不能", "過剰主張(件)"]
    L += tab(main_rows, keys2, names2) + [""]
    L += ["語釈は、対訳が与えられていない held-out 文書の内容語（数詞を除く）について、正解・誤答・判定不能の割合。", ""]
    prev_p = Path("results/v1.0.1-on-eval-200-219.json")
    if prev_p.exists():
        prev = {c["label"]: c for c in json.loads(prev_p.read_text(encoding="utf-8"))["conditions"]}

        def mv(table, label, key):
            x = table.get(label, {}).get("metrics", {}).get(key, {})
            return "–" if x.get("mean") is None else f"{x['mean']:.3f}"
        ks = [("numeral_accuracy", "数詞値 正解"), ("numeral_wrong_rate", "数詞値 誤答"), ("lexicon_precision", "語彙適合率"),
              ("lexicon_claimed", "主張語数"), ("gloss_correct", "語釈 正解"), ("gloss_wrong", "語釈 誤答"), ("pos_accuracy", "品詞")]
        L += ["### 旧版との比較（同じ評価シード、v1.0.1 → v1.1）", "",
              "v1.1 の凍結後に、旧版 v1.0.1 の解読器を同じ評価シードで実行した。シードの違いによる差を除いた比較になる。", "",
              "| 条件 | " + " | ".join(n for _, n in ks) + " |", "|---|" + "---|" * len(ks)]
        for lab, name in main_rows:
            if lab in conds:
                L.append(f"| {name} | " + " | ".join(f"{mv(prev, lab, k)} → {mv(conds, lab, k)}" for k, _ in ks) + " |")
        L.append("")
    L += ["## 3. 単純な方法との比較（LEVEL A）", ""]
    k3 = ["seg_f1", "lexicon_precision", "lexicon_claimed", "gloss_correct", "gloss_wrong"]
    n3 = ["形態素境界F1", "語彙適合率", "主張語数", "語釈 正解", "語釈 誤答"]
    L += tab([("levelA", "解読エンジン"), ("levelA:baseline_freq", "頻度法"), ("levelA:baseline_ngram", "n-gram法")], k3, n3) + [""]
    L += ["## 4. アブレーション（LEVEL A）", ""]
    L += tab([("levelA", "全機能"), ("levelA:no_variants", "表記揺れ正規化なし"), ("levelA:no_morph", "形態素解析なし"),
              ("levelA:no_em", "EMなし"), ("levelA:no_numeral_solver", "数詞ソルバーなし"), ("levelA:no_abstain", "保留なし"),
              ("levelA:no_pos_constraint", "品詞制約なし")],
             k3 + ["numeral_accuracy"], n3 + ["数詞値"]) + [""]
    L += ["## 5. データ量・対訳量・ノイズ", ""]
    k5 = ["seg_f1", "numeral_accuracy", "lexicon_precision", "gloss_correct", "gloss_wrong"]
    n5 = ["形態素境界F1", "数詞値", "語彙適合率", "語釈 正解", "語釈 誤答"]
    L += tab([("levelA:docs=33", "文書25%"), ("levelA:docs=65", "文書50%"), ("levelA", "文書100%"),
              ("levelA:parallel=0.01", "対訳1%"), ("levelA:parallel=0.025", "対訳2.5%"), ("levelA:parallel=0.1", "対訳10%"),
              ("levelA:variation=0.0", "表記揺れなし"), ("levelA:variation=0.9", "表記揺れ2倍"),
              ("levelA:lacuna=0.0", "欠損なし"), ("levelA:lacuna=0.08", "欠損4倍"),
              ("levelA:heldout=late_era", "後期時代のみ held-out")], k5, n5) + [""]
    L += tab([("levelC:docs=33", "LEVEL C 文書25%"), ("levelC:docs=65", "LEVEL C 文書50%"), ("levelC", "LEVEL C 文書100%"),
              ("levelC:no_numeral_solver", "LEVEL C 数詞ソルバーなし"), ("levelC:no_variants", "LEVEL C 正規化なし")],
             ["seg_f1", "pos_accuracy", "numeral_accuracy", "numeral_wrong_rate"], ["形態素境界F1", "品詞", "数詞値 正解", "数詞値 誤答"]) + [""]
    if "no_word_boundaries" in conds:
        L += ["単語境界のない連続書記での単語境界F1: 単語ユニグラム反復推定 "
              f"{v('no_word_boundaries', 'word_boundary_f1_engine')}、n-gram 遷移確率 "
              f"{v('no_word_boundaries', 'word_boundary_f1_ngram_baseline')}。", ""]
    L += ["## 6. 解読不能性の判断", ""]
    for kind, name in (("random", "文法のない無作為な語列"), ("shuffled", "実コーパスの語順を破壊")):
        n = results.get("null", {}).get(kind)
        if n:
            L.append(f"- {name}（{n['runs']} 実行）: 「統語あり」と誤判定 {n['syntax_detected_rate']:.2f}、"
                     f"意味の主張 {n['meaning_claims_total']} 件、構造の主張 平均 {n['structure_claims_mean']} 件")
    ca = conds.get("levelA", {})
    L += [f"- LEVEL A の語彙主張の期待較正誤差 ECE = {ca.get('lexicon_ece')}",
          f"- 対訳に現れない語を保留した割合 = {v('levelA', 'abstain_without_evidence')}", "",
          "| 信頼度の帯 | 件数 | 平均信頼度 | 実際の正解率 |", "|---|---|---|---|"]
    L += [f"| {c['bin']} | {c['n']} | {c['confidence']} | {c['accuracy']} |" for c in ca.get("lexicon_calibration", [])]
    L += ["", "## 7. 誤答例と失敗原因（LEVEL A）", "", "| 系統-シード | 語幹 | 推定 | 正解 | 信頼度 | 原因 |", "|---|---|---|---|---|---|"]
    L += [f"| {e['family']}-{e['seed']} | `{e['stem']}` | {e['predicted']} | {e['gold']} | {e['confidence']} | {e['cause']} |"
          for e in ca.get("error_examples", [])[:12]]
    causes = Counter(e["cause"] for e in ca.get("error_examples", []))
    L += ["", "原因の内訳（抽出した誤答例）: " + "、".join(f"{k} {n}件" for k, n in causes.most_common()), "",
          "## 8. 初見言語テスト", ""]
    if ext_scores:
        L += ["| コーパス | 試行 | 独立性 | 構造（正/誤/保留） | 語彙主張（正/誤） | 数詞（正/誤/保留） | 語釈 正解/誤答 |", "|---|---|---|---|---|---|---|"]
        for s_ in ext_scores:
            st = Counter(s_.get("structure", {}).values())
            lx, nm, gh = s_.get("lexicon", {}), s_.get("numerals", {}), s_.get("gloss_heldout", {})
            L.append(f"| {s_['attempt']['name']} | {s_['attempt']['attempt']} | {'独立' if s_['independent'] else '独立でない（準初見）'} | "
                     f"{st['correct']}/{st['wrong']}/{st['abstain']} | {lx.get('correct')}/{lx.get('wrong')} | "
                     f"{nm.get('correct')}/{nm.get('wrong')}/{nm.get('abstain')} | {gh.get('correct', '–')}/{gh.get('wrong', '–')} |")
        L += ["", "構造の内訳:", ""]
        for s_ in ext_scores:
            L.append(f"- {s_['attempt']['name']}: " + "、".join(f"{k}={x}" for k, x in s_.get("structure", {}).items()))
        L.append("")
    if not any(s_.get("independent") for s_ in ext_scores):
        L += ["**独立した初見テストは未評価。** 上の表にあるのは、解読器と同じAIが解読器の凍結後に作った系統（準初見）だけで、"
              "独立テストの代わりにはならない。第三者が作ったコーパスを `external/inbox/` に投入すると評価される。", ""]
    L += ["## 9. 採点（100点満点、算出式は事前登録）", "", "| 項目 | 得点 | 配点 | 状態 |", "|---|---|---|---|"]
    L += [f"| {i['name']} | {'未評価' if i['points'] is None else i['points']} | {i['max']} | {i['status']} |" for i in card["items"]]
    L += ["", f"**内部テスト: {card['internal_points']} / {card['evaluated_max']} 点**（未評価の配点 {card['unevaluated_max']} 点を除く）。"
          "独立初見テストの点数は別枠で、現時点では未評価。", ""]
    for i in card["items"]:
        L += [f"### {i['name']}", ""] + [f"- {d}" for d in ([i["detail"]] if isinstance(i["detail"], str) else i["detail"])] + [""]
    L += ["## 10. 研究上の限界", ""] + [f"- {x}" for x in limits] + [""]
    return "\n".join(L)


def build(results_path="results/latest.json", out_dir="docs/data", with_showcase=True):
    out = Path(out_dir)
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    issues = json.loads(Path("research/issues.json").read_text(encoding="utf-8"))
    limits = json.loads(Path("research/limitations.json").read_text(encoding="utf-8"))
    extras_p = Path("research/verification.json")
    extras = json.loads(extras_p.read_text(encoding="utf-8")) if extras_p.exists() else {}
    ext_scores = []
    sdir = external.ROOT / "scores"
    for f in sorted(sdir.rglob("attempt-*.json")) if sdir.exists() else []:
        ext_scores.append(json.loads(f.read_text(encoding="utf-8")))
    indep = [s for s in ext_scores if s.get("independent") and s.get("blind") and s["attempt"]["first_attempt"]]
    if indep:
        pts, detail = [], []
        for s in indep:
            lx, nm = s.get("lexicon", {}), s.get("numerals", {})
            st = [v for v in s.get("structure", {}).values()]
            acc = sum(v == "correct" for v in st) / len(st) if st else 0.0
            p = 10 * (0.5 * acc + 0.25 * (lx.get("precision") or 0.0) + 0.25 * nm.get("accuracy", 0.0))
            pts.append(p)
            detail.append(f"{s['attempt']['name']}: 構造 {acc:.2f}, 語彙適合率 {lx.get('precision')}, 数詞 {nm.get('accuracy')}")
        extras["external_independent"] = {"points": sum(pts) / len(pts), "detail": detail}
    card = scorecard.build(results, issues, extras)
    slim = dict(results)
    slim["runs"] = [{k: r.get(k) for k in ("experiment", "label", "family", "seed", "ok", "seconds", "error")}
                    for r in results["runs"]]
    _dump(out / "results.json", slim)
    _dump(out / "scorecard.json", card)
    _dump(out / "issues.json", {"issues": issues, "limitations": limits})
    _dump(out / "external.json", {"ledger": external.read_ledger(), "scores": ext_scores,
                                  "problems": external.verify(),
                                  "engine_fingerprint": external.engine_fingerprint()})
    hist_p = Path("results/history.jsonl")
    hist = [json.loads(x) for x in hist_p.read_text(encoding="utf-8").splitlines() if x.strip()] if hist_p.exists() else []
    _dump(out / "history.json", hist)
    if with_showcase:
        _dump(out / "showcase.json", showcase())
    Path("docs/ISSUES.md").write_text(issues_markdown(issues), encoding="utf-8")
    Path("docs/RESEARCH_REPORT.md").write_text(report_markdown(results, card, ext_scores, limits), encoding="utf-8")
    Path("results/scorecard.json").write_text(json.dumps(card, ensure_ascii=False, indent=1), encoding="utf-8")
    return card
