"""100点満点の採点表。算出式は docs/RESEARCH_PLAN.md に事前登録したもの。未実施項目は「未評価」。"""
from __future__ import annotations

import ast
import json
from pathlib import Path

from babel import external
from babel.experiments.run import DEV_SEEDS


def check_import_isolation():
    """解読器が生成器・採点器を import していないこと、正解ファイルを開く文字列を持たないこと。"""
    bad = []
    for p in sorted((Path(__file__).parents[1] / "decipher").glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "")] + [f"{node.module or ''}.{a.name}" for a in node.names]
            for n in names:
                if any(x in n for x in ("generator", "evaluate", "experiments")):
                    bad.append(f"{p.name}: import {n}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and \
                    any(x in node.value.lower() for x in ("secret", "answer.json")):
                bad.append(f"{p.name}: 文字列 {node.value!r}")
    return bad


def check_public_only():
    from babel.generator.corpus import build_corpus
    bad = []
    for lv in "ABC":
        pub, sec = build_corpus("KES", 0, lv, n_docs=12)
        blob = json.dumps(pub, ensure_ascii=False)
        if any(k in pub for k in ("tokens", "lexicon", "grammar", "structure", "numerals")):
            bad.append(f"level {lv}: public に正解キーが含まれる")
        if lv == "C" and (pub["parallel"] or pub["known"] or any(d["meta"] for d in pub["documents"])):
            bad.append("level C: 意味の手がかりが漏れている")
        if '"concept"' in blob or '"canon"' in blob:
            bad.append(f"level {lv}: public にトークン正解が含まれる")
    return bad


def check_heldout():
    from babel.generator.corpus import build_corpus
    pub, _ = build_corpus("SOT", 1, "A", n_docs=40)
    tr, he = set(pub["split"]["train"]), set(pub["split"]["heldout"])
    bad = []
    if tr & he or not he:
        bad.append("train と held-out が重複、または held-out が空")
    if any(p["doc"] in he for p in pub["parallel"]):
        bad.append("held-out 文書に対訳が与えられている")
    return bad


def build(results: dict, issues: list, extras: dict | None = None) -> dict:
    extras = extras or {}
    conds = {c["label"]: c for c in results["conditions"]}

    def m(label, key, default=None):
        v = conds.get(label, {}).get("metrics", {}).get(key, {}).get("mean")
        return default if v is None else v

    items = []

    def item(name, mx, pts, detail, status="評価済み"):
        items.append({"name": name, "max": mx, "points": None if pts is None else round(pts, 2),
                      "status": status, "detail": detail})

    # 1. 実験設計・データ分離
    seeds = results["meta"]["seeds"]
    checks = [
        ("解読器は生成器・採点器を import しない", not check_import_isolation()),
        ("解読器への入力に正解が含まれない", not check_public_only()),
        ("開発シードと評価シードが分離されている", not (set(seeds) & set(DEV_SEEDS))),
        ("held-out 文書が分離され対訳も与えられない", not check_heldout()),
        ("外部予測の台帳に改竄がない", not external.verify()),
    ]
    item("実験設計・データ分離", 15, 3 * sum(ok for _, ok in checks),
         [f"{'○' if ok else '×'} {n}" for n, ok in checks])
    # 2. 文法・言語構造
    parts, tot = [], 0.0
    for lv in "ABC":
        lab = f"level{lv}"
        sc = (m(lab, "seg_f1", 0) * 5 + m(lab, "affix_f1", 0) * 3 + m(lab, "pos_accuracy", 0) * 4 +
              (m(lab, "verb_position_correct", 0) + m(lab, "affix_position_correct", 0)) / 2 * 4 +
              (m(lab, "numeral_base_correct", 0) + m(lab, "numeral_accuracy", 0)) / 2 * 4)
        tot += sc / 3
        parts.append(f"LEVEL {lv}: {sc:.2f}/20（境界F1 {m(lab, 'seg_f1', 0):.2f}, 接辞F1 {m(lab, 'affix_f1', 0):.2f}, "
                     f"品詞 {m(lab, 'pos_accuracy', 0):.2f}, 動詞位置 {m(lab, 'verb_position_correct', 0):.2f}, "
                     f"接辞側 {m(lab, 'affix_position_correct', 0):.2f}, 基数 {m(lab, 'numeral_base_correct', 0):.2f}, "
                     f"数詞値 {m(lab, 'numeral_accuracy', 0):.2f}）")
    item("文法・言語構造の推定", 20, tot, parts)
    # 3. 意味
    a_gloss, a_prec = m("levelA", "gloss_correct", 0), m("levelA", "lexicon_precision", 0)
    b_top = min(1.0, m("levelB", "lexicon_correct", 0) / 5)
    b_num = m("levelB", "numeral_accuracy", 0)
    item("証拠に基づく意味の推定", 20, a_gloss * 10 + a_prec * 5 + (b_top + b_num) / 2 * 5,
         [f"LEVEL A held-out 語釈正解率 {a_gloss:.3f} ×10", f"LEVEL A 語彙主張の適合率 {a_prec:.3f} ×5",
          f"LEVEL B 地名の再現率 {b_top:.3f}・数詞値 {b_num:.3f} の平均 ×5"])
    # 4. 一般化
    base, era = m("levelA", "gloss_correct", 0), m("levelA:heldout=late_era", "gloss_correct", 0)
    keep = min(1.0, era / base) if base else 0.0
    item("未知言語への一般化（内部）", 5, keep * 5,
         [f"後期時代のみ held-out での語釈正解率 {era:.3f} / 通常 held-out {base:.3f} = 維持率 {keep:.2f}"])
    ext = extras.get("external_independent")
    if ext:
        item("未知言語への一般化（独立初見テスト）", 10, ext["points"], ext["detail"])
    else:
        item("未知言語への一般化（独立初見テスト）", 10, None,
             ["独立した第三者が作った未知言語コーパスがまだ採点されていない。"
              "external/inbox に投入し、予測の提出後に正解を追加すると評価される。"], "未評価")
    # 5. 不確実性
    c_runs = [r for r in results["runs"] if r.get("experiment") == "E1_main" and r.get("level") == "C" and r["ok"]]
    c_clean = sum(1 for r in c_runs if r["metrics"]["overclaims"] == 0) / max(1, len(c_runs))
    nul = results.get("null", {})
    det = [1 - nul[k]["syntax_detected_rate"] for k in ("random", "shuffled") if k in nul]
    null_rate = sum(det) / len(det) if det else 0.0
    ece = conds.get("levelA", {}).get("lexicon_ece")
    cal = max(0.0, 1 - (ece if ece is not None else 0.25) / 0.25)
    abst = m("levelA", "abstain_without_evidence", 0)
    item("不確実性・解読不能性の判断", 15, c_clean * 5 + null_rate * 4 + cal * 3 + abst * 3,
         [f"LEVEL C で意味の過剰主張が0件だった実行の割合 {c_clean:.2f} ×5"
          "（注: LEVEL C では解読器に意味ラベルの供給源がなく、構造上0件になる。能力の証明としては弱い）",
          f"無構造・語順破壊コーパスを「統語なし」と判定した割合 {null_rate:.2f} ×4",
          f"語彙主張の較正 ECE={ece} → {cal:.2f} ×3", f"対訳に現れない語を保留した割合 {abst:.3f} ×3"])
    # 6. 問題発見と修正（自己採点）
    fixed = [i for i in issues if i.get("status") == "修正済み"]
    item("自律的な問題発見と修正", 10, min(10, len(fixed)),
         [f"記録した問題 {len(issues)} 件、うち修正済み {len(fixed)} 件（1件1点、上限10点）。"
          "自己採点であり利益相反がある。docs/ISSUES.md を読んで人間が再評価してほしい。"], "暫定（自己採点）")
    # 7. 再現性
    rep = results.get("reproducibility", {})
    pts, det7, status = (2.0 if rep.get("identical") else 0.0), [f"同一シード再実行の出力ハッシュ一致: {rep.get('identical')}"], "評価済み"
    cross = extras.get("cross_env")
    digest = (rep.get("digests") or [None])[0]
    if cross and cross.get("local_digest") != digest:
        # 検証記録が別の実行（旧版など）のものなら、今回の結果の再現性の証拠にはならない
        det7.append(f"ローカルと GitHub Actions の一致: 未評価（検証記録は別の実行 {cross.get('local_digest')} のもので、"
                    f"今回の出力 {digest} とは照合していない）")
        cross = None
        status = "一部未評価"
    elif cross:
        pts += 2.0 if cross["match"] else 0.0
        det7.append(f"ローカルと GitHub Actions のハッシュ一致: {cross['match']}（{cross.get('note', '')}）")
    else:
        det7.append("ローカルと GitHub Actions の一致: 未評価")
        status = "一部未評価"
    pages = extras.get("pages")
    if pages:
        pts += 1.0
        det7.append(f"GitHub Pages 公開確認: {pages}")
    else:
        det7.append("GitHub Pages の公開確認: 未評価（リポジトリ設定での有効化が必要）")
        status = "一部未評価"
    mx7 = 2 + (2 if cross else 0) + (1 if pages else 0)
    item("再現性・成果物の公開", 5, pts, det7 + [f"評価できた配点は {mx7}/5 点分"], status)
    scored = [i for i in items if i["points"] is not None]
    unev = 10 * (ext is None) + (5 - mx7)
    return {"items": items, "internal_points": round(sum(i["points"] for i in scored), 2),
            "evaluated_max": 100 - unev, "unevaluated_max": unev,
            "note": "内部テストの点数。生成器と解読器を同じAIが作っているため、未知言語への一般化の証明にはならない。",
            "seed_role": results["meta"]["seed_role"], "commit": results["meta"]["commit"]}
