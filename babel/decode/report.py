"""Human-readable Markdown report for a decode result (schema babel-decode-v2)."""
from __future__ import annotations

REASONS = {"incomplete_translation": "対訳に主語か目的語が無い", "below_threshold": "確信度が閾値未満",
           "no_triple": "完全な組の候補が無い", "no_model": "対訳が少なくモデル未学習", "no_posterior": "事後確率なし"}


def _fmt_role(rec, name):
    role = (rec.get("roles") or {}).get(name) or {}
    if role.get("absent_in_translation"):
        return "（対訳に無し）"
    c = role.get("candidates") or []
    if not c:
        return "—"
    top = c[0]
    extra = f"、非表出 {role['p_not_expressed']:.2f}" if "p_not_expressed" in role else ""
    return f"`{top['token']}`@{top['position']}（{top['p']:.2f}{extra}）"


def render(result, max_sentences=40):
    s = result["summary"]
    inp = result["input"]
    st = result["structure"]
    lines = [
        f"# 解読レポート: {result.get('language_id') or '（無名のコーパス）'}",
        "",
        f"エンジン {result['schema']} v{result['version']}。入力 {inp['documents']} 文書・{inp['sentences']} 文、"
        f"うち対訳つき {inp['translated_sentences']} 文。役割モデル: {'学習済み' if inp['role_model_trained'] else '未学習（対訳が少ない）'}。",
        "",
        "## 分かったこと・分からないこと",
        "",
        f"- 対訳つきの文で主語・述語・目的語の組を**主張**: {s['claimed']} 文、**判定不能**: {s['undetermined']} 文。",
        "- 判定不能の理由: " + ("、".join(f"{REASONS.get(k, k)} {v}" for k, v in
                                       sorted(s["undetermined_reasons"].items(), key=lambda x: -x[1])) or "なし") + "。",
        f"- 対訳のない文への**構造仮説**（意味は主張しない）: {s['structural_hypotheses_claimed']} / "
        f"{s['structural_hypotheses_total']} 文。",
        "- 確信度は役割モデルの事後確率（較正済みのときは較正値）で、正解確率の保証ではない。"
        "開発データでの実測の関係は docs/DECODE_ENGINE.md にある。",
        "",
        "## 構造（記号列だけから。意味は使わない）",
        "",
        f"- 語境界: {st['boundaries']['source']}（{st['boundaries']['tokens']} 語、{st['boundaries']['types']} 種）",
        f"- 接辞の位置: {st['morphology']['affix_position']}。頻度の高い接辞: "
        + "、".join(f"`{a['affix']}`（語幹 {a['stem_types']} 種）" for a in st["morphology"]["affixes"][:10]),
    ]
    wo = result.get("word_order") or {}
    if wo:
        v = wo["verb_position_votes"]
        lines += ["", "## 語順（主張した組から）", "",
                  f"- 述語の位置の票: 前 {v['initial']}・間 {v['medial']}・後 {v['final']}（n={wo['n']}、"
                  f"一様に対する片側 p={wo['p_value_vs_uniform']:.3g}）→ "
                  + (f"**{wo['verb_position']}**" if wo["verb_position"] else "判定不能（票が足りないか偏りが有意でない）"),
                  f"- 主語と目的語の前後: {wo['subject_object_votes']}"]
    if st.get("role_profiles"):
        lines += ["", "## 役割ごとの手がかり（対訳つきの文で学習）", ""]
        for role, rows in st["role_profiles"].items():
            if rows:
                lines.append(f"- {role}: " + "、".join(f"{r['template']}=`{r['value']}`×{r['ratio']}" for r in rows[:6]))
    lex = result.get("lexicon") or []
    lines += ["", f"## 語彙（対訳から。{len(lex)} 件）", ""]
    if lex:
        lines += ["| 英語の概念 | 対象の語幹 | 対応の量 | 割合 |", "|---|---|---:|---:|"]
        lines += [f"| {r['concept']} | `{r['stem']}` | {r['mass']} | {r['share']} |" for r in lex[:30]]
    else:
        lines.append("対応を主張できる語はない。")
    an = result.get("anchors")
    if an:
        lines += ["", "## 固有名詞・借用語のアンカー", "",
                  f"記号と英字の対応 {an['symbols_mapped']} 個を学習、確度 0.5 以上の対 {an['confident_pairs']} 件。例: "
                  + "、".join(f"{e['concept']}≈`{e['token']}`（{e['p']}）" for e in an["examples"][:10])]
    lines += ["", f"## 文ごとの結果（対訳つき、先頭 {max_sentences} 文）", "",
              "| 文 | 概念 (主/述/目) | 主語 | 述語 | 目的語 | 確信度 | 状態 |", "|---|---|---|---|---|---:|---|"]
    for rec in result["sentences"][:max_sentences]:
        c = rec.get("concepts") or {}
        conf = rec.get("calibrated_confidence", rec.get("confidence", 0.0))
        status = "主張" if rec["status"] == "claimed" else f"判定不能: {REASONS.get(rec.get('reason_code'), rec['reason'])}"
        lines.append(f"| {rec['doc']}#{rec['sent']} | {c.get('subject') or '—'} / {c.get('predicate')} / "
                     f"{c.get('object') or '—'} | {_fmt_role(rec, 'subject')} | {_fmt_role(rec, 'predicate')} | "
                     f"{_fmt_role(rec, 'object')} | {conf:.2f} | {status} |")
    hyp = [r for r in result.get("structural_hypotheses", []) if r["status"] == "claimed"]
    if hyp:
        lines += ["", f"## 構造仮説（対訳なし、意味は主張しない。先頭 {min(len(hyp), 15)} 件）", ""]
        for rec in hyp[:15]:
            t = rec["triple_tokens"]
            lines.append(f"- {rec['doc']}#{rec['sent']}: 主語候補 `{t['subject']}`・述語候補 `{t['predicate']}`・"
                         f"目的語候補 `{t['object']}`（{rec.get('calibrated_confidence', rec['confidence']):.2f}）")
    if result.get("warnings"):
        lines += ["", "## 警告", ""] + [f"- {w}" for w in result["warnings"]]
    return "\n".join(lines) + "\n"
