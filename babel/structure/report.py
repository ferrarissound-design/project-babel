"""解析結果（JSON）から人が読む Markdown を作る。"""
from __future__ import annotations


def _p(x):
    return "—" if x is None else (f"{x:.4f}" if x >= 0.0001 else f"{x:.1e}")


def _pct(x):
    return "—" if x is None else f"{x:.0%}"


def _loc(e):
    if not isinstance(e, dict) or "line" not in e:
        return ""
    s = f"行{e['line']}"
    if "start" in e:
        s += f":{e['start']}"
    if e.get("image_box"):
        s += f" 画像{e['image_box']}"
    return s


def markdown(a: dict) -> str:
    st, v = a["statistics"], a["verdict"]
    L = [f"# 構造解析レポート（{a['input'].get('language_id') or a['input'].get('image') or a['input'].get('format')}）", ""]
    L += [f"**判定**: {v['text']}", "",
          f"- 解析モード: {a['mode']}　/　置換回数: {a['config']['permutations']}　/　有意水準 {a['config']['alpha']}"
          f"（族ごとの閾値 {a['multiple_testing']['per_family_threshold']}）",
          f"- 計算時間: {a['seconds']['total']} 秒", ""]
    L += ["## 1. コーパスの基本統計", "",
          "| 項目 | 値 |", "|---|---|",
          f"| 行（文） | {st['lines']} |", f"| 文書 | {st['documents']} |", f"| 記号（延べ） | {st['symbols_total']} |",
          f"| 記号（異なり） | {st['symbol_types']} |", f"| 異なり/延べ | {st['type_token_ratio']} |",
          f"| 1回だけの記号 | {st['hapax_types']} |",
          f"| 行の長さ（最小/中央/最大） | {st['line_length']['min']} / {st['line_length']['median']} / {st['line_length']['max']} |",
          f"| 最長反復部分列 | {st['longest_repeat']['length']} 記号（{' '.join(st['longest_repeat']['symbols'])}） |", ""]
    if st["repeated_ngrams"]:
        L += ["反復する n-gram（2回以上）の種類数: " + "、".join(f"{n}記号 {x['types']}種" for n, x in st["repeated_ngrams"].items()), ""]
    L += ["## 2. コーパス全体の検定（無作為な並びとの比較）", "",
          "| 検定 | 観測 | 帰無の平均 | 帰無モデル | p | p（Holm） |", "|---|---:|---:|---|---:|---:|"]
    for k, g in a["global_tests"].items():
        null = "行内シャッフル" if g["null"] == "within" else "全体シャッフル"
        L.append(f"| {g['description']} | {g['observed']} | {g['null_mean']} | {null} | {_p(g['p'])} | {_p(g['p_holm'])} |")
    L += ["", "## 3. 構造の仮説（多重検定を通ったもの）", ""]
    if not a["hypotheses"]:
        L += ["多重検定の補正後に残った仮説はない。", ""]
    else:
        L += ["段階: A = 安定かつ別の方法とも一致 / B = 安定 / C = 有意だが部分標本または転写の攪乱で再現しにくい", "",
              "| ID | 段階 | 種類 | 観測 | 支持 | 反例 | p（補正後） | 標本の安定性 | 転写の安定性 |",
              "|---|---|---|---|---:|---:|---:|---:|---:|"]
        for h in a["hypotheses"]:
            L.append(f"| {h['id']} | {h['tier']} | {h['kind']} | {h['observation']} | {h['support']} | "
                     f"{h['counterexamples'].get('count', '—')} | {_p(h['statistics']['p_family_corrected'])} | "
                     f"{_pct(h['stability']['sampling'])} | {_pct(h['stability']['transcription'])} |")
        L.append("")
        for h in a["hypotheses"]:
            L += [f"### {h['id']} {h['kind']}（段階 {h['tier']}）", "",
                  f"- **観測**: {h['observation']}",
                  "- **根拠の位置**: " + ("、".join(filter(None, (_loc(e) for e in h["evidence"][:6]))) or "—"),
                  f"- **反例**: {h['counterexamples'].get('description')}: {h['counterexamples'].get('count')}",
                  f"- **統計的な強さ**: 補正後 p = {_p(h['statistics']['p_family_corrected'])}（補正前 {_p(h['statistics']['p_raw'])}、"
                  f"帰無モデル: {h['statistics']['null_model']}）",
                  f"- **他の方法との一致**: {'; '.join(h['corroborated_by']) or 'なし'}",
                  f"- **解釈の候補（未検証）**: {' / '.join(h['interpretation']['candidates'])}",
                  f"- **信頼度の限界**: {'; '.join(h['limits']) or '—'}",
                  f"- **追加検証に必要なデータ**: {h['needed_data']}", ""]
    if a.get("experimental_observations"):
        L += ["### 実験的な観察（接辞、規則として扱わない）", "",
              "接辞の検定は評価で偽の率が94%だったため、仮説に含めない。分割候補に依存する参考表示。", ""]
        for h in a["experimental_observations"][:10]:
            L.append(f"- {h['observation']}（補正後 p = {_p(h['statistics']['p_family_corrected'])}）")
        L.append("")
    s = a["structures"]
    if s["substitution_sets"] or s["line_templates"]:
        L += ["### まとめた構造", ""]
        for c in s["substitution_sets"]:
            L.append(f"- 置き換え可能に見える記号の集合: {{{', '.join(c)}}}")
        for t in s["line_templates"]:
            L.append(f"- 似た行のまとまり（行 {', '.join(map(str, t['lines']))}）: `{t['pattern_on_first_line']}`（_ = 変わる部分）")
        L.append("")
    L += ["## 4. 記述的観察（偶然と区別できないもの）", ""]
    if a["descriptive_observations"]:
        L += ["以下は補正前には目立つが、多重検定の補正後または支持数の点で偶然と区別できない。規則として扱わないこと。", ""]
        for d in a["descriptive_observations"]:
            L.append(f"- {d['observation']}（補正前 p = {_p(d['p_raw'])}、補正後 {_p(d['p_family_corrected'])}、{d['status']}）")
    else:
        L.append("なし")
    b = a["boundaries"]
    L += ["", "## 5. 単語境界の候補", "",
          "どれも正解ではない。記述長（ビット）が小さいほど、繰り返し使われる単位でコーパスを短く書ける分割。", "",
          "| 順位 | 方式 | 記述長 | 語の異なり | 他方式との一致（F1） | 部分標本での再現 |", "|---:|---|---:|---:|---:|---:|"]
    for c in b["candidates"]:
        L.append(f"| {c['rank_by_description_length']} | {c['method']} | {c['description_length_bits']} | {c['word_types']} | "
                 f"{c['mean_agreement_with_others']} | {c['subsample_reproducibility'] if c['subsample_reproducibility'] is not None else '—'} |")
    L.append(f"\n基準: 区切りなし {b['baselines_bits']['no_cuts']} ビット / 1記号＝1語 {b['baselines_bits']['every_symbol']} ビット")
    ct = b.get("compression_test")
    if ct:
        L.append(f"\n圧縮の検定: MDL 分割の圧縮率 {ct['observed_ratio']}（行内シャッフル平均 {ct['null_mean_ratio']}）、p = {_p(ct['p'])}"
                 + ("。**境界の統計的な根拠は弱い**" if ct["p"] > a["config"]["alpha"] else "。シャッフルより再利用される単位が多い"))
    if a.get("given_boundaries_agreement"):
        L.append("\n入力に書かれていた区切りとの一致（F1、参考）: " + "、".join(f"{k} {v}" for k, v in a["given_boundaries_agreement"].items()))
    d = a["detectability"]
    L += ["", "## 6. 検出力とデータ不足による制限", ""]
    for k, x in (d.get("bigram_occurrences_needed") or {}).items():
        L.append(f"- {'最頻の2記号' if 'most' in k else '中央頻度の2記号'}（{' '.join(x['symbols'])}）の連接が規則として検出されるには、"
                 f"約 {x['occurrences_needed']} 回以上の出現が必要（偶然の期待値 {x['expected_count_under_null']} 回）")
    for x in a["data_limits"]:
        L.append(f"- {x}")
    L += ["", "## 7. 人間が次に確認すべき箇所", ""]
    for r in a["review_points"] or [{"what": "特になし", "why": "", "where": []}]:
        where = "、".join(filter(None, (_loc(e) if "start" in e or "line" in e and "cut_before_position" not in e else
                                        f"行{e.get('line')} の位置{e.get('cut_before_position', e.get('position'))}"
                                        for e in r["where"][:6])))
        L.append(f"- **{r['what']}** — {r['why']}" + (f"（{where}）" if where else ""))
    L += ["", "## 注意", ""] + [f"- {n}" for n in a["notes"]]
    return "\n".join(L) + "\n"
