"""評価結果の Markdown 表。"""
from __future__ import annotations


def _f(x):
    return "—" if x is None else x


def write_markdown(res, path):
    m = res["meta"]
    L = [f"# 構造解析の評価（{m['split']}、シード {m['seeds']}）", "",
         f"- コミット: {m['commit']}　/　Python {m['python']}　/　全体 {m['seconds']} 秒（{m['jobs']} 並列）",
         f"- 設定: {m['config']}", "",
         "正解（設計した規則・単語境界）は採点にだけ使い、解析器には境界を除いた記号列だけを渡した。", "",
         "## 行数ごとのまとめ", "",
         "| 行数 | 無作為: 偽の仮説を出したコーパス | 無作為: 構造あり判定 | 無作為: 旧BABEL構造あり | 設計した規則の発見 | 判定不能 "
         "| 規則のあるコーパスで構造あり | 同・旧BABEL | 偽の率 A / B / C | 仮説数 A / B / C | 平均秒 |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"]
    for o in res.get("overall", []):
        fr, ht = o["false_rate_by_tier"], o["hypotheses_by_tier"]
        L.append(f"| {o['size']} | {o['null_corpora_with_false_hypothesis']}/{o['null_corpora']} | {o['null_corpora_judged_structured']}/{o['null_corpora']} | "
                 f"{o['null_corpora_old_babel_structured']}/{o['null_corpora']} | {o['designed_rules_found']}/{o['designed_rules']} | "
                 f"{o['designed_rules_undecided']} | {o['structured_corpora_detected']}/{o['structured_corpora']} | "
                 f"{o['structured_corpora_old_babel_detected']}/{o['structured_corpora']} | "
                 f"{_f(fr['A'])} / {_f(fr['B'])} / {_f(fr['C'])} | {ht.get('A', 0)} / {ht.get('B', 0)} / {ht.get('C', 0)} | {o['mean_seconds']} |")
    L += ["", "## 規則の発見と誤検出", "",
         "| 生成器 | 原理 | 行数 | 実行 | 規則 | 発見 | 判定不能 | 見逃し | 仮説 | 真 | 偽 | 検証不能 | 偽の率 "
         "| 偽を出したコーパス | 構造あり判定 | 旧BABEL構造あり | 平均秒 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in res["summary"]:
        L.append(f"| {r['generator']} | {r['principle']} | {r['mean_lines']} | {r['runs']} | {r['gold_rules']} | {r['gold_found']} | "
                 f"{r['gold_undecided']} | {r['gold_missed']} | {r['hypotheses']} | {r['true']} | {r['false']} | {r['unverifiable']} | "
                 f"{_f(r['false_discovery_rate'])} | {r['corpora_with_any_false_hypothesis']}/{r['runs']} | {r['structure_detected']}/{r['runs']} | "
                 f"{r['old_babel_structure_detected']}/{r['runs'] - r['old_babel_errors']}"
                 f"{' (失敗' + str(r['old_babel_errors']) + ')' if r['old_babel_errors'] else ''} | {r['mean_seconds']} |")
    L += ["", "## 単語境界の F1（正解の境界との一致、平均）", "",
          "unigram は既存 BABEL の分割（変更前）。", ""]
    methods = ["unigram", "transition", "branching", "mdl", "repeats", "consensus"]
    L += ["| 生成器 | 行数 | " + " | ".join(methods) + " |", "|---|---:|" + "---:|" * len(methods)]
    for r in res["summary"]:
        if r["segmentation_f1"]:
            L.append(f"| {r['generator']} | {r['mean_lines']} | " + " | ".join(str(r["segmentation_f1"].get(x, "—")) for x in methods) + " |")
    L += ["", "## 生成器", ""] + [f"- **{k}**: {v}" for k, v in m["generators"].items()]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
