"""docs/ROBUST_STRUCTURE.md を results/robust/ の JSON から生成する（数値は JSON からだけ取る）。

python -m babel.robust_eval.report
"""
from __future__ import annotations

import json
from pathlib import Path

from . import summary

R = Path("results/robust")
SPLIT_NAMES = {"dev": "dev", "holdout-synth": "holdout-synth", "holdout-font": "holdout-font"}
METHOD_LABELS = {
    "oracle-ngram": "正解の転写 + 既存の検定（上限）", "oracle-lcs": "正解の転写 + 新しい検定（上限）",
    "ngram:baseline": "baseline + 既存の検定", "ngram:v2": "improved-v2 + 既存の検定", "ngram:v3": "improved-v3 + 既存の検定（現状）",
    "lcs:baseline": "baseline + 新しい検定", "lcs:v2": "improved-v2 + 新しい検定", "lcs:v3": "improved-v3 + 新しい検定",
    "soft@0.2": "分類を経ない検定（tau 0.2）", "soft@0.3": "分類を経ない検定（tau 0.3）", "soft@0.45": "分類を経ない検定（tau 0.45）",
}
MAIN = ["oracle-ngram", "oracle-lcs", "ngram:baseline", "ngram:v2", "ngram:v3", "lcs:baseline", "lcs:v2", "lcs:v3", "soft@0.3"]


def _load(name):
    p = R / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _label(m):
    if m.startswith("vote>="):
        return f"方式の集まり: {m.split('>=')[1]}/8 以上で有意"
    return METHOD_LABELS.get(m, m)


def _f(v, nd=3):
    return "—" if v is None else f"{v:.{nd}f}"


def method_table(S, methods):
    L = ["| 判定 | 仕込みのない画像で1つでも報告（95% 区間） | 仕込みのない画像1枚あたりの報告 | 仕込んだ組の発見率 "
         "| 仕込みのある画像1枚あたりの偽の報告 | 報告のうち真 |",
         "|---|---:|---:|---:|---:|---:|"]
    for m in methods:
        s = S["overall"][m]
        ci = s["fwer_ci95"]
        L.append(f"| {_label(m)} | {_f(s['fwer'])}（{_f(ci[0], 2)}〜{_f(ci[1], 2)}） | {_f(s['false_per_negative_image'])} | {_f(s['recall'])} "
                 f"| {_f(s['false_per_positive_image'])} | {_f(s['precision'])} |")
    return "\n".join(L)


def group_table(S, key, methods, metric):
    groups = list(S[key])
    L = ["| 判定 | " + " | ".join(groups) + " |", "|---|" + "---:|" * len(groups)]
    for m in methods:
        L.append(f"| {_label(m)} | " + " | ".join(_f(S[key][g][m][metric], 2) for g in groups) + " |")
    return "\n".join(L)


def vote_table(S):
    d = S["vote_discrimination"]
    L = ["| 有意だった方式の数 | 真（仕込んだ組） | 偽 | 真の割合（95% 区間） |", "|---:|---:|---:|---:|"]
    for k, v in d["by_votes"].items():
        L.append(f"| {k} | {v['true']} | {v['false']} | {_f(v['p_true'], 2)}（{_f(v['ci95'][0], 2)}〜{_f(v['ci95'][1], 2)}） |")
    return "\n".join(L)


def check_predictions(S_all, k):
    """holdout の結果を見る前に書いた予測（results/robust/predictions.json）を、結果と照らす。"""
    out = []
    hs = {n: S_all[n] for n in ("holdout-synth", "holdout-font") if S_all.get(n)}
    if not hs:
        return out
    vk = f"vote>={k}"

    def each(fn):
        return {n: fn(S) for n, S in hs.items()}
    v = each(lambda S: S["overall"]["oracle-lcs"]["fwer"])
    out.append(("P1", all(0.02 <= x <= 0.10 for x in v.values()), "、".join(f"{n} {_f(x)}" for n, x in v.items())))
    v = each(lambda S: S["overall"][vk]["fwer"])
    out.append(("P2", all(x <= 0.08 for x in v.values()), "、".join(f"{n} {_f(x)}" for n, x in v.items())))
    v = each(lambda S: (S["overall"][vk]["recall"], S["overall"]["ngram:v3"]["recall"]))
    out.append(("P3", all(a > b for a, b in v.values()), "、".join(f"{n} {_f(a)} 対 {_f(b)}" for n, (a, b) in v.items())))

    def p4(S):
        d = S["vote_discrimination"]
        hi_t = sum(x["true"] for kk, x in d["by_votes"].items() if int(kk) >= k)
        hi_f = sum(x["false"] for kk, x in d["by_votes"].items() if int(kk) >= k)
        one = d["by_votes"].get("1", {"true": 0, "false": 0})
        return d["auc"], hi_t / max(1, hi_t + hi_f), one["true"] / max(1, one["true"] + one["false"])
    v = each(p4)
    out.append(("P4", all(a is not None and a >= 0.8 and h >= 0.85 and o <= 0.3 for a, h, o in v.values()),
                "、".join(f"{n} AUC {_f(a)}・票 {k} 以上 {_f(h, 2)}・票 1 {_f(o, 2)}" for n, (a, h, o) in v.items())))
    if "holdout-synth" in hs:
        x = hs["holdout-synth"]["overall"]["lcs:v2"]["fwer"]
        out.append(("P5", x >= 0.2, f"holdout-synth {_f(x)}"))
    v = each(lambda S: (S["overall"]["lcs:v3"]["fwer"], S["overall"][vk]["fwer"]))
    out.append(("P6", all(a > b for a, b in v.values()), "、".join(f"{n} {_f(a)} 対 {_f(b)}" for n, (a, b) in v.items())))
    return out


def build():
    sel = _load("selected.json")
    k = sel["vote_threshold"]
    vk = f"vote>={k}"
    S_all = {}
    for name in SPLIT_NAMES:
        res = _load(f"bench-{name}.json")
        if res:
            S_all[name] = summary.summarize(res)
            S_all[name]["config"] = res["config"]
            S_all[name]["opencv"] = res.get("opencv")
            (R / f"summary-{name}.json").write_text(json.dumps(S_all[name], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    inv = {name: _load(f"inventory-{name}.json") for name in SPLIT_NAMES}
    cb = {name: _load(f"compare-bench-{name}.json") for name in SPLIT_NAMES}
    cmp_ = _load("real-img4494/compare-baseline-v3.json")
    preds = _load("predictions.json")
    checks = check_predictions(S_all, k)
    (R / "prediction-check.json").write_text(json.dumps([{"id": i, "held": ok, "observed": obs} for i, ok, obs in checks],
                                                        ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    from . import prose
    text = prose.render({"S": S_all, "sel": sel, "k": k, "vk": vk, "inv": inv, "cb": cb, "cmp": cmp_, "preds": preds, "checks": checks,
                         "method_table": method_table, "group_table": group_table, "vote_table": vote_table, "f": _f, "label": _label,
                         "MAIN": MAIN})
    Path("docs/ROBUST_STRUCTURE.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    t = build()
    print(f"docs/ROBUST_STRUCTURE.md: {len(t.splitlines())} 行")
