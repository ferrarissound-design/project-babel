"""Tables comparing decoder v2 with the PR #31 methods A–E and simple baselines on identical inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from babel.experiments.decode_eval import KNOWN19, LOCK, TUNED8, UNTUNED11

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("eligible", "claimed", "correct", "wrong", "abstain")


def load_old(path):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for m in "ABCDE":
        out[m] = [{"code": r["code"], "seed": r["seed"], "mismatch": r["mismatch"], "public_sha256": r["public_sha256"],
                   "metrics": r["methods"][m]} for r in d["rows"]]
    return out


def load_new(path, tag=None):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = d["rows"]
    for r in rows:
        if "public_sha256" not in r and tag:
            lock = LOCK / tag / r["phase"] / f"{r['code']}-{r['seed']}-{int(r['mismatch'])}.json"
            if lock.exists():
                r["public_sha256"] = json.loads(lock.read_text(encoding="utf-8"))["public_sha256"]
    return rows


def aggregate(rows, codes, seed, mismatch):
    a = dict.fromkeys(FIELDS, 0)
    n = 0
    for r in rows:
        if r["code"] in codes and r["seed"] == seed and r["mismatch"] == mismatch:
            n += 1
            for f in FIELDS:
                a[f] += r["metrics"][f]
    if not n:
        return None
    a["languages"] = n
    a["precision"] = a["correct"] / a["claimed"] if a["claimed"] else None
    a["accuracy"] = a["correct"] / a["eligible"] if a["eligible"] else None
    a["abstain_rate"] = a["abstain"] / a["eligible"] if a["eligible"] else None
    return a


def check_same_inputs(methods):
    """All methods must have been run on byte-identical public inputs."""
    keys = {}
    bad = []
    for name, rows in methods.items():
        for r in rows:
            k = (r["code"], r["seed"], r["mismatch"])
            if "public_sha256" not in r:
                continue
            if k in keys and keys[k] != r["public_sha256"]:
                bad.append((name, k))
            keys.setdefault(k, r["public_sha256"])
    return bad


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def markdown(methods, groups, seeds=(0, 1, 2)):
    lines = []
    for gname, codes in groups:
        for mismatch in (False, True):
            lines += ["", f"### {gname} — {'一文ずらした対訳（誤対応の対照）' if mismatch else '通常の対訳'}", "",
                      "| 方式 | 換字seed | N | 正解 | 誤答 | 保留 | 適合率 | 正解率 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
            for name, rows in methods.items():
                for seed in seeds:
                    a = aggregate(rows, codes, seed, mismatch)
                    if a:
                        lines.append(f"| {name} | {seed} | {a['eligible']} | {a['correct']} | {a['wrong']} | {a['abstain']} | "
                                     f"{pct(a['precision'])} | {pct(a['accuracy'])} |")
    return "\n".join(lines)


def per_language(methods, codes, seed=0):
    names = list(methods)
    lines = ["| 言語 | " + " | ".join(f"{n} 正/誤" for n in names) + " |", "|---|" + "---|" * len(names)]
    for c in codes:
        cells = []
        for n in names:
            a = aggregate(methods[n], [c], seed, False)
            cells.append(f"{a['correct']}/{a['wrong']}" if a else "—")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


GROUPS = [("既知19言語", KNOWN19), ("設計判断に使った8言語", TUNED8), ("設計判断に使っていない11言語", UNTUNED11),
          ("ガリシア語", ["gl_pud"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("dev", "final"), default="dev")
    ap.add_argument("--new", action="append", default=[], metavar="NAME=PATH[:TAG]")
    ap.add_argument("--out")
    args = ap.parse_args()
    old = load_old(ROOT / f"results/role_recovery/{args.phase}.json")
    methods = {f"旧{m}": rows for m, rows in old.items()}
    for spec in args.new:
        name, rest = spec.split("=", 1)
        path, _, tag = rest.partition(":")
        methods[name] = load_new(path, tag or None)
    bad = check_same_inputs(methods)
    text = f"## {args.phase}\n\n入力の一致（public_sha256）: {'すべて一致' if not bad else f'不一致 {len(bad)} 件'}\n"
    text += markdown(methods, GROUPS) + "\n\n### 言語別（換字seed 0、通常の対訳）\n\n" + per_language(methods, KNOWN19 + ["gl_pud"])
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    if bad:
        raise SystemExit(f"inputs differ: {bad[:5]}")


if __name__ == "__main__":
    main()
