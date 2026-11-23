"""Fit the decoder's confidence calibrator on DEVELOPMENT predictions only, with leave-one-language-out CV.

Input: results/decode_v2/dev-raw.json (raw posteriors, calibration off). Labels come from the
development annotations already joined by decode_eval (after predictions were locked).

Threshold rule (fixed before the final split is run, docs/PREREG_DECODE_V2.md):
  tau = the smallest grid value whose LOLO precision over the 19 known languages (seeds 0-2 pooled)
  is >= 0.60 with at least 30 claims; otherwise the tau with the most correct claims among those with
  LOLO precision >= 0.50; otherwise 0.5 and the target is reported as not reached.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from babel.decode import calibration
from babel.experiments.decode_eval import KNOWN19

ROOT = Path(__file__).resolve().parents[2]
GRID = [round(0.05 * k, 2) for k in range(2, 20)]


def collect(dev):
    data = defaultdict(list)    # (code, seed, mismatch) -> [(feat, label)]
    for row in dev["rows"]:
        for conf, ok, meta in row["curve"]:
            if meta is not None:
                data[(row["code"], row["seed"], row["mismatch"])].append((meta, ok))
    return data


def lolo(data, codes):
    preds = defaultdict(list)
    for held in codes:
        train = [x for (c, s, m), rows in data.items() if not m and c != held and c in codes for x in rows]
        model = calibration.fit(train)
        for (c, s, m), rows in data.items():
            if c == held:
                preds[(c, s, m)] = [(calibration.predict(model, f), y) for f, y in rows]
    return preds


def table(preds, codes, mismatch=False):
    out = []
    for tau in GRID:
        claimed = correct = 0
        for (c, s, m), rows in preds.items():
            if c in codes and m == mismatch:
                for p, y in rows:
                    if p >= tau:
                        claimed += 1
                        correct += y
        out.append({"tau": tau, "claimed": claimed, "correct": correct,
                    "precision": round(correct / claimed, 4) if claimed else None})
    return out


def choose(tab):
    ok = [r for r in tab if r["precision"] is not None and r["precision"] >= 0.60 and r["claimed"] >= 30]
    if ok:
        return min(ok, key=lambda r: r["tau"]), "precision >= 0.60 reached"
    ok = [r for r in tab if r["precision"] is not None and r["precision"] >= 0.50]
    if ok:
        return max(ok, key=lambda r: (r["correct"], -r["tau"])), "0.60 not reached; best with precision >= 0.50"
    return next(r for r in tab if r["tau"] == 0.5), "precision 0.50 not reached at any threshold"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", default=str(ROOT / "results/decode_v2/dev-raw.json"))
    ap.add_argument("--out", default=str(ROOT / "results/decode_v2/calibration-cv.json"))
    ap.add_argument("--write-model", action="store_true", help="fit on all development rows and write babel/decode/calibration.json")
    args = ap.parse_args()
    dev = json.loads(Path(args.dev).read_text(encoding="utf-8"))
    data = collect(dev)
    preds = lolo(data, KNOWN19)
    normal = table(preds, KNOWN19)
    mismatch = table(preds, KNOWN19, mismatch=True)
    chosen, why = choose(normal)
    res = {"rule": __doc__.split("Threshold rule")[1].strip(), "chosen": chosen, "why": why,
           "lolo_normal": normal, "lolo_mismatch": mismatch, "dev_source_files": dev.get("source_files")}
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for r in normal:
        mm = next(x for x in mismatch if x["tau"] == r["tau"])
        print(r, "mismatch claims", mm["claimed"], "mismatch correct", mm["correct"])
    print("CHOSEN", chosen, why)
    # LOLO predictions at the chosen threshold, per condition, in the comparison-table format
    elig = {(r["code"], r["seed"], r["mismatch"]): r["metrics"]["eligible"] for r in dev["rows"]}
    lolo_rows = []
    for (c, s, m), rows in sorted(preds.items()):
        claimed = sum(p >= chosen["tau"] for p, _ in rows)
        correct = sum(y for p, y in rows if p >= chosen["tau"])
        e = elig[(c, s, m)]
        lolo_rows.append({"code": c, "phase": "dev", "seed": s, "mismatch": m, "metrics": {
            "eligible": e, "claimed": claimed, "correct": correct, "wrong": claimed - correct, "abstain": e - claimed}})
    Path(args.out).with_name("dev-lolo.json").write_text(json.dumps({"tau": chosen["tau"], "rows": lolo_rows}) + "\n",
                                                         encoding="utf-8")
    if args.write_model:
        train = [x for (c, s, m), rows in data.items() if not m and c in KNOWN19 for x in rows]
        model = calibration.fit(train)
        model.update(threshold=chosen["tau"], fitted_on="development split, 19 known PUD languages, cipher seeds 0-2, "
                     "normal translations", rule_outcome=why)
        calibration.PATH.write_text(json.dumps(model, indent=1) + "\n", encoding="utf-8")
        print("wrote", calibration.PATH)


if __name__ == "__main__":
    main()
