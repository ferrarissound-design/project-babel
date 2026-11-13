"""dev の結果だけから、票の閾値 k と字形の類似度の tau を選ぶ。holdout を走らせる前に実行し、結果を固定する。

規則（dev の結果を見る前に決めた）:
- 票の閾値 k: 仕込みのない dev 画像で「1つでも組を報告する割合」が 0.05 以下になる最小の k（妥当性を満たす範囲で、見落としが最も少ない）
- tau: 同じ割合が 0.05 以下の候補のうち、仕込んだ組の発見率が最大のもの。満たす候補がなければ None（分類を経ない検定は不採用）
"""
from __future__ import annotations

import json
from pathlib import Path

from . import bench, summary

LIMIT = 0.05


def select(dev):
    runs = dev["runs"]
    votes = {k: summary.score(runs, f"vote>={k}") for k in range(1, summary.N_MEMBERS + 1)}
    k_star = next((k for k in range(1, summary.N_MEMBERS + 1) if votes[k]["fwer"] <= LIMIT), None)
    soft = {tau: summary.score(runs, f"soft@{tau}") for tau in bench.SOFT_TAUS}
    ok = [tau for tau in bench.SOFT_TAUS if soft[tau]["fwer"] <= LIMIT]
    tau_star = max(ok, key=lambda t: (soft[t]["recall"], -t)) if ok else None
    return {"rule": {"limit": LIMIT, "vote": "FWER <= limit となる最小の k", "tau": "FWER <= limit の候補で recall が最大"},
            "vote_threshold": k_star, "soft_tau": tau_star,
            "dev_votes": {str(k): {x: v[x] for x in ("fwer", "recall", "precision", "false_per_positive_image")} for k, v in votes.items()},
            "dev_soft": {str(t): {x: v[x] for x in ("fwer", "recall", "precision", "false_per_positive_image")} for t, v in soft.items()},
            "dev_cases": len(runs)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="results/robust/bench-dev.json")
    ap.add_argument("--out", default="results/robust/selected.json")
    a = ap.parse_args()
    sel = select(json.loads(Path(a.dev).read_text(encoding="utf-8")))
    Path(a.out).write_text(json.dumps(sel, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(sel, ensure_ascii=False, indent=1))
