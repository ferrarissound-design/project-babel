"""Equal-data comparison of the old method A (PR #31 RoleInference) and decoder v2 as translations are thinned.

Development split only. For keep = 1/k, every k-th translated sentence (in corpus order) keeps its
translation; both methods receive the identical public JSON. Predictions are made before annotations
are read, and both are scored with role_recovery_score.score_predictions.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from babel.experiments import real_parallel as rp
from babel.experiments import role_recovery as rr
from babel.experiments.decode_eval import KNOWN19, to_rows

ROOT = Path(__file__).resolve().parents[2]


def job(args):
    from babel.decipher.role_inference import RoleInference
    from babel.decode.engine import Decoder
    from babel.experiments.candidate_audit import direct_triples
    from babel.experiments.clause_probe import target_view
    from babel.experiments.role_recovery_score import score_predictions

    code, k = args
    manifest = json.loads(rr.MANIFEST.read_text())
    public, meta, sid_for_key = rr.build_input(code, "dev", 0, False, manifest)
    public["parallel"] = public["parallel"][::k]
    public = json.loads(rr.canonical(public))
    old = RoleInference(json.loads(json.dumps(public))).predict_all()["A"]
    new = to_rows(Decoder(json.loads(json.dumps(public))).fit().grounded())
    repo = dict((c, r) for r, c in rr.TARGETS)[code]
    raw = {s["sent_id"]: s for s in rp.parse(rp.fetch(repo, code)[0])}
    gold = {key: direct_triples(target_view(raw[sid])[0]) for key, sid in sid_for_key.items()}
    keep = ("eligible", "claimed", "correct", "wrong", "abstain")
    return {"code": code, "keep": f"1/{k}", "translations": len(public["parallel"]),
            "A": {f: score_predictions(old, gold)[f] for f in keep},
            "v2": {f: score_predictions(new, gold)[f] for f in keep}}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keeps", default="1,2,4")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results/decode_v2/dev-learning-curve.json"))
    args = ap.parse_args()
    todo = [(c, int(k)) for k in args.keeps.split(",") for c in KNOWN19]
    with ProcessPoolExecutor(args.jobs) as pool:
        rows = list(pool.map(job, todo))
    Path(args.out).write_text(json.dumps({"rows": rows}, indent=1) + "\n", encoding="utf-8")
    for k in args.keeps.split(","):
        sub = [r for r in rows if r["keep"] == f"1/{k}"]
        agg = {m: {f: sum(r[m][f] for r in sub) for f in ("eligible", "claimed", "correct", "wrong")} for m in ("A", "v2")}
        print(f"keep 1/{k}", json.dumps(agg))


if __name__ == "__main__":
    main()
