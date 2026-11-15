"""対訳の帰無対照の探索（問題記録 #100）。

本物の対訳での語順の票（動詞の位置・主語と目的語の順序）と、対訳を文の間で無作為に入れ替えた対照での票を比べる。
解読器は変えずに、fit 済みの Decipherer でアラインメントだけをやり直す（状態は戻す）。

使うデータ（PUD の 14 言語、合成言語の開発シード）は結果を見た開発用のもの。
"""
from __future__ import annotations

import copy
import json
import random
import sys
from collections import Counter
from pathlib import Path

from babel.decipher.engine import Decipherer
from babel.experiments import real_languages as rl
from babel.experiments import real_parallel as rp
from babel.generator.corpus import build_corpus
from babel.generator.languages import FAMILIES

STATE = ("lexicon", "affix_fn", "slot_pos", "order_counts", "role_stats", "aligner")


def permuted(parallel, seed):
    """対訳を文の間で入れ替える（どの文も元の対訳にならない入れ替え）。"""
    rng = random.Random(seed)
    trs = [p["translation"] for p in parallel]
    idx = list(range(len(trs)))
    while True:
        rng.shuffle(idx)
        if all(i != j for i, j in enumerate(idx)) or len(idx) < 2:
            break
    return [dict(p, translation=trs[j]) for p, j in zip(parallel, idx)]


def order_counts_for(d, parallel):
    """fit 済みの d で、与えた対訳だけを使ってアラインメントをやり直し、語順の票を返す（d の状態は元に戻す）。"""
    saved = {k: copy.copy(getattr(d, k)) for k in STATE}
    pub = d.pub
    d.pub = dict(pub, parallel=parallel)
    d.lexicon, d.affix_fn, d.slot_pos = {}, {}, {}
    d.order_counts = {"so": Counter(), "adj": Counter(), "verb": Counter()}
    try:
        d._fit_alignment()
        return {k: dict(v) for k, v in d.order_counts.items()}
    finally:
        d.pub = pub
        for k, v in saved.items():
            setattr(d, k, v)


def verb_class_support(d, parallel):
    """品詞の制約を外したアラインメントで、述語の概念に最も強く結び付く語幹が「動詞とみなしたクラス」に入る割合。"""
    saved = {k: copy.copy(getattr(d, k)) for k in STATE}
    pub, cfg = d.pub, d.cfg
    d.pub = dict(pub, parallel=parallel)
    d.cfg = dict(cfg, pos_constraint=False)
    d.lexicon, d.affix_fn, d.slot_pos = {}, {}, {}
    d.order_counts = {"so": Counter(), "adj": Counter(), "verb": Counter()}
    try:
        d._fit_alignment()
        hits = total = 0
        for p, link in zip(parallel, d.aligner.links):
            pred = p["translation"].get("pred")
            if not pred:
                continue
            cand = [(a, s) for (s, c), a in link.items() if c == "C:" + pred and s.startswith("S:")]
            if not cand:
                continue
            a, s = max(cand)
            if a < 0.25:
                continue
            total += 1
            hits += d.stem_pos.get(s[2:]) == "V"
        return {"hits": hits, "total": total, "share": round(hits / total, 3) if total else None}
    finally:
        d.pub, d.cfg = pub, cfg
        for k, v in saved.items():
            setattr(d, k, v)


def study(public, k=19):
    d = Decipherer(public).fit()
    real = {k_: dict(v) for k_, v in d.order_counts.items()}
    real["verb_class_support"] = verb_class_support(d, public["parallel"])
    nulls = []
    for s in range(k):
        perm = permuted(public["parallel"], s)
        x = order_counts_for(d, perm)
        x["verb_class_support"] = verb_class_support(d, perm)
        nulls.append(x)
    return real, nulls


def _stats(counts):
    n = sum(counts.values())
    top = max(counts.values()) if counts else 0
    return {"n": n, "top": top, "share": round(top / n, 3) if n else None,
            "value": max(counts, key=counts.get) if counts else None}


def _show(row):
    real, nulls = row["real"], row["nulls"]
    sup = [x["verb_class_support"]["share"] for x in nulls if x["verb_class_support"]["share"] is not None]
    print(row["corpus"], "verb", _stats(real["verb"]), "so", _stats(real["so"]),
          "| null so n max", max(sum(x["so"].values()) for x in nulls),
          "| support real", real["verb_class_support"], "null max", max(sup) if sup else None, flush=True)


def main(out="results/alignment_null/study-dev.json"):
    rows = []
    for fam in FAMILIES:
        for seed in range(10):
            public, secret = build_corpus(fam, seed, "A")
            real, nulls = study(public)
            rows.append({"corpus": f"{fam}-{seed}-A", "kind": "synthetic", "gold": secret["structure"],
                         "parallel": len(public["parallel"]), "real": real, "nulls": nulls})
            _show(rows[-1])
    src, _ = rp.fetch(*rp.SOURCE)
    en_tr = {s["sent_id"]: t for s in rp.parse(src) if (t := rp.translation(s))}
    for repo, code in rp.TARGETS:
        try:
            path, _ = rp.fetch(repo, code)
            ts = rp.target_sentences(rp.parse(path))
            sents = [ws for _, ws, _ in ts]
            words, _ = rl.cipher(sents)
        except ValueError:
            continue                      # 文字の種類が多い言語（評価用に取っておく）は使わない
        gold = {"verb_position": rl.gold_verb_position(sents)["value"], "so_order": rp.gold_so(ts)["value"]}
        public, _ = rp.build_public(f"real-{code}", words, [s for s, _, _ in ts], en_tr, 1.0)
        real, nulls = study(public)
        rows.append({"corpus": code, "kind": "pud", "gold": gold, "parallel": len(public["parallel"]),
                     "real": real, "nulls": nulls})
        _show(rows[-1])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"note": "開発用（結果を見たデータ）", "rows": rows}, ensure_ascii=False) + "\n",
                         encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:2])
