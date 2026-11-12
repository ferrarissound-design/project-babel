"""コーパスの組み立て。public（解読器に渡す）と secret（正解）を完全に分けて返す。"""
from __future__ import annotations

import random

from .languages import STRUCTURE, apply_rules, make_language
from .world import GENRES, World, translate

LACUNA = "[…]"
N_SCRIBES = 6


def _pick(rng, weighted):
    x, acc = rng.random(), 0.0
    for k, w in weighted:
        acc += w
        if x < acc:
            return k
    return weighted[-1][0]


def build_corpus(family: str, seed: int, level: str = "A", n_docs: int = 130, parallel_frac: float = 0.05,
                 dropout: float = 0.3, var_rate: float = 0.45, lacuna: float = 0.02, boundaries: bool = True,
                 split: str = "random", n_known: int = 6, totals: bool = True, same_concept_rate: float = 0.0):
    """(public, secret) を返す。level は A / B / C。"""
    lang = make_language(family, seed)
    rng = random.Random(f"corpus-{family}-{seed}")
    world = World(rng, totals=totals, same_concept_rate=same_concept_rate)
    scribes = [[r for r in lang.scribe_rules if rng.random() < var_rate] for _ in range(N_SCRIBES)]
    docs, gold_docs, props_by_sent, eras = [], {}, {}, {}
    for i in range(n_docs):
        did = f"d{i + 1:03d}"
        genre, site, era = _pick(rng, GENRES), rng.choice(world.sites), rng.randrange(3)
        scribe = rng.randrange(N_SCRIBES)
        eras[did] = era
        rules = lang.sound[: era] + scribes[scribe]
        sents, gsents = [], []
        for j, prop in enumerate(world.document(genre, site, era)):
            gtoks = []
            for w in lang.render(prop, era, rng):
                canon = "".join(m[0] for m in w["morphs"])
                morphs = apply_rules(w["morphs"], rules) if rng.random() < 0.95 else w["morphs"]
                surface = "".join(m[0] for m in morphs)
                cuts, pos = [], 0
                for m in morphs[:-1]:
                    pos += len(m[0])
                    cuts.append(pos)
                tok = {"s": surface, "canon": canon, "cuts": sorted(set(c for c in cuts if 0 < c < len(surface))),
                       "morphs": morphs, "concept": w["concept"], "pos": w["pos"], "feats": w["feats"],
                       "sem": w["sem"]}
                if boundaries and rng.random() < lacuna:
                    tok = dict(tok, s=LACUNA, lacuna=True)
                gtoks.append(tok)
            sents.append((" " if boundaries else "").join(t["s"] for t in gtoks))
            gsents.append(gtoks)
            props_by_sent[(did, j)] = prop
        meta = {"site": site, "era": era, "genre": genre} if level in ("A", "B") else {}
        docs.append({"id": did, "meta": meta, "sentences": sents})
        gold_docs[did] = gsents
    ids = [d["id"] for d in docs]
    if split == "era":      # 後期の文書だけを held-out にする（未観測の音変化・語尾変化）
        held = [d for d in ids if eras[d] == 2]
    else:
        held = sorted(random.Random(f"split-{family}-{seed}").sample(ids, max(1, len(ids) // 5)))
    train = [i for i in ids if i not in set(held)]
    public = {"language_id": f"{family}-{seed}", "level": level, "has_word_boundaries": boundaries,
              "documents": docs, "parallel": [], "known": [], "split": {"train": train, "heldout": held}}
    if level == "A":
        cand = [(d, j) for d in train for j in range(len(gold_docs[d]))]
        prng = random.Random(f"par-{family}-{seed}")
        for d, j in sorted(prng.sample(cand, max(1, round(len(cand) * parallel_frac))) if parallel_frac > 0 else []):
            public["parallel"].append({"doc": d, "sent": j,
                                       "translation": translate(props_by_sent[(d, j)], prng, dropout)})
        for s in world.sites[:4]:
            p = world.site_place[s]
            public["known"].append({"form": lang.roots[p], "meaning": p})
        for v in (1, 2):
            public["known"].append({"form": lang.units[v], "meaning": f"#{v}"})
    elif level == "B":
        for con in ["king", "temple", "grain", "give", "silver"][: max(0, n_known - 1)]:
            public["known"].append({"form": lang.roots[con], "meaning": con})
        if n_known > 0:
            public["known"].append({"form": lang.units[1], "meaning": "#1"})
    secret = {"language_id": public["language_id"], "structure": STRUCTURE[family],
              "lexicon": lang.answer_lexicon(), "numerals": lang.numeral_atoms(),
              "aliases": {world.site_place[s]: [f"place@{s}"] for s in world.sites},
              "grammar": lang.describe(), "tokens": gold_docs}
    return public, secret


def null_corpus(seed: int, kind: str = "random", n_docs: int = 130):
    """対照コーパス。random: 文法のない独立同分布の語列。shuffled: 実コーパスの語を全体でシャッフル（統語のみ破壊）。"""
    rng = random.Random(f"null-{kind}-{seed}")
    public, _ = build_corpus("KES", seed, "C", n_docs=n_docs, lacuna=0.0, var_rate=0.0)
    lens = [[len(s.split()) for s in d["sentences"]] for d in public["documents"]]
    if kind == "random":
        lang = make_language("KES", seed + 7919)
        vocab = [lang.new_root(rng.choice([1, 2, 2, 3])) for _ in range(600)]
        weights = [1.0 / (i + 1) for i in range(len(vocab))]
        pool = rng.choices(vocab, weights, k=sum(map(sum, lens)))
    else:
        pool = [t for d in public["documents"] for s in d["sentences"] for t in s.split()]
        rng.shuffle(pool)
    it = iter(pool)
    for d, ls in zip(public["documents"], lens):
        d["sentences"] = [" ".join(next(it) for _ in range(n)) for n in ls]
    public["language_id"] = f"NULL-{kind}-{seed}"
    return public, {"language_id": public["language_id"], "null": kind}
