"""動詞のクラスの検証の探索（問題記録 #94・#98）。

解読器が作る語形のクラス（パラダイムのグループ）ごとに、解読器に見える特徴量と、
正解の品詞から数えた「動詞だった割合」を並べる。品詞は採点・診断のためだけに使い、解読器には渡さない。

ここで使う 20 言語と開発シードは、結果を見た開発用のデータ。見つけた手がかりの評価は、別の言語で行う。
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from babel.decipher.engine import Decipherer
from babel.experiments import real_languages as rl
from babel.generator.corpus import build_corpus
from babel.generator.languages import FAMILIES

V14 = {"v15_gates": False}


def _entropy(counter):
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values() if c) if n else 0.0


def group_features(d, word_pos, verb_label):
    """d: fit 済みの Decipherer。word_pos: 語形 → 品詞の Counter（診断用）。"""
    info = d.class_info
    groups = info["groups"]
    tok = defaultdict(Counter)            # group -> stem -> tokens
    forms = defaultdict(lambda: defaultdict(set))
    pos3 = defaultdict(Counter)
    relpos = defaultdict(list)
    left, right = defaultdict(Counter), defaultdict(Counter)
    bare = Counter()
    lens = defaultdict(list)
    upos = defaultdict(Counter)
    per_sent = defaultdict(list)
    total = 0
    for s in d.sents:
        total += len(s)
        k = Counter()
        for i, w in enumerate(s):
            if w not in d.infos:
                continue
            pre, stem, suf = d.infos[w]
            g = d.stem_group.get(stem)
            if g is None:
                continue
            k[g] += 1
            tok[g][stem] += 1
            forms[g][stem].add(w)
            L = len(s)
            pos3[g]["initial" if i == 0 else ("final" if i == L - 1 else "medial")] += 1
            relpos[g].append(i / (L - 1) if L > 1 else 0.0)
            left[g][s[i - 1] if i else "<s>"] += 1
            right[g][s[i + 1] if i < L - 1 else "</s>"] += 1
            bare[g] += not pre and not suf
            lens[g].append(len(w))
            for p, c in word_pos.get(w, {}).items():
                upos[g][p] += c
        for g, c in k.items():
            per_sent[g].append(c)
    n_aff = Counter(d.group_of.values())
    rows = []
    for g, st in groups.items():
        n = sum(tok[g].values())
        if not n:
            continue
        nu = sum(upos[g].values())
        rows.append({
            "group": g, "chosen_as_verb": g == info["verb_group"],
            "verb_share": round(upos[g][verb_label] / nu, 3) if nu else None,
            "top_pos": upos[g].most_common(1)[0][0] if nu else None,
            "top_pos_share": round(upos[g].most_common(1)[0][1] / nu, 3) if nu else None,
            "stems": st["stems"], "coverage": round(st["coverage"], 3),
            "one_per_sentence": round(st["one_per_sentence"], 3),
            "tokens": n, "token_share": round(n / max(1, total), 4),
            "effective_stems": round(2 ** _entropy(tok[g]), 2),
            "affix_tags": n_aff[g],
            "forms_per_stem": round(sum(len(v) for v in forms[g].values()) / max(1, len(forms[g])), 2),
            "bare_share": round(bare[g] / n, 3),
            "mean_relpos": round(sum(relpos[g]) / n, 3),
            "pos3_entropy": round(_entropy(pos3[g]), 3),
            "mean_len": round(sum(lens[g]) / n, 2),
            "left_types_per_token": round(len(left[g]) / n, 3),
            "right_types_per_token": round(len(right[g]) / n, 3),
            "mean_per_sentence": round(sum(per_sent[g]) / max(1, len(per_sent[g])), 3),
        })
    return rows


def synthetic(seeds=range(5), level="C"):
    out = []
    for fam in FAMILIES:
        for seed in seeds:
            public, secret = build_corpus(fam, seed, level)
            word_pos = defaultdict(Counter)
            for d in public["documents"]:
                for si, sent in enumerate(d["sentences"]):
                    for w, t in zip(sent.split(), secret["tokens"][d["id"]][si]):
                        word_pos[w][t["pos"]] += 1
            dec = Decipherer(public, V14).fit()
            # 正規化後の語形にも品詞を写す（表記揺れの統合で語形が変わるため）
            norm_pos = defaultdict(Counter)
            for w, c in word_pos.items():
                norm_pos[dec.nz.norm(w)[0]].update(c)
            for r in group_features(dec, norm_pos, "V"):
                out.append(dict(r, corpus=f"{fam}-{seed}-{level}", kind="synthetic"))
    return out


def real(treebanks):
    out = []
    for repo, code in treebanks:
        path, _ = rl.fetch(repo, code)
        sents = rl.read_sentences(path)
        words, _ = rl.cipher(sents)
        word_pos = defaultdict(Counter)
        for s, ws in zip(sents, words):
            for (_, u, _), c in zip(s, ws):
                word_pos[c][u] += 1
        dec = Decipherer(rl.to_public(f"real-{code}", words), V14).fit()
        norm_pos = defaultdict(Counter)
        for w, c in word_pos.items():
            norm_pos[dec.nz.norm(w)[0]].update(c)
        for r in group_features(dec, norm_pos, "VERB"):
            out.append(dict(r, corpus=code, kind="real"))
        print(code, flush=True)
    return out


def main(out="results/verb_class/features-dev.json"):
    rows = synthetic() + real(rl.TREEBANKS + rl.TREEBANKS_HELDOUT)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"note": "開発用（20 言語と開発シード 0〜4 は結果を見たデータ）", "rows": rows},
                                    ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(len(rows), "groups ->", out)


if __name__ == "__main__":
    main(*sys.argv[1:2])
