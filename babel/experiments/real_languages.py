"""実在言語（UD r2.15）を換字した「未知言語」での評価（docs/PREREG_REAL_LANGUAGES.md）。

ツリーバンクも換字後のコーパスもコミットしない（ライセンスの扱いのため）。その都度取得し、SHA-256 を記録する。
正解は UD の注釈（品詞・見出し語）から機械的に作る。解読器には換字後の記号列だけを渡す。
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

from babel.decipher.engine import VERSION, decipher

UD_TAG = "r2.15"
TREEBANKS = [
    ("UD_Turkish-IMST", "tr_imst"), ("UD_Basque-BDT", "eu_bdt"), ("UD_Finnish-TDT", "fi_tdt"),
    ("UD_Hungarian-Szeged", "hu_szeged"), ("UD_Irish-IDT", "ga_idt"), ("UD_Welsh-CCG", "cy_ccg"),
    ("UD_Indonesian-GSD", "id_gsd"), ("UD_Spanish-AnCora", "es_ancora"), ("UD_English-EWT", "en_ewt"),
    ("UD_Wolof-WTB", "wo_wtb"),
]
N_SENTS = 1000
DROP_UPOS = {"PUNCT", "SYM", "X"}
SYMBOLS = ("abcdefghijklmnopqrstuvwxyz" + "αβγδεζηθικλμνξοπρστυφχψω"
           + "абвгдежзийклмнопрстуфхцчшщъыьэюя")
CACHE = Path(".cache/ud")
TREEBANKS_HELDOUT = [   # docs/PREREG_v1.5_VERB_EVIDENCE.md の評価用の言語
    ("UD_German-GSD", "de_gsd"), ("UD_Swedish-Talbanken", "sv_talbanken"), ("UD_Polish-PDB", "pl_pdb"),
    ("UD_Czech-CAC", "cs_cac"), ("UD_Italian-ISDT", "it_isdt"), ("UD_Persian-Seraji", "fa_seraji"),
    ("UD_Hindi-HDTB", "hi_hdtb"), ("UD_Estonian-EDT", "et_edt"), ("UD_Arabic-PADT", "ar_padt"),
    ("UD_Latvian-LVTB", "lv_lvtb"),
]
ALL_VERSIONS = {"v1.2": {"null_gates": False, "v15_gates": False}, "v1.4": {"v15_gates": False}, "v1.5": {}}
VERSIONS = {k: ALL_VERSIONS[k] for k in ("v1.4", "v1.2")}      # 最初の 10 言語（PREREG_REAL_LANGUAGES）。v1.4.0 以前の版に固定


def fetch(repo, code):
    """取得したファイルのパスと SHA-256。キャッシュがあれば再利用する（SHA-256 は毎回計算して記録する）。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{UD_TAG}-{code}-ud-train.conllu"
    if not path.exists():
        url = f"https://raw.githubusercontent.com/UniversalDependencies/{repo}/{UD_TAG}/{code}-ud-train.conllu"
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        path.write_bytes(data)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(form):
    form = unicodedata.normalize("NFC", form.lower())
    return "".join(ch for ch in form if unicodedata.category(ch)[0] in "LM")


def read_sentences(path, n=N_SENTS):
    """[(語形, 品詞, 見出し語)] の文のリスト。手順 1〜3。"""
    out, cur = [], []
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if not line.strip():
            words = [w for w in cur if w[0]]
            if len(words) >= 2:
                out.append(words)
                if len(out) >= n:
                    break
            cur = []
            continue
        if line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 4 or not cols[0].isdigit():
            continue        # 複数語トークン（1-2）と空ノード（1.1）
        if cols[3] in DROP_UPOS:
            continue
        cur.append((_clean(cols[1]), cols[3], _clean(cols[2])))
    return out


def gold_verb_position(sents):
    c = Counter()
    for s in sents:
        idx = [i for i, (_, upos, _) in enumerate(s) if upos == "VERB"]
        if len(idx) == 1 and len(s) >= 3:
            i = idx[0]
            c["initial" if i == 0 else ("final" if i == len(s) - 1 else "medial")] += 1
    n = sum(c.values())
    if not n:
        return {"value": None, "n": 0, "shares": {}}
    shares = {k: round(c[k] / n, 3) for k in ("initial", "medial", "final")}
    best = max(shares, key=lambda k: shares[k])
    return {"value": best if shares[best] >= 0.5 else None, "n": n, "shares": shares}


def _lcp(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def gold_affix_position(sents):
    c = Counter()
    for s in sents:
        for form, upos, lemma in s:
            if upos in ("NOUN", "VERB", "ADJ") and lemma and form != lemma and len(lemma) >= 3:
                lcp, lcs = _lcp(form, lemma), _lcp(form[::-1], lemma[::-1])
                if lcp > lcs:
                    c["suffix"] += 1
                elif lcs > lcp:
                    c["prefix"] += 1
    n = c["suffix"] + c["prefix"]
    if not n:
        return {"value": None, "n": 0, "shares": {}}
    shares = {k: round(c[k] / n, 3) for k in ("suffix", "prefix")}
    value = "suffix" if shares["suffix"] >= 0.75 else ("prefix" if shares["prefix"] >= 0.75 else "both")
    return {"value": value, "n": n, "shares": shares}


# 文字の多い言語用（日本語・中国語など）。Unicode の私用領域（U+E000〜）。allow_extended=True のときだけ使う
EXTENDED_SYMBOLS = SYMBOLS + "".join(chr(0xE000 + i) for i in range(6000))


def cipher(sents, seed=0, allow_extended=False):
    """文字を無作為な1対1の記号に置き換える。

    allow_extended=False（従来）: 記号は SYMBOLS（82）。超えたら ValueError。
    allow_extended=True: 文字の種類が SYMBOLS を超える言語にだけ、私用領域を足した記号を使う。
    超えない言語では従来と同じ換字になる（過去の評価の再現性を保つ）。
    """
    chars = sorted({ch for s in sents for w, _, _ in s for ch in w})
    pool = SYMBOLS
    if len(chars) > len(SYMBOLS):
        if not allow_extended or len(chars) > len(EXTENDED_SYMBOLS):
            raise ValueError(f"文字の種類 {len(chars)} が記号の数 {len(SYMBOLS)} を超える")
        pool = EXTENDED_SYMBOLS
    rng = random.Random(f"cipher-{seed}")
    syms = list(pool)
    rng.shuffle(syms)
    table = dict(zip(chars, syms))
    return [["".join(table[ch] for ch in w) for w, _, _ in s] for s in sents], len(chars)


def to_public(lang_id, sents_words):
    docs = [{"id": f"d{i // 10 + 1:03d}", "sentences": []} for i in range(0, len(sents_words), 10)]
    for i, ws in enumerate(sents_words):
        docs[i // 10]["sentences"].append(" ".join(ws))
    return {"language_id": lang_id, "level": "C", "has_word_boundaries": True, "documents": docs}


def shuffled_control(sents_words, seed=0):
    rng = random.Random(f"shuffle-{seed}")
    pool = [w for s in sents_words for w in s]
    rng.shuffle(pool)
    it = iter(pool)
    return [[next(it) for _ in s] for s in sents_words]


def _judge(pred_value, gold_value):
    if gold_value is None:
        return "no_gold" if pred_value is None else "claim_without_gold"
    if pred_value is None:
        return "abstain"
    return "correct" if str(pred_value) == str(gold_value) else "wrong"


def _verb_class_share(decipherer, upos):
    """動詞とみなしたクラスのトークンのうち、UD で VERB だった割合（換字を戻して数える診断。解読器には渡さない）。"""
    vt = Counter(upos[w].most_common(1)[0][0] for s in decipherer.sents for w in s
                 if decipherer.stem_pos.get(decipherer.infos[w][1]) == "V")
    n = sum(vt.values())
    return {"v_tokens": n, "verb_share": round(vt["VERB"] / n, 3) if n else None,
            "upos_share": {k: round(c / n, 3) for k, c in vt.most_common(4)} if n else {}}


def run_one(repo, code, versions=None):
    from babel.decipher.engine import Decipherer
    versions = versions or VERSIONS
    t0 = time.time()
    path, sha = fetch(repo, code)
    sents = read_sentences(path)
    gold = {"verb_position": gold_verb_position(sents), "affix_position": gold_affix_position(sents)}
    words, n_chars = cipher(sents)
    upos = {}
    for sent, ws in zip(sents, words):
        for (_, u, _), c in zip(sent, ws):
            upos.setdefault(c, Counter())[u] += 1
    public = to_public(f"real-{code}", words)
    control = to_public(f"real-{code}-shuffled", shuffled_control(words))
    row = {"treebank": repo, "code": code, "ud_tag": UD_TAG, "sha256": sha, "sentences": len(sents),
           "tokens": sum(map(len, sents)), "characters": n_chars, "gold": gold, "versions": {}}
    for name, cfg in versions.items():
        dec = Decipherer(public, cfg).fit()
        pred = dec.predict()
        st = pred["structure"]
        ctrl_pred = decipher(control, cfg)
        ctrl = ctrl_pred["structure"]
        row["versions"][name] = {
            "verb_class": _verb_class_share(dec, upos),
            "verb_candidate_position": st.get("verb_candidate_position"),
            "control_morphology_present": ctrl_pred["verdict"]["morphology"]["present"],
            "verb_position": {"pred": st["verb_position"].get("value"),
                              "confidence": st["verb_position"].get("confidence"),
                              "alternatives": st["verb_position"].get("alternatives", {}),
                              "null_gate": st["verb_position"].get("null_gate"),
                              "judge": _judge(st["verb_position"].get("value"), gold["verb_position"]["value"])},
            "affix_position": {"pred": st["affix_position"].get("value"),
                               "confidence": st["affix_position"].get("confidence"),
                               "support": st["affix_position"].get("support"),
                               "null_gate": st["affix_position"].get("null_gate"),
                               "judge": _judge(st["affix_position"].get("value"), gold["affix_position"]["value"])},
            "syntax": pred["verdict"]["syntax"],
            "morphology": pred["verdict"]["morphology"],
            "control_verb_position_claim": ctrl["verb_position"].get("value") is not None,
            "control_claims": sorted(k for k, v in ctrl.items() if k != "sum_marker" and v.get("value") is not None),
        }
    row["seconds"] = round(time.time() - t0, 1)
    return row


def summarize(rows, versions=None):
    s = {}
    for name in (versions or VERSIONS):
        for key in ("verb_position", "affix_position"):
            s[f"{name}:{key}"] = dict(Counter(r["versions"][name][key]["judge"] for r in rows))
        s[f"{name}:syntax_detected"] = sum(1 for r in rows if r["versions"][name]["syntax"]["present"])
        s[f"{name}:control_verb_claims"] = sum(1 for r in rows if r["versions"][name]["control_verb_position_claim"])
        s[f"{name}:languages"] = len(rows)
        s[f"{name}:control_morphology_present"] = sum(1 for r in rows if r["versions"][name]["control_morphology_present"])
        s[f"{name}:morphology_present"] = sum(1 for r in rows if r["versions"][name]["morphology"]["present"])
    return s


def main(out="results/real_languages/real-languages.json", treebanks=None, versions=None, prereg="docs/PREREG_REAL_LANGUAGES.md"):
    rows, excluded = [], []
    for repo, code in (treebanks or TREEBANKS):
        try:
            rows.append(run_one(repo, code, versions))
            print(repo, "ok", flush=True)
        except Exception as e:          # 取得できない言語は除外して記録する（置き換えない）
            excluded.append({"treebank": repo, "error": f"{type(e).__name__}: {e}"})
            print(repo, "excluded", e, flush=True)
    res = {"engine_version": VERSION, "ud_tag": UD_TAG, "preregistration": prereg,
           "summary": summarize(rows, versions), "excluded": excluded, "rows": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))


def verb_class_diagnostic(out="results/real_languages/verb-class-diagnostic.json"):
    """補足の診断（事前登録にない）: 解読器が動詞とみなした語の、UD の品詞の内訳。

    換字を戻して確かめる。解読器には UD の品詞を渡していない（診断のためだけに対応を取る）。
    """
    from babel.decipher.engine import Decipherer
    res = {}
    for repo, code in TREEBANKS:
        path, _ = fetch(repo, code)
        sents = read_sentences(path)
        words, _ = cipher(sents)
        upos = {}
        for s, ws in zip(sents, words):
            for (_, u, _), c in zip(s, ws):
                upos.setdefault(c, Counter())[u] += 1
        d = Decipherer(to_public(f"real-{code}", words), ALL_VERSIONS["v1.4"]).fit()
        vt = Counter(upos[w].most_common(1)[0][0] for s in d.sents for w in s
                     if d.stem_pos.get(d.infos[w][1]) == "V")
        n = sum(vt.values())
        res[code] = {"v_tokens": n, "verb_share": round(vt["VERB"] / n, 3) if n else None,
                     "upos_share": {k: round(c / n, 3) for k, c in vt.most_common(5)},
                     "class_verb_confidence": d.class_info.get("verb_confidence")}
        print(code, res[code], flush=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:2] == ["verb-class"]:
        verb_class_diagnostic()
    elif sys.argv[1:2] == ["heldout"]:
        main("results/real_languages/real-languages-heldout.json", TREEBANKS_HELDOUT,
             {k: ALL_VERSIONS[k] for k in ("v1.5", "v1.4")}, "docs/PREREG_v1.5_VERB_EVIDENCE.md")
    else:
        main(*sys.argv[1:2])
