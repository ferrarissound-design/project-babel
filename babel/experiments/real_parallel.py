"""実在言語（Parallel UD を換字）で、対訳があるときの評価（docs/PREREG_REAL_PARALLEL.md）。

対訳は英語 PUD の注釈（根の動詞・主語・目的語の見出し語）から機械的に作る。正解は対象の言語の PUD の注釈から作る。
ツリーバンクも換字後のコーパスもコミットしない。解読器には換字後の記号列と構造化対訳だけを渡す。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

from babel.decipher.engine import VERSION, decipher
from babel.experiments import real_languages as rl

SOURCE = ("UD_English-PUD", "en_pud")
TARGETS = [("UD_Arabic-PUD", "ar_pud"), ("UD_Czech-PUD", "cs_pud"), ("UD_German-PUD", "de_pud"),
           ("UD_Spanish-PUD", "es_pud"), ("UD_Finnish-PUD", "fi_pud"), ("UD_French-PUD", "fr_pud"),
           ("UD_Hindi-PUD", "hi_pud"), ("UD_Indonesian-PUD", "id_pud"), ("UD_Icelandic-PUD", "is_pud"),
           ("UD_Italian-PUD", "it_pud"), ("UD_Japanese-PUD", "ja_pud"), ("UD_Korean-PUD", "ko_pud"),
           ("UD_Polish-PUD", "pl_pud"), ("UD_Portuguese-PUD", "pt_pud"), ("UD_Russian-PUD", "ru_pud"),
           ("UD_Swedish-PUD", "sv_pud"), ("UD_Thai-PUD", "th_pud"), ("UD_Turkish-PUD", "tr_pud"),
           ("UD_Chinese-PUD", "zh_pud")]
FRACTIONS = (0.1, 0.25, 1.0)
# v1.6 の評価用（docs/PREREG_v1.6_ALIGNMENT_NULL.md）。先の評価で文字の種類のため除外し、出力を見ていない 5 言語
TARGETS16 = [("UD_Hindi-PUD", "hi_pud"), ("UD_Thai-PUD", "th_pud"), ("UD_Japanese-PUD", "ja_pud"),
             ("UD_Korean-PUD", "ko_pud"), ("UD_Chinese-PUD", "zh_pud")]
SETTINGS = [("10%", 0.1, False), ("25%", 0.25, False), ("100%", 1.0, False), ("100%-mismatch", 1.0, True)]
SETTINGS16 = [("100%", 1.0, False), ("100%-mismatch", 1.0, True)]
VERSIONS16 = {"v1.5": {"v16_gates": False}, "v1.6": {}}
CACHE = Path(".cache/ud")


def fetch(repo, code):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{rl.UD_TAG}-{code}-ud-test.conllu"
    if not path.exists():
        url = f"https://raw.githubusercontent.com/UniversalDependencies/{repo}/{rl.UD_TAG}/{code}-ud-test.conllu"
        with urllib.request.urlopen(url, timeout=120) as r:
            path.write_bytes(r.read())
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def parse(path):
    """[{sent_id, tokens: [{id, form, lemma, upos, head, deprel, feats}]}]。複数語トークンと空ノードは除く。"""
    out, cur, sid = [], [], None
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if not line.strip():
            if cur:
                out.append({"sent_id": sid, "tokens": cur})
            cur, sid = [], None
            continue
        if line.startswith("#"):
            if line.startswith("# sent_id"):
                sid = line.split("=", 1)[1].strip()
            continue
        c = line.split("\t")
        if len(c) < 8 or not c[0].isdigit():
            continue
        feats = dict(kv.split("=", 1) for kv in c[5].split("|") if "=" in kv) if c[5] != "_" else {}
        cur.append({"id": int(c[0]), "form": c[1], "lemma": c[2], "upos": c[3], "head": int(c[6]) if c[6].isdigit() else -1,
                    "deprel": c[7], "feats": feats})
    return out


def _lemma(tok):
    lem = unicodedata.normalize("NFC", tok["lemma"].lower())
    return lem if lem.isalpha() and lem.isascii() else None


def translation(sent):
    """英語の文 → 構造化対訳（作れなければ None）。事前登録の手順 1〜6。"""
    toks = sent["tokens"]
    roots = [t for t in toks if t["head"] == 0]
    if len(roots) != 1 or roots[0]["upos"] != "VERB":
        return None
    root = roots[0]
    kids = [t for t in toks if t["head"] == root["id"]]
    if any(k["deprel"] == "nsubj:pass" or k["deprel"] == "aux:pass" for k in kids):
        return None
    pred = _lemma(root)
    if not pred:
        return None
    args = []
    for role, rel in (("subject", "nsubj"), ("object", "obj")):
        for k in kids:
            if k["deprel"] == rel and _lemma(k):
                a = {"role": role, "head": _lemma(k)}
                if k["feats"].get("Number") == "Plur":
                    a["pl"] = True
                args.append(a)
                break
    if not args:
        return None
    tr = {"kind": "clause", "pred": pred, "args": args}
    tense = {"Past": "PAST", "Pres": "PRES"}.get(root["feats"].get("Tense"))
    if tense:
        tr["tense"] = tense
    return tr


def target_sentences(parsed):
    """対象の言語: (sent_id, 清掃後の語 [(語形, 品詞, 見出し語)], 主語と目的語の順序の数え上げ)。2 語未満の文は除く。"""
    out = []
    for s in parsed:
        words = [(rl._clean(t["form"]), t["upos"], rl._clean(t["lemma"])) for t in s["tokens"] if t["upos"] not in rl.DROP_UPOS]
        words = [w for w in words if w[0]]
        if len(words) < 2:
            continue
        so = Counter()
        for v in s["tokens"]:
            if v["upos"] != "VERB":
                continue
            subj = [t["id"] for t in s["tokens"] if t["head"] == v["id"] and t["deprel"] == "nsubj"]
            obj = [t["id"] for t in s["tokens"] if t["head"] == v["id"] and t["deprel"] == "obj"]
            if subj and obj:
                so["SO" if subj[0] < obj[0] else "OS"] += 1
        out.append((s["sent_id"], words, so))
    return out


def gold_so(tsents):
    c = Counter()
    for _, _, so in tsents:
        c.update(so)
    n = c["SO"] + c["OS"]
    if not n:
        return {"value": None, "n": 0, "shares": {}}
    share = c["SO"] / n
    value = "SO" if share >= 0.75 else ("OS" if share <= 0.25 else None)
    return {"value": value, "n": n, "shares": {"SO": round(share, 3), "OS": round(1 - share, 3)}}


def select(eligible, fraction):
    """対訳を作れる文から、sent_id 順に等間隔に選ぶ（10% は 10 文ごと、25% は 4 文ごと）。"""
    step = round(1 / fraction)
    return eligible[::step]


def build_public(lang_id, words, sent_ids, tr_by_id, fraction, mismatch=False):
    public = rl.to_public(lang_id, words)
    public["level"] = "A"
    pos = {sid: (i // 10, i % 10) for i, sid in enumerate(sent_ids)}      # sent_id → (文書番号, 文書内の番号)
    eligible = [sid for sid in sorted(sent_ids) if sid in tr_by_id]
    chosen = select(eligible, fraction)
    trs = [tr_by_id[sid] for sid in chosen]
    if mismatch:
        trs = trs[1:] + trs[:1]
    public["parallel"] = [{"doc": f"d{pos[sid][0] + 1:03d}", "sent": pos[sid][1], "translation": tr}
                          for sid, tr in zip(chosen, trs)]
    return public, len(chosen)


def _judge(pred, gold):
    if gold is None:
        return "no_gold" if pred is None else "claim_without_gold"
    if pred is None:
        return "abstain"
    return "correct" if str(pred) == str(gold) else "wrong"


def run_language(repo, code, en_tr, cfg=None, settings=SETTINGS, allow_extended=False):
    t0 = time.time()
    path, sha = fetch(repo, code)
    tsents = target_sentences(parse(path))
    sent_ids = [s for s, _, _ in tsents]
    sents = [[w for w in ws] for _, ws, _ in tsents]
    gold = {"verb_position": rl.gold_verb_position(sents), "so_order": gold_so(tsents)}
    words, n_chars = rl.cipher(sents, allow_extended=allow_extended)
    row = {"treebank": repo, "code": code, "sha256": sha, "sentences": len(sents), "characters": n_chars, "gold": gold,
           "settings": {}}
    for label, frac, mis in settings:
        public, n_par = build_public(f"real-{code}", words, sent_ids, en_tr, frac, mis)
        st = decipher(public, cfg)["structure"]
        vp, so = st["verb_position"], st["so_order"]
        row["settings"][label] = {
            "translations": n_par,
            "verb_position": {"pred": vp.get("value"), "confidence": vp.get("confidence"), "n": vp.get("n"),
                              "judge": _judge(vp.get("value"), gold["verb_position"]["value"])},
            "so_order": {"pred": so.get("value"), "confidence": so.get("confidence"), "n": so.get("n"),
                         "judge": _judge(so.get("value"), gold["so_order"]["value"])}}
    row["seconds"] = round(time.time() - t0, 1)
    return row


def summarize(rows):
    s = {}
    for label in ("10%", "25%", "100%", "100%-mismatch"):
        for key in ("verb_position", "so_order"):
            s[f"{label}:{key}"] = dict(Counter(r["settings"][label][key]["judge"] for r in rows))
    return s


def main(out="results/real_parallel/real-parallel.json"):
    src_path, src_sha = fetch(*SOURCE)
    en = parse(src_path)
    en_tr = {s["sent_id"]: t for s in en if (t := translation(s))}
    print("英語の文", len(en), "対訳を作れる文", len(en_tr), flush=True)
    rows, excluded = [], []
    for repo, code in TARGETS:
        try:
            rows.append(run_language(repo, code, en_tr, {"v16_gates": False}))    # PREREG_REAL_PARALLEL は v1.5.0 で実行した
            print(code, "ok", flush=True)
        except Exception as e:                     # 取得できない・文字の種類が多い言語は除外して記録する（置き換えない）
            excluded.append({"treebank": repo, "error": f"{type(e).__name__}: {e}"})
            print(code, "excluded", e, flush=True)
    res = {"engine_version": VERSION, "ud_tag": rl.UD_TAG, "preregistration": "docs/PREREG_REAL_PARALLEL.md",
           "source": {"treebank": SOURCE[0], "sha256": src_sha, "sentences": len(en), "translations": len(en_tr)},
           "summary": summarize(rows), "excluded": excluded, "rows": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))


def main16(out="results/real_parallel/real-parallel-v16.json"):
    """v1.6 の評価: 新しい 5 言語、100% とずらした対照、v1.5.0 と v1.6.0。"""
    src_path, src_sha = fetch(*SOURCE)
    en_tr = {s["sent_id"]: t for s in parse(src_path) if (t := translation(s))}
    rows, excluded = [], []
    for repo, code in TARGETS16:
        try:
            row = None
            for name, cfg in VERSIONS16.items():
                r = run_language(repo, code, en_tr, cfg, SETTINGS16, allow_extended=True)
                if row is None:
                    row = {k: v for k, v in r.items() if k != "settings"}
                    row["versions"] = {}
                row["versions"][name] = r["settings"]
            rows.append(row)
            print(code, "ok", flush=True)
        except Exception as e:
            excluded.append({"treebank": repo, "error": f"{type(e).__name__}: {e}"})
            print(code, "excluded", e, flush=True)
    summary = {f"{name}:{label}:{key}": dict(Counter(r["versions"][name][label][key]["judge"] for r in rows))
               for name in VERSIONS16 for label, _, _ in SETTINGS16 for key in ("verb_position", "so_order")}
    res = {"engine_version": VERSION, "ud_tag": rl.UD_TAG, "preregistration": "docs/PREREG_v1.6_ALIGNMENT_NULL.md",
           "source": {"treebank": SOURCE[0], "sha256": src_sha, "translations": len(en_tr)},
           "summary": summary, "excluded": excluded, "rows": rows}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["v16"]:
        main16()
    else:
        main(*sys.argv[1:2])
