"""初見言語テストの手順。予測は追記専用の台帳で管理し、後から書き換えられないようにする。

external/inbox/<名前>/corpus.json | corpus.txt   投入された未知言語（正解は入れない）
external/predictions/<名前>/attempt-NNN.json     解読器の予測（上書き禁止）
external/ledger.jsonl                            予測ファイルの SHA-256 台帳（追記のみ）
external/answers/<名前>/answer.json              正解（予測の提出後に追加する）
external/scores/<名前>/attempt-NNN.json          採点結果
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("external")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def engine_fingerprint() -> str:
    """解読器ソースのハッシュ。解読器を修正すると変わり、別試行として扱われる。"""
    h = hashlib.sha256()
    for p in sorted((Path(__file__).parent / "decipher").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def load_corpus(folder: Path) -> dict:
    """JSON（内部と同じ public 形式）または UTF-8 テキスト（1行1文、空行で文書区切り）を読む。"""
    js, txt = folder / "corpus.json", folder / "corpus.txt"
    if js.exists():
        pub = json.loads(js.read_text(encoding="utf-8-sig"))      # スマホのエディタが付ける BOM を除く
    elif txt.exists():
        docs, cur = [], []
        for line in txt.read_text(encoding="utf-8-sig").splitlines() + [""]:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line:
                cur.append(line)
            elif cur:
                docs.append({"id": f"d{len(docs) + 1:03d}", "meta": {}, "sentences": cur})
                cur = []
        pub = {"documents": docs}
    else:
        raise FileNotFoundError(f"{folder} に corpus.json も corpus.txt もない")
    pub.setdefault("language_id", folder.name)
    pub.setdefault("has_word_boundaries", True)
    pub.setdefault("parallel", [])
    pub.setdefault("known", [])
    has_meta = any(d.get("meta") for d in pub["documents"])
    pub.setdefault("level", "A" if pub["parallel"] else ("B" if pub["known"] or has_meta else "C"))
    for i, d in enumerate(pub["documents"]):
        d.setdefault("id", f"d{i + 1:03d}")
        d.setdefault("meta", {})
    for forbidden in ("tokens", "lexicon", "answer", "grammar", "structure"):
        if forbidden in pub:
            raise ValueError(f"投入コーパスに正解らしきキー '{forbidden}' が含まれている。正解は answers/ に分けること")
    return pub


def read_ledger(root: Path = ROOT):
    p = root / "ledger.jsonl"
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def predict(root: Path = ROOT, commit: str = "unknown"):
    from babel.decipher.engine import VERSION, decipher
    made, ledger, fp = [], read_ledger(root), engine_fingerprint()
    inbox = root / "inbox"
    for folder in sorted(p for p in inbox.iterdir() if p.is_dir()) if inbox.exists() else []:
        src = next((folder / n for n in ("corpus.json", "corpus.txt") if (folder / n).exists()), None)
        if src is None:
            continue
        csha = _sha(src)
        mine = [e for e in ledger if e["name"] == folder.name]
        if any(e["engine_fingerprint"] == fp and e["corpus_sha256"] == csha for e in mine):
            continue        # 同じ解読器・同じコーパスの予測は既にある（再実行しても上書きしない）
        pred = decipher(load_corpus(folder))
        pred.pop("timing", None)
        n = len(mine) + 1
        out = root / "predictions" / folder.name / f"attempt-{n:03d}.json"
        if out.exists():
            raise RuntimeError(f"{out} は既に存在する。予測の上書きは禁止")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(pred, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
        entry = {"name": folder.name, "attempt": n, "file": str(out.relative_to(root)), "sha256": _sha(out),
                 "corpus_sha256": csha, "engine_fingerprint": fp, "engine_version": VERSION, "commit": commit,
                 "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "answer_already_public": (root / "answers" / folder.name / "answer.json").exists(),
                 "first_attempt": n == 1}
        with (root / "ledger.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
        ledger.append(entry)
        made.append(entry)
    return made


def verify(root: Path = ROOT):
    """台帳と予測ファイルの整合性。問題のリストを返す（空なら正常）。"""
    problems, ledger = [], read_ledger(root)
    seen = {}
    for e in ledger:
        f = root / e["file"]
        if not f.exists():
            problems.append(f"台帳にある予測ファイルが存在しない: {e['file']}")
        elif _sha(f) != e["sha256"]:
            problems.append(f"予測ファイルが台帳記録後に書き換えられている: {e['file']}")
        if e["attempt"] != seen.get(e["name"], 0) + 1:
            problems.append(f"試行番号が連続していない: {e['name']} attempt {e['attempt']}")
        seen[e["name"]] = e["attempt"]
        folder = root / "inbox" / e["name"]
        src = next((folder / n for n in ("corpus.json", "corpus.txt") if (folder / n).exists()), None)
        if src is None or _sha(src) != e["corpus_sha256"]:
            problems.append(f"予測時のコーパスが見つからないか変更されている: {e['name']} attempt {e['attempt']}")
    listed = {e["file"] for e in ledger}
    pred_dir = root / "predictions"
    for f in sorted(pred_dir.rglob("attempt-*.json")) if pred_dir.exists() else []:
        if str(f.relative_to(root)) not in listed:
            problems.append(f"台帳に記録のない予測ファイル: {f.relative_to(root)}")
    return problems


def score_all(root: Path = ROOT):
    from babel.evaluate.score import score
    out = []
    for e in read_ledger(root):
        ans = root / "answers" / e["name"] / "answer.json"
        if not ans.exists():
            continue
        dst = root / "scores" / e["name"] / f"attempt-{e['attempt']:03d}.json"
        pred = json.loads((root / e["file"]).read_text(encoding="utf-8"))
        gold = json.loads(ans.read_text(encoding="utf-8"))
        folder = root / "inbox" / e["name"]
        src = next((folder / n for n in ("corpus.json", "corpus.txt") if (folder / n).exists()), None)
        if src is None or _sha(src) != e["corpus_sha256"]:
            # 予測時と違うコーパスで採点すると、過去の試行の結果を黙って書き換えてしまう
            raise RuntimeError(f"{e['name']} attempt {e['attempt']}: 予測時のコーパス（sha256 {e['corpus_sha256'][:12]}）が"
                               "見つからないか変更されている。コーパスを変えるなら新しい名前で投入すること")
        public = load_corpus(folder)
        r = score(pred, gold, public)
        r.pop("calibration", None)
        r["attempt"] = {k: e[k] for k in ("name", "attempt", "sha256", "engine_fingerprint", "commit", "created_utc",
                                           "answer_already_public", "first_attempt")}
        r["blind"] = not e["answer_already_public"]
        r["independent"] = bool(gold.get("independent", False))
        r["provenance"] = gold.get("provenance", "不明")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(r, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
        out.append(r)
    return out
