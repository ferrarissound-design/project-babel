"""解読エンジン本体。入力は public コーパス（dict）だけ。正解には一切触れない。"""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict

from . import baselines, classes, nulltest, numerals, segment, semantics
from .align import Aligner, flatten
from .morph import Morphology
from .variants import LACUNA, Normalizer

VERSION = "1.0.1"
DEFAULT_CONFIG = {
    "use_variants": True,        # 表記揺れの正規化
    "use_morph": True,           # 形態素解析
    "use_em": True,              # EM アラインメント（False なら共起の一様配分）
    "use_numeral_solver": True,  # 合計行の算術による数詞推定
    "use_metadata": True,        # 出土地などのメタデータ
    "abstain": True,             # 信頼度が低い主張を保留する
    "claim_threshold": 0.5,
    "morph_method": "paradigm",  # paradigm / freq / successor（後2つは比較用ベースライン）
    "align_method": "em",        # em / count / dice（後2つは比較用ベースライン）
    "pos_constraint": True,      # 推定した品詞で対応付けの候補を絞る（v1.1）
}


def sanitize_public(public: dict):
    """外部から来る public コーパスの不整合を直す。直した内容は warnings として返し、出力に残す。

    解読器は第三者が作ったコーパスを受け取るので、手書きの JSON の小さな誤りで落ちてはいけない。
    ただし黙って直すと結果の解釈を誤らせるため、何をしたかを必ず記録する。
    """
    pub, warnings = dict(public), []
    docs = []
    for i, d in enumerate(pub.get("documents") or []):
        d = dict(d)
        d.setdefault("id", f"d{i + 1:03d}")
        if not isinstance(d.get("meta"), dict):
            if d.get("meta") is not None:
                warnings.append(f"文書 {d['id']}: meta が辞書でないので無視した")
            d["meta"] = {}
        d["sentences"] = [str(x) for x in (d.get("sentences") or [])]
        docs.append(d)
    pub["documents"] = docs
    ids = {d["id"] for d in docs}
    split = pub.get("split") or {}
    tr = [x for x in (split.get("train") or []) if x in ids]
    he = [x for x in (split.get("heldout") or []) if x in ids and x not in set(tr)]
    dropped = len(split.get("train") or []) + len(split.get("heldout") or []) - len(tr) - len(he)
    if dropped:
        warnings.append(f"split にある存在しない（または重複した）文書ID {dropped} 件を無視した")
    pub["split"] = {"train": tr, "heldout": he} if tr else {}
    n_sent = {d["id"]: len(d["sentences"]) for d in docs}
    par = []
    for p in pub.get("parallel") or []:
        try:
            ok = p["doc"] in n_sent and 0 <= int(p["sent"]) < n_sent[p["doc"]] and isinstance(p.get("translation"), dict)
        except (KeyError, TypeError, ValueError):
            ok = False
        if ok:
            par.append(dict(p, sent=int(p["sent"])))
        else:
            warnings.append(f"対訳 {str(p)[:60]} は文書・文番号・訳のどれかが不正なので無視した")
    pub["parallel"] = par
    known = []
    for k in pub.get("known") or []:
        m = str(k.get("meaning", "")) if isinstance(k, dict) else ""
        if not (isinstance(k, dict) and k.get("form")) or not m or (m.startswith("#") and not m[1:].isdigit()):
            warnings.append(f"既知語 {str(k)[:60]} は形式が不正なので無視した（数詞は \"#3\" のように書く）")
            continue
        known.append(k)
    pub["known"] = known
    return pub, warnings


class Decipherer:
    def __init__(self, public: dict, config: dict | None = None):
        public, self.input_warnings = sanitize_public(public)
        self.pub = public
        self.cfg = dict(DEFAULT_CONFIG, **(config or {}))
        self.level = public.get("level", "C")
        self.docs = {d["id"]: d for d in public["documents"]}
        split = public.get("split") or {}
        self.train_ids = split.get("train") or [d["id"] for d in public["documents"]]
        self.heldout_ids = split.get("heldout") or []
        self.boundaries = public.get("has_word_boundaries", True)
        self.tokens: dict[str, list[list[str]]] = {}
        if self.boundaries:
            for d in public["documents"]:
                self.tokens[d["id"]] = [s.split() for s in d["sentences"]]
        else:   # 連続書記: 全文書をまとめて分割する（教師なし・トランスダクティブ）
            lines = [(d["id"], s) for d in public["documents"] for s in d["sentences"]]
            segs = segment.unigram_segment([s for _, s in lines])
            for (did, _), words in zip(lines, segs):
                self.tokens.setdefault(did, []).append(words)
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self):
        t0 = time.time()
        cfg = self.cfg
        train = [self.tokens[d] for d in self.train_ids]
        self.nz = Normalizer([[t for s in doc for t in s] for doc in train], enable=cfg["use_variants"])
        self.norm_docs = {d: [[self.nz.norm(t)[0] if t != LACUNA else LACUNA for t in s] for s in self.tokens[d]]
                          for d in self.train_ids}
        sents = [[w for w in s if w != LACUNA] for d in self.train_ids for s in self.norm_docs[d]]
        self.sents = [s for s in sents if s]
        self.counts = Counter(w for s in self.sents for w in s)
        if cfg["morph_method"] == "paradigm":
            self.morph = Morphology(self.counts, enable=cfg["use_morph"])
        else:
            self.morph = baselines.make_morphology(cfg["morph_method"], self.counts)
        self.infos = {w: self.morph.analyze(w) for w in self.counts}
        self.group_of, self.stem_group = classes.paradigm_groups(self.infos)
        self.stem_pos, self.class_info = classes.label_classes(self.sents, self.infos, self.stem_group)
        self._fit_numerals()
        self.func_words = classes.function_words(self.sents, self.infos, self.stem_pos, self.num_values)
        self.verb_pos = classes.verb_position(self.sents, self.infos, self.stem_pos)
        self.lexicon: dict[str, dict] = {}
        self.affix_fn: dict[str, dict] = {}
        self.slot_pos: dict[str, str] = {}
        self.order_counts = {"so": Counter(), "adj": Counter(), "verb": Counter()}
        self.aligner = None
        if self.pub.get("parallel"):
            self._fit_alignment()
        self._fit_known()
        self.toponyms = {}
        if cfg["use_metadata"] and any(self.docs[d].get("meta", {}).get("site") for d in self.train_ids):
            self._fit_toponyms()
        self.num_noun = self._num_noun_order()
        self.syntax = nulltest.syntax_test(self.sents)
        tok_total = max(1, sum(self.counts.values()))
        affixed = sum(c for w, c in self.counts.items() if self.infos[w][0] or self.infos[w][2])
        self.morph_test = nulltest.morphology_test(affixed / tok_total, len(self.morph.affixes))
        sig = [[(self.infos[w][1], "|".join(self.infos[w][0]) + "_" + "|".join(self.infos[w][2])) for w in s]
               for s in self.sents]
        self.related = semantics.relatedness(sig)
        self.fit_seconds = time.time() - t0
        self.fitted = True
        return self

    def _is_inflected(self, w):
        info = self.infos.get(w)
        return bool(info) and self.stem_pos.get(info[1], "X") in ("N", "V")

    def _fit_numerals(self):
        docs = []
        for d in self.train_ids:
            ss = self.norm_docs[d]
            damaged = any(LACUNA in s for s in ss)
            docs.append((d, [[w for w in s if w != LACUNA] for s in ss], damaged))
        anchors = []
        for k in self.pub.get("known", []):
            if str(k["meaning"]).startswith("#"):
                anchors.append((self.nz.norm(k["form"])[0], int(k["meaning"][1:])))
        solver = self.cfg["use_numeral_solver"]
        res = numerals.solve_numerals(docs, self._is_inflected, self.counts, anchors, enable_solver=solver)
        extra = []
        if solver and self.pub.get("parallel") and res["variables"]:
            cand = set(res["variables"])
            for p in self.pub["parallel"]:
                if p["doc"] not in self.norm_docs:
                    continue
                _, _, counts = flatten(p["translation"])
                sent = [w for w in self.norm_docs[p["doc"]][p["sent"]]]
                if len(counts) == 1 and LACUNA not in sent:
                    ws = [w for w in sent if w in cand]
                    if ws:
                        extra.append((ws, counts[0][0]))
            if extra:
                res = numerals.solve_numerals(docs, self._is_inflected, self.counts, anchors, extra, True)
        if solver and res["status"] != "solved" and res.get("sum_marker") is None and self.pub.get("parallel"):
            res2 = self._numerals_from_parallel(docs, anchors)
            if res2 is not None:
                res = res2
        self.num = res
        self.num_values = dict(res["values"])
        for w, v in anchors:
            self.num_values.setdefault(w, v)

    def _numerals_from_parallel(self, docs, anchors):
        """合計行がない場合: 対訳の数量を概念 #NUM として対応付けに混ぜ、#NUM に結び付く語を数詞候補にする。

        数詞語は「数量のある文」にだけ規則的に現れるので、EM の対応付けで #NUM に吸着される。
        候補が決まれば、対訳の文ごとに「数詞語の値の和＝数量」という方程式が立つ。
        """
        items, counted = [], []
        for p in self.pub["parallel"]:
            if p["doc"] not in self.norm_docs:
                continue
            sent = self.norm_docs[p["doc"]][p["sent"]]
            concepts, feats, counts = flatten(p["translation"])
            if counts:
                concepts = concepts + [("#NUM", "num")]
            units = []
            for i, w in enumerate(sent):
                if w != LACUNA:
                    units.append(("S:" + w, i))      # 数詞は屈折しないことが多いので、語形全体を単位にする
            items.append({"units": units, "concepts": concepts, "feats": [], "sent": sent, "counts": counts})
            if len(counts) == 1 and LACUNA not in sent:
                counted.append((sent, counts[0][0]))
        if len(counted) < 3:
            return None
        al = Aligner(items, mode="em")
        cand = set()
        for u in al.n:
            meaning, conf, _ = al.best(u, "C:")
            if meaning == "#NUM" and al.joint[u]["C:#NUM"] >= 1:
                cand.add(u[2:])
        cand |= {w for w, _ in anchors}
        if len(cand) < 2:
            return None
        extra = [([w for w in sent if w != LACUNA], n) for sent, n in counted]   # 絞り込みはソルバーが候補拡張の後に行う
        res = numerals.solve_numerals(docs, self._is_inflected, self.counts, anchors, extra, True, seed_variables=cand)
        res["parallel_candidates"] = len(cand)
        return res

    def _units(self, sent):
        """文 → [(unit, 語の位置)]。数詞と欠損は除く。"""
        out = []
        for i, w in enumerate(sent):
            if w == LACUNA or w in self.num_values:
                continue
            pre, stem, suf = self.infos.get(w) or self.morph.analyze(w)
            for part in self.morph.compounds.get(stem, (stem,)):
                out.append(("S:" + part, i))
            out += [("A:P:" + a, i) for a in pre] + [("A:S:" + a, i) for a in suf]
        return out

    def _fit_alignment(self):
        items = []
        for p in self.pub["parallel"]:
            if p["doc"] not in self.norm_docs:
                continue
            sent = self.norm_docs[p["doc"]][p["sent"]]
            concepts, feats, counts = flatten(p["translation"])
            items.append({"units": self._units(sent), "concepts": concepts, "feats": feats, "sent": sent,
                          "counts": counts})
        mode = self.cfg["align_method"] if self.cfg["use_em"] else "count"
        constraint = self._pos_constraint(items) if self.cfg["pos_constraint"] else None
        self.aligner = al = Aligner(items, mode=mode, constraint=constraint)
        thr = self.cfg["claim_threshold"]
        slot_votes = defaultdict(Counter)
        for it, link in zip(items, al.links):
            pos_of = {}
            slot_of = dict(it["concepts"])
            for u, i in it["units"]:
                if not u.startswith("S:"):
                    continue
                best = max(((a, c) for (s, c), a in link.items() if s == u and c.startswith("C:")), default=None)
                if best and best[0] >= 0.25:
                    slot = slot_of.get(best[1][2:], "")
                    slot_votes[u[2:]][slot.split(":")[0]] += 1
                    pos_of.setdefault(slot, i)
            if "head:subject" in pos_of and "head:object" in pos_of:
                self.order_counts["so"]["SO" if pos_of["head:subject"] < pos_of["head:object"] else "OS"] += 1
            for role in ("subject", "object", "recipient", "location"):
                if "adj:" + role in pos_of and "head:" + role in pos_of:
                    self.order_counts["adj"]["AN" if pos_of["adj:" + role] < pos_of["head:" + role] else "NA"] += 1
            if "pred" in pos_of and len(it["sent"]) >= 3:
                i = pos_of["pred"]
                self.order_counts["verb"]["initial" if i == 0 else "final" if i == len(it["sent"]) - 1 else "medial"] += 1
        for u in sorted(al.n):
            if u.startswith("A:"):
                fn = al.functions(u)
                if fn:
                    self.affix_fn[u[2:]] = fn
                continue
            stem = u[2:]
            meaning, conf, alts = al.best(u, "C:")
            fn = al.functions(u, thr=0.5)
            entry = {"meaning": meaning, "confidence": conf, "alternatives": alts, "source": "alignment",
                     "evidence": al.joint[u]["C:" + meaning] if meaning else 0, "feats": {}}
            if fn and conf < thr:
                entry.update(meaning=None, feats=fn, confidence=max(fn.values()), source="alignment-function")
            if alts and meaning and alts[0][1] >= 0.5 and al.joint[u]["C:" + alts[0][0]] >= 2:
                entry["polysemy_candidate"] = True
            self.lexicon[stem] = entry
            if slot_votes[stem]:
                slot = slot_votes[stem].most_common(1)[0][0]
                self.slot_pos[stem] = {"pred": "V", "head": "N", "title": "N", "adj": "ADJ", "func": "FUNC"}.get(slot, "X")

    def _pos_constraint(self, items):
        """意味の手がかりなしで推定した品詞（パラダイムの型）で、語幹が対応できる概念を絞る。

        動詞型の語幹は対訳で述語になる概念にだけ、名詞型の語幹は述語以外の概念にだけ対応できる。
        品詞が不明（X）の語幹と、述語にも項にもなる概念には制約をかけない。
        """
        slots = defaultdict(set)
        for it in items:
            for c, slot in it["concepts"]:
                slots["C:" + c].add("pred" if slot == "pred" else "arg")

        def ok(unit, concept):
            if not unit.startswith("S:") or not concept.startswith("C:"):
                return True
            sl = slots.get(concept)
            if not sl or len(sl) > 1:
                return True
            pos = self.stem_pos.get(unit[2:], "X")
            if pos == "V":
                return "pred" in sl
            if pos == "N":
                return "arg" in sl
            return True
        return ok

    def _stem_of_form(self, form):
        return self.morph.analyze(self.nz.norm(form)[0])[1]

    def _fit_known(self):
        for k in self.pub.get("known", []):
            m = str(k["meaning"])
            if m.startswith("#"):
                continue
            self.lexicon[self._stem_of_form(k["form"])] = {"meaning": m, "confidence": 0.97, "alternatives": [],
                                                           "source": "given", "evidence": 0, "feats": {}}

    def _fit_toponyms(self):
        doc_stems, doc_site = [], []
        for d in self.train_ids:
            site = self.docs[d].get("meta", {}).get("site")
            if site is None:
                continue
            doc_site.append(site)
            doc_stems.append({self.infos[w][1] for s in self.norm_docs[d] for w in s if w != LACUNA})
        thr = self.cfg["claim_threshold"]

        def eligible(stem):
            if stem in self.num_values or stem in self.func_words or self.stem_pos.get(stem) == "V":
                return False
            e = self.lexicon.get(stem)
            return not (e and e["meaning"] and e["confidence"] >= thr)

        self.toponyms = semantics.toponym_hypotheses(doc_stems, doc_site, eligible)
        for site, h in self.toponyms.items():
            self.lexicon[h["stem"]] = {"meaning": f"place@{site}", "confidence": h["confidence"],
                                       "alternatives": h["alternatives"], "source": "metadata-site",
                                       "evidence": 0, "feats": {}}

    def _num_noun_order(self):
        """合計行（標識＋数詞＋名詞）での数詞と名詞の順序。"""
        marker, c = self.num.get("sum_marker"), Counter()
        if not marker or not self.num_values:
            return {"value": None, "confidence": 0.0, "n": 0}
        for s in self.sents:
            if marker in s:
                nums = [i for i, w in enumerate(s) if w in self.num_values]
                rest = [i for i, w in enumerate(s) if w not in self.num_values and w != marker]
                if nums and len(rest) == 1:
                    c["NumN" if max(nums) < rest[0] else "NNum" if min(nums) > rest[0] else "?"] += 1
        n = c["NumN"] + c["NNum"]
        if n < 5:
            return {"value": None, "confidence": 0.0, "n": n}
        best = "NumN" if c["NumN"] >= c["NNum"] else "NNum"
        return {"value": best, "confidence": round(c[best] / n * min(1, n / 15), 3), "n": n}

    # -------------------------------------------------------------- outputs
    def _pos(self, w, stem):
        if w in self.num_values:
            return "NUM"
        if stem in self.slot_pos and self.slot_pos[stem] != "X":
            return self.slot_pos[stem]
        if w in self.func_words:
            return "FUNC"
        return self.stem_pos.get(stem, "X")

    def _claim(self, stem):
        """(意味 or None, 信頼度)。信頼度が閾値未満なら主張しない。"""
        parts = self.morph.compounds.get(stem) if self.morph.enable else None
        if parts and stem not in self.lexicon:
            ms = [self._claim(p) for p in parts]
            if all(m for m, _ in ms):
                return "+".join(m for m, _ in ms), round(min(c for _, c in ms), 3)
            return None, 0.0
        e = self.lexicon.get(stem)
        if not e or not e["meaning"]:
            return None, 0.0
        if self.cfg["abstain"] and e["confidence"] < self.cfg["claim_threshold"]:
            return None, e["confidence"]
        return e["meaning"], e["confidence"]

    def analyze_token(self, t):
        if t == LACUNA:
            return {"s": t, "lacuna": True}
        n, mp = self.nz.norm(t)
        pre, stem, suf = self.morph.analyze(n)
        cuts = sorted({mp[c] for c in self.morph.cuts(n)} - {0, len(t)})
        out = {"s": t, "norm": n, "cuts": cuts, "stem": stem, "pos": self._pos(n, stem), "concept": None,
               "conf": 0.0, "feats": []}
        if n in self.num_values:
            out.update(concept=f"#{self.num_values[n]}", conf=self.num.get("confidence", 0.0) or 0.97)
            return out
        m, c = self._claim(stem)
        out.update(concept=m, conf=c)
        feats = set()
        for a in ["P:" + a for a in pre] + ["S:" + a for a in suf]:
            feats.update(self.affix_fn.get(a, {}))
        e = self.lexicon.get(stem)
        if e and e.get("feats"):
            feats.update(e["feats"])
        out["feats"] = sorted(feats)
        return out

    def _structure(self):
        A = bool(self.aligner)
        vp = dict(self.verb_pos)
        if vp["value"] is None and A and sum(self.order_counts["verb"].values()) >= 5:
            c = self.order_counts["verb"]
            n = sum(c.values())
            best = c.most_common(1)[0][0]
            vp = {"value": best, "confidence": round(c[best] / n * min(1, n / 20), 3),
                  "alternatives": {k: round(v / n, 3) for k, v in c.items()}, "n": n, "source": "alignment"}

        def from_counts(c, min_n=4):
            n = sum(c.values())
            if n < min_n:
                return {"value": None, "confidence": 0.0, "n": n,
                        "reason": "役割の分かる対訳が足りない" if A else "意味の手がかりがなく、主語と目的語を区別できない"}
            best = c.most_common(1)[0][0]
            return {"value": best, "confidence": round(c[best] / n * min(1, n / 12), 3), "n": n,
                    "alternatives": {k: round(v / n, 3) for k, v in c.items()}}

        sup = getattr(self.morph, "support", {"suffix": 0, "prefix": 0})
        tot = sup["suffix"] + sup["prefix"]
        pos = self.morph.position
        return {
            "verb_position": vp,
            "so_order": from_counts(self.order_counts["so"]),
            "adj_noun": from_counts(self.order_counts["adj"]),
            "num_noun": self.num_noun,
            "affix_position": {"value": None if pos == "none" else pos,
                               "confidence": round(max(sup.values()) / tot, 3) if tot else 0.0, "support": sup},
            "numeral_base": {"value": self.num.get("base"), "confidence": self.num.get("confidence", 0.0)
                             if self.num.get("base") else 0.0, "alternatives": self.num.get("base_scores", {})},
            "sum_marker": {"value": self.num.get("sum_marker"),
                           "note": "合計行を示す機能語（算術的に検証できる機能であり、実世界の意味の主張ではない）"},
        }

    def _verdict(self):
        syn, mor = self.syntax["present"], self.morph_test["present"]
        sem = "partial" if (self.aligner or self.pub.get("known")) else "none"
        if not syn and not mor:
            text = "言語的な構造が検出されない。解読不能と判定する。"
        elif not syn:
            text = "語形の規則性はあるが、語順に規則性がない。文法と意味は判定不能。"
        elif sem == "none":
            text = "内部構造（語形・語順・数詞）は推定できるが、実世界の意味に対応づける証拠がない。意味は判定不能。"
        else:
            text = "構造を推定でき、証拠のある語に限って意味を主張できる。"
        return {"syntax": self.syntax, "morphology": self.morph_test, "semantic_evidence": sem,
                "decipherable_structure": bool(syn or mor), "text": text}

    def _hypotheses(self, st):
        hs = []

        def add(topic, value, conf, alts, evidence):
            status = "保留（判定不能）" if value is None else ("採用" if conf >= 0.6 else "暫定")
            hs.append({"topic": topic, "value": value, "confidence": conf, "alternatives": alts,
                       "evidence": evidence, "status": status})

        vp = st["verb_position"]
        add("動詞の位置", vp["value"], vp["confidence"], vp.get("alternatives", {}),
            f"動詞クラスが1つだけ現れる文 {vp.get('n', 0)} 文での位置")
        ap = st["affix_position"]
        add("接辞の付く側", ap["value"], ap["confidence"], ap["support"], "接頭辞・接尾辞として解析された語の延べ数")
        so = st["so_order"]
        add("主語と目的語の順序", so["value"], so["confidence"], so.get("alternatives", {}),
            so.get("reason") or f"対訳で役割が分かった {so['n']} 文")
        an = st["adj_noun"]
        add("形容詞と名詞の順序", an["value"], an["confidence"], an.get("alternatives", {}),
            an.get("reason") or f"対訳で対応が取れた {an['n']} 例")
        nn = st["num_noun"]
        add("数詞と名詞の順序", nn["value"], nn["confidence"], {}, f"合計行 {nn['n']} 行")
        nb = st["numeral_base"]
        add("数詞の基数", nb["value"], nb["confidence"], nb["alternatives"],
            f"合計の方程式 {self.num['equations']} 本（冗長 {self.num['redundant']}、棄却 {self.num['rejected']}）"
            + ("。単位は最小値=1と仮定した相対値" if self.num.get("relative") else "") + (self.num.get("reason") or ""))
        for op in self.nz.ops:
            add(f"表記の交替 {op['a']}→{op['b']}", "同一語の異表記", op["b_in_those_docs"], {},
                f"{op['pairs']} 組の語形。{op['a']} が現れない文書 {op['docs_without_a']} 件")
        for site, h in sorted(self.toponyms.items()):
            add(f"出土地 {site} の地名", h["stem"], h["confidence"], dict((a, p) for a, p in h["alternatives"]),
                f"この語を含む文書の {h['precision']:.0%} が {site} 出土（地元の人名の可能性が残る）")
        return hs

    def predict(self):
        assert self.fitted
        t0 = time.time()
        st = self._structure()
        thr = self.cfg["claim_threshold"]
        forms = defaultdict(Counter)
        for w, c in self.counts.items():
            forms[self.infos[w][1]][w] += c
        lex = []
        for stem in sorted(forms):
            if any(w in self.num_values for w in forms[stem]):
                continue
            m, c = self._claim(stem)
            e = self.lexicon.get(stem, {})
            if m:
                status, reason = "claimed", {"given": "既知語として与えられた", "metadata-site": "出土地との偏り",
                                              "alignment": "対訳との対応"}.get(e.get("source"), "複合語の構成要素から")
            else:
                status = "undetermined"
                reason = "信頼度が閾値未満" if e.get("meaning") else "意味に結びつく証拠がない"
            lex.append({"stem": stem, "forms": [w for w, _ in forms[stem].most_common(5)],
                        "count": sum(forms[stem].values()), "pos": self._pos(forms[stem].most_common(1)[0][0], stem),
                        "meaning": m, "confidence": c, "status": status, "reason": reason,
                        "candidate": e.get("meaning"), "alternatives": e.get("alternatives", []),
                        "polysemy_candidate": bool(e.get("polysemy_candidate")), "evidence": e.get("evidence", 0),
                        "compound": list(self.morph.compounds.get(stem, ())) if self.morph.enable else []})
        affixes = [{"affix": a, "types": v[0], "tokens": v[1], "group": self.group_of.get(a),
                    "function": self.affix_fn.get(a, {})}
                   for a, v in sorted(self.morph.affixes.items(), key=lambda kv: -kv[1][1])]
        tokens = {d: [[self.analyze_token(t) for t in s] for s in self.tokens[d]]
                  for d in self.train_ids + self.heldout_ids}
        chars = Counter(ch for w, c in self.counts.items() for ch in w for _ in range(c))
        n_tok = sum(self.counts.values())
        ent = -sum(c / n_tok * math.log2(c / n_tok) for c in self.counts.values()) if n_tok else 0.0
        claims = [x for x in lex if x["status"] == "claimed"]
        return {
            "engine": {"name": "babel-decipher", "version": VERSION, "config": self.cfg},
            "input_warnings": self.input_warnings,
            "language_id": self.pub.get("language_id"), "level": self.level,
            "verdict": self._verdict(),
            "frequency": {"tokens": n_tok, "types": len(self.counts), "word_entropy_bits": round(ent, 3),
                          "top_words": self.counts.most_common(15), "top_chars": chars.most_common(15)},
            "structure": st,
            "variants": {"rules": self.nz.ops, "merged_types": sum(1 for w, c in self.nz.canon.items() if w != c)},
            "morphology": {"affixes": affixes, "allomorph_candidates": self.morph.allomorph_candidates()
                           if self.morph.enable else [], "compounds": len(self.morph.compounds)},
            "classes": self.class_info,
            "numerals": {k: v for k, v in self.num.items() if k != "variables"},
            "lexicon": lex,
            "summary": {"stems": len(lex), "claimed": len(claims), "undetermined": len(lex) - len(claims),
                        "claim_threshold": thr},
            "hypotheses": self._hypotheses(st),
            "relatedness": self.related,
            "tokens": tokens,
            "timing": {"fit_seconds": round(self.fit_seconds, 3), "predict_seconds": round(time.time() - t0, 3)},
        }


def decipher(public: dict, config: dict | None = None) -> dict:
    return Decipherer(public, config).fit().predict()
