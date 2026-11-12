"""意味側の世界モデル。言語に依存しない命題（proposition）を生成する。

命題は各言語の文法で別々に描画されるため、生成される文は文字置換暗号にならない。
"""
from __future__ import annotations

import random

# 概念 -> 意味クラス
NOUN_CLASSES = {
    "human": ["king", "priest", "scribe", "farmer", "soldier", "merchant", "servant",
              "judge", "governor", "man", "woman", "child", "son", "father", "brother", "enemy"],
    "group": ["army", "people"],
    "divine": ["god"],
    "building": ["temple", "house", "wall", "gate"],
    "place": ["city", "land", "field", "river", "mountain", "road", "sea"],
    "commodity": ["grain", "barley", "oil", "wine", "wool", "cloth", "silver", "gold",
                  "copper", "wood", "stone", "bread"],
    "animal": ["ox", "sheep", "goat", "horse", "fish"],
    "artifact": ["ship", "sword", "spear", "jar", "tablet"],
    "time": ["year", "month", "day", "night"],
    "abstract": ["law", "word", "name", "tribute"],
}
TITLES = ["king", "priest", "scribe", "governor", "judge"]
ADJECTIVES = ["great", "small", "new", "old", "good", "bad", "holy", "strong", "white", "black"]
# 複合語: (修飾部, 主要部, 意味クラス)
COMPOUNDS = [("grain", "house", "building"), ("king", "house", "building"),
             ("sheep", "man", "human"), ("water", "road", "place"), ("war", "ship", "artifact")]
EXTRA_ROOTS = ["water", "war"]  # 複合語の中にだけ現れる語根もある
FUNCTION = ["if", "total", "not"]

# 動詞: (行為者クラス, 対象クラス or None, 受け手を取るか, 場所を取るか, 意味クラス)
VERBS = {
    "give": (["human"], ["commodity", "animal"], True, False, "transfer"),
    "bring": (["human"], ["commodity", "animal", "artifact"], True, False, "transfer"),
    "send": (["human"], ["commodity", "artifact", "human"], True, False, "transfer"),
    "pay": (["human"], ["commodity"], True, False, "transfer"),
    "buy": (["human"], ["commodity", "animal", "artifact"], False, True, "transfer"),
    "sell": (["human"], ["commodity", "animal", "artifact"], False, True, "transfer"),
    "steal": (["human"], ["commodity", "animal", "artifact"], False, False, "transfer"),
    "offer": (["human"], ["commodity", "animal"], True, False, "ritual"),
    "build": (["human", "group"], ["building", "artifact"], False, True, "make"),
    "destroy": (["human", "group"], ["building", "place"], False, False, "conflict"),
    "conquer": (["human", "group"], ["place"], False, False, "conflict"),
    "guard": (["human", "group", "divine"], ["place", "building", "human"], False, False, "conflict"),
    "strike": (["human", "group"], ["human", "group"], False, True, "conflict"),
    "rule": (["human"], ["place", "group"], False, False, "state"),
    "judge_v": (["human"], ["human"], False, False, "state"),
    "see": (["human"], ["human", "place", "animal", "artifact"], False, True, "perceive"),
    "hear": (["human"], ["abstract", "human"], False, False, "perceive"),
    "write": (["human"], ["abstract", "artifact"], False, False, "make"),
    "count": (["human"], ["commodity", "animal"], False, True, "make"),
    "plow": (["human"], ["place"], False, False, "farm"),
    "harvest": (["human"], ["commodity"], False, True, "farm"),
    "eat": (["human", "animal"], ["commodity", "animal"], False, False, "consume"),
    "drink": (["human", "animal"], ["commodity"], False, False, "consume"),
    "carry": (["human", "animal"], ["commodity", "artifact"], False, True, "transfer"),
    "come": (["human", "group"], None, False, True, "motion"),
    "go": (["human", "group"], None, False, True, "motion"),
    "die": (["human", "animal"], None, False, True, "state"),
    "pray": (["human"], None, True, True, "ritual"),
}
AGENTIVE = ["plow", "guard", "write", "carry", "build"]  # 派生名詞（〜する者）を作る動詞

N_PERSONS, N_PLACES, N_GODS, N_SITES = 24, 10, 4, 5
ROLE_TAG = {"agent": "subject", "theme": "subject", "patient": "object", "recipient": "recipient",
            "location": "location", "time": "location"}
GENRES = [("account", 0.35), ("chronicle", 0.20), ("law", 0.12), ("letter", 0.18), ("ritual", 0.15)]


def sem_class_of(concept: str) -> str:
    """人名（P..）も人として扱う意味クラス。"""
    if concept[0] == "P" and concept[1:].isdigit():
        return "human"
    return next((k for k, v in NOUN_CLASSES.items() if concept in v), "")


def sem_class(concept: str) -> str:
    for k, v in NOUN_CLASSES.items():
        if concept in v:
            return k
    if concept in VERBS:
        return "verb_" + VERBS[concept][4]
    if concept in ADJECTIVES:
        return "adj"
    if concept[0] == "P" and concept[1:].isdigit():
        return "name_person"
    if concept[0] == "L" and concept[1:].isdigit():
        return "name_place"
    if concept[0] == "G" and concept[1:].isdigit():
        return "name_god"
    return "function"


def all_concepts():
    nouns = [c for v in NOUN_CLASSES.values() for c in v] + EXTRA_ROOTS
    persons = [f"P{i:02d}" for i in range(1, N_PERSONS + 1)]
    places = [f"L{i:02d}" for i in range(1, N_PLACES + 1)]
    gods = [f"G{i:02d}" for i in range(1, N_GODS + 1)]
    return {"nouns": nouns, "verbs": list(VERBS), "adjs": ADJECTIVES, "persons": persons,
            "places": places, "gods": gods, "function": FUNCTION}


class World:
    """1つの文明の記録（文書→命題列）を作る。"""

    def __init__(self, rng: random.Random, totals: bool = True, same_concept_rate: float = 0.0):
        self.rng = rng
        # >0: 行為者と同じ概念を、受け手・目的語（人を取れる場合）にも使う（「商人が商人に魚を渡した」）。
        # 0 のときは乱数を余分に消費しないので、既存のコーパスは変わらない
        self.same_concept_rate = same_concept_rate
        self.totals = totals    # False: 会計記録に合計行を書かない（独立初見テストで外れた仮定の再現用）
        c = all_concepts()
        self.persons, self.places, self.gods = c["persons"], c["places"], c["gods"]
        self.sites = [f"S{i}" for i in range(1, N_SITES + 1)]
        self.site_place = {s: self.places[i] for i, s in enumerate(self.sites)}
        # 各出土地には地元の人物が2人いる（地名推定の交絡要因）
        self.site_person = {s: self.persons[3 + 2 * i: 5 + 2 * i] for i, s in enumerate(self.sites)}
        self.kings = self.persons[:3]  # 時代ごとの王

    # ---- 名詞句 ----
    def np(self, head, pl=False, adj=None, count=None, poss=None, title=None, deriv=None):
        return {"head": head, "pl": pl, "adj": adj, "count": count, "poss": poss,
                "title": title, "deriv": deriv}

    def person(self, site, titled=0.3):
        r = self.rng
        if r.random() < 0.35:
            p = r.choice(self.site_person[site])
        else:
            p = self.persons[min(int(r.expovariate(0.12)), len(self.persons) - 1)]
        return self.np(p, title=r.choice(TITLES) if r.random() < titled else None)

    def place(self, site):
        r = self.rng
        return self.np(self.site_place[site] if r.random() < 0.6 else r.choice(self.places))

    def common(self, classes, allow_pl=True, p_adj=0.18, p_poss=0.12, site=None):
        r = self.rng
        cls = r.choice(classes)
        pool = list(NOUN_CLASSES[cls])
        pool += ["+".join(c[:2]) for c in COMPOUNDS if c[2] == cls]
        head = r.choice(pool)
        deriv = None
        if cls == "human" and r.random() < 0.12:
            head, deriv = r.choice(AGENTIVE), "AGENT"
        pl = allow_pl and cls != "commodity" and r.random() < 0.3
        adj = r.choice(ADJECTIVES) if r.random() < p_adj else None
        poss = None
        if r.random() < p_poss and site:
            poss = self.person(site, 0.0) if r.random() < 0.5 else self.np(r.choice(["king", "god", "city", "temple"]))
        return self.np(head, pl=pl, adj=adj, poss=poss, deriv=deriv)

    def agent(self, classes, site):
        if "human" in classes and self.rng.random() < 0.45:
            return self.person(site)
        return self.common([c for c in classes], p_poss=0.08, site=site)

    def clause(self, verb, site, tense, neg=False, cond=False, agent=None, patient=None):
        r = self.rng
        a_cls, p_cls, takes_rec, takes_loc, _ = VERBS[verb]
        args = [("agent", agent or self.agent(a_cls, site))]
        if p_cls:
            if patient is None:
                patient = self.common(p_cls, site=site)
                if patient["head"] in NOUN_CLASSES["commodity"] + NOUN_CLASSES["animal"] and r.random() < 0.35:
                    patient["count"] = self.amount(small=True)
                    patient["pl"] = patient["count"] > 1
            args.append(("patient", patient))
        if takes_rec and r.random() < 0.6:
            rec = self.np(r.choice(self.gods)) if verb in ("offer", "pray") else \
                (self.person(site) if r.random() < 0.6 else self.np(r.choice(["temple", "king", "god"])))
            args.append(("recipient", rec))
        if takes_loc and r.random() < 0.35:
            loc = self.place(site) if r.random() < 0.6 else self.np(r.choice(["city", "field", "temple", "river", "mountain", "gate"]))
            args.append(("location", loc))
        if self.same_concept_rate > 0 and r.random() < self.same_concept_rate:
            head = args[0][1]["head"]
            if "+" not in head and sem_class_of(head) == "human":
                for i, (role, np) in enumerate(args):
                    if role == "recipient" or (role == "patient" and "human" in (p_cls or [])):
                        args[i] = (role, self.np(head, pl=r.random() < 0.2))
                        break
        return {"kind": "clause", "pred": verb, "tense": tense, "neg": neg, "cond": cond,
                "args": [{"role": k, "np": v} for k, v in args]}

    def amount(self, small=False):
        r = self.rng.random()
        if small or r < 0.5:
            return self.rng.randint(1, 12)
        if r < 0.85:
            return self.rng.randint(13, 40)
        return self.rng.randint(41, 60)

    # ---- 文書 ----
    def document(self, genre, site, era):
        r = self.rng
        props = []
        if genre == "account":
            item = r.choice(NOUN_CLASSES["commodity"] + NOUN_CLASSES["animal"])
            verb = r.choice(["give", "bring", "pay"])
            rec = self.np(r.choice(["temple", "king"])) if r.random() < 0.7 else None
            total = 0
            for _ in range(r.randint(3, 7)):
                n = self.amount()
                total += n
                pat = self.np(item, pl=n > 1, count=n)
                ag = self.person(site, 0.15)
                if r.random() < 0.3:
                    ag["poss"] = self.place(site)
                c = self.clause(verb, site, "PAST", agent=ag, patient=pat)
                c["args"] = [a for a in c["args"] if a["role"] in ("agent", "patient")]
                if rec:
                    c["args"].append({"role": "recipient", "np": rec})
                props.append(c)
            if self.totals:
                props.append({"kind": "total", "pred": None, "tense": None, "neg": False, "cond": False,
                              "args": [{"role": "theme", "np": self.np(item, pl=total > 1, count=total)}]})
        elif genre == "chronicle":
            king = self.np(self.kings[era], title="king")
            props.append({"kind": "heading", "pred": None, "tense": None, "neg": False, "cond": False,
                          "args": [{"role": "time", "np": self.np("year", count=r.randint(1, 30), poss=king)}]})
            for _ in range(r.randint(5, 11)):
                v = r.choice(["conquer", "destroy", "build", "strike", "guard", "rule", "come", "go", "die", "see", "send"])
                ag = king if r.random() < 0.4 else None
                pat = self.place(site) if v in ("conquer", "rule") and r.random() < 0.6 else None
                props.append(self.clause(v, site, "PAST", neg=r.random() < 0.06, agent=ag, patient=pat))
        elif genre == "law":
            for _ in range(r.randint(3, 6)):
                v = r.choice(["steal", "strike", "destroy", "sell", "buy", "harvest", "plow", "carry"])
                ag = self.np(r.choice(["man", "servant", "merchant", "farmer", "soldier"]))
                props.append(self.clause(v, site, "PRES", cond=True, agent=ag))
                fine = self.np(r.choice(["silver", "grain", "copper", "barley"]), count=self.amount(), pl=True)
                v2 = r.choice(["pay", "give"])
                c = self.clause(v2, site, "FUT", agent=dict(ag), patient=fine)
                props.append(c)
                if r.random() < 0.3:
                    props.append(self.clause("judge_v", site, "FUT", agent=self.np("judge"), patient=dict(ag)))
        elif genre == "letter":
            a, b = self.person(site, 0.2), self.person(site, 0.2)
            props.append({"kind": "clause", "pred": "write", "tense": "PRES", "neg": False, "cond": False,
                          "args": [{"role": "agent", "np": a}, {"role": "recipient", "np": b}]})
            for _ in range(r.randint(5, 10)):
                v = r.choice(["send", "bring", "buy", "sell", "come", "go", "see", "hear", "count", "carry", "give"])
                props.append(self.clause(v, site, r.choice(["PAST", "PRES", "FUT", "FUT"]), neg=r.random() < 0.15))
        else:  # ritual
            for _ in range(r.randint(5, 10)):
                v = r.choice(["offer", "pray", "guard", "eat", "drink", "bring", "see", "hear"])
                ag = self.np("priest", pl=r.random() < 0.3) if r.random() < 0.4 else None
                if v == "guard" and r.random() < 0.6:
                    ag = self.np(r.choice(self.gods), adj=r.choice(["great", "holy", "good"]) if r.random() < 0.4 else None)
                props.append(self.clause(v, site, r.choice(["PRES", "PRES", "FUT"]), agent=ag))
        return props


def translate(prop, rng: random.Random, dropout: float):
    """命題の部分対訳（構造化）。要素を確率 dropout で欠落させる。"""
    def keep():
        return rng.random() >= dropout

    def np_tr(np):
        out = {"head": np["head"] if keep() else None}
        if np["pl"] and keep():
            out["pl"] = True
        if np["adj"] and keep():
            out["adj"] = np["adj"]
        if np["count"] is not None and keep():
            out["count"] = np["count"]
        if np["title"] and keep():
            out["title"] = np["title"]
        if np["deriv"]:
            out["deriv"] = np["deriv"]
        if np["poss"] and keep():
            out["poss"] = np_tr(np["poss"])
        return out

    tr = {"kind": prop["kind"], "args": []}
    if prop["pred"]:
        tr["pred"] = prop["pred"] if keep() else None
        tr["tense"] = prop["tense"] if keep() else None
    if prop["neg"]:
        tr["neg"] = True
    if prop["cond"]:
        tr["cond"] = True
    for a in prop["args"]:
        if keep() or len(prop["args"]) == 1:
            tr["args"].append({"role": ROLE_TAG[a["role"]], **np_tr(a["np"])})
    return tr
