"""実在言語（PUD）の対訳つき評価（docs/PREREG_REAL_PARALLEL.md）の部品。実データは使わない。"""
from babel.experiments import real_parallel as rp

EN = """# sent_id = n01
1\tKings\tking\tNOUN\t_\tNumber=Plur\t2\tnsubj\t_\t_
2\tgave\tgive\tVERB\t_\tTense=Past\t0\troot\t_\t_
3\tgrain\tgrain\tNOUN\t_\tNumber=Sing\t2\tobj\t_\t_
4\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_

# sent_id = n02
1\tGrain\tgrain\tNOUN\t_\t_\t3\tnsubj:pass\t_\t_
2\twas\tbe\tAUX\t_\t_\t3\taux:pass\t_\t_
3\teaten\teat\tVERB\t_\tTense=Past\t0\troot\t_\t_

# sent_id = n03
1\tIt\tit\tPRON\t_\t_\t2\tnsubj\t_\t_
2\trains\train\tVERB\t_\tTense=Pres\t0\troot\t_\t_

# sent_id = n04
1\tThe\tthe\tDET\t_\t_\t2\tdet\t_\t_
2\tking\tking\tNOUN\t_\t_\t0\troot\t_\t_
"""

TR = """# sent_id = n01
1\tgrain\tgrain\tNOUN\t_\t_\t3\tobj\t_\t_
2\tking\tking\tNOUN\t_\t_\t3\tnsubj\t_\t_
3\tgave\tgive\tVERB\t_\t_\t0\troot\t_\t_

# sent_id = n02
1\tx\tx\tNOUN\t_\t_\t3\tnsubj\t_\t_
2\ty\ty\tNOUN\t_\t_\t3\tobj\t_\t_
3\tz\tz\tVERB\t_\t_\t0\troot\t_\t_
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return rp.parse(p)


def test_translation_rules(tmp_path):
    en = {s["sent_id"]: rp.translation(s) for s in _write(tmp_path, "en.conllu", EN)}
    assert en["n01"] == {"kind": "clause", "pred": "give", "tense": "PAST",
                         "args": [{"role": "subject", "head": "king", "pl": True}, {"role": "object", "head": "grain"}]}
    assert en["n02"] is None            # 受動
    assert en["n03"]["args"] == [{"role": "subject", "head": "it"}]   # 目的語なし、主語だけ
    assert en["n04"] is None            # 根が動詞でない


def test_gold_so_and_verb(tmp_path):
    ts = rp.target_sentences(_write(tmp_path, "tr.conllu", TR))
    assert [s for s, _, _ in ts] == ["n01", "n02"]
    g = rp.gold_so(ts)
    assert g["n"] == 2 and g["shares"]["SO"] == 0.5 and g["value"] is None        # OS と SO が半々 → 正解なし


def test_select_even_spacing_and_mismatch():
    ids = [f"n{i:02d}" for i in range(20)]
    assert rp.select(ids, 0.1) == ["n00", "n10"] and rp.select(ids, 0.25)[:3] == ["n00", "n04", "n08"]
    tr = {s: {"kind": "clause", "pred": s, "args": [{"role": "subject", "head": "k"}]} for s in ids}
    words = [["ab", "cd"] for _ in ids]
    pub, n = rp.build_public("t", words, ids, tr, 0.25, mismatch=False)
    pubm, _ = rp.build_public("t", words, ids, tr, 0.25, mismatch=True)
    assert n == 5 and [p["translation"]["pred"] for p in pub["parallel"]] == ["n00", "n04", "n08", "n12", "n16"]
    assert [p["translation"]["pred"] for p in pubm["parallel"]] == ["n04", "n08", "n12", "n16", "n00"]
    assert [p["doc"] for p in pub["parallel"]] == [p["doc"] for p in pubm["parallel"]]


def test_gold_verb_rel(tmp_path):
    ja = """# sent_id = a
1\tkoku\tkoku\tNOUN\t_\t_\t3\tnsubj\t_\t_
2\tmizu\tmizu\tNOUN\t_\t_\t3\tobj\t_\t_
3\tnomu\tnomu\tVERB\t_\t_\t0\troot\t_\t_
4\tta\tta\tAUX\t_\t_\t3\taux\t_\t_

# sent_id = b
1\tx\tx\tNOUN\t_\t_\t3\tnsubj\t_\t_
2\ty\ty\tNOUN\t_\t_\t3\tobj\t_\t_
3\tz\tz\tVERB\t_\t_\t0\troot\t_\t_
"""
    parsed = _write(tmp_path, "ja.conllu", ja)
    g = rp.gold_verb_rel(parsed)
    assert g["value"] == "final" and g["n"] == 2          # 動詞のあとに助動詞があっても、主語・目的語より後なら final
    en = _write(tmp_path, "en.conllu", EN)
    assert rp.gold_verb_rel(en)["value"] == "medial"       # SVO
