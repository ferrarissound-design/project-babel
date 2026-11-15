"""実在言語の評価（docs/PREREG_REAL_LANGUAGES.md）の部品。実データは使わない。"""
from babel.experiments import real_languages as rl

CONLLU = """# sent_id = 1
1\tThe\tthe\tDET\t_\t_\t2\tdet\t_\t_
2\tDogs\tdog\tNOUN\t_\t_\t3\tnsubj\t_\t_
3\tbarked\tbark\tVERB\t_\t_\t0\troot\t_\t_
4\t.\t.\tPUNCT\t_\t_\t3\tpunct\t_\t_

1-2\tdel\t_\t_\t_\t_\t_\t_\t_\t_
1\tde\tde\tADP\t_\t_\t2\tcase\t_\t_
2\tel\tel\tDET\t_\t_\t3\tdet\t_\t_
3\tgatos\tgato\tNOUN\t_\t_\t0\troot\t_\t_

1\tx\tx\tX\t_\t_\t0\troot\t_\t_
2\tno\tno\tPART\t_\t_\t1\tdep\t_\t_
"""


def _write(tmp_path):
    p = tmp_path / "t.conllu"
    p.write_text(CONLLU, encoding="utf-8")
    return p


def test_read_skips_multiword_punct_and_short(tmp_path):
    sents = rl.read_sentences(_write(tmp_path))
    assert [[w for w, _, _ in s] for s in sents] == [["the", "dogs", "barked"], ["de", "el", "gatos"]]


def test_gold_verb_and_affix(tmp_path):
    sents = rl.read_sentences(_write(tmp_path))
    assert rl.gold_verb_position(sents)["value"] == "final"
    g = rl.gold_affix_position(sents)
    assert g["value"] == "suffix" and g["n"] == 3     # dogs, barked, gatos


def test_affix_gold_prefix_and_both():
    s = [[("bhean", "NOUN", "bean"), ("mbád", "NOUN", "bád"), ("bheag", "ADJ", "beag")]]
    assert rl.gold_affix_position(s)["value"] == "prefix"
    s = [[("bhean", "NOUN", "bean"), ("cats", "NOUN", "cat")]]
    assert rl.gold_affix_position(s)["value"] == "both"


def test_verb_gold_none_when_no_majority():
    s = [[("a", "VERB", "a"), ("b", "NOUN", "b"), ("c", "NOUN", "c")],
         [("b", "NOUN", "b"), ("a", "VERB", "a"), ("c", "NOUN", "c")],
         [("b", "NOUN", "b"), ("c", "NOUN", "c"), ("a", "VERB", "a")]]
    assert rl.gold_verb_position(s)["value"] is None


def test_cipher_is_bijective_and_keeps_shape(tmp_path):
    sents = rl.read_sentences(_write(tmp_path))
    words, n = rl.cipher(sents)
    flat_src = [w for s in sents for w, _, _ in s]
    flat = [w for s in words for w in s]
    assert [len(w) for w in flat] == [len(w) for w in flat_src]
    pairs = {(a, b) for x, y in zip(flat_src, flat) for a, b in zip(x, y)}
    assert len({a for a, _ in pairs}) == len(pairs) == len({b for _, b in pairs}) == n
