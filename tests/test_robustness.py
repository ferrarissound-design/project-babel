"""外部から入る不正・極端なコーパスで落ちないこと、黙って結果を変えないこと。"""
import json
from pathlib import Path

import pytest

from babel import external
from babel.decipher.engine import decipher
from babel.generator.corpus import build_corpus


def _pub(docs, **kw):
    d = {"documents": [{"id": f"d{i}", "meta": {}, "sentences": s} for i, s in enumerate(docs)]}
    d.update(kw)
    return d


@pytest.mark.parametrize("public, expect_warning", [
    (_pub([]), False),
    (_pub([["ka"]]), False),
    (_pub([["", "  ", "[…] […]"]]), False),
    (_pub([["a" * 3000 + " b"]]), False),
    (_pub([["katamiso", "rakatami"]], has_word_boundaries=False), False),
    (_pub([["ka ta"]], known=[{"form": "ka", "meaning": "#x"}]), True),
    (_pub([["ka ta"]], known=["ka"]), True),
    (_pub([["ka ta"]], parallel=[{"doc": "d0", "sent": 9, "translation": {"kind": "clause", "args": []}}]), True),
    (_pub([["ka ta"]], parallel=[{"doc": "d0", "sent": "x"}]), True),
    (_pub([["ka ta"]], split={"train": ["zzz"], "heldout": []}), True),
    ({"documents": [{"id": "d0", "meta": None, "sentences": ["ka ta"]}]}, False),
    ({"documents": [{"id": "d0", "meta": "S1", "sentences": ["ka ta"]}]}, True),
])
def test_malformed_input_does_not_crash_and_is_reported(public, expect_warning):
    pred = decipher(json.loads(json.dumps(public)))
    assert bool(pred["input_warnings"]) == expect_warning


def test_valid_corpus_produces_no_warnings():
    public, _ = build_corpus("UMB", 3, "A", n_docs=30)
    assert decipher(public)["input_warnings"] == []


def test_bom_is_stripped_from_text_and_json(tmp_path):
    f = tmp_path / "x"
    f.mkdir()
    (f / "corpus.txt").write_text("﻿ka ta\nmi\n", encoding="utf-8")
    assert external.load_corpus(f)["documents"][0]["sentences"][0] == "ka ta"
    (f / "corpus.txt").unlink()
    (f / "corpus.json").write_text("﻿" + json.dumps({"documents": [{"sentences": ["ka"]}]}), encoding="utf-8")
    assert external.load_corpus(f)["documents"][0]["sentences"] == ["ka"]


def test_external_workflow_is_serialised_and_uses_branch_head():
    """並行実行すると、正解を見た予測が最初の試行として残りうる。直列化と先頭の取得を守る。"""
    wf = (Path(__file__).parents[1] / ".github/workflows/external.yml").read_text(encoding="utf-8")
    assert "concurrency:" in wf and "cancel-in-progress: false" in wf
    assert "ref: ${{ github.ref }}" in wf
