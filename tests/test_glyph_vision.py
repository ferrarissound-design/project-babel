"""Vision pipeline tests use entirely synthetic images, no external service."""
import json

import pytest

from babel.vision.glyphs import analyze_image, repeated_patterns, save_analysis

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")


def _draw_test_image(tmp_path, light=False):
    background, ink = (255, 0) if light else (0, 255)
    frame = np.full((90, 160, 3), background, dtype=np.uint8)
    for y in (22, 60):
        for x, glyph in ((8, "X"), (40, "O"), (72, "X"), (104, "O")):
            cv2.putText(frame, glyph, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (ink, ink, ink), 2, cv2.LINE_8)
    target = tmp_path / ("light.png" if light else "dark.png")
    assert cv2.imwrite(str(target), frame)
    return target


@pytest.mark.parametrize("light", [False, True])
def test_repeated_shapes_and_patterns(tmp_path, light):
    img = _draw_test_image(tmp_path, light)
    data = analyze_image(img)
    assert data["stats"]["rows"] == 2
    assert data["stats"]["glyph_candidates"] == 8
    assert data["stats"]["visual_classes"] == 2
    first, second = [row["classes"] for row in data["rows"]]
    assert first == second
    assert first[0] == first[2] and first[1] == first[3] and first[0] != first[1]
    assert any(p["occurrences"] >= 2 and len(p["symbols"]) == 2 for p in data["repeated_patterns"])
    assert "meaning" not in data["glyphs"][0]


def test_save_json_corpus_hypotheses_and_annotation(tmp_path):
    img = _draw_test_image(tmp_path)
    out = tmp_path / "result"
    data = save_analysis(img, out, regions=[{"id": "panel", "box": [0, 0, 160, 90]}])
    assert (out / "glyph-annotated.png").exists()
    assert json.loads((out / "glyph-report.json").read_text())["stats"] == data["stats"]
    continuous = json.loads((out / "corpus-continuous.json").read_text())
    spaced = json.loads((out / "corpus-glyph-tokens.json").read_text())
    assert not continuous["has_word_boundaries"] and spaced["has_word_boundaries"]
    assert continuous["known"] == spaced["known"] == []
    assert continuous["parallel"] == spaced["parallel"] == []
    assert [s.replace(" ", "") for s in spaced["documents"][0]["sentences"]] == continuous["documents"][0]["sentences"]
    assert all(s for s in continuous["documents"][0]["sentences"])


def test_region_validation_and_repeated_pattern_boundaries(tmp_path):
    img = _draw_test_image(tmp_path)
    with pytest.raises(ValueError, match="bounds"):
        analyze_image(img, regions=[{"id": "wrong", "box": [-1, 0, 10, 10]}])
    with pytest.raises(ValueError, match="unique"):
        analyze_image(img, regions=[{"id": "dup", "box": [0, 0, 10, 10]},
                                    {"id": "dup", "box": [10, 0, 20, 10]}])
    with pytest.raises(ValueError, match="threshold"):
        analyze_image(img, threshold=999)
    with pytest.raises(ValueError, match="similarity"):
        analyze_image(img, similarity=0)
    assert repeated_patterns([{"classes": [1, 2]}, {"classes": [3, 4]}]) == []

def test_cli_runs_offline(tmp_path, capsys):
    from babel.__main__ import main

    img = _draw_test_image(tmp_path)
    output = tmp_path / "cli-result"
    assert main(["vision", str(img), "--out-dir", str(output),
                 "--region", "words:0,0,160,90"]) == 0
    assert (output / "glyph-report.json").exists()
    assert "glyph_candidates" in capsys.readouterr().out


# --- Regression tests for PR #6 review (each fails on c350f89) ---

def _canvas(width, height):
    return np.full((height, width, 3), 255, dtype=np.uint8)


def test_tall_glyph_stems_are_not_erased_as_rules(tmp_path):
    """Stems of large glyphs (>=45px) in a 70px-high region were removed as vertical rules."""
    frame = _canvas(200, 70)
    for x in (20, 80, 140):
        cv2.putText(frame, "H", (x, 62), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3, cv2.LINE_8)
    img = tmp_path / "tall.png"
    assert cv2.imwrite(str(img), frame)
    data = analyze_image(img)
    assert data["stats"]["glyph_candidates"] == 3
    assert data["stats"]["visual_classes"] == 1


def test_divider_crossing_glyph_keeps_its_class(tmp_path):
    """Features were cropped from the raw mask, so a removed divider re-entered the shape."""
    frame = _canvas(240, 40)
    for x in (10, 60, 110, 220):                           # four identical 11x11 X glyphs
        cv2.line(frame, (x, 14), (x + 10, 24), (0, 0, 0), 1)
        cv2.line(frame, (x + 10, 14), (x, 24), (0, 0, 0), 1)
    cv2.line(frame, (0, 17), (205, 17), (0, 0, 0), 1)      # divider (>=80% of width) crossing all but the last
    img = tmp_path / "divider.png"
    assert cv2.imwrite(str(img), frame)
    data = analyze_image(img)
    assert data["stats"]["glyph_candidates"] == 4
    assert data["stats"]["visual_classes"] == 1, [g["match_iou"] for g in data["glyphs"]]


def test_full_width_divider_still_removed_and_can_be_kept(tmp_path):
    frame = _canvas(200, 90)
    for y in (25, 70):
        for x in (20, 80, 140):
            cv2.putText(frame, "O", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_8)
    cv2.line(frame, (0, 45), (199, 45), (0, 0, 0), 1)
    img = tmp_path / "rule.png"
    assert cv2.imwrite(str(img), frame)
    data = analyze_image(img)
    assert data["stats"]["rows"] == 2 and data["stats"]["glyph_candidates"] == 6
    assert data["settings"]["remove_rules"] is True
    kept = analyze_image(img, remove_rules=False)
    assert kept["settings"]["remove_rules"] is False
    assert kept["stats"]["glyph_candidates"] != 6       # the divider now shows up as ink
