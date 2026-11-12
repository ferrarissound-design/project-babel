"""Offline glyph discovery, not OCR or a decipherment claim.

Requires optional OpenCV + NumPy. Results are provisional visual clusters:
interactive corrections and multiple independent samples remain necessary.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _dependencies():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Image analysis requires: pip install '.[vision]' (no API key needed)") from exc
    return cv2, np


def _runs(active, max_gap=0):
    """Inclusive (start, end) runs; bridge at most max_gap empty positions."""
    xs = [int(i) for i, value in enumerate(active) if value]
    if not xs:
        return []
    out = []
    start = end = xs[0]
    for x in xs[1:]:
        if x - end - 1 > max_gap:
            out.append((start, end))
            start = x
        end = x
    out.append((start, end))
    return out


def _mask(image, threshold=None):
    cv2, np = _dependencies()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if threshold is None:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        if not 0 <= threshold <= 255:
            raise ValueError("threshold must be between 0 and 255")
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Text is normally the minority. Auto-invert for black glyphs on light paper.
    if np.count_nonzero(binary) > binary.size // 2:
        binary = cv2.bitwise_not(binary)
    return binary


def _without_rules(region):
    """Remove long straight UI dividers; never assume borders are glyphs."""
    cv2, _ = _dependencies()
    height, width = region.shape[:2]
    cleaned = region.copy()
    if width >= 80:
        horizontal = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (max(45, width // 3), 1)))
        cleaned = cv2.subtract(cleaned, horizontal)
    if height >= 60:
        vertical = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(45, height // 2))))
        cleaned = cv2.subtract(cleaned, vertical)
    return cleaned


def _feature(pixels):
    cv2, _ = _dependencies()
    return cv2.resize(pixels, (32, 32), interpolation=cv2.INTER_NEAREST) > 0


def _iou(a, b):
    _, np = _dependencies()
    u = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / u) if u else 1.0


def _rectangles(image_shape, regions):
    height, width = image_shape[:2]
    if not regions:
        return [{"id": "page", "box": [0, 0, width, height]}]
    parsed = []
    used = set()
    for j, r in enumerate(regions):
        if isinstance(r, dict):
            name, box = str(r.get("id", f"region-{j+1}")), r["box"]
        else:
            name, box = f"region-{j+1}", r
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError("Each region must provide [x1, y1, x2, y2]")
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in box):
            raise ValueError("Region coordinates must be integers")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Region {name} is outside image bounds")
        if name in used:
            raise ValueError("Region ids must be unique")
        used.add(name)
        parsed.append({"id": name, "box": [x1, y1, x2, y2]})
    return parsed


def _extract_regions(mask, regions, min_pixels=3, row_gap=6, glyph_gap=2):
    """Projection-based candidate segmentation; every rectangle is auditable."""
    _, np = _dependencies()
    rows, shapes = [], []
    for region in regions:
        x1, y1, x2, y2 = region["box"]
        m = _without_rules(mask[y1:y2, x1:x2])
        for top, bottom in _runs(np.any(m > 0, axis=1), max_gap=row_gap):
            strip = m[top:bottom + 1]
            if int(np.count_nonzero(strip)) < min_pixels:
                continue
            ids = []
            for left, right in _runs(np.any(strip > 0, axis=0), max_gap=glyph_gap):
                crop = strip[:, left:right + 1]
                ys, xs = np.where(crop > 0)
                if len(xs) < min_pixels:
                    continue
                gx1, gy1 = x1 + left + int(xs.min()), y1 + top + int(ys.min())
                gx2, gy2 = x1 + left + int(xs.max()) + 1, y1 + top + int(ys.max()) + 1
                shape = mask[gy1:gy2, gx1:gx2]
                ids.append(len(shapes))
                shapes.append({"box": [gx1, gy1, gx2, gy2], "feature": _feature(shape),
                               "width": gx2 - gx1, "height": gy2 - gy1, "pixels": int(len(xs))})
            if ids:
                rows.append({"region": region["id"], "box": [x1, y1 + top, x2, y1 + bottom + 1],
                             "glyph_indices": ids})
    return rows, shapes


def _cluster(shapes, similarity=0.7):
    if not 0 < similarity <= 1:
        raise ValueError("similarity must be in (0, 1]")
    prototypes = []
    for g in shapes:
        best_id, best_score = None, -1.0
        for label, prototype in enumerate(prototypes):
            if abs(g["width"] - prototype["width"]) > max(2, 0.2 * prototype["width"]):
                continue
            if abs(g["height"] - prototype["height"]) > max(2, 0.2 * prototype["height"]):
                continue
            score = _iou(g["feature"], prototype["feature"])
            if score > best_score:
                best_id, best_score = label, score
        if best_score < similarity:
            best_id = len(prototypes)
            prototypes.append(g)
            best_score = 1.0
        g["class_id"] = int(best_id)
        g["match_iou"] = round(best_score, 3)
    return prototypes


def repeated_patterns(rows, sizes=(2, 3, 4), limit=20):
    counts = Counter()
    for row in rows:
        symbols = row["classes"]
        for n in sizes:
            for j in range(len(symbols) - n + 1):
                counts[tuple(symbols[j:j + n])] += 1
    repeated = [(phrase, count) for phrase, count in counts.items() if count >= 2]
    repeated.sort(key=lambda item: (-item[1], -len(item[0]), item[0]))
    return [{"symbols": list(symbols), "occurrences": count} for symbols, count in repeated[:limit]]


def analyze_image(path, regions=None, threshold=None, similarity=0.7):
    """Analyze an image without network access; output never claims meaning.

    regions: optional [{"id":"left", "box":[x1,y1,x2,y2]}].
    Coordinates use half-open intervals. One-to-many component/glyph
    ambiguity is intentionally reported, not silently resolved.
    """
    cv2, _ = _dependencies()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    selected = _rectangles(image.shape, regions)
    rows, shapes = _extract_regions(_mask(image, threshold), selected)
    _cluster(shapes, similarity)
    if len({g["class_id"] for g in shapes}) > 6400:
        raise ValueError("Too many visual classes for Private Use Area encoding")
    for row in rows:
        row["classes"] = [shapes[i]["class_id"] for i in row.pop("glyph_indices")]
    frequencies = Counter(g["class_id"] for g in shapes)
    visible = [{k: g[k] for k in ("box", "class_id", "match_iou", "pixels")} for g in shapes]
    return {
        "schema": "babel-glyph-pilot-v1", "source_image": Path(path).name,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "regions": selected, "rows": rows, "glyphs": visible,
        "stats": {"rows": len(rows), "glyph_candidates": len(visible),
                  "visual_classes": len(frequencies),
                  "classes_repeated": sum(n > 1 for n in frequencies.values())},
        "frequencies": [{"class_id": c, "count": n} for c, n in sorted(frequencies.items(), key=lambda x: (-x[1], x[0]))],
        "repeated_patterns": repeated_patterns(rows),
        "warnings": ["Visual glyph clusters are not validated characters, words, or translations.",
                     "Projection grouping can split one glyph or merge neighboring glyphs.",
                     "Without translations or other anchors, meanings are not identifiable."],
    }


def _corpus(analysis, spaced):
    """Generate an explicit segmentation HYPOTHESIS for BABEL, not ground truth."""
    documents = []
    for region in analysis["regions"]:
        sentences = []
        for row in analysis["rows"]:
            if row["region"] != region["id"]:
                continue
            symbols = [chr(0xE000 + k) for k in row["classes"]]
            sentences.append(" ".join(symbols) if spaced else "".join(symbols))
        if sentences:
            documents.append({"id": region["id"], "meta": {}, "sentences": sentences})
    return {"language_id": "image-pilot-" + ("glyph-tokens" if spaced else "continuous"),
            "level": "C", "has_word_boundaries": spaced,
            "known": [], "parallel": [], "documents": documents,
            "notes": "Image-derived, unverified segmentation hypothesis. No gold or meaning supplied."}


def save_analysis(path, out_dir, regions=None, threshold=None, similarity=0.7):
    """Write JSON report, two distinct corpus hypotheses, and annotated PNG."""
    cv2, _ = _dependencies()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = analyze_image(path, regions=regions, threshold=threshold, similarity=similarity)
    (out_dir / "glyph-report.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for spaced, filename in ((False, "corpus-continuous.json"), (True, "corpus-glyph-tokens.json")):
        (out_dir / filename).write_text(json.dumps(_corpus(data, spaced), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    for glyph in data["glyphs"]:
        x1, y1, x2, y2 = glyph["box"]
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (255, 170, 40), 1)
        cv2.putText(image, str(glyph["class_id"]), (x1, max(8, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 90, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "glyph-annotated.png"), image)
    return data
