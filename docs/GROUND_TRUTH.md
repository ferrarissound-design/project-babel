# 画像の正解転写（ground truth）の仕様と作り方

画像認識（babel vision）の精度を測るための、人が確認した正解転写の形式。schema は `babel-glyph-gt-v1`。
実装と検査は `babel/vision/groundtruth.py`、評価は `babel/vision/evaluate.py`、UI は `babel/vision/annotate.py`。

## 原則

- **AI（認識器）の分類を正解として登録しない。** 認識器の案は `proposed` に入れるだけで、`label` には人が付けた名前だけを入れる
- **人が確認していない項目は `verified: false`。** 評価は `verified: true` の項目だけを使い、1つもなければ精度は「測定不能」
- 判断できない記号は `uncertain: true`（クラスの評価から除く）
- 人工画像の正解は `source: "synthetic"`（生成時の座標とクラス）。評価専用で、認識器に渡さない

## 形式

```json
{
  "schema": "babel-glyph-gt-v1",
  "image": "page.png",
  "image_size": [670, 376],
  "source": "human",
  "lines": [
    {"id": "L1", "box": [10, 5, 320, 30], "verified": true, "continues_sentence": false}
  ],
  "items": [
    {"id": "g1", "box": [12, 8, 24, 28], "line": "L1", "kind": "glyph",
     "label": "A", "verified": true, "uncertain": false, "same_as": null,
     "proposed": {"class": 12, "match_iou": 0.83, "by": "babel-vision similarity=0.7"}}
  ],
  "annotators": ["初期の担当者名"],
  "notes": "自由記述"
}
```

| 項目 | 意味 |
|---|---|
| `box` | 画素の座標 `[x1, y1, x2, y2]`、右下は含まない（babel vision と同じ） |
| `lines[].continues_sentence` | `true` なら、この行の終わりは文の区切りではない（折り返し）。babel vision は1行を1文として出すので、この行は「改行を文の区切りと誤認」として数える |
| `items[].kind` | `glyph`（文字）、`decoration`（装飾・枠・アイコン）、`rule`（罫線）、`noise`（汚れ） |
| `items[].label` | 字形の名前。**同じ形には同じ名前**。名前そのものに意味はない（`A`, `B`, `x1` など何でもよい） |
| `items[].outline` | 枠のような中空の要素なら `true`。評価では内部ではなく縁の帯（幅は短い辺の2%、最低2画素）に重なる予測だけを文字以外の誤検出に数える |
| `items[].same_as` | 別の名前で付けたが同じ形だと後で分かった場合の統合先（例: `"A"`）。評価では統合先の名前で数える |
| `items[].line` | 属する行の `id`（文字以外は `null` でよい） |
| `items[].proposed` | 認識器の案（参考）。評価には使わない |

1つの文字が離れた部品からなる場合（点と線など）は、部品をまとめた1つの枠にする。
2つの文字が接触している場合も、文字ごとに別の枠にする（認識器の結合の誤りを測るため）。

## 作り方

```bash
# 1. 認識器で下書きを作る（すべて未確認・label なしで取り込まれる）
python -m babel vision page.png --out-dir v
# 2. ブラウザで直す（127.0.0.1 だけで待ち受ける）
python -m babel annotate page.png --gt page.gt.json --report v/glyph-report.json --open
# 3. 認識器の結果を評価する
python -m babel vision-eval v/glyph-report.json page.gt.json
```

UI の操作: 枠をクリックで選択、中をドラッグで移動、右下の角でリサイズ、`n` の後にドラッグで新しい枠、`Delete` で削除、
`v` で確認済み、`u` で不確か、`s` で保存、`←` `→` で前後の記号。保存のたびに形式を検査し、不正なら保存しない。

## 評価の指標（`babel vision-eval`）

| 指標 | 定義 |
|---|---|
| 検出の precision / recall / F1 | IoU ≥ 0.5 の組を IoU の高い順に **1対1** で対応付ける（1つの正解に複数の予測を対応させない） |
| 分割の誤り | 1つの正解の枠に、面積の半分以上が入る予測が2つ以上ある |
| 結合の誤り | 1つの予測が、面積の半分以上を覆う正解の枠を2つ以上含む |
| 文字以外の誤検出 | 対応しなかった予測のうち、面積の半分以上が罫線・装飾・汚れの枠に入るもの（`outline` の要素は縁の帯） |
| 未確認の位置の予測 | 未確認の正解と対応する（または面積の半分以上が入る）予測は、偽陽性にも precision の分母にも入れず、件数だけを報告する |
| ARI / NMI / 純度 | 対応した記号について、正解の字形名と予測のクラス番号の分類の構造を比べる（番号の一致は見ない） |
| 過剰分割 / 誤統合 | 同じ字形が複数のクラスに分かれた数 / 1つのクラスに複数の字形が入った数 |
| 行の precision / recall | 行の枠を縦方向の IoU ≥ 0.5 で1対1に対応付ける |
| 位置のずれ | 対応した組の IoU の平均と、中心の距離 |
