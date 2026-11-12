# 実画像の評価手順（画像を受け取ったら、このとおりに進める）

対象: 以前の実験で使った未知文字の画像 IMG_4494（13行・114記号候補・74視覚クラスという過去の数値は**正解ではない**）。
**画像本体はリポジトリにも実験ブランチにもない。** 画像が届くまで、精度は「測定不能」、構造解析の比較は「評価待ち」のまま。

画像はリポジトリに入れない（`study.json` に sha256 と大きさだけを記録する）。正解転写 `gt.json` も、公開してよいか確認してからコミットする。

## 0. 準備

```bash
python -m pip install -e '.[vision]'      # OpenCV 4.x（pyproject.toml の範囲。評価の数値は版で変わりうる: docs/ISSUES.md #61）
```

## 1. 両方式で認識する（prepare）

```bash
python -m babel image-study prepare study-img4494 --image IMG_4494.jpeg \
  --region left:0,0,330,375 --region right:337,0,670,165        # 文字のある領域（画面の仕切りを避ける）
```

作られるもの（`study-img4494/`）:

| ファイル | 内容 |
|---|---|
| `baseline/`、`improved/`、`improved-v2/` | 各方式の `glyph-report.json`（記号の座標とクラス）と `annotated.png` |
| `comparison.png` | 3方式の認識結果を縦に並べた画像（人が見比べるため） |
| `methods-comparison.json` | 方式どうしの一致（枠の対応・クラスの ARI）。**正しさではない** |
| `gt.json` | 正解転写の下書き。**すべて未確認（verified=false）・字形名なし**。各記号の `proposed` に両方式のクラスを参考として残す。既にあれば上書きしない |
| `study.json` | 画像の sha256・大きさ・領域・OpenCV の版 |

## 2. 人が正解転写を作る（annotate）

```bash
python -m babel annotate IMG_4494.jpeg --gt study-img4494/gt.json --open
```

- 記号の枠を、1つの字に1つの枠になるよう直す（分割された字はまとめ、結合された字は分ける）。離れた部品を持つ字は、部品全体を1つの枠にする
- 同じ形の字に同じ字形名（`label`）を付ける。名前に意味はない（`A`, `B`, … でよい）
- 後で同じ形だと分かった名前は `same_as` で統合する（**評価に使われるのは確認済みの項目の統合だけ**）
- 判断できない字は「不確か」（`uncertain`）。罫線・枠・アイコンは種類を `rule` / `decoration` にし、枠は「中空」に印を付ける
- 行の枠を確かめ、改行が文の区切りでなければ `continues_sentence` にする
- **確認した項目だけを `verified` にする。** 認識器の案（proposed）をそのまま正解にしない

確認の優先順位（docs/VISION_EVAL.md 第4節）: 行3・4（唯一の有意な仮説の根拠）→ 行頭の記号 → 頻度の高いクラス → 1回だけのクラス → 改行 → 文字以外の要素。

## 3. 進み具合を見る（status）

```bash
python -m babel image-study status study-img4494
```

`state` が `not_started`（確認済み0）/ `partial`（一部）/ `complete`（すべての記号と行を確認済み、字形名あり）。
`scope` に、確認済み・未確認の数、使った統合、無視した未確認の `same_as`、統合の矛盾（`alias_conflicts`）が出る。

## 4. 認識の精度を測る（evaluate）

```bash
python -m babel image-study evaluate study-img4494      # → study-img4494/evaluation.json
```

各方式について、検出の precision / recall / F1（1対1の対応付け）、分割・結合の誤り、ARI・NMI・過剰分割・誤統合、行の精度、位置のずれ。
確認済みの範囲だけで計算し、未確認の位置の予測は数えない（件数を報告）。確認済みが0なら「測定不能」。

## 5. 構造解析への影響を比べる（structure）

```bash
python -m babel image-study structure study-img4494     # → structure-comparison.json / .md
```

- A = 人が確認した正解転写、B = baseline、C = improved、C2 = improved-v2。**同じ構造解析・同じ設定**（置換 999 回、部分標本 10、転写の攪乱 6、seed 0）で、入力の転写だけを変える
- 転写ごとに視覚クラスの番号は違うので、仮説は**画像上の位置（根拠の枠）と族ごとの鍵**で照合する。自動転写の記号には、位置で対応した確認済みの字形名を添える（評価用の読み替え）
- 出るもの: 各転写の仮説（補正後 p・安定性）、A の仮説のうち自動転写でも見つかったもの / 見逃したもの、自動転写だけの仮説、行の分け方の違い、単語境界（合議）の A との一致
- 確認済みの正解がなければ「評価待ち」で、結果を作らない。一部だけ確認済み（partial）なら参考値と明記する

## 結果の記録

- 研究記録にするときは、`study-img4494/` の JSON と、使ったコミット・OpenCV の版を `results/vision/real-img4494/` の新しいサブディレクトリに複写する（既存の記録は上書きしない）
- 正解転写の作成者・確認日を `gt.json` の `annotators` / `notes` に書く
