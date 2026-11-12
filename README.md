# PROJECT BABEL

架空文明の未知言語を、限られた証拠からどこまで妥当に解読できるかを調べる研究システム。
高得点を演出することではなく、**再現可能な証拠で「どこまで分かり、どこから分からないか」を示す**ことが目的。

- 研究ダッシュボード（GitHub Pages）: `docs/` を公開する（下の「公開方法」参照）
- 研究計画（実装前に事前登録）: [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)
- 研究報告書（結果から自動生成）: [docs/RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md)
- 発見した問題と対応: [docs/ISSUES.md](docs/ISSUES.md)
- 初見言語の投入方法: [external/README.md](external/README.md)
- API不要の画像→記号候補・反復パターン解析: [docs/VISION.md](docs/VISION.md)
- 画像認識の定量評価・正解転写の作り方: [docs/VISION_EVAL.md](docs/VISION_EVAL.md)、[docs/GROUND_TRUTH.md](docs/GROUND_TRUTH.md)
- 意味を使わない構造発見（反復・境界・文法的な規則の仮説）: [docs/STRUCTURE.md](docs/STRUCTURE.md)、評価結果: [docs/STRUCTURE_RESULTS.md](docs/STRUCTURE_RESULTS.md)

## 重要な注意

生成器と解読器は同じAIが設計した。内部テストの成績は、完全な未知言語への一般化を証明しない。
独立した初見テストは、第三者が作ったコーパスが投入されるまで**未評価**である。

## 構成

```
babel/generator/    言語と記録文の生成（正解を保持）
babel/decipher/     解読器（generator / evaluate を import しない。テストで強制）
babel/evaluate/     採点（予測と正解を比較する唯一の場所）
babel/experiments/  実験ランナーと採点表
babel/report/       ダッシュボード用 JSON と報告書の生成
babel/structure/    構造発見（意味・辞書・対訳を使わない。翻訳しない。generator / evaluate を import しない）
babel/structure_eval/ 構造発見の評価（生成器・正解・母集団オラクル。解析器には記号列だけを渡す）
babel/vision/       画像→記号候補（既存の方式 baseline と比較用の方式）、正解転写の形式、評価器、アノテーション UI
babel/vision_eval/  画像認識の評価（人工画像と正解、方式の比較、構造解析への影響の実験。認識器には画像のパスだけを渡す）
babel/external.py   初見コーパスの予測台帳（上書き禁止）
docs/               静的サイト（HTML/CSS/JS が docs/data/*.json を表示）
external/           投入コーパス・予測・台帳・正解・採点
research/           問題記録・限界・検証記録（サイトと報告書の元データ）
results/            実験結果
tests/              テスト
```

コアの解読器は Python 標準ライブラリのみで動く（外部API・LLM不要）。オプションの画像解析のみ OpenCV/NumPy をローカルにインストールする。

## 実行方法

```bash
export PYTHONHASHSEED=0
python -m pytest                              # 基本テスト
python -m babel generate --family KES --seed 0 --level A --out data
python -m babel decipher data/KES-0-A/public/corpus.json --out pred.json
python -m babel evaluate pred.json data/KES-0-A/secret/answer.json --public data/KES-0-A/public/corpus.json
python -m babel experiment --mode quick       # 開発シード3個（約1分）
python -m babel experiment --mode full        # 評価シード20個
python -m babel report                        # docs/data と報告書を生成
python -m babel external predict              # external/inbox のコーパスを予測
python -m babel analyze corpus.json --out analysis.json   # 構造発見（analysis.md も書く）
python -m babel structure-benchmark --split eval          # 構造発見の評価（約30分、results/structure/）
python -m babel annotate page.png --gt page.gt.json --report v/glyph-report.json   # 正解転写を作る UI
python -m babel vision-eval v/glyph-report.json page.gt.json  # 認識結果を正解転写と比べる
python -m babel vision-benchmark --split test            # 人工画像で認識方式を比べる（results/vision/）
```

## GitHub Actions

| ワークフロー | 内容 | 起動 |
|---|---|---|
| `ci.yml` | 静的検査、テスト、予測台帳の改竄検査 | push / PR / 手動 |
| `experiment.yml` | コーパス生成と実験（quick / full） | push / 手動 |
| `evaluate.yml` | 評価シードでの採点表とベースライン比較 | push / 手動 |
| `report.yml` | 実験を再実行してサイトを生成し Pages に公開 | main への push / 手動 |
| `external.yml` | 初見コーパスの予測・採点を記録してコミット | `external/` への push / 手動 |

手動実行（workflow_dispatch）は、ワークフローが既定ブランチ（main）に入った後に使える。

## 公開方法（スマートフォンから可能）

1. Pull Request をマージする。
2. リポジトリの Settings → Pages → Source を **GitHub Actions** にする（`report.yml` が公開する）。
   または **Deploy from a branch** で `main` / `docs` を選ぶ（コミット済みの `docs/` をそのまま公開する）。

Pages の有効化はリポジトリ設定の操作なので、AI からは実行できない。

## シードの役割

- 開発シード 0〜4: 解読器の調整に使用
- 評価シード 100〜119: 最終報告専用
