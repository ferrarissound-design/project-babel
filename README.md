# PROJECT BABEL

架空文明の未知言語を、限られた証拠からどこまで妥当に解読できるかを調べる研究システム。
高得点を演出することではなく、**再現可能な証拠で「どこまで分かり、どこから分からないか」を示す**ことが目的。

- 研究ダッシュボード（GitHub Pages）: `docs/` を公開する（下の「公開方法」参照）
- 研究計画（実装前に事前登録）: [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)
- 研究報告書（結果から自動生成）: [docs/RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md)
- 発見した問題と対応: [docs/ISSUES.md](docs/ISSUES.md)
- 初見言語の投入方法: [external/README.md](external/README.md)
- API不要の画像→記号候補・反復パターン解析: [docs/VISION.md](docs/VISION.md)

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
