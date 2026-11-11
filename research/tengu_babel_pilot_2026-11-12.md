# 天狗の詫び証文 × PROJECT BABEL v1.0.1 独立パイロット (2026-11-12)

## データと出典
- 画像原本ではなく、公開された試行転写の **A–H / S1–S3 計145個の字形ラベル**を使用: https://satoru.net/crypt/tengu/transcription_v0.html
- 仮ラベルは見た目が似た漢字・かなであり、日本語の意味や音を示さない。
- 欠損箇所と丸い印は除外、11行/15連続区間。未収録区画が多い。
- 両コーパスとも対訳/既知語は与えず、単語境界なし (LEVEL C) で実行。

## BABEL実測
| 入力 | 記号数 | BABEL推定トークン | 統語構造 present | z (6シャッフル) | 語義主張 |
| --- | ---: | ---: | --- | ---: | ---: |
| 詫び証文の仮転写 | 145 | 56 | false | 0 | 0 |
| 同区間内の字順を攪拌 | 145 | 54 | false | 0 | 0 |
| 規則的に生成した S-O-V 正の対照、空白あり | 145 (空白除く) | 55 | true | 18.16 | 0 |
| 同じ正の対照、空白なし | 145 | 52 | true | 26.37 | 0 |

## 別系統の置換検定
[再現用 Python](../../) は別添。20,000回 × 2条件の乱数入替テストで、
- 145記号・異なり97種、重複する2記号連続は「以女」×2 の1種、3記号連続の重複は0。
- 区間内シャッフルの重複2記号列平均0.6915種。観測値1以上の頻度 0.5131。
- 全体シャッフルの重複2記号列平均0.4543種。観測値1以上の頻度 0.3720。

## 解釈
本物由来の部分転写には、少なくとも今回のBABELが抽出できる明瞭な統語規則は見つからなかった。
同じ短さで規則の明瞭な人工データを検出できることは確認したが、これは本データが意味のない擬似文字であることの証明ではない。

限界:
1. 写真からの字形同定は人手による仮符号化。細部・同一字の崩し・連続部分が不明。
2. 145記号のごく小さな非連続標本。97種類あり、言語データとして極めて疎。
3. 語の境界不明、正解のない初見入力。意味・文法の採点はできない。
4. BABELの構文検定はランダムシャッフル6回しか使わず、z値は概略である。短い陽性対照は非常に規則的であり、自然な文章の性能を保証しない。

## 証拠
- [仮転写入力](https://github.com/ferrarissound-design/project-babel/blob/main/external/inbox/tengu-pilot-v0-real/corpus.json)
- [攪拌入力](https://github.com/ferrarissound-design/project-babel/blob/main/external/inbox/tengu-pilot-v0-shuffled-control/corpus.json)
- [仮転写のBABEL出力](https://github.com/ferrarissound-design/project-babel/blob/main/external/predictions/tengu-pilot-v0-real/attempt-001.json)
- [攪拌のBABEL出力](https://github.com/ferrarissound-design/project-babel/blob/main/external/predictions/tengu-pilot-v0-shuffled-control/attempt-001.json)
- [空白あり正の対照](https://github.com/ferrarissound-design/project-babel/blob/main/external/predictions/tengu-positive-control-boundaries/attempt-001.json)
- [空白なし正の対照](https://github.com/ferrarissound-design/project-babel/blob/main/external/predictions/tengu-positive-control-unspaced/attempt-001.json)
