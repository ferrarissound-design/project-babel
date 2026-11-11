# 初見言語テスト

解読器を固定した後に、**解読器の作者が見たことのない言語**を投入して性能を測る。これが最も重要なテスト。

## 手順（スマートフォンの GitHub アプリ／ブラウザから可能）

1. **コーパスを追加**: `external/inbox/<名前>/corpus.txt` を作る（Add file → Create new file）。
   正解はまだ入れない。
2. **予測が自動で保存される**: `external.yml` が `external/predictions/<名前>/attempt-001.json` を作り、
   その SHA-256 を `external/ledger.jsonl` に追記してコミットする。
3. **正解を追加**: `external/answers/<名前>/answer.json` を作る。
4. **採点が自動で保存される**: `external/scores/<名前>/attempt-001.json`。

守られること:

- 予測ファイルは上書きされない。書き換えると CI の台帳検査が失敗する。
- 解読器を修正してから再実行すると、新しい試行番号（attempt-002 …）になる。最初の予測は残る。
- 正解の公開後に作られた予測は `answer_already_public: true` と記録され、初見の成績には数えない。

## コーパスの形式

### テキスト（最も簡単）

UTF-8。1行に1文、語は空白区切り。空行で文書を区切る。`#` で始まる行は無視される。欠損は `[…]` と書く。

```
# 文書1
kamirta tozan belud
sorak minetlar

# 文書2
paluk derimko
```

### JSON（手がかりを付ける場合）

```json
{
  "has_word_boundaries": true,
  "documents": [
    {"id": "d001", "meta": {"site": "S1", "genre": "account"}, "sentences": ["kamirta tozan belud", "sorak minetlar"]}
  ],
  "known": [{"form": "tozan", "meaning": "king"}, {"form": "du", "meaning": "#1"}],
  "parallel": [
    {"doc": "d001", "sent": 0,
     "translation": {"kind": "clause", "pred": "give", "tense": "PAST",
                     "args": [{"role": "subject", "head": "king"}, {"role": "object", "head": "grain", "count": 3, "pl": true}]}}
  ]
}
```

`known` と `parallel` と `meta` は省略できる（省略すると LEVEL C 相当）。
対訳の `role` は subject / object / recipient / location、`tense` は PAST / PRES / FUT。

## 正解の形式（answer.json）

分かる範囲だけでよい。書いた項目だけ採点される。

```json
{
  "independent": true,
  "provenance": "誰が・どうやって作った言語か",
  "structure": {"verb_position": "final", "affix_position": "suffix", "numeral_base": 10,
                "so_order": "SO", "adj_noun": "AN", "num_noun": "NumN"},
  "lexicon": [{"form": "tozan", "meaning": "king"}],
  "numerals": [{"form": "du", "value": 1}]
}
```

- `verb_position`: initial / medial / final、`affix_position`: prefix / suffix / both
- `independent` は、解読器の作者（このリポジトリのAI）が関与していない言語なら `true`。
- 対訳を与えていない場合、解読器は意味を主張しないはずである（主張したら過剰主張）。

## 独立性を高めるには

- 別のAIや人に「名詞の格変化・動詞の時制・数詞体系・会計記録（合計行つき）を持つ人工言語で、100文書ほどの記録を作って」と依頼する。
- このリポジトリの生成器（`babel/generator`）を使わない。
- 合計行のない言語、接辞を持たない言語、語順が自由な言語など、解読器の仮定を外す言語ほど良いテストになる。
