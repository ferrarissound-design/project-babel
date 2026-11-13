"""docs/ROBUST_STRUCTURE.md の本文。数値はすべて引数（results/robust/ の JSON から計算したもの）から埋める。"""
from __future__ import annotations

HELD = {True: "当たった", False: "**外れた**"}


def _pct(v):
    return "—" if v is None else f"{100 * v:.0f}%"


def render(c):
    S, k, vk, f, cmp_ = c["S"], c["k"], c["vk"], c["f"], c["cmp"]
    dev = S["dev"]
    hs, hf = S.get("holdout-synth"), S.get("holdout-font")
    L = []
    w = L.append

    def ov(split, m, key):
        return S[split]["overall"][m][key]

    def trio(m, key, nd=3):
        return " / ".join(f(S[n]["overall"][m][key], nd) for n in ("dev", "holdout-synth", "holdout-font") if S.get(n))
    splits_present = "・".join(n for n in ("dev", "holdout-synth", "holdout-font") if S.get(n))
    held_all = all(ok for _, ok, _ in c["checks"]) if c["checks"] else None
    failed = [i for i, ok, _ in c["checks"] if not ok]

    w("# 認識の誤りに強い構造検定: 本当にある構造と、認識の誤りが作った構造を区別できるか\n")
    w("このファイルは `python -m babel.robust_eval.report` が results/robust/ の JSON から生成する。"
      "既存の記録（docs/VISION_EVAL*.md、results/vision/、results/structure/）と既存の認識方式・構造解析のコードは変えていない。\n")
    w("問い: **PROJECT BABEL は、未知の文字列から、本当に存在する構造と、画像認識の誤りが生み出した見かけ上の構造を区別できるのか。**\n")

    # ------------------------------------------------------------------ 0
    w("## 0. 結論（先に）\n")
    w("事実（測ったこと）・仮説（測っていないこと）を分けて書く。\n")
    w("**測ったこと**\n")
    pd = {tuple(d["rows"]): d for d in cmp_["pair_details"]}
    w(f"1. IMG_4494 で「認識方式を変えると見つかる規則が変わった」原因は、**規則が入れ替わったのではなく、既存の検定が記号1つの取り違えで切れること**。"
      f"左4・5行目（行 3・4）は improved-v3 が記号1つを別のクラスとみなしたために、共有する 2〜4 記号列が {pd[(3, 4)]['shared_ngrams_a']} 種から "
      f"{pd[(3, 4)]['shared_ngrams_b']} 種になった。左1・2行目（行 0・1）は逆に baseline が記号1つを別のクラスとみなしていた"
      f"（{pd[(0, 1)]['shared_ngrams_a']} 種と {pd[(0, 1)]['shared_ngrams_b']} 種）。"
      "記号の抜け・取り違えがあっても切れない検定（順序を保った共通部分列）では、**2組とも、baseline・improved-v3 のどちらの転写でも有意**で、"
      "行を過剰に分割した improved-v2 の転写（25行）でも同じ2組の本体の行が有意（第2節）")
    w(f"2. 正解つきの画像（{splits_present}）では、**1つの転写だけを見ていると、認識の誤りが作った規則を本物と区別できない**。"
      f"仕込みのない画像で1つでも行の組を報告する割合は、行を過剰に分割する improved-v2 の転写で {trio('lcs:v2', 'fwer', 2)}、"
      f"improved-v3 の転写でも {trio('lcs:v3', 'fwer', 2)}（名目は 0.05。正解の転写なら {trio('oracle-lcs', 'fwer', 2)}）（第5・6節）")
    w(f"3. **複数の認識方式で転写し、{k}/8 以上の方式で有意な規則だけを採ると区別できる**（この種類の規則＝行の組の類似について）。"
      f"仕込みのない画像での誤報告 {trio(vk, 'fwer', 2)}、仕込んだ組の発見率 {trio(vk, 'recall', 2)}"
      f"（既存の方式 {trio('ngram:v3', 'recall', 2)}、正解の転写での上限 {trio('oracle-lcs', 'recall', 2)}）、報告のうち真 {trio(vk, 'precision', 2)}。"
      "値は dev / holdout-synth / holdout-font の順。holdout は閾値を固定してから1回だけ走らせた")
    if c["checks"]:
        w(f"4. holdout の結果を見る前に書いた予測 {len(c['checks'])} 件のうち、" +
          ("すべて当たった" if held_all else f"{len(c['checks']) - len(failed)} 件が当たり、**{'・'.join(failed)} が外れた**") + "（第6節）")
    w("\n**測っていないこと（仮説のまま）**\n")
    w("- IMG_4494 の2組（行 0・1、行 3・4）が画像に本当にあるか。**画像がリポジトリにないので、方式の集まりを実画像に使えていない。**"
      "転写2本で残ることは確かめたが、2つの方式に共通する誤り（同じ二値化・似た切り出し）が作った可能性は除けていない")
    w("- 79 クラス・58 個の「1回だけのクラス」が、記号の種類の多さなのか、分けすぎなのか（第8節。転写の比較と人工画像からは、どちらとも決まらない）")
    w("- 見つかった構造の意味（文法・定型句・見た目の繰り返しのどれか）。この研究は意味に到達していない（第10節）\n")

    # ------------------------------------------------------------------ 1
    w("## 1. 手元にあるもの・ないもの\n")
    sa, sb = cmp_["stats"]["a"], cmp_["stats"]["b"]
    w("| もの | 状態 |\n|---|---|")
    w("| IMG_4494 の画像本体 | **ない**（リポジトリにも作業環境にもない。docs/ISSUES.md #60） |")
    w(f"| baseline の転写（{sa['rows']} 行・{sa['symbols']} 記号・{sa['classes']} クラス、1回だけ {sa['hapax']}） | results/vision/real-img4494/corpus-continuous.json |")
    w(f"| improved-v3 の転写（{sb['rows']} 行・{sb['symbols']} 記号・{sb['classes']} クラス、1回だけ {sb['hapax']}） | results/robust/real-img4494/inputs/v3-continuous.json |")
    w("| improved-v2 の転写（25 行） | results/robust/real-img4494/inputs/v2-continuous.json |")
    w("| 記号の位置（枠）・認識結果（glyph-report） | **ない** |")
    w("| 人が確かめた正解の転写 | **ない** |\n")
    w("- v2・v3 の転写は、マージされていない実験ブランチ（experiment/img4494-v3-structure-20261113）から内容を変えずに写した。"
      "出どころとハッシュは results/robust/real-img4494/PROVENANCE.json（テストで照合する）")
    w(f"- 依頼にあった「13行・119記号候補・79視覚クラス・58クラスが1回だけ」は、improved-v3 の転写と一致する（{sb['rows']}・{sb['symbols']}・{sb['classes']}・{sb['hapax']}）")
    w("- したがって実画像について言えるのは、**3本の転写どうしの関係**まで。どの転写が正しいかは、この研究では分からない\n")

    # ------------------------------------------------------------------ 2
    w("## 2. 実画像の転写の比較（画像なしでできる範囲）\n")
    w("詳細は results/robust/real-img4494/compare-baseline-v3.md（`babel transcription-compare` の出力）。行の番号は 0 から数える"
      "（依頼の「左側4行目と5行目」は行 3・4、「左側1行目と2行目」は行 0・1）。\n")
    w("### 2.1 観測の再現\n")
    w("既存の構造解析（`babel analyze`、置換 999 回）の「似た記号列を共有する行」の仮説:\n")
    w("| 転写 | 有意な行の組 | 記録 |\n|---|---|---|")
    w("| baseline | (3, 4) | results/vision/real-img4494/analysis.json（H01） |")
    w("| improved-v3 | (0, 1) | results/robust/real-img4494/analyze-v3.json |")
    w("| improved-v2（25 行） | (6, 8)、(0, 2) | results/robust/real-img4494/analyze-v2.json |\n")
    w("依頼の観測（H01 が improved-v3 で消え、行 0・1 の類似が出た）は再現した。\n")
    w("### 2.2 原因: 既存の検定は、記号1つの取り違えで切れる\n")
    w("既存の検定は、2つの行が共有する**連続した** 2〜4 記号列の数を数える。2つの転写を位置ごとに対応させて（2.4）、同じ行の組を見比べた:\n")
    names = {"a": "baseline", "b": "improved-v3"}
    word = {"same": "同じ", "different": "**別のクラス**", "split_or_merged": "**分割・結合**", "missing": "相手なし"}
    for key in ((3, 4), (0, 1)):
        d = pd[key]
        w(f"**行 {key}**（共有する 2〜4 記号列: baseline {d['shared_ngrams_a']} 種、improved-v3 {d['shared_ngrams_b']} 種。既存の検定は 2 種以上を要する）\n")
        w(f"- baseline: `{' '.join(map(str, d['row_a'][0]))}` / `{' '.join(map(str, d['row_a'][1]))}`")
        w(f"- improved-v3: `{' '.join(map(str, d['row_b'][0]))}` / `{' '.join(map(str, d['row_b'][1]))}`")
        for side, other in (("a_matches", "b"), ("b_matches", "a")):
            me = "a" if side == "a_matches" else "b"
            w(f"- {names[me]} が「2行に共通」とみなした {len(d[side])} 記号を {names[other]} で見ると: "
              + "、".join(f"{m['class']}→{word[m['other']]}" for m in d[side]))
        w("")
    w("どちらの組も、2つの転写は共通の記号 4 個のうち 3 個で一致し、1 個で食い違う。その1個が連続した記号列の途中にあると、"
      "共有する記号列が 0〜1 種になって既存の検定の下限（2 種）を割る。**規則が入れ替わったのではなく、同じ2組の規則が、"
      "転写ごとに別の1記号で切れていた。**\n")
    w("### 2.3 切れない検定での結果\n")
    t = cmp_["tests"]["a"]
    w(f"順序を保った共通部分列の長さを統計量にし（第3節）、全体シャッフル（置換 {t['permutations']} 回）と比べた。多重検定は行の組 {t['n_pairs']} 個すべてにわたる maxT。\n")
    w("| 行の組 | baseline | improved-v3 | 細かい側 | 粗い側 | 位置を保つ帰無モデルでも有意 |\n|---|---|---|---|---|---|")

    def cell(key, name):
        for p in cmp_["tests"][name]["pairs"]:
            if tuple(p["rows"]) == key:
                return f"有意（一致 {p['score']:g}、p={p['p_adj']:.4f}）"
        for p in cmp_["tests"][name]["top"]:
            if tuple(p["rows"]) == key:
                return f"有意でない（一致 {p['score']:g}、p={p['p_adj']:.3f}）"
        return "有意でない"
    for v in cmp_["verdicts"]:
        key = tuple(v["rows"])
        w(f"| {key} | {cell(key, 'a')} | {cell(key, 'b')} | {cell(key, 'fine')} | {cell(key, 'coarse')} | "
          f"{'、'.join(names[x] for x in v['column_null']) or '—'} |")
    w("")
    w(f"- 細かい側（{cmp_['stats']['fine_classes']} クラス）: 2つの転写の**両方**が同じとみなした位置だけを同じクラスにした転写。"
      f"粗い側（{cmp_['stats']['coarse_classes']} クラス）: **どちらか**が同じとみなした位置を同じクラスにした転写。"
      "2つの方式の分類の食い違いを、どちらに倒しても残るかを見る")
    w("- 位置を保つ帰無モデル: 行の先頭から数えた位置が同じ記号どうしだけを入れ替える（「どの行も同じ書き出しで始まる」だけで似て見える場合を除く感度分析）")
    ex = list(cmp_["extra"].values())[0]
    w(f"- improved-v2 の転写（25 行、行の分け方が違うので単独で検定）: 有意な組は {'、'.join(str(tuple(p['rows'])) for p in ex['pairs'])}。"
      "行の長さと記号の並びから、v2 の行 0・2 は v3 の行 0・1 の本体、v2 の行 6・8 は v3 の行 3・4 の本体に当たる（位置の情報がないので、並びからの対応）")
    w("- 行 (0, 9)（左1行目と右1行目）は baseline と細かい側・粗い側で有意、improved-v3 単独では有意でない。行 (4, 6) は baseline だけ。"
      "この2組は「一方の転写に依存する」ので、規則として扱わない\n")
    w("### 2.4 2つの転写は、どの位置を「同じ記号」とみなしたか\n")
    pa, pall = cmp_["pair_agreement"]["anchor_rows"], cmp_["pair_agreement"]["all_rows"]
    h = cmp_["hapax"]
    w("クラス番号は転写ごとに別なので、番号ではなく「同じ記号とみなした位置の組」を比べた。長さが同じ行"
      f"（{len(cmp_['alignment']['anchor_rows'])} 行）は位置ごとに1対1と**仮定**し、ほかの行はそこで学んだクラスの対応を手掛かりに推定した。\n")
    w("| 範囲 | 位置の組 | 両方が同じ | baseline だけが同じ | improved-v3 だけが同じ | 調整ランド指数 |\n|---|---:|---:|---:|---:|---:|")
    w(f"| 長さが同じ行だけ | {pa['pairs']} | {pa['same_in_both']} | {pa['same_only_in_a']} | {pa['same_only_in_b']} | {pa['adjusted_rand']} |")
    w(f"| 全行（推定の対応を含む） | {pall['pairs']} | {pall['same_in_both']} | {pall['same_only_in_a']} | {pall['same_only_in_b']} | {pall['adjusted_rand']} |\n")
    w(f"- baseline が同じとみなした組の {_pct(pa['a_same_confirmed_by_b'])} は improved-v3 も同じとみなす。逆は {_pct(pa['b_same_confirmed_by_a'])}。"
      "improved-v3 には、baseline が 3〜4 クラスに分けた記号をまとめた頻出クラス（クラス 2・26）がある。"
      "**improved-v3 は baseline より「分けすぎ」でもあり「まとめすぎ」でもある**（1回だけのクラスも多く、頻出クラスも大きい）")
    n12 = sum(1 for cs in cmp_["alignment"]["columns"] for a, b in cs if a and b and len(a) != len(b))
    w(f"- 一方の1記号がもう一方の2記号に当たる箇所が {n12}（推定）。問題1の「一方を1記号、もう一方を複数記号」に当たる。どちらが正しいかは画像がないと分からない")
    w(f"- 1回だけのクラス: 1対1に対応した位置のうち、improved-v3 で1回だけの {h['b_hapax_aligned']} 個中 {h['hapax_in_both']} 個は baseline でも1回だけ"
      f"（{h['b_hapax_repeated_in_a']} 個は baseline では繰り返すクラス）\n")
    w("### 2.5 この節から言えること・言えないこと\n")
    w("- 言える: 行 0・1 と行 3・4 の類似は、3本の転写のどれでも見える。「方式を変えたら規則が変わった」のは検定の作りによる")
    w("- 言えない: この2組が画像に本当にあるか。3本の転写は同じ画像・同じ二値化の系統から作られていて、共通の誤りを除けない。"
      "また、**新しい統計量は、この転写を見てから選んだ**（最初に試した1つで、その後は変えていない）。実データでの p 値は探索的なものとして読むこと。"
      "統計量の妥当性（誤報告の率）と検出力は、この転写と無関係な人工データで測った（第5・6節）")
    w("- 言えない: 2組の意味。同じ書き出しの2行、同じ枠の2行という以上のことは分からない\n")

    # ------------------------------------------------------------------ 3
    w("## 3. 方法\n")
    w("### 3.1 行の組の類似の検定（babel/robust/rowsim.py）\n")
    w("- 統計量: 2つの行の、順序を保って対応する記号の数の最大値（最長共通部分列の長さ）。記号の取り違え・抜け・余分があっても、残りの一致を数える")
    w("- 帰無モデル: 全体シャッフル（行の長さを保ち、全記号の位置を入れ替える）。既存の linepair 族と同じ")
    w("- 多重検定: 組ごとに置換分布で標準化し、行の組すべての最大値の置換分布と比べる（Westfall–Young の maxT）。「1つでも誤って有意とする確率」を制御する")
    w("- 支持の下限: 一致が 3 記号未満の組は報告しない")
    w("- 標準ライブラリだけで動く（NumPy があれば置換をまとめて計算する。同じ値になることをテストで確かめている）\n")
    w("### 3.2 認識方式の集まりと票（babel/robust/ensemble.py）\n")
    w("1枚の画像を 8 通りに転写し、方式ごとに 3.1 の検定を行う。仮説は、行の枠の上下の重なりで基準の方式（improved-v3）の行に対応させる（クラス番号は使わない）。"
      "ある行の組が何個の方式で有意かを「票」とする。\n")
    w("| 方式 | 切り出し | 分類 | 変えたもの |\n|---|---|---|---|")
    w("| v3 | improved-v3 | template（閾値 0.20） | 基準 |")
    w("| v3-fine | 同じ | template（閾値 0.12） | 分類を細かく |")
    w("| v3-coarse | 同じ | template（閾値 0.30） | 分類を粗く |")
    w("| v3-shape | 同じ | shape（特徴量） | 分類の方式 |")
    w("| v3-iou | 同じ | iou（baseline の分類） | 分類の方式 |")
    w("| v3-scaled | improved-v3（画像を 1.3 倍） | template | 画像の大きさ |")
    w("| v2 | improved-v2（行のまとめ方が旧方式） | template | 行のまとめ方 |")
    w("| baseline | 投影 | iou | 切り出しと分類の両方 |\n")
    w("構成は dev の結果を見る前に決め、その後は変えていない。既存の方式（METHODS）は書き換えず、同じ部品を呼ぶだけ（テストで確かめている）。\n")
    w("### 3.3 分類を経ない検定（不採用）\n")
    w("記号をクラスに分けず、字形の類似度（0〜1）を重みにして同じ検定を行う方式も作った。分類の閾値の近くで「同じ」と「違う」に倒れるのを避ける狙い。"
      "dev では、クラスの一致による検定より良くならなかった（第5節）ので、判定には使っていない。\n")
    w("### 3.4 dev だけで選んだもの（results/robust/selected.json）\n")
    sel = c["sel"]
    w(f"- 票の閾値: 規則は「仕込みのない dev 画像で1つでも報告する割合が {sel['rule']['limit']} 以下になる最小の k」。結果は **k = {k}**")
    w("- 分類を経ない検定の tau: 規則は「同じ割合が 0.05 以下の候補のうち発見率が最大」。"
      + ("満たす候補がなく、**不採用**" if sel["soft_tau"] is None else f"結果は {sel['soft_tau']}"))
    w("- 選択の規則は dev の結果を見る前に決めた（babel/robust_eval/select.py）。holdout では何も選び直していない\n")

    # ------------------------------------------------------------------ 4
    w("## 4. 評価データ（babel/robust_eval/）\n")
    w("| 分割 | 字 | 行の作り・劣化 | シード | 画像 | 用途 |\n|---|---|---|---|---:|---|")
    for n, use in (("dev", "方式の設計、閾値の選択"), ("holdout-synth", "閾値を固定してから1回だけ"), ("holdout-font", "閾値を固定してから1回だけ")):
        if not S.get(n):
            continue
        cf = S[n]["config"]
        w(f"| {n} | {'・'.join(cf['families'])} | {'・'.join(cf['conditions'])} | {cf['seeds'][0]}〜{cf['seeds'][-1]} | {S[n]['cases']} | {use} |")
    w("")
    w("- 文（texts.py）: IMG_4494 の転写と同じ規模（13 行、行の長さ 6〜13）。語彙 30・80・250 の Zipf 分布から引く。"
      "**仕込みあり**の文は 2 組の行に類似を仕込む（一方を写し、各位置を 15〜45% の確率で別の記号に替え、ときどき1記号を抜くか足す）。"
      "**仕込みなし**の文は陰性対照。各分割の半分ずつ")
    w("- dev の字は babel/vision_eval/layout.py の font・stroke・stack。holdout-synth は dev に出ない字の作り（radical・curve）と劣化（photo）")
    w("- holdout-font は、**実在の文字体系を実在のフォントで描いた画像**（realfont.py: グルジア文字・アルメニア文字・タイ文字・片仮名・平仮名・漢字・ハングル、"
      "アンチエイリアスあり、degraded はぼかしと JPEG）。字形の設計に、このリポジトリの作者が関わっていない。ただし実画像（写真・手書き）の代わりにはならない")
    w("- 採点: 見つかった行の組を、記号の重なりで正解の行に戻す。仕込んだ組なら真、それ以外（仕込んでいない組・同じ行の破片どうし・正解の行に戻せない行）は偽")
    w("- 認識器には画像（と領域の枠）だけを渡す。babel/robust は正解・評価用のコードを import しない（テストで確かめている）")
    w("- **holdout について開発中に見たもの**: realfont の描画が正しいかを確かめるために、シード 1 の画像 5 枚（グルジア・タイ・片仮名・漢字・ハングル）で "
      "baseline・improved-v2・improved-v3 の認識の指標（記号数・検出 F1・ARI）を見た。行の組の検定は走らせていない。評価のシード（9200〜）とは別\n")

    # ------------------------------------------------------------------ 5
    votes = [f"vote>={x}" for x in range(2, 9)]
    w("## 5. 結果: dev\n")
    w(c["method_table"](dev, c["MAIN"] + ["soft@0.2", "soft@0.45"] + votes))
    w("")
    w("- 列の意味: 「仕込みのない画像で1つでも報告」は族全体の誤り率（名目 0.05）。「仕込んだ組の発見率」は検出力")
    w(f"- **新しい検定は、正解の転写でも既存の検定の約 {ov('dev', 'oracle-lcs', 'recall') / max(1e-9, ov('dev', 'oracle-ngram', 'recall')):.1f} 倍の組を見つける**"
      f"（{f(ov('dev', 'oracle-lcs', 'recall'), 2)} 対 {f(ov('dev', 'oracle-ngram', 'recall'), 2)}）。誤報告は {f(ov('dev', 'oracle-lcs', 'fwer'))} で名目の近く。"
      "既存の検定は誤報告がほぼ 0 で、保守的すぎる")
    w(f"- **認識した転写に使うと、誤報告が名目を超える**。improved-v3 で {f(ov('dev', 'lcs:v3', 'fwer'))}、improved-v2 では {f(ov('dev', 'lcs:v2', 'fwer'))}。"
      "improved-v2 は行を過剰に分割し、字の破片だけの行どうし（字の上の記号だけが並んだ行など）が似て見える。**認識の誤りが、統計的に有意な構造を作る**")
    w(f"- 票で絞ると誤報告が下がる。{k}/8 以上で {f(ov('dev', vk, 'fwer'))}、発見率 {f(ov('dev', vk, 'recall'))}")
    w("- 分類を経ない検定は、tau のどの値でも improved-v3 + 新しい検定と同じ程度で、良くならなかった\n")
    w("行の作りごと（dev、仕込みのない画像で1つでも報告する割合）:\n")
    sub = ["oracle-lcs", "ngram:v3", "lcs:v3", "lcs:v2", vk]
    w(c["group_table"](dev, "by_condition", sub, "fwer"))
    w("\n行の作りごと（dev、仕込んだ組の発見率）:\n")
    w(c["group_table"](dev, "by_condition", sub, "recall"))
    w("\n- 行を詰めた画像（tight）では認識そのものが崩れ、どの判定でも発見率が低い。improved-v3 単独の誤報告もここで多い\n")

    # ------------------------------------------------------------------ 6
    if hs or hf:
        w("## 6. 結果: holdout（閾値を固定してから1回だけ）\n")
        w("### 6.1 結果を見る前に書いた予測（results/robust/predictions.json、holdout の結果より前のコミット）\n")
        claims = {p["id"]: p for p in c["preds"]["predictions"]}
        w("| 予測 | 内容 | 反証の条件 | 観測 | 結果 |\n|---|---|---|---|---|")
        for i, ok, obs in c["checks"]:
            w(f"| {i} | {claims[i]['claim']} | {claims[i]['refuted_if']} | {obs} | {HELD[ok]} |")
        w("")
        for n, S_ in (("holdout-synth", hs), ("holdout-font", hf)):
            if not S_:
                continue
            w(f"### 6.{2 if n == 'holdout-synth' else 3} {n}（{S_['cases']} 枚、OpenCV {S_['opencv']}）\n")
            w(c["method_table"](S_, c["MAIN"] + votes))
            w("\n字ごと（仕込みのない画像で1つでも報告 / 仕込んだ組の発見率）:\n")
            groups = list(S_["by_family"])
            w("| 判定 | " + " | ".join(groups) + " |\n|---|" + "---|" * len(groups))
            for m in ["oracle-lcs", "ngram:v3", "lcs:v3", "lcs:v2", vk]:
                w(f"| {c['label'](m)} | " + " | ".join(f"{f(S_['by_family'][g][m]['fwer'], 2)} / {f(S_['by_family'][g][m]['recall'], 2)}" for g in groups) + " |")
            w("\n行の作り・劣化ごと（同じ書き方）:\n")
            groups = list(S_["by_condition"])
            w("| 判定 | " + " | ".join(groups) + " |\n|---|" + "---|" * len(groups))
            for m in ["oracle-lcs", "ngram:v3", "lcs:v3", "lcs:v2", vk]:
                w(f"| {c['label'](m)} | " + " | ".join(f"{f(S_['by_condition'][g][m]['fwer'], 2)} / {f(S_['by_condition'][g][m]['recall'], 2)}" for g in groups)
                  + " |")
            w("")

    # ------------------------------------------------------------------ 7
    w("## 7. 仮説の真偽を分けるものは何か\n")
    w("### 7.1 票の数\n")
    w("どれか1つの方式で有意になった行の組すべて（基準の行に戻せたもの）を、票の数で分けた。\n")
    for n in ("dev", "holdout-synth", "holdout-font"):
        if not S.get(n):
            continue
        d = S[n]["vote_discrimination"]
        w(f"**{n}**（候補 {d['candidates']}、真 {d['true']}、偽 {d['false']}、AUC {d['auc']}）\n")
        w(c["vote_table"](S[n]))
        w("")
    w("- 1つの方式でしか有意にならない組は、ほとんどが偽。6 つ以上の方式で有意な組は、ほとんどが真")
    w("- 票が 2〜5 の組は真偽が半々に近い。「保留」として扱い、画像で照合する対象にするのが妥当\n")
    w("### 7.2 既存の安定性の指標（参考）\n")
    w("既存の `babel analyze` が付ける段階（A/B/C）と安定性（部分標本・転写の攪乱での再現率）が、improved-v3 の転写から出た linepair 仮説の真偽を分けるか。\n")
    w("| 分割 | 段階 B（真 / 偽） | 段階 C（真 / 偽） | 部分標本の安定性の AUC | 転写の安定性の AUC |\n|---|---|---|---:|---:|")
    for n in ("dev", "holdout-synth", "holdout-font"):
        if not S.get(n):
            continue
        e = S[n]["existing_stability"]
        b, cc = e["by_tier"].get("B", {"true": 0, "false": 0}), e["by_tier"].get("C", {"true": 0, "false": 0})
        w(f"| {n} | {b['true']} / {b['false']} | {cc['true']} / {cc['false']} | {e['auc_sampling_stability']} | {e['auc_transcription_stability']} |")
    w("\n- 既存の検定は偽の仮説をほとんど出さない（そのかわり見落としが多い）ので、偽の例が少なく、AUC の値は不確か。"
      "部分標本の安定性は真偽を分けない（docs/VISION_EVAL_V3.md 第8節の結論と同じ）\n")
    w("### 7.3 同じ画像で、方式によって見つかる組が変わる頻度（IMG_4494 で起きた現象）\n")
    w("improved-v3 と baseline の転写から見つかる行の組の集合を、同じ画像で比べた。\n")
    w("| 分割 | 検定 | 両方で見つかる | v3 だけ | baseline だけ | 集合の一致（Jaccard） | 集合が違う画像 / どちらかで見つかる画像 | 両方で見つかる組が真 | 片方だけの組が真 |\n"
      "|---|---|---:|---:|---:|---:|---|---:|---:|")
    for n in ("dev", "holdout-synth", "holdout-font"):
        if not S.get(n):
            continue
        for key, lab in (("dependence_ngram", "既存"), ("dependence_lcs", "新しい検定")):
            d = S[n][key]
            w(f"| {n} | {lab} | {d['pairs_in_both']} | {d['pairs_only_in_a']} | {d['pairs_only_in_b']} | {f(d['jaccard'], 2)} | "
              f"{d['images_with_different_sets']} / {d['images_with_any']} | {f(d['p_true_if_both'], 2)} | {f(d['p_true_if_only_one'], 2)} |")
    w("\n- **方式を変えると見つかる組が変わるのは、人工画像でも普通に起きる**。IMG_4494 だけの異常ではない")
    w("- 新しい検定にしても、2つの方式の集合の一致は高くならない（baseline は行の作りを変えた画像で記号の検出が崩れる）。"
      "一致を上げるのは検定ではなく、方式の集まりで票を数えること\n")

    # ------------------------------------------------------------------ 8
    w("## 8. 「1回だけのクラス」は本当に1回だけの記号か（問題3）\n")
    if any(c["inv"].values()):
        w("正解つきの画像で、認識器が「1回だけ」としたクラスの記号が、正解でも1回だけの記号である割合（適合率）を測った（babel/robust_eval/inventory.py）。\n")
        w("| 分割 | 語彙 | 正解の種類 / 枚 | 正解の1回だけ / 枚 | v3 のクラス / 枚 | v3 の1回だけ / 枚 | v3 の1回だけの適合率 | 繰り返す記号を1回だけとした率（v3） | "
          "両方式とも1回だけ / 枚 | その適合率 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for n in ("dev", "holdout-synth", "holdout-font"):
            iv = c["inv"].get(n)
            if not iv:
                continue
            for vocab, o in iv["summary"]["by_vocab"].items():
                w(f"| {n} | {vocab} | {o['true_types_per_image']} | {o['true_hapax_per_image']} | {o['v3']['classes_per_image']} | {o['v3']['hapax_classes_per_image']} "
                  f"| {f(o['v3']['hapax_precision'], 2)} | {f(o['v3']['repeated_called_hapax_rate'], 2)} | {o['both']['hapax_in_both_per_image']} "
                  f"| {f(o['both']['precision'], 2)} |")
        w("")
        w("字ごと（holdout-font、語彙をまとめた値）:\n")
        iv = c["inv"].get("holdout-font")
        if iv:
            w("| 字 | 正解の種類 / 枚 | v3 のクラス / 枚 | v3 の1回だけの適合率 | 両方式とも1回だけの適合率 |\n|---|---:|---:|---:|---:|")
            for fam, o in iv["summary"]["by_family"].items():
                w(f"| {fam} | {o['true_types_per_image']} | {o['v3']['classes_per_image']} | {f(o['v3']['hapax_precision'], 2)} | {f(o['both']['precision'], 2)} |")
            w("")
        w("読み取り（数値は上の表）:\n")
        w("- 認識器が「1回だけ」としたクラスには、正解では繰り返す記号が混ざる（分けすぎ）。混ざる割合は、字と語彙によって大きく変わる")
        w("- 2つの方式がどちらも「1回だけ」とした記号に絞ると適合率は上がるが、1 にはならない。**2つの方式の一致は、分けすぎの否定にはならない**")
        w(f"- IMG_4494: improved-v3 で1回だけの記号のうち、baseline と1対1に対応した {h['b_hapax_aligned']} 個中 {h['hapax_in_both']} 個は baseline でも1回だけ。"
          "上の表の適合率を当てはめるには、IMG_4494 の字が表のどの行に近いかを知る必要があるが、画像がないので分からない")
        w("- したがって問題3の答えは**未確定**。分かったのは「79 クラスという数は、認識器の出力としては、記号の種類が多い資料でも、分けすぎでも出る」ことと、"
          "「どちらなのかは、クラスの数からも、2方式の一致からも決まらない」こと。決めるには、1回だけのクラスの記号の画像を人が見比べる必要がある\n")
    else:
        w("（inventory の結果がまだない）\n")
    w("### 分類方式どうしの一致は、正解との一致を予測するか（正解なしの指標の検討）\n")
    w("同じ切り出しで分類だけを変えた 5 方式の間の一致（調整ランド指数の平均）が、improved-v3 の分類の正しさ（正解との調整ランド指数）と相関するか。\n")
    w("| 分割 | 画像 | 順位相関 | 方式どうしの一致の平均 | 正解との一致の平均 |\n|---|---:|---:|---:|---:|")
    for n in ("dev", "holdout-synth", "holdout-font"):
        if S.get(n):
            a = S[n]["agreement_vs_truth"]
            w(f"| {n} | {a['images']} | {a['spearman']} | {a['mean_agreement']} | {a['mean_true_ari']} |")
    w("\n- 相関は弱い。**分類方式どうしが一致していても、分類が正しいとは言えない**（同じ切り出しの誤りを共有する）。正解なしで分類の精度を見積もる指標としては使えない\n")

    # ------------------------------------------------------------------ 9
    w("## 9. うまくいかなかったこと・評価の欠陥の検討\n")
    w("| 何をしたか | 何が起きたか | どうしたか |\n|---|---|---|")
    w("| 多重検定の補正を minP（最小 p 値の置換分布）で実装した | 統計量が整数で置換回数が 999 だと、補正後の p 値の下限が 0.02 前後になり、"
      "行の多い転写（improved-v2 の 25 行）では有意水準に届かないことがあった | 標準化した統計量の maxT に替えた（dev を走らせる前） |")
    w("| 予測の行を、IoU 0.5 以上で対応する正解の記号から正解の行に戻した | 字の破片だけの行は正解の行に戻せず、全部「偽」と数えられた"
      "（仕込んだ行の破片どうしの類似も偽になり、認識の誤りによる偽を多く見積もる） | 重なる面積が最大の正解の記号の行に戻すように替えた（dev を走らせる前） |")
    w(f"| 新しい検定を1つの転写に使う | 仕込みのない画像での誤報告が名目 0.05 を超えた（improved-v3 で dev {f(ov('dev', 'lcs:v3', 'fwer'))}）。"
      "置換検定は「記号の位置が入れ替え可能」を前提にするが、認識の誤りは行ごとに偏る（行の誤結合、字の部品の行）ので前提が崩れる | "
      "単独では使わず、方式の集まりの票と組み合わせる。単独の結果は「候補」と呼ぶ |")
    w("| 分類を経ない検定（字形の類似度を重みにする） | dev で、クラスの一致による検定と発見率・誤報告が変わらなかった。"
      "見落としの主因は分類の閾値ではなく、切り出しと行の誤り | 不採用（選択の規則どおり）。コードは残し、参考として出すだけ |")
    w("| 分類方式どうしの一致から、分類の精度を正解なしで見積もる | 正解との一致との相関が弱かった（第8節） | 指標として使わない |")
    w("| 2つの方式で見つかる組の集合の一致を、新しい検定で上げる | 上がらなかった（第7.3節） | 「検定を替えれば方式に依存しなくなる」とは言わない |")
    w("")
    w("評価そのものの限界:\n")
    w("- **正解の定義**: 「真」は仕込んだ組。仕込んでいない組が偶然似ることがあり（正解の転写でも 5% 前後の文で1組が有意）、それは認識の誤りではないが偽に数えている")
    w("- **方式の集まりの偏り**: 8 方式のうち 6 方式は improved-v3 の切り出しを使う（1つは画像を拡大してからなので、切り出しの結果は少し違う）。"
      f"閾値 {k} は、切り出しを共有する方式だけでも届く。improved-v3 の切り出しの誤りが作る規則は、票では除けない。"
      "誤報告が名目の近くまで下がったのは dev・holdout の画像での観測で、切り出しが系統的に誤る資料では成り立たない可能性がある")
    w("- **仕込んだ構造は1種類**（同じ枠の行の組）。位置の規則・連接・共起など、ほかの族の仮説には票の方法を使っていない")
    w("- **人工画像と実在のフォントの画像は、同じ字が画素までほぼ同じ**。手書き・写真・劣化した資料では、分類の誤りがずっと多いはずで、ここでの数値は当てはまらない")
    w("- **統計量を実データを見てから選んだ**（第2.5節）\n")

    # ------------------------------------------------------------------ 10
    w("## 10. 5つの問題への答え\n")
    w("**問題1（同じように見える記号が、異なるものとして認識される）** — 一部だけ分かった。"
      f"baseline と improved-v3 は、対応が付いた位置の組のうち {pa['same_only_in_a'] + pa['same_only_in_b']} 組（長さが同じ行だけで）について「同じ／違う」が食い違い、"
      f"1記号と2記号が対応する箇所が {n12} ある。食い違いの場所は compare-baseline-v3.md に列挙した。"
      "**どちらが字形の違いで、どちらが認識器の誤りかは、画像がないので確定していない。** 人が見るべき位置は絞れた\n")
    w("**問題2（認識方式によって構造仮説が変わる）** — 原因を特定した（第2.2節）。既存の検定が、記号1つの取り違えで切れるため。"
      "切れない検定では、行 0・1 と行 3・4 の2組が、どの転写でも有意。ただし「画像に本当にある」ことの確認には画像が要る\n")
    w("**問題3（視覚クラスの分類が不安定）** — 未確定（第8節）。認識器の「1回だけのクラス」には分けすぎが混ざること、"
      "2方式の一致でも分けすぎを否定できないことを、正解つきの画像で示した\n")
    gen = []
    if hs:
        gen.append(f"dev に出ない字の作りと劣化（holdout-synth）で発見率 {f(ov('holdout-synth', vk, 'recall'), 2)}・誤報告 {f(ov('holdout-synth', vk, 'fwer'), 2)}")
    if hf:
        gen.append(f"実在のフォント（holdout-font）で発見率 {f(ov('holdout-font', vk, 'recall'), 2)}・誤報告 {f(ov('holdout-font', vk, 'fwer'), 2)}")
    w("**問題4（人工データと実画像の性能差）** — 一部だけ答えた。方式の集まり（" + f"{k}/8 以上）は、" + "、".join(gen) +
      f"（dev は {f(ov('dev', vk, 'recall'), 2)}・{f(ov('dev', vk, 'fwer'), 2)}）。"
      "字の作りが変わっても方法は働くが、**実画像（写真・手書き）への一般化は測っていない**。実画像の正解転写が1枚もないため\n")
    w("**問題5（構造を発見しても意味には到達していない）** — 到達していない。この研究が足したのは「見つかった構造が認識の誤りによるものかどうか」の判定で、"
      "構造の種類（文法・定型句・見た目の繰り返し）は区別していない。行の組の類似は、同じ枠の文でも、箇条書きの記号でも、画面の飾りの繰り返しでも同じように出る。"
      "13 行・119 記号では、どれであるかを統計で決めるだけの量がない。区別するには、(a) 画像そのもの（文字か飾りかは見れば分かることが多い）、"
      "(b) 同じ資料の続き（枠が別の行にも現れるか）、(c) 文脈や対訳、のどれかが要る\n")

    # ------------------------------------------------------------------ 11
    w("## 11. 最終の問いへの答えと、反証の条件\n")
    w("**答え: 条件つきで、できる。**\n")
    w(f"- 1つの転写からは、できない。認識の誤りは、置換検定で有意になる構造を作る（仕込みのない画像の {_pct(ov('dev', 'lcs:v2', 'fwer'))}〔improved-v2〕、"
      f"{_pct(ov('dev', 'lcs:v3', 'fwer'))}〔improved-v3〕で、dev）。既存の安定性の指標も、それを見分けない")
    w(f"- 同じ画像を複数の方式で転写し、方式をまたいで再現する規則だけを採れば、できる（行の組の類似について、{k}/8 以上で、報告のうち真が {trio(vk, 'precision', 2)}）。"
      "1つの方式でしか出ない規則は、認識の誤りによるものとして扱うのが正しい（真である割合は 1 割前後）")
    w("- ただし「方式をまたいで再現する」は「画像に本当にある」の近似で、どの方式にも共通する誤りは見分けられない。"
      "また「画像に本当にある」は「言語の構造である」を意味しない\n")
    w("**この答えが誤りだと分かる条件**\n")
    w("1. 第6.1節の予測が、新しい holdout（別のシード・別の字）で外れる")
    w(f"2. 実画像の正解転写ができたとき、票 {k}/8 以上の行の組が、正解の転写での検定で有意でない")
    w("3. IMG_4494 の画像で `babel robust-structure` を走らせたとき、行 0・1 と行 3・4 の票が少ない（5 以下）。"
      "その場合、第2節の「どの転写でも有意」は、baseline と improved-v3 に共通する誤りによるものだったことになる")
    w("4. 人が画像で行 3・4 の対応する記号を見比べて、同じ字ではないと判断する\n")
    w("**足りないもの（これがあれば解決できること）**\n")
    w("| 足りないもの | あれば分かること |\n|---|---|")
    w("| IMG_4494 の画像 | 2組の規則が方式をまたいで再現するか（上の条件 3）。食い違う位置の字形を直接見比べられる |")
    w("| IMG_4494 の正解転写（人が確かめたもの。docs/GROUND_TRUTH.md の手順） | 問題1・3 の確定（どちらの方式が正しいか、1回だけのクラスが本物か）。問題4 の実画像での精度 |")
    w("| 別の実画像と正解転写 | 実画像への一般化。今は 0 枚 |")
    w("| 同じ資料の続き・文脈 | 問題5（構造の種類と意味） |\n")

    # ------------------------------------------------------------------ 12
    w("## 12. 画像が届いたら・再現方法\n")
    w("```bash\nexport PYTHONHASHSEED=0\n"
      "# 画像1枚を 8 方式で転写し、方式をまたいで再現する行の組を出す（正解は要らない）\n"
      "python -m babel robust-structure IMG_4494.png --region left:X1,Y1,X2,Y2 --region right:X1,Y1,X2,Y2 --out study/robust.json\n"
      "# 2本の転写の比較（画像は要らない）\n"
      "python -m babel transcription-compare results/vision/real-img4494/corpus-continuous.json results/robust/real-img4494/inputs/v3-continuous.json \\\n"
      "    --extra results/robust/real-img4494/inputs/v2-continuous.json --name-a baseline --name-b improved-v3 --out results/robust/real-img4494/compare-baseline-v3.json\n"
      "# 評価（dev → 閾値の選択 → holdout。holdout-font は Pillow とフォントが要る）\n"
      "python -m babel.robust_eval.bench --split dev --out results/robust/bench-dev.json\n"
      "python -m babel.robust_eval.select\n"
      "python -m babel.robust_eval.bench --split holdout-synth --out results/robust/bench-holdout-synth.json\n"
      "python -m babel.robust_eval.bench --split holdout-font --out results/robust/bench-holdout-font.json\n"
      "python -m babel.robust_eval.inventory --split dev --out results/robust/inventory-dev.json      # holdout-synth・holdout-font も同じ\n"
      "python -m babel.robust_eval.report\n```\n")
    w("- 画像の比較では、行の番号は基準の方式（improved-v3）の行。票が 5 以下の組は「保留」、1 の組は認識の誤りとして扱う")
    w("- 結果の JSON には、画像ごとの各方式の発見・票・認識の指標が入っている（results/robust/bench-*.json）。集計は summary-*.json")
    w("- holdout-font は、フォント（DejaVu Sans・Loma・IPA ゴシック・WenQuanYi Zen Hei）と Pillow の版によって画素が変わる。記録した結果は、この PR の作業環境のもの")
    return "\n".join(L) + "\n"
