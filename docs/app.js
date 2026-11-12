// PROJECT BABEL 研究ダッシュボード。docs/data/*.json を読んで表示するだけの静的スクリプト。
const D = {};
const state = { tab: "overview", fam: "KES", lev: "A" };
const TABS = [["overview", "概要"], ["samples", "記録サンプル"], ["grammar", "語彙と文法"], ["gloss", "解読前後"],
  ["hyp", "仮説と信頼度"], ["results", "成績と比較"], ["external", "初見テスト"], ["history", "実験履歴"], ["issues", "問題と限界"]];
const NEED_PICKER = new Set(["samples", "grammar", "gloss", "hyp"]);
const LEVELS = { A: "LEVEL A（部分対訳あり）", B: "LEVEL B（断片的な手がかり）", C: "LEVEL C（手がかりなし）" };
const STRUCT = { verb_position: "動詞の位置", so_order: "主語・目的語の順", affix_position: "接辞の側",
  numeral_base: "数詞の基数", adj_noun: "形容詞・名詞の順", num_noun: "数詞・名詞の順" };
const METRIC = { seg_f1: "形態素境界 F1", affix_f1: "接辞目録 F1", pos_accuracy: "品詞 正解率", variant_f1: "表記揺れ統合 F1",
  verb_position_correct: "動詞位置 正解率", affix_position_correct: "接辞側 正解率", numeral_base_correct: "基数 正解率",
  numeral_accuracy: "数詞の値 正解率", numeral_wrong_rate: "数詞の値 誤答率", lexicon_precision: "語彙主張の適合率",
  lexicon_claimed: "意味を主張した語数", gloss_correct: "語釈 正解率", gloss_wrong: "語釈 誤答率", gloss_abstain: "語釈 判定不能率",
  feature_jaccard: "文法素性の一致度", relatedness_p1: "意味的近傍 P@1", overclaims: "過剰主張の件数",
  abstain_without_evidence: "証拠なし語の保留率" };

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v, d = 2) => (v === null || v === undefined ? "–" : Number(v).toFixed(d));
const pct = (v) => (v === null || v === undefined ? "–" : (v * 100).toFixed(1) + "%");
const bar3 = (c, x, a) => `<div class="bar" title="正解 ${pct(c)} / 誤答 ${pct(x)} / 判定不能 ${pct(a)}"><i class="c" style="width:${c * 100}%"></i><i class="x" style="width:${x * 100}%"></i><i class="a" style="width:${a * 100}%"></i></div>`;
const bar1 = (v) => `<div class="bar"><i class="p" style="width:${Math.max(0, Math.min(1, v || 0)) * 100}%"></i></div>`;
const table = (head, rows) => `<div class="scroll"><table><thead><tr>${head.map((h) => `<th${h.n ? ' class="n"' : ""}>${esc(h.t ?? h)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c, i) => `<td${head[i].n ? ' class="n"' : ""}>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
const cond = (label) => (D.results.conditions || []).find((c) => c.label === label);
const mm = (label, key) => cond(label)?.metrics?.[key]?.mean;
const ms = (label, key) => { const m = cond(label)?.metrics?.[key]; return m && m.mean !== null ? `${num(m.mean, 3)} <span class="muted">±${num(m.std, 3)}</span>` : "–"; };
const show = () => D.showcase.find((s) => s.family === state.fam && s.level === state.lev);
const LEGEND = '<p class="legend"><i style="background:var(--ok)"></i>正解<i style="background:var(--bad)"></i>誤答<i style="background:var(--hold);opacity:.45"></i>判定不能</p>';

const views = {
  overview() {
    const c = D.scorecard, m = D.results.meta, n = D.results.null || {};
    const rows = c.items.map((i) => [esc(i.name), i.points === null ? '<span class="hold">未評価</span>' : `<b>${num(i.points, 1)}</b> / ${i.max}`,
      esc(i.status), `<ul>${[].concat(i.detail).map((d) => `<li>${esc(d)}</li>`).join("")}</ul>`]);
    return `<h2>結論</h2>
<div class="card"><div class="grid">
<div class="stat"><b>${num(c.internal_points, 1)} <small>/ ${c.evaluated_max}</small></b><span>内部テストの得点（評価済みの配点に対して）</span></div>
<div class="stat"><b class="hold">未評価</b><span>独立した初見テスト（${c.unevaluated_max} 点分が未評価）</span></div>
<div class="stat"><b>${pct(mm("levelA", "gloss_correct"))}</b><span>LEVEL A：対訳のない held-out 文の語釈正解率</span></div>
<div class="stat"><b>${pct(mm("levelA", "gloss_wrong"))}</b><span>同 誤答率（残りは「判定不能」と報告）</span></div>
<div class="stat"><b>${num(mm("levelC", "overclaims"), 0)}</b><span>LEVEL C：証拠のない意味の主張（件／実行）</span></div>
<div class="stat"><b>${pct(mm("levelC", "numeral_accuracy"))}</b><span>LEVEL C：算術だけで復元した数詞の値</span></div>
</div></div>
<p class="note">${esc(c.note)} シードの役割: <b>${m.seed_role === "evaluation" ? "評価用（調整に未使用）" : "開発用"}</b>。</p>
<h2>採点表（100点満点、算出式は事前登録）</h2>${table(["項目", "得点", "状態", "根拠"], rows)}
<h2>解読できないものを見抜けたか</h2>
${table(["対照コーパス", { t: "実行数", n: 1 }, { t: "統語ありと誤判定", n: 1 }, { t: "意味の主張", n: 1 }],
    ["random", "shuffled"].filter((k) => n[k]).map((k) => [k === "random" ? "文法のない無作為な語列" : "実コーパスの語順を破壊", n[k].runs, pct(n[k].syntax_detected_rate), n[k].meaning_claims_total]))}`;
  },
  samples() {
    const s = show();
    return `<h2>未知文明の記録（${esc(s.language_id)}）</h2><p class="muted">${esc(s.typology)}。解読器にはこの文字列${s.level === "C" ? "だけ" : "と下の周辺情報"}が渡される。</p>
${s.samples.map((d) => `<div class="card"><p class="muted">${esc(d.id)} ${Object.entries(d.meta).map(([k, v]) => `<span class="tag">${esc(k)}: ${esc(v)}</span>`).join("")}</p>${d.sentences.map((x) => `<p class="raw">${esc(x)}</p>`).join("")}</div>`).join("")}
<h3>頻度分析</h3><div class="card"><p>延べ ${s.frequency.tokens} 語・異なり ${s.frequency.types} 語・語のエントロピー ${s.frequency.word_entropy_bits} bit</p>
<p>${s.frequency.top_words.map(([w, c]) => `<span class="tag w">${esc(w)} ${c}</span>`).join("")}</p></div>
<h3>解読可能性の判定</h3><div class="card"><p>${esc(s.verdict.text)}</p><p class="muted">語順の規則性 z=${s.verdict.syntax.z}（無作為化との比較）／接辞付きの語の割合 ${pct(s.verdict.morphology.affixed_token_share)}</p></div>`;
  },
  grammar() {
    const s = show();
    const st = Object.entries(s.structure).map(([k, v]) => [STRUCT[k] || k, v.value === null ? `<span class="hold">判定不能</span>${v.reason ? `<br><span class="muted">${esc(v.reason)}</span>` : ""}` : `<b class="${String(v.value) === String(v.gold) ? "ok" : "bad"}">${esc(v.value)}</b>`, num(v.confidence), esc(v.gold)]);
    const nums = s.numerals.values.map((n) => `<span class="tag w ${n.gold === n.value ? "ok" : "bad"}">${esc(n.form)} = ${n.value}${n.gold === n.value ? "" : `（正解 ${n.gold ?? "数詞ではない"}）`}</span>`).join("");
    return `<h2>推定された文法</h2>${table(["項目", "推定", { t: "信頼度", n: 1 }, "正解"], st)}
<h3>数詞（${s.numerals.status === "solved" ? "解けた" : "判定不能"}）</h3><div class="card">
<p class="muted">合計行の算術から立てた方程式 ${s.numerals.equations} 本。${s.numerals.relative ? "意味の手がかりがないため、最小の単位を 1 と仮定した相対値。" : "既知の数詞・対訳で単位を固定。"} ${esc(s.numerals.reason || "")}</p><p>${nums || '<span class="hold">値を主張しない</span>'}</p>
${s.numerals.affix_hypotheses.map((h) => `<p class="muted">形成要素 <span class="w">${esc(h.affix)}</span>: ${h.chosen === null ? "決められない" : h.chosen === "add" ? "加法（足す）" : `乗法 ×${h.chosen}`}／競合 ${h.rivals.map((r) => `${r[0]}(${r[1]})`).join(", ")}</p>`).join("")}</div>
<h3>接辞</h3>${table(["接辞", { t: "語幹数", n: 1 }, { t: "出現", n: 1 }, "推定した機能"], s.affixes.map((a) => [`<span class="w">${a.affix.startsWith("P:") ? esc(a.affix.slice(2)) + "-" : "-" + esc(a.affix.slice(2))}</span>`, a.types, a.tokens, Object.keys(a.function).length ? Object.entries(a.function).map(([f, p]) => `<span class="tag">${esc(f)} ${num(p)}</span>`).join("") : '<span class="hold">機能は判定不能</span>']))}
<h3>語彙（頻度上位）</h3><p class="muted">主張 ${s.summary.claimed} 語／判定不能 ${s.summary.undetermined} 語（信頼度 ${s.summary.claim_threshold} 未満は主張しない）</p>
${table(["語幹", { t: "出現", n: 1 }, "品詞", "推定した意味", { t: "信頼度", n: 1 }, "正解"], s.lexicon.map((e) => [`<span class="w">${esc(e.stem)}</span>`, e.count, esc(e.pos), e.meaning === null ? `<span class="hold">判定不能</span><br><span class="muted">${esc(e.reason)}</span>` : `<b class="${e.correct ? "ok" : "bad"}">${esc(e.meaning)}</b>`, e.meaning === null ? "–" : num(e.confidence), esc(e.gold)]))}`;
  },
  gloss() {
    const s = show(), g = s.score.gloss_heldout;
    return `<h2>解読前後の比較（held-out 文書）</h2><p class="muted">上段: 原文を形態素に分割／中段: 解読器の語釈／下段: 正解。学習に使っていない文書。</p>
<div class="card">${bar3(g.correct, g.wrong, g.abstain)}${LEGEND}<p class="muted">この言語・条件での内容語: 正解 ${pct(g.correct)}／誤答 ${pct(g.wrong)}／判定不能 ${pct(g.abstain)}</p></div>
${s.examples.map((ex) => `<div class="sent">${ex.tokens.map((t) => t.lacuna ? '<div class="tok"><span class="s hold">[…]</span><span class="g">欠損</span></div>' : `<div class="tok"><span class="s">${t.parts.map(esc).join("·")}</span><span class="${t.ok === null ? "hold" : t.ok ? "ok" : "bad"}">${t.concept === null ? "？" + (t.pos !== "X" ? `(${t.pos})` : "") : esc(t.concept)}${t.feats.length ? " " + t.feats.map(esc).join(",") : ""}</span><span class="g">${esc(t.gold)}${t.gold_feats.length ? " " + t.gold_feats.map(esc).join(",") : ""}</span></div>`).join("")}</div>`).join("")}
${s.errors.length ? `<h3>誤答例と原因</h3>${table(["語幹", "推定", "正解", { t: "信頼度", n: 1 }, "失敗原因"], s.errors.map((e) => [`<span class="w">${esc(e.stem)}</span>`, esc(e.predicted), esc(e.gold), num(e.confidence), esc(e.cause)]))}` : ""}`;
  },
  hyp() {
    const s = show();
    const rows = s.hypotheses.map((h) => [esc(h.topic), h.value === null ? '<span class="hold">判定不能</span>' : `<b>${esc(h.value)}</b>`, `${bar1(h.confidence)}<span class="muted">${num(h.confidence)}</span>`, esc(h.status),
      `${esc(h.evidence)}${Object.keys(h.alternatives || {}).length ? `<br><span class="muted">競合: ${Object.entries(h.alternatives).map(([k, v]) => `${esc(k)}=${esc(typeof v === "number" ? num(v) : v)}`).join(", ")}</span>` : ""}`]);
    const hist = s.history.map((h) => [`${Math.round(h.fraction * 100)}%（${h.documents} 文書）`, `${esc(h.verb_position[0] ?? "–")} <span class="muted">${num(h.verb_position[1])}</span>`,
      `${esc(h.affix_position[0] ?? "–")} <span class="muted">${num(h.affix_position[1])}</span>`, `${esc(h.numeral_base[0] ?? "判定不能")} <span class="muted">${num(h.numeral_base[1])}</span>`, h.numerals_solved, h.meanings_claimed]);
    return `<h2>仮説と信頼度</h2>${table(["仮説", "採用した値", "信頼度", "状態", "証拠"], rows)}
<h3>新しい証拠による更新</h3><p class="muted">発掘が進む（読める文書が増える）につれて仮説がどう変わるか。各行はその時点の文書だけで推定し直した結果。</p>
${table(["証拠の量", "動詞位置", "接辞の側", "基数", { t: "値の決まった数詞", n: 1 }, { t: "意味を主張した語", n: 1 }], hist)}`;
  },
  results() {
    const main = ["A", "B", "C"].map((l) => { const k = "level" + l; return [LEVELS[l], bar3(mm(k, "gloss_correct") || 0, mm(k, "gloss_wrong") || 0, mm(k, "gloss_abstain") || 0), pct(mm(k, "gloss_correct")), pct(mm(k, "gloss_wrong")), pct(mm(k, "gloss_abstain"))]; });
    const keys = ["seg_f1", "affix_f1", "pos_accuracy", "variant_f1", "verb_position_correct", "affix_position_correct", "numeral_base_correct", "numeral_accuracy", "numeral_wrong_rate", "lexicon_precision", "lexicon_claimed", "feature_jaccard", "relatedness_p1", "abstain_without_evidence", "overclaims"];
    const struct = keys.map((k) => [METRIC[k], ms("levelA", k), ms("levelB", k), ms("levelC", k)]);
    const fam = ["KES", "UMB", "SOT"].map((f) => { const b = (l, k) => num(cond(l)?.by_family?.[f]?.[k]?.mean, 3); return [f, b("levelA", "seg_f1"), b("levelA", "pos_accuracy"), b("levelC", "numeral_accuracy"), b("levelA", "lexicon_precision"), b("levelA", "gloss_correct"), b("levelA", "gloss_wrong")]; });
    const cmp = (labels, ks) => table(["条件"].concat(ks.map((k) => ({ t: METRIC[k], n: 1 }))), labels.filter(([l]) => cond(l)).map(([l, name]) => [esc(name)].concat(ks.map((k) => ms(l, k)))));
    const ck = ["seg_f1", "lexicon_precision", "lexicon_claimed", "gloss_correct", "gloss_wrong"];
    const cal = cond("levelA")?.lexicon_calibration || [];
    const errs = (cond("levelA")?.error_examples || []).slice(0, 10);
    const b9 = cond("no_word_boundaries");
    return `<h2>成績（${D.results.meta.seeds.length} シード × 3 系統の平均 ± 標準偏差）</h2>
<h3>held-out 文の語釈：正解・誤答・判定不能</h3>${LEGEND}${table(["条件", "内訳", { t: "正解", n: 1 }, { t: "誤答", n: 1 }, { t: "判定不能", n: 1 }], main)}
<h3>構造と意味の指標</h3>${table(["指標", { t: "LEVEL A", n: 1 }, { t: "LEVEL B", n: 1 }, { t: "LEVEL C", n: 1 }], struct)}
<h3>言語系統別</h3>${table(["系統", { t: "境界F1", n: 1 }, { t: "品詞", n: 1 }, { t: "数詞(C)", n: 1 }, { t: "語彙適合率", n: 1 }, { t: "語釈正解", n: 1 }, { t: "語釈誤答", n: 1 }], fam)}
<h3>ベースラインとの比較（LEVEL A）</h3><p class="muted">頻度法＝頻出語尾の切り落とし＋共起回数の最大値。n-gram 法＝後続文字の多様性で分割＋Dice 係数。</p>
${cmp([["levelA", "解読エンジン"], ["levelA:baseline_freq", "頻度法"], ["levelA:baseline_ngram", "n-gram 法"]], ck)}
<h3>アブレーション（機能を1つずつ外す）</h3>${cmp([["levelA", "全機能"], ["levelA:no_variants", "表記揺れ正規化なし"], ["levelA:no_morph", "形態素解析なし"], ["levelA:no_em", "EMなし（共起回数）"], ["levelA:no_numeral_solver", "数詞ソルバーなし"], ["levelA:no_abstain", "保留なし（必ず答える）"], ["levelA:no_pos_constraint", "品詞制約なし"]], ck.concat(["numeral_accuracy"]))}
<h3>データ量を減らす</h3>${cmp([["levelA:docs=33", "LEVEL A 文書25%"], ["levelA:docs=65", "LEVEL A 文書50%"], ["levelA", "LEVEL A 文書100%"], ["levelC:docs=33", "LEVEL C 文書25%"], ["levelC:docs=65", "LEVEL C 文書50%"], ["levelC", "LEVEL C 文書100%"]], ["seg_f1", "pos_accuracy", "numeral_accuracy", "gloss_correct", "gloss_wrong"])}
<h3>対訳を減らす</h3>${cmp([["levelA:parallel=0.01", "対訳 1%"], ["levelA:parallel=0.025", "対訳 2.5%"], ["levelA", "対訳 5%"], ["levelA:parallel=0.1", "対訳 10%"]], ["lexicon_precision", "lexicon_claimed", "gloss_correct", "gloss_wrong"])}
<h3>表記揺れ・欠損への耐性</h3>${cmp([["levelA:variation=0.0", "表記揺れなし"], ["levelA", "標準"], ["levelA:variation=0.9", "表記揺れ 2倍"], ["levelA:lacuna=0.0", "欠損なし"], ["levelA:lacuna=0.08", "欠損 4倍"]], ["seg_f1", "variant_f1", "numeral_accuracy", "gloss_correct", "gloss_wrong"])}
<h3>未観測の構造（後期時代の文書だけを held-out）</h3>${cmp([["levelA", "無作為 held-out"], ["levelA:heldout=late_era", "後期時代 held-out"]], ["gloss_correct", "gloss_wrong", "gloss_abstain", "feature_jaccard"])}
${b9 ? `<h3>単語境界のない連続書記</h3>${table(["手法", { t: "単語境界 F1", n: 1 }], [["単語ユニグラム反復推定", ms("no_word_boundaries", "word_boundary_f1_engine")], ["n-gram 遷移確率（ベースライン）", ms("no_word_boundaries", "word_boundary_f1_ngram_baseline")]])}` : ""}
<h3>信頼度の較正（LEVEL A の語彙主張）</h3><p class="muted">期待較正誤差 ECE = ${cond("levelA")?.lexicon_ece ?? "–"}</p>${table(["信頼度の帯", { t: "件数", n: 1 }, { t: "平均信頼度", n: 1 }, { t: "実際の正解率", n: 1 }], cal.map((c) => [c.bin, c.n, num(c.confidence), num(c.accuracy)]))}
<h3>誤答例と失敗原因（LEVEL A）</h3>${table(["系統", "語幹", "推定", "正解", { t: "信頼度", n: 1 }, "失敗原因"], errs.map((e) => [e.family + "-" + e.seed, `<span class="w">${esc(e.stem)}</span>`, esc(e.predicted), esc(e.gold), num(e.confidence), esc(e.cause)]))}`;
  },
  external() {
    const e = D.external;
    const led = e.ledger.map((l) => [esc(l.name), l.attempt, `<code>${esc(l.sha256.slice(0, 12))}</code>`, `<code>${esc((l.commit || "").slice(0, 8))}</code>`, esc(l.created_utc), l.answer_already_public ? '<span class="bad">正解公開後</span>' : '<span class="ok">正解公開前</span>']);
    const sc = e.scores.map((s) => [esc(s.attempt.name) + " #" + s.attempt.attempt, s.independent ? '<span class="ok">独立</span>' : '<span class="hold">独立でない</span>', esc(s.provenance),
      Object.entries(s.structure || {}).map(([k, v]) => `<span class="tag ${v === "correct" ? "ok" : v === "wrong" ? "bad" : "hold"}">${esc(STRUCT[k] || k)}</span>`).join(""), s.lexicon ? `${s.lexicon.correct}/${s.lexicon.wrong}/${s.lexicon.abstain}` : "–", s.numerals ? `${s.numerals.correct}/${s.numerals.wrong}/${s.numerals.abstain}` : "–"]);
    return `<h2>初見言語テスト</h2><p class="note">最も重要なテスト。解読器を固定した後に、第三者が作った未知言語を投入する。予測は台帳に SHA-256 で記録され、後から書き換えると CI が検出する。</p>
<div class="card"><p>解読器の指紋 <code>${esc(e.engine_fingerprint)}</code>／台帳の整合性: ${e.problems.length ? `<span class="bad">問題 ${e.problems.length} 件</span>` : '<span class="ok">問題なし</span>'}</p>
<p class="muted">投入方法: <code>external/inbox/名前/corpus.txt</code>（1行1文、空行で文書区切り）を GitHub に追加 → 予測が自動保存 → その後 <code>external/answers/名前/answer.json</code> を追加 → 採点。</p></div>
<h3>予測台帳</h3>${led.length ? table(["名前", { t: "試行", n: 1 }, "予測のハッシュ", "コミット", "日時 (UTC)", "時点"], led) : '<p class="hold">まだ投入されていない。</p>'}
<h3>採点結果</h3>${sc.length ? table(["コーパス", "独立性", "出所", "構造", "語彙 正/誤/保留", "数詞 正/誤/保留"], sc) : '<p class="hold">未評価（正解が追加されたコーパスがない）。</p>'}`;
  },
  history() {
    const m = D.results.meta, runs = D.results.runs || [];
    const byExp = {};
    runs.forEach((r) => { const k = r.experiment; byExp[k] = byExp[k] || { n: 0, fail: 0, sec: 0 }; byExp[k].n++; byExp[k].fail += r.ok ? 0 : 1; byExp[k].sec += r.seconds || 0; });
    const failed = runs.filter((r) => !r.ok);
    return `<h2>今回の実行</h2>${table(["項目", "値"], [["実行日時 (UTC)", esc(m.started_utc)], ["Git コミット", `<code>${esc(m.commit)}</code>`], ["実行環境", esc(m.environment) + (m.run_id ? ` (run ${esc(m.run_id)})` : "")], ["モード", esc(m.mode)], ["シード", esc(m.seeds.join(", ")) + `（${m.seed_role === "evaluation" ? "評価用" : "開発用"}）`], ["Python", esc(m.python)], ["1言語あたりの文数", m.sentences_per_language], ["総実行数 / 失敗", `${m.total_runs} / <span class="${m.failed_runs ? "bad" : "ok"}">${m.failed_runs}</span>`], ["所要時間", m.seconds + " 秒"], ["再現性（同条件の再実行でハッシュ一致）", D.results.reproducibility.identical ? '<span class="ok">一致</span>' : '<span class="bad">不一致</span>'], ["出力ハッシュ", `<code>${esc(D.results.reproducibility.digests[0])}</code>`]])}
<h3>実験ごとの実行数</h3>${table(["実験", { t: "実行", n: 1 }, { t: "失敗", n: 1 }, { t: "合計秒", n: 1 }], Object.entries(byExp).map(([k, v]) => [esc(k), v.n, v.fail, num(v.sec, 0)]))}
${failed.length ? `<h3>失敗した実行</h3>${table(["実験", "条件", "系統", "シード", "エラー"], failed.map((r) => [esc(r.experiment), esc(r.label), esc(r.family), r.seed, esc(r.error)]))}` : ""}
<h3>過去の実行</h3>${D.history.length ? table(["日時 (UTC)", "コミット", "環境", "モード", { t: "実行", n: 1 }, { t: "失敗", n: 1 }, { t: "秒", n: 1 }], D.history.slice().reverse().map((h) => [esc(h.started_utc), `<code>${esc((h.commit || "").slice(0, 8))}</code>`, esc(h.environment), esc(h.mode) + "/" + (h.seed_role === "evaluation" ? "評価" : "開発"), h.total_runs, h.failed_runs, h.seconds])) : '<p class="hold">記録なし</p>'}`;
  },
  issues() {
    const i = D.issues;
    return `<h2>発見された問題</h2>${i.issues.map((x) => `<div class="card"><p><b>${esc(x.id)}. ${esc(x.title)}</b> <span class="tag ${x.status === "修正済み" ? "ok" : "bad"}">${esc(x.status)}</span></p><p class="muted">発見: ${esc(x.found)}</p><p>原因: ${esc(x.cause)}</p><p>対応: ${esc(x.fix)}</p><p class="muted">検証: ${esc(x.verified)}</p></div>`).join("")}
<h2>研究上の限界</h2><ul>${i.limitations.map((l) => `<li>${esc(l)}</li>`).join("")}</ul>`;
  },
};

function render() {
  document.getElementById("picker").hidden = !NEED_PICKER.has(state.tab);
  document.querySelectorAll("nav button").forEach((b) => b.setAttribute("aria-selected", b.dataset.t === state.tab));
  try { document.getElementById("main").innerHTML = views[state.tab](); } catch (e) { document.getElementById("main").innerHTML = `<p class="bad">表示エラー: ${esc(e.message)}</p>`; }
  window.scrollTo(0, 0);
}

async function boot() {
  try {
    const names = ["results", "scorecard", "showcase", "issues", "external", "history"];
    const got = await Promise.all(names.map((n) => fetch(`data/${n}.json`).then((r) => { if (!r.ok) throw new Error(`${n}.json: ${r.status}`); return r.json(); })));
    names.forEach((n, i) => { D[n] = got[i]; });
  } catch (e) {
    document.getElementById("meta").innerHTML = `<span class="bad">データを読み込めない（${esc(e.message)}）。<code>python -m babel report</code> で docs/data を生成すること。</span>`;
    return;
  }
  const m = D.results.meta;
  document.getElementById("meta").textContent = `${m.started_utc} 実行 ／ コミット ${m.commit.slice(0, 8)} ／ ${m.seeds.length} シード ／ ${m.total_runs} 実行（失敗 ${m.failed_runs}）／ ${m.environment}`;
  document.getElementById("tabs").innerHTML = TABS.map(([t, n]) => `<button data-t="${t}">${n}</button>`).join("");
  document.getElementById("tabs").addEventListener("click", (e) => { if (e.target.dataset.t) { state.tab = e.target.dataset.t; render(); } });
  const fam = document.getElementById("fam"), lev = document.getElementById("lev");
  fam.innerHTML = [...new Set(D.showcase.map((s) => s.family))].map((f) => `<option value="${f}">${f}（${esc(D.showcase.find((s) => s.family === f).typology)}）</option>`).join("");
  lev.innerHTML = Object.entries(LEVELS).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
  fam.onchange = () => { state.fam = fam.value; render(); };
  lev.onchange = () => { state.lev = lev.value; render(); };
  render();
}
boot();
