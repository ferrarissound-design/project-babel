"""記号列の統計量と、置換（ランダム化）検定。

二つの帰無モデルを使い分ける。
- within（行内シャッフル）: 各行の記号の構成（どの記号が何回）を保ったまま、行の中で並びだけを入れ替える。
  「並び・位置・反復」の規則性は、この帰無モデルで偶然に起きる程度と比べる。記号の頻度の偏り（Zipf 的な分布）や
  行ごとの記号の偏りだけでは有意にならない。
- global（全体シャッフル）: 行の長さと全体の記号頻度を保ったまま、記号をコーパス全体で入れ替える。
  「行どうしの類似」「同じ行に現れやすい記号の組」はこちらと比べる。

多重検定: 家族（family）ごとに Westfall–Young の single-step maxT（全仮説の最大統計量の置換分布）で
家族単位の誤り率（FWER）を制御し、さらに家族の数で Bonferroni 補正する。全体の誤検出率は名目上 alpha 以下。
統計量のスケール: 解析的に期待値が分かる族（n-gram・位置・間隔）は Poisson 上側確率の -log10 を、
そうでない族は置換分布の平均・標準偏差で標準化した z を使う。p 値の妥当性はどちらも置換分布が保証する。
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

# --------------------------------------------------------------------------------------------- 基本関数


def ngrams(line, n):
    return [line[i:i + n] for i in range(len(line) - n + 1)]


def repeated_ngrams(lines, n_min=2, n_max=8):
    """2回以上現れる n-gram（n_min ≤ n ≤ n_max）の出現数。長さ n の列が繰り返すには、その先頭 n-1 記号も
    繰り返す必要があるので、短い列から順に、繰り返す列の位置だけを伸ばして数える（結果は全列挙と同じ）。"""
    out = Counter()
    c = Counter(g for ln in lines for g in ngrams(ln, 2))
    live = {g for g, k in c.items() if k >= 2}
    if n_min <= 2:
        out.update({g: c[g] for g in live})
    starts = [[i for i in range(len(ln) - 1) if ln[i:i + 2] in live] for ln in lines]
    n = 2
    while n < n_max and any(starts):
        n += 1
        c = Counter()
        for ln, st in zip(lines, starts):
            for i in st:
                if i + n <= len(ln):
                    c[ln[i:i + n]] += 1
        live = {g for g, k in c.items() if k >= 2}
        if not live:
            break
        if n >= n_min:
            out.update({g: c[g] for g in live})
        starts = [[i for i in st if i + n <= len(ln) and ln[i:i + n] in live] for ln, st in zip(lines, starts)]
    return out


def poisson_logsf(k, mu):
    """log P(X >= k), X ~ Poisson(mu)。"""
    if k <= 0:
        return 0.0
    if mu <= 0:
        return -math.inf
    # 項 log(e^-mu mu^k / k!) から上側を足す（下側を1から引くより小さい確率で正確）
    log_t = -mu + k * math.log(mu) - math.lgamma(k + 1)
    s, t, i = 1.0, 1.0, k
    while True:
        i += 1
        t *= mu / i
        s += t
        if t < 1e-12 * s or i > k + 2000:
            break
    return min(0.0, log_t + math.log(s))


def surprise(k, mu):
    """-log10 P(X >= k)。期待値 mu に対して k 回の観測がどれだけ珍しいか。"""
    return -poisson_logsf(k, mu) / math.log(10)


def shuffle_within(lines, rng):
    out = []
    for ln in lines:
        x = list(ln)
        rng.shuffle(x)
        out.append(tuple(x))
    return out


def shuffle_global(lines, rng):
    pool = [s for ln in lines for s in ln]
    rng.shuffle(pool)
    out, k = [], 0
    for ln in lines:
        out.append(tuple(pool[k:k + len(ln)]))
        k += len(ln)
    return out


# --------------------------------------------------------------------------------- 行内シャッフルでの期待値

class Expectation:
    """行内シャッフルのもとでの期待値。行の構成は並べ替えで変わらないので、置換の間で固定できる。"""

    def __init__(self, lines):
        self.lines = [(len(ln), dict(Counter(ln))) for ln in lines]
        self.by_symbol = defaultdict(list)          # 記号 → それを含む行の番号
        for i, (_, c) in enumerate(self.lines):
            for s in c:
                self.by_symbol[s].append(i)
        self.sets = {s: set(v) for s, v in self.by_symbol.items()}
        self.cache = {}

    def _lines_with(self, seq):
        sets = sorted((self.sets.get(s, ()) for s in set(seq)), key=len)
        if not sets:
            return ()
        if len(sets) == 1:
            return sets[0]
        return set.intersection(*map(set, sets))

    @staticmethod
    def _prob(L, c, seq):
        """長さ L・構成 c の行を一様に並べ替えたとき、決まった連続位置に seq が来る確率。"""
        if len(seq) == 1:
            return c.get(seq[0], 0) / L
        if len(seq) == 2:
            a, b = seq
            return c.get(a, 0) * (c.get(b, 0) - (a == b)) / (L * (L - 1))
        used, p = {}, 1.0
        for j, s in enumerate(seq):
            u = used.get(s, 0)
            avail = c.get(s, 0) - u
            if avail <= 0:
                return 0.0
            p *= avail / (L - j)
            used[s] = u + 1
        return p

    def ngram(self, seq):
        key = ("n", seq)
        if key not in self.cache:
            n = len(seq)
            e = 0.0
            for i in self._lines_with(seq):
                L, c = self.lines[i]
                if L >= n:
                    e += (L - n + 1) * self._prob(L, c, seq)
            self.cache[key] = e
        return self.cache[key]

    def edge(self, seq):
        """行頭（または行末）に seq が来る行の数の期待値。左右対称なので両方に使える。"""
        key = ("e", seq)
        if key not in self.cache:
            n = len(seq)
            e = 0.0
            for i in self._lines_with(seq):
                L, c = self.lines[i]
                if L >= n:
                    e += self._prob(L, c, seq)
            self.cache[key] = e
        return self.cache[key]

    def gap(self, s, d):
        """同じ記号 s が距離 d で並ぶ組の数の期待値。"""
        key = ("g", s, d)
        if key not in self.cache:
            e = 0.0
            for i in self.by_symbol.get(s, []):
                L, c = self.lines[i]
                k = c[s]
                if L > d and k >= 2:
                    e += (L - d) * k * (k - 1) / (L * (L - 1))
            self.cache[key] = e
        return self.cache[key]


# ------------------------------------------------------------------------------------------------ 族（family）

class Family:
    """仮説の族。compute(lines) → {key: 統計量}。"""
    name = ""
    null = "within"
    scale = "analytic"          # analytic: 統計量そのものを比べる / studentized: 置換分布で標準化
    min_support = 3
    sd_floor = 0.5              # 標準化の分母の下限（統計量の単位に合わせる）

    def __init__(self, lines, cfg):
        self.cfg = cfg

    def compute(self, lines):
        raise NotImplementedError

    def support(self, key, lines):
        raise NotImplementedError


class NgramFamily(Family):
    """2〜max_n 記号の連続（n-gram）の反復。2回以上出たものだけを仮説にする。"""
    name = "ngram"
    min_support = 2

    def __init__(self, lines, cfg):
        super().__init__(lines, cfg)
        self.exp = Expectation(lines)
        self.max_n = cfg["max_n"]
        self.memo = {}

    def counts(self, lines, rep=None):
        rep = repeated_ngrams(lines, 2, self.max_n) if rep is None else rep
        return {g: k for g, k in rep.items() if len(g) <= self.max_n}

    def compute(self, lines, rep=None):
        out = {}
        for g, k in self.counts(lines, rep).items():
            if k >= 2:
                m = self.memo.get((g, k))
                if m is None:
                    m = self.memo[(g, k)] = surprise(k, self.exp.ngram(g))
                out[g] = m
        return out


class PositionFamily(Family):
    """行頭・行末に集中する記号列（1〜3記号）。"""
    name = "position"

    def __init__(self, lines, cfg):
        super().__init__(lines, cfg)
        self.exp = Expectation(lines)
        self.rexp = Expectation([ln[::-1] for ln in lines])
        self.memo = {}

    def counts(self, lines):
        c = Counter()
        for ln in lines:
            for n in range(1, min(3, len(ln)) + 1):
                c[("^", ln[:n])] += 1
                c[("$", ln[-n:])] += 1
        return c

    def compute(self, lines):
        out = {}
        for key, k in self.counts(lines).items():
            if k >= 2:
                m = self.memo.get((key, k))
                if m is None:
                    side, seq = key
                    mu = self.exp.edge(seq) if side == "^" else self.rexp.edge(seq[::-1])
                    m = self.memo[(key, k)] = surprise(k, mu)
                out[key] = m
        return out


class GapFamily(Family):
    """同じ記号が一定の間隔（2〜max_gap）で繰り返される。"""
    name = "gap"

    def __init__(self, lines, cfg):
        super().__init__(lines, cfg)
        self.exp = Expectation(lines)
        self.max_gap = cfg["max_gap"]
        self.memo = {}

    def counts(self, lines):
        c = Counter()
        for ln in lines:
            pos = defaultdict(list)
            for i, s in enumerate(ln):
                pos[s].append(i)
            for s, ps in pos.items():
                for a in range(len(ps)):
                    for b in range(a + 1, len(ps)):
                        d = ps[b] - ps[a]
                        if d > self.max_gap:
                            break
                        if d >= 2:
                            c[(s, d)] += 1
        return c

    def compute(self, lines):
        out = {}
        for key, k in self.counts(lines).items():
            if k >= 2:
                m = self.memo.get((key, k))
                if m is None:
                    m = self.memo[(key, k)] = surprise(k, self.exp.gap(*key))
                out[key] = m
        return out


def hypergeom_logsf(k, N, K, n):
    """log P(X >= k), X ~ 超幾何（全 N 行、記号 a を含む行 K、記号 b を含む行 n）。"""
    lo, hi = max(0, n - (N - K)), min(K, n)
    if k <= lo:
        return 0.0
    if k > hi:
        return -math.inf

    def lp(x):
        return (math.lgamma(K + 1) - math.lgamma(x + 1) - math.lgamma(K - x + 1)
                + math.lgamma(N - K + 1) - math.lgamma(n - x + 1) - math.lgamma(N - K - n + x + 1)
                - math.lgamma(N + 1) + math.lgamma(n + 1) + math.lgamma(N - n + 1))
    terms = [lp(x) for x in range(k, hi + 1)]
    m = max(terms)
    return min(0.0, m + math.log(sum(math.exp(t - m) for t in terms)))


def length_strata(lines, per_stratum=8, max_strata=4):
    """行を長さの順に並べて、ほぼ同じ数の行からなる層に分ける（短いコーパスでは1層）。"""
    n = len(lines)
    k = max(1, min(max_strata, n // per_stratum))
    order = sorted(range(n), key=lambda i: (len(lines[i]), i))
    return [order[j * n // k:(j + 1) * n // k] for j in range(k)]


def shuffle_presence(lines, rng, strata=None):
    """各記号について「その記号を含む行の数」を層（行の長さの近い行の集まり）ごとに保ったまま、
    含む行を層の中で無作為に選び直す（記号どうしは独立）。長い行ほどどの記号も含みやすいことによる
    見かけの共起を、層に分けることで帰無モデルに含める。"""
    strata = strata or [list(range(len(lines)))]
    out = [set() for _ in lines]
    for idx in strata:
        where = defaultdict(int)
        for i in idx:
            for x in set(lines[i]):
                where[x] += 1
        for x, k in where.items():
            for i in rng.sample(idx, k):
                out[i].add(x)
    return out


class CooccurrenceFamily(Family):
    """同じ行に現れやすい記号の組。帰無モデル: 行の長さの近い層ごとに、各記号を含む行の数を保ち、
    記号どうしは独立に配置する。「どちらもほぼ全ての行に出る」組は有意にならない（分散の族で扱う）。
    長さの違う行が混ざると、長い行ほど多くの記号を含むので、層に分けないと見かけの共起が生じる
    （Codex レビュー、PR #7。docs/ISSUES.md #49）。層ごとの超幾何の和は解析的に扱いにくいので、置換分布で標準化する。"""
    name = "cooccurrence"
    null = "presence"
    scale = "studentized"

    def __init__(self, lines, cfg):
        super().__init__(lines, cfg)
        nlines = Counter(x for ln in lines for x in set(ln))
        self.eligible = {x for x, k in nlines.items() if k >= 3}

    def compute(self, lines):
        c = Counter()
        for ln in lines:
            ss = sorted(set(ln) & self.eligible)
            for a in range(len(ss)):
                for b in range(a + 1, len(ss)):
                    c[(ss[a], ss[b])] += 1
        return {k: v for k, v in c.items() if v >= 2}


class DispersionFamily(Family):
    """記号が行に均等に散らばる（ほぼ全ての行に1回ずつ: 必須の要素の可能性）か、少数の行に固まるか。
    全体シャッフル（行の長さと記号の頻度を保つ）と比べる。"""
    name = "dispersion"
    null = "global"
    scale = "studentized"

    def __init__(self, lines, cfg):
        super().__init__(lines, cfg)
        freq = Counter(s for ln in lines for s in ln)
        self.eligible = {s for s, k in freq.items() if k >= 3}

    def compute(self, lines):
        c = Counter(x for ln in lines for x in set(ln) if x in self.eligible)
        out = {}
        for x in self.eligible:
            out[("even", x)] = c[x]
            out[("clump", x)] = -c[x]
        return out


class SubstitutionFamily(Family):
    """互いに置き換え可能に見える記号の組: 直前・直後の文脈（行頭・行末を含む）の分布が似ている。

    文脈の数え上げから「全体の頻度から期待される分」を引いた残差ベクトルのコサイン類似度を使う。
    頻度の高い記号どうしが偶然に同じ文脈を持つ分は差し引かれる。行内シャッフルと比べる。"""
    name = "substitution"
    scale = "studentized"
    sd_floor = 0.05
    START, END = -1, -2

    def __init__(self, lines, cfg):
        super().__init__(lines, cfg)
        freq = Counter(s for ln in lines for s in ln)
        self.eligible = sorted(s for s, k in freq.items() if k >= 3)

    def compute(self, lines):
        el = set(self.eligible)
        L, R = defaultdict(Counter), defaultdict(Counter)
        tl, tr = Counter(), Counter()
        for ln in lines:
            for i, s in enumerate(ln):
                lc = ln[i - 1] if i else self.START
                rc = ln[i + 1] if i + 1 < len(ln) else self.END
                tl[lc] += 1
                tr[rc] += 1
                if s in el:
                    L[s][lc] += 1
                    R[s][rc] += 1
        nt = sum(tl.values())
        vec = {}
        for s in self.eligible:
            ns = sum(L[s].values())
            v = {}
            for c in set(L[s]) | set(tl):
                v[("L", c)] = L[s][c] - ns * tl[c] / nt
            for c in set(R[s]) | set(tr):
                v[("R", c)] = R[s][c] - ns * tr[c] / nt
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            vec[s] = (v, norm)
        out = {}
        el = self.eligible
        for a in range(len(el)):
            va, na = vec[el[a]]
            for b in range(a + 1, len(el)):
                vb, nb = vec[el[b]]
                dot = sum(x * vb.get(k, 0.0) for k, x in va.items() if x)
                # 負の値も残す（0 で切ると、まれな記号の組の帰無分布が「ほぼ0・まれに大」になり、標準化が過敏になる）
                out[(el[a], el[b])] = dot / (na * nb)
        return out


class LinePairFamily(Family):
    """似た記号列を共有する行の組（同じ文型・定型句の可能性）。全体シャッフルと比べる。"""
    name = "linepair"
    null = "global"
    scale = "studentized"

    def compute(self, lines):
        index = defaultdict(set)
        for i, ln in enumerate(lines):
            for n in (2, 3, 4):
                for g in ngrams(ln, n):
                    index[g].add(i)
        c = Counter()
        for g, ls in index.items():
            if 1 < len(ls) <= 40:
                ls = sorted(ls)
                for a in range(len(ls)):
                    for b in range(a + 1, len(ls)):
                        c[(ls[a], ls[b])] += 1
        return {k: v for k, v in c.items() if v >= 2}


FAMILIES = [NgramFamily, PositionFamily, GapFamily, CooccurrenceFamily, DispersionFamily, SubstitutionFamily, LinePairFamily]


# --------------------------------------------------------------------------------------------- 大域検定

def _mi_adjacent(lines):
    big, left, right = Counter(), Counter(), Counter()
    for ln in lines:
        for a, b in zip(ln, ln[1:]):
            big[(a, b)] += 1
            left[a] += 1
            right[b] += 1
    n = sum(big.values())
    if not n:
        return 0.0
    return sum(k / n * math.log2(k * n / (left[a] * right[b])) for (a, b), k in big.items())


def _repeat_excess(rep):
    return sum(k - 1 for g, k in rep.items() if 3 <= len(g) <= 6)


def _edge_chi2(lines, exp):
    obs = Counter()
    for ln in lines:
        if ln:
            obs[("^", ln[0])] += 1
            obs[("$", ln[-1])] += 1
    tot = 0.0
    for s in exp.by_symbol:
        e = exp.edge((s,))
        if e > 0:
            tot += (obs[("^", s)] - e) ** 2 / e + (obs[("$", s)] - e) ** 2 / e
    return tot


def _shared_bigrams(lines):
    index = defaultdict(set)
    for i, ln in enumerate(lines):
        for g in ngrams(ln, 2):
            index[g].add(i)
    return sum(len(v) * (len(v) - 1) / 2 for v in index.values())


def longest_repeat(lines):
    """2回以上（重ならずに、別の位置で）現れる最長の連続記号列の長さと例。行をまたがない。"""
    def has(L):
        seen = {}
        for li, ln in enumerate(lines):
            for i in range(len(ln) - L + 1):
                g = ln[i:i + L]
                if g in seen:
                    pl, pi = seen[g]
                    if pl != li or i - pi >= L:
                        return g
                else:
                    seen[g] = (li, i)
        return None
    lo, hi, best = 1, max((len(ln) for ln in lines), default=0), None
    while lo <= hi:
        mid = (lo + hi) // 2
        g = has(mid)
        if g is not None:
            best, lo = g, mid + 1
        else:
            hi = mid - 1
    return (len(best) if best else 0), best


GLOBAL_TESTS = {
    "sequential_dependence": ("within", "隣り合う記号の相互情報量（並びの依存）"),
    "repetition": ("within", "3〜6記号列の反復の総数"),
    "edge_position": ("within", "行頭・行末の記号の偏り（カイ二乗）"),
    "longest_repeat": ("within", "最長反復部分列の長さ"),
    "line_similarity": ("global", "行の間で共有される2記号列の数"),
}


def global_stats(lines, exp, rep=None):
    """rep: repeated_ngrams(lines, 2, 64)。最長反復は「2回以上現れる最長の n-gram」（重なりを許す）。"""
    rep = repeated_ngrams(lines, 2, 64) if rep is None else rep
    return {"sequential_dependence": _mi_adjacent(lines), "repetition": _repeat_excess(rep),
            "edge_position": _edge_chi2(lines, exp), "longest_repeat": max((len(g) for g in rep), default=0),
            "line_similarity": _shared_bigrams(lines)}


# --------------------------------------------------------------------------------------------- 置換検定の実行

def run_tests(lines, cfg, rng_seed=0, with_global=True):
    """全族の置換検定。戻り値: {family: {"keys": {key: {...}}, ...}, "_global": {...}}"""
    rng = random.Random(rng_seed)
    B = cfg["permutations"]
    fams = [F(lines, cfg) for F in FAMILIES if F.name in cfg["families"]]
    rep_obs = repeated_ngrams(lines, 2, 64)
    obs = {f.name: (f.compute(lines, rep_obs) if f.name == "ngram" else f.compute(lines)) for f in fams}
    exp = Expectation(lines)
    g_obs = global_stats(lines, exp, rep_obs) if with_global else {}
    # 置換分布
    null_max = {f.name: [] for f in fams if f.scale == "analytic"}
    raw_ge = {f.name: Counter() for f in fams if f.scale == "analytic"}
    stud_samples = {f.name: [] for f in fams if f.scale == "studentized"}
    g_ge = Counter()
    g_null = defaultdict(list)
    strata = length_strata(lines) if any(f.null == "presence" for f in fams) else None
    for _ in range(B):
        w = shuffle_within(lines, rng)
        gl = shuffle_global(lines, rng)
        pr = shuffle_presence(lines, rng, strata) if strata else None
        rep = repeated_ngrams(w, 2, 64)
        for f in fams:
            data = w if f.null == "within" else gl if f.null == "global" else pr
            st = f.compute(data, rep) if f.name == "ngram" else f.compute(data)
            if f.scale == "analytic":
                null_max[f.name].append(max(st.values(), default=0.0))
                o = obs[f.name]
                ge = raw_ge[f.name]
                for k, v in o.items():
                    if st.get(k, 0.0) >= v - 1e-12:
                        ge[k] += 1
            else:
                stud_samples[f.name].append(st)
        if with_global:
            gw = global_stats(w, exp, rep)
            gg = {"line_similarity": _shared_bigrams(gl)}
            for k, (null, _) in GLOBAL_TESTS.items():
                v = gg[k] if null == "global" else gw[k]
                g_null[k].append(v)
                if v >= g_obs[k] - 1e-12:
                    g_ge[k] += 1
    out = {}
    for f in fams:
        o = obs[f.name]
        res = {}
        if f.scale == "analytic":
            nm = sorted(null_max[f.name])
            for k, v in o.items():
                n_ge = len(nm) - _bisect_left(nm, v - 1e-12)
                res[k] = {"stat": v, "p_family": (1 + n_ge) / (B + 1), "p_raw": (1 + raw_ge[f.name][k]) / (B + 1)}
            out[f.name] = {"keys": res, "null_max_q95": _quantile(nm, 0.95), "null_max": nm}
        else:
            samples = stud_samples[f.name]
            keys = set(o)
            for st in samples:
                keys.update(st)
            mu, sd = {}, {}
            for k in keys:
                vals = [st.get(k, 0) for st in samples] + [o.get(k, 0)]
                m = sum(vals) / len(vals)
                mu[k] = m
                sd[k] = max(f.sd_floor, math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals)))
            maxes = []
            for st in samples:
                maxes.append(max(((v - mu[k]) / sd[k] for k, v in st.items()), default=0.0))
            maxes.sort()
            for k, v in o.items():
                z = (v - mu[k]) / sd[k]
                n_ge = len(maxes) - _bisect_left(maxes, z - 1e-12)
                raw = sum(1 for st in samples if st.get(k, 0) >= v) + 1
                res[k] = {"stat": z, "value": v, "null_mean": mu[k], "p_family": (1 + n_ge) / (B + 1),
                          "p_raw": raw / (B + 1)}
            out[f.name] = {"keys": res, "null_max_q95": _quantile(maxes, 0.95)}
    if with_global:
        glob = {}
        for k, (null, desc) in GLOBAL_TESTS.items():
            vals = g_null[k]
            m = sum(vals) / len(vals) if vals else 0.0
            glob[k] = {"observed": round(float(g_obs[k]), 4), "null_mean": round(m, 4), "null": null, "description": desc,
                       "p": (1 + g_ge[k]) / (B + 1)}
        # Holm 補正
        order = sorted(glob, key=lambda k: glob[k]["p"])
        prev = 0.0
        for r, k in enumerate(order):
            adj = min(1.0, max(prev, glob[k]["p"] * (len(order) - r)))
            glob[k]["p_holm"] = adj
            prev = adj
        out["_global"] = glob
    out["_permutations"] = B
    return out


def _bisect_left(a, x):
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    return round(float(sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]), 4)
