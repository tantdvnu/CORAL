import os, sys, glob, zipfile, time, json, warnings, subprocess
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import QuantileRegressor
warnings.filterwarnings('ignore')
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
try:
    import lightgbm as lgb
    GBDT_BACKEND = 'lightgbm'
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    GBDT_BACKEND = 'sklearn'
try:
    from numba import njit
    NUMBA = True
except ImportError:
    NUMBA = False

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f
print(f'GBDT backend: {GBDT_BACKEND} | numba: {NUMBA}')
DATA_ROOT = '5Gdataset'
EXTRACT = os.path.join(DATA_ROOT, 'extracted')
RESULTS = 'results'
os.makedirs(RESULTS, exist_ok=True)
if not os.path.isdir(DATA_ROOT):
    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/uccmisl/5Gdataset.git', DATA_ROOT], check=True)
if not os.path.isdir(EXTRACT):
    with zipfile.ZipFile(os.path.join(DATA_ROOT, '5G-production-dataset.zip')) as zf:
        zf.extractall(EXTRACT)
TRACE_FILES = sorted((f for f in glob.glob(os.path.join(EXTRACT, '**', '*.csv'), recursive=True) if '__MACOSX' not in f))
print('Trace files found:', len(TRACE_FILES))
CFG = dict(H=3, ALPHA=0.1, R=10, TEST_FRAC=0.3, VAL_FRAC=0.3, GAP_HOLD=2, HO_WIN=5, SPEED_TH=5.0, N_TREES=400, LR=0.05, LEAVES=31, ETA_GRID=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2], B=5.0, KNOTS=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9], ALPHAS_FIG2B=[0.05, 0.1, 0.2, 0.3], MIN_TRACE_LEN=30)
KPI = ['RSRP', 'RSRQ', 'SNR', 'CQI', 'RSSI', 'NRxRSRP', 'NRxRSRQ', 'Speed']
EPS = 1e-06

def load_trace(path, H):
    df = pd.read_csv(path, na_values=['-', '', ' '], low_memory=False)
    t = pd.to_datetime(df['Timestamp'], format='%Y.%m.%d_%H.%M.%S', errors='coerce')
    df = df.loc[t.notna()].copy()
    t = t[t.notna()]
    df['sec'] = (t - t.iloc[0]).dt.total_seconds().astype(int)
    for c in KPI + ['DL_bitrate']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['rate'] = df['DL_bitrate'] / 1000.0
    num = df.groupby('sec')[KPI + ['rate']].mean()
    cat = df.groupby('sec')[['CellID', 'NetworkMode']].last()
    g = num.join(cat)
    full = np.arange(g.index.min(), g.index.max() + 1)
    g = g.reindex(full)
    observed = g['rate'].notna().values
    valid = observed.copy()
    i = 0
    n = len(full)
    while i < n:
        if not observed[i]:
            j = i
            while j < n and (not observed[j]):
                j += 1
            if j - i <= CFG['GAP_HOLD'] and i > 0:
                valid[i:j] = True
            i = j
        else:
            i += 1
    g = g.ffill()
    seg = np.cumsum(~valid)
    r = g['rate'].values.astype(float)
    ell = np.log1p(np.maximum(r, 0.0))
    kap = g[KPI].values.astype(float)
    mode = g['NetworkMode'].astype(str).values
    cell = g['CellID'].values
    change = np.zeros(n, bool)
    change[1:] = (cell[1:] != cell[:-1]) & pd.notna(cell[1:]) & pd.notna(cell[:-1])
    tau = np.zeros(n)
    last = -10000
    for k in range(n):
        if change[k]:
            last = k
        tau[k] = k - last if last > -10000 else k + 300.0
    tau = np.minimum(tau, 300.0)
    cs = np.cumsum(change)
    n_ch30 = cs - np.concatenate([np.zeros(30), cs[:-30]])[:n]
    rows = []
    for k in range(4, n - H):
        s0 = seg[k]
        if not (valid[k - 4:k + H + 1].all() and (seg[k - 4:k + H + 1] == s0).all()):
            continue
        lags = ell[k - 4:k + 1][::-1]
        feats = np.concatenate([lags, [lags.mean(), lags.std(), lags.max()], kap[k], kap[k] - kap[k - 2], [mode[k] == '5G', mode[k] == 'LTE', tau[k], n_ch30[k]]]).astype(float)
        y = r[k + 1:k + H + 1].mean()
        past = np.maximum(r[k - 4:k + 1], 0.001)
        rows.append((feats, y, float(tau[k] <= CFG['HO_WIN']), float(kap[k, 7] > CFG['SPEED_TH']) if np.isfinite(kap[k, 7]) else 0.0, float(mode[k] != '5G'), 5.0 / np.sum(1.0 / past), lags.std()))
    return (rows, len(df), set(pd.unique(df['CellID'].dropna())))

def build_dataset(H):
    recs, all_cells, meta = ([], set(), [])
    total_records = 0
    for tid, f in enumerate(TRACE_FILES):
        rows, nrec, cells = load_trace(f, H)
        total_records += nrec
        all_cells |= cells
        parts = f.replace('\\', '/').split('/')
        app = [p for p in parts if p in ('Download', 'Netflix', 'Amazon_Prime')][0]
        mob = 'Driving' if '/Driving/' in f.replace('\\', '/') else 'Static'
        meta.append(dict(tid=tid, file=os.path.basename(f), app=app, mob=mob, n=len(rows)))
        for x, y, h, m, l, hm, sl in rows:
            recs.append((tid, x, y, h, m, l, hm, sl))
    tid = np.array([r[0] for r in recs])
    D = dict(tid=tid, X=np.vstack([r[1] for r in recs]), y=np.array([r[2] for r in recs]), h=np.array([r[3] for r in recs]), m=np.array([r[4] for r in recs]), l=np.array([r[5] for r in recs]), hm5=np.array([r[6] for r in recs]), sig=np.array([r[7] for r in recs]))
    D['z'] = np.log1p(D['y'])
    meta = pd.DataFrame(meta)
    info = dict(traces=len(TRACE_FILES), records=total_records, cells=len(all_cells), samples=len(tid))
    return (D, meta, info)
t0 = time.time()
D, META, INFO = build_dataset(CFG['H'])
print(f"Dataset: {INFO['traces']} traces, {INFO['records']:,} records, {INFO['cells']} cells, {INFO['samples']:,} usable samples (H={CFG['H']} s)  [{time.time() - t0:.0f}s]")
print(META.groupby(['app', 'mob']).agg(traces=('tid', 'size'), samples=('n', 'sum')))
DL_TIDS = META.loc[META.app == 'Download', 'tid'].values
VID_TIDS = META.loc[META.app != 'Download', 'tid'].values
print(f"P1 download traces: {len(DL_TIDS)} ({META.loc[META.app == 'Download', 'n'].sum():,} samples) | P2 video traces: {len(VID_TIDS)} ({META.loc[META.app != 'Download', 'n'].sum():,} samples)")

def split_traces(seed):
    rng = np.random.default_rng(seed)
    tr, va, te = ([], [], [])
    for mob in ('Static', 'Driving'):
        sub = META[(META.app == 'Download') & (META.mob == mob)]
        ids = sub.tid.values.copy()
        rng.shuffle(ids)
        n_of = dict(zip(sub.tid, sub.n))
        N = sum(n_of.values())
        test_ids, acc = ([], 0)
        for i in ids:
            if len(ids) - len(test_ids) <= 2:
                break
            if abs(acc + n_of[i] - CFG['TEST_FRAC'] * N) < abs(acc - CFG['TEST_FRAC'] * N):
                test_ids.append(i)
                acc += n_of[i]
        if not test_ids:
            test_ids = [ids[0]]
        rest = [i for i in ids if i not in test_ids]
        Nr = sum((n_of[i] for i in rest))
        val_ids, acc = ([], 0)
        for i in rest:
            if len(rest) - len(val_ids) <= 1:
                break
            if abs(acc + n_of[i] - CFG['VAL_FRAC'] * Nr) < abs(acc - CFG['VAL_FRAC'] * Nr):
                val_ids.append(i)
                acc += n_of[i]
        if not val_ids:
            val_ids = [rest[0]]
        te.extend(test_ids)
        va.extend(val_ids)
        rest = [i for i in rest if i not in val_ids]
        j = 0
        tr.extend(rest[j:])
    return (np.array(tr), np.array(va), np.array(te))

def fit_gbdt(X, target, quantile=None, seed=0):
    if GBDT_BACKEND == 'lightgbm':
        kw = dict(n_estimators=CFG['N_TREES'], learning_rate=CFG['LR'], num_leaves=CFG['LEAVES'], random_state=seed, verbose=-1)
        mdl = lgb.LGBMRegressor(objective='quantile', alpha=quantile, **kw) if quantile is not None else lgb.LGBMRegressor(objective='regression', **kw)
    else:
        kw = dict(max_iter=CFG['N_TREES'], learning_rate=CFG['LR'], max_leaf_nodes=CFG['LEAVES'], random_state=seed, early_stopping=False)
        mdl = HistGradientBoostingRegressor(loss='quantile', quantile=quantile, **kw) if quantile is not None else HistGradientBoostingRegressor(loss='squared_error', **kw)
    return mdl.fit(X, target)

@njit(cache=True)
def run_coral(q01, z, Phi, w1, eta, p, B, starts):
    T, d = Phi.shape
    mu = np.empty(T)
    w = w1.copy()
    s = 0
    for t in range(T):
        if s < len(starts) and t == starts[s]:
            for j in range(d):
                w[j] = w1[j]
            s += 1
        bo = 0.0
        for j in range(d):
            bo += w[j] * Phi[t, j]
        mu[t] = q01[t] - bo
        e = 1.0 if z[t] < mu[t] else 0.0
        for j in range(d):
            v = w[j] + eta * (e - p) * Phi[t, j]
            w[j] = min(max(v, -B), B)
    return mu

@njit(cache=True)
def run_aci(q01, z, scores_sorted, alpha, gamma, starts):
    T = len(q01)
    n = len(scores_sorted)
    mu = np.empty(T)
    a = alpha
    s = 0
    for t in range(T):
        if s < len(starts) and t == starts[s]:
            a = alpha
            s += 1
        lev = 1.0 - a
        if lev >= 1.0:
            off = scores_sorted[n - 1] + 1.0
        elif lev <= 0.0:
            off = scores_sorted[0] - 1.0
        else:
            k = int(np.ceil(lev * (n + 1))) - 1
            k = min(max(k, 0), n - 1)
            off = scores_sorted[k]
        mu[t] = q01[t] - off
        e = 1.0 if z[t] < mu[t] else 0.0
        a = a + gamma * (alpha - e)
    return mu

def trace_starts(tids):
    return np.flatnonzero(np.r_[True, tids[1:] != tids[:-1]]).astype(np.int64)

def violation(y, L):
    return float(np.mean(y < L))

def utilization(y, L):
    return float(np.sum(np.minimum(np.maximum(L, 0), y)) / np.sum(y))

def goodput(y, r):
    r = np.maximum(r, 0)
    return float(np.sum(r * (y >= r)) / np.sum(y))

def worst_trace(y, L, tids):
    v = []
    for t in np.unique(tids):
        k = tids == t
        if k.sum() >= CFG['MIN_TRACE_LEN']:
            v.append(np.mean(y[k] < L[k]))
    return float(np.max(v)) if v else np.nan

def per_trace_U(y, L, tids):
    out = {}
    for t in np.unique(tids):
        k = tids == t
        if k.sum() >= CFG['MIN_TRACE_LEN']:
            out[t] = utilization(y[k], L[k])
    return out

def density_cdf(zq, MU, P, lamL, lamU):
    MU = np.sort(MU, axis=1)
    MU = MU + np.arange(MU.shape[1]) * 1e-06
    T, K = MU.shape
    zq = np.atleast_2d(zq)
    if zq.shape[0] == 1 and T > 1:
        zq = np.repeat(zq, T, 0)
    F = np.empty_like(zq)
    lo = zq < MU[:, [0]]
    hi = zq >= MU[:, [K - 1]]
    F[lo] = (P[0] * np.exp((zq - MU[:, [0]]) / lamL))[lo]
    F[hi] = (1 - (1 - P[-1]) * np.exp(-(zq - MU[:, [K - 1]]) / lamU))[hi]
    mid = ~(lo | hi)
    if mid.any():
        for k in range(K - 1):
            sel = mid & (zq >= MU[:, [k]]) & (zq < MU[:, [k + 1]])
            slope = (P[k + 1] - P[k]) / (MU[:, [k + 1]] - MU[:, [k]])
            val = P[k] + slope * (zq - MU[:, [k]])
            F[sel] = val[sel]
    return F

def density_logpdf_y(y, MU, P, lamL, lamU):
    MU = np.sort(MU, axis=1) + np.arange(MU.shape[1]) * 1e-06
    z = np.log1p(y)
    K = MU.shape[1]
    f = np.empty(len(z))
    lo = z < MU[:, 0]
    hi = z >= MU[:, K - 1]
    f[lo] = P[0] / lamL * np.exp((z[lo] - MU[lo, 0]) / lamL)
    f[hi] = (1 - P[-1]) / lamU * np.exp(-(z[hi] - MU[hi, K - 1]) / lamU)
    mid = ~(lo | hi)
    for k in range(K - 1):
        sel = mid & (z >= MU[:, k]) & (z < MU[:, k + 1])
        f[sel] = (P[k + 1] - P[k]) / (MU[sel, k + 1] - MU[sel, k])
    return np.log(np.maximum(f, 1e-300)) - z

def tail_scales(MU_va, z_va):
    MU = np.sort(MU_va, axis=1)
    lo = z_va < MU[:, 0]
    hi = z_va > MU[:, -1]
    lamL = float(np.mean(MU[lo, 0] - z_va[lo])) if lo.any() else 0.1
    lamU = float(np.mean(z_va[hi] - MU[hi, -1])) if hi.any() else 0.1
    return (max(lamL, 0.001), max(lamU, 0.001))
PIT_LEVELS = np.arange(0.05, 1.0, 0.1)

def pit_curve(z, MU, P, lamL, lamU):
    u = density_cdf(z[:, None], MU, P, lamL, lamU)[:, 0]
    emp = np.array([np.mean(u <= p) for p in PIT_LEVELS])
    ce = float(np.mean(np.abs(emp - PIT_LEVELS)))
    return (emp, ce)

def goodput_optimal_rate(MU, P, lamL, lamU, n_grid=200):
    MUs = np.sort(MU, axis=1)
    a = MUs[:, [0]] - 3 * lamL
    b = MUs[:, [-1]] + 3 * lamU
    g = np.linspace(0, 1, n_grid)[None, :]
    best = np.empty(len(MU))
    CH = 20000
    for s in range(0, len(MU), CH):
        zz = a[s:s + CH] + (b[s:s + CH] - a[s:s + CH]) * g
        F = density_cdf(zz, MU[s:s + CH], P, lamL, lamU)
        G = np.expm1(zz) * (1 - F)
        best[s:s + CH] = np.expm1(zz[np.arange(len(zz)), np.argmax(G, axis=1)])
    return np.maximum(best, 0)

def phi_matrix(q01, q05, sig, h, m, l, sbar, vbar, kind='coral'):
    one = np.ones_like(q01)
    s_t = (q05 - q01) / sbar
    v_t = sig / vbar
    if kind == 'coral':
        return np.column_stack([one, s_t, v_t, h, m, l])
    if kind == 'noctx':
        return np.column_stack([one, s_t, v_t])
    if kind == 'noshape':
        return np.column_stack([one, h, m, l])
    if kind == 'olla':
        return one[:, None]
    raise ValueError(kind)

def offline_qr(Phi, c, p):
    qr = QuantileRegressor(quantile=1 - p, alpha=0.0, fit_intercept=False, solver='highs')
    return qr.fit(Phi, c).coef_.astype(float)

def tune_eta(run_fn, y_va, alpha):
    best, best_u, fallback, fb_gap = (None, -1, None, 9)
    for eta in CFG['ETA_GRID']:
        L = np.expm1(run_fn(eta))
        v, u = (violation(y_va, L), utilization(y_va, L))
        if abs(v - alpha) <= 0.01 and u > best_u:
            best, best_u = (eta, u)
        if abs(v - alpha) < fb_gap:
            fallback, fb_gap = (eta, abs(v - alpha))
    return best if best is not None else fallback

def subset(idx_tids, tids_all):
    mask = np.isin(D['tid'], idx_tids)
    order = np.argsort(D['tid'][mask], kind='stable')
    return np.flatnonzero(mask)[order]

def run_partition(seed, alphas=(0.1,), want_density=True):
    tr, va, te = split_traces(seed)
    I_tr, I_va, I_te = (subset(tr, D['tid']), subset(va, D['tid']), subset(te, D['tid']))
    I_p2 = subset(VID_TIDS, D['tid'])
    X, z, y = (D['X'], D['z'], D['y'])
    m01 = fit_gbdt(X[I_tr], z[I_tr], 0.1, seed)
    m05 = fit_gbdt(X[I_tr], z[I_tr], 0.5, seed)
    mse = fit_gbdt(X[I_tr], z[I_tr], None, seed)
    sets = {}
    for name, I in (('va', I_va), ('P1', I_te), ('P2', I_p2)):
        q01, q05 = (m01.predict(X[I]), m05.predict(X[I]))
        sets[name] = dict(I=I, q01=q01, q05=q05, qm=mse.predict(X[I]), z=z[I], y=y[I], tid=D['tid'][I], h=D['h'][I], m=D['m'][I], l=D['l'][I], hm5=D['hm5'][I], sig=D['sig'][I], st=trace_starts(D['tid'][I]))
    V = sets['va']
    sbar = np.sqrt(np.mean((V['q05'] - V['q01']) ** 2)) + EPS
    vbar = np.sqrt(np.mean(V['sig'] ** 2)) + EPS
    for S in sets.values():
        S['c'] = S['q01'] - S['z']
        for kind in ('coral', 'noctx', 'noshape', 'olla'):
            S['Phi_' + kind] = phi_matrix(S['q01'], S['q05'], S['sig'], S['h'], S['m'], S['l'], sbar, vbar, kind)
    cal_scores = np.sort(V['c'])
    out = dict(seed=seed, n_tr=len(I_tr), n_va=len(I_va), n_te=len(I_te), test_traces=te, res={})
    for alpha in alphas:
        L = {}
        k = int(np.ceil((1 - alpha) * (len(cal_scores) + 1))) - 1
        q_cqr = cal_scores[min(max(k, 0), len(cal_scores) - 1)]
        L['HM-5'] = {s: sets[s]['hm5'] for s in ('P1', 'P2')}
        L['GBDT-MSE'] = {s: np.expm1(sets[s]['qm']) for s in ('P1', 'P2')}
        L['GBDT-QR'] = {s: np.expm1(sets[s]['q01']) for s in ('P1', 'P2')}
        L['Split CQR'] = {s: np.expm1(sets[s]['q01'] - q_cqr) for s in ('P1', 'P2')}
        fn = lambda g, S=V: run_aci(S['q01'], S['z'], cal_scores, alpha, g, S['st'])
        g_star = tune_eta(fn, V['y'], alpha)
        L['ACI'] = {s: np.expm1(run_aci(sets[s]['q01'], sets[s]['z'], cal_scores, alpha, g_star, sets[s]['st'])) for s in ('P1', 'P2')}
        etas = {'ACI': g_star}
        for name, kind in (('OLLA', 'olla'), ('w/o contexts', 'noctx'), ('w/o shape', 'noshape'), ('CORAL', 'coral')):
            w1 = offline_qr(V['Phi_' + kind], V['c'], alpha)
            fn = lambda eta, S=V, w1=w1, kind=kind: run_coral(S['q01'], S['z'], S['Phi_' + kind], w1, eta, alpha, CFG['B'], S['st'])
            eta = tune_eta(fn, V['y'], alpha)
            etas[name] = eta
            L[name] = {s: np.expm1(run_coral(sets[s]['q01'], sets[s]['z'], sets[s]['Phi_' + kind], w1, eta, alpha, CFG['B'], sets[s]['st'])) for s in ('P1', 'P2')}
        M = {}
        for name, Ls in L.items():
            row = {}
            for s in ('P1', 'P2'):
                S = sets[s]
                Lb = Ls[s]
                row[f'{s} V'] = violation(S['y'], Lb)
                row[f'{s} Worst'] = worst_trace(S['y'], Lb, S['tid'])
                row[f'{s} HO'] = violation(S['y'][S['h'] == 1], Lb[S['h'] == 1])
                row[f'{s} Mobile'] = violation(S['y'][S['m'] == 1], Lb[S['m'] == 1])
                row[f'{s} NonNR'] = violation(S['y'][S['l'] == 1], Lb[S['l'] == 1])
                row[f'{s} U'] = utilization(S['y'], Lb)
                row[f'{s} pertraceU'] = per_trace_U(S['y'], Lb, S['tid'])
            M[name] = row
        out['res'][alpha] = dict(metrics=M, etas=etas)
        if want_density and abs(alpha - CFG['ALPHA']) < 1e-09:
            P = np.array(CFG['KNOTS'])
            eta = etas['CORAL']
            W1 = {p: offline_qr(V['Phi_coral'], V['c'], p) for p in P}

            def bank(S, online=True):
                return np.column_stack([run_coral(S['q01'], S['z'], S['Phi_coral'], W1[p], eta if online else 0.0, p, CFG['B'], S['st']) for p in P])
            MU_va = bank(V)
            lamL, lamU = tail_scales(MU_va, V['z'])
            MU_va_static = bank(V, online=False)
            lamL_s, lamU_s = tail_scales(MU_va_static, V['z'])
            dens = dict(lamL=lamL, lamU=lamU)
            for s in ('P1', 'P2'):
                S = sets[s]
                for tag, online, lL, lU in (('online', True, lamL, lamU), ('static', False, lamL_s, lamU_s)):
                    MU = bank(S, online)
                    emp, ce = pit_curve(S['z'], MU, P, lL, lU)
                    lnf = float(np.mean(density_logpdf_y(S['y'], MU, P, lL, lU)))
                    r_star = goodput_optimal_rate(MU, P, lL, lU)
                    dens[f'{s} {tag}'] = dict(pit=emp, CE=ce, lnf=lnf, Gamma=goodput(S['y'], r_star), V=violation(S['y'], r_star))
                    if tag == 'online' and s == 'P1':
                        mu_a = MU[:, list(P).index(CFG['ALPHA'])]
                        dz = 0.001
                        fz = (density_cdf((mu_a + dz)[:, None], MU, P, lL, lU)[:, 0] - density_cdf((mu_a - dz)[:, None], MU, P, lL, lU)[:, 0]) / (2 * dz)
                        Ph = S['Phi_coral']
                        J = (Ph * fz[:, None]).T @ Ph / len(Ph)
                        lam = np.clip(np.linalg.eigvalsh(J), 1e-12, None)
                        dens['tau_modes'] = np.sort(1.0 / (eta * lam))
                        dens['tau_olla'] = 1.0 / (etas['OLLA'] * np.mean(fz))
            for nm in ('Split CQR', 'OLLA', 'CORAL'):
                dens[nm] = {s: dict(Gamma=goodput(sets[s]['y'], L[nm][s]), V=violation(sets[s]['y'], L[nm][s])) for s in ('P1', 'P2')}
            out['density'] = dens
    return out
ALL = []
t0 = time.time()
alphas = sorted(set(CFG['ALPHAS_FIG2B']) | {CFG['ALPHA']})
for r in range(CFG['R']):
    res = run_partition(r, alphas=alphas)
    ALL.append(res)
    M = res['res'][CFG['ALPHA']]['metrics']
    tot = res['n_tr'] + res['n_va'] + res['n_te']
    print(f"partition {r}: split {100 * res['n_tr'] / tot:.0f}:{100 * res['n_va'] / tot:.0f}:{100 * res['n_te'] / tot:.0f} | CORAL P1 V={M['CORAL']['P1 V']:.3f} U={M['CORAL']['P1 U']:.3f} | P2 V={M['CORAL']['P2 V']:.3f} U={M['CORAL']['P2 U']:.3f}  [{time.time() - t0:.0f}s]")
ORDER = ['HM-5', 'GBDT-MSE', 'GBDT-QR', 'Split CQR', 'ACI', 'OLLA', 'w/o contexts', 'w/o shape', 'CORAL']
COLS = ['P1 V', 'P1 Worst', 'P1 HO', 'P1 U', 'P2 V', 'P2 Worst', 'P2 U']

def mean_metric(name, col, alpha=CFG['ALPHA']):
    return 100 * np.nanmean([R['res'][alpha]['metrics'][name][col] for R in ALL])
T1 = pd.DataFrame({c: [mean_metric(n, c) for n in ORDER] for c in COLS}, index=ORDER).round(1)
T1.columns = ['P1 V', 'Worst', 'HO', 'P1 U', 'P2 V', 'Worst ', 'P2 U']
print('\nTABLE I  Assurance at alpha = 0.1, H = 3 s, mean of 10 trace partitions (%)')
print(T1.to_string())
T1.to_csv(os.path.join(RESULTS, 'table1.csv'))
ctx = pd.DataFrame({c: [mean_metric('CORAL', c)] for c in ['P1 HO', 'P2 HO', 'P2 Mobile', 'P2 NonNR']}, index=['CORAL']).round(1)
print('\nCORAL per-context violation (%):\n', ctx.to_string())
rng = np.random.default_rng(2026)
stat_lines = []

def paired(a, b, label):
    d = np.array(a) - np.array(b)
    wins = int(np.sum(d > 0))
    try:
        p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        p = np.nan
    boots = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    s = f'{label}: wins {wins}/{len(d)}, mean gain {100 * d.mean():.2f} pts, Wilcoxon p = {p:.3g}, bootstrap 95% CI [{100 * lo:.2f}, {100 * hi:.2f}]'
    print(s)
    stat_lines.append(s)
A = CFG['ALPHA']
for s in ('P1', 'P2'):
    cor = [R['res'][A]['metrics']['CORAL'][f'{s} U'] for R in ALL]
    for base in ('ACI', 'OLLA'):
        b = [R['res'][A]['metrics'][base][f'{s} U'] for R in ALL]
        paired(cor, b, f'{s} U, CORAL vs {base} (partitions)')
for base in ('ACI', 'OLLA'):
    a_, b_ = ([], [])
    for R in ALL:
        pc = R['res'][A]['metrics']['CORAL']['P1 pertraceU']
        pb = R['res'][A]['metrics'][base]['P1 pertraceU']
        for t in pc:
            a_.append(pc[t])
            b_.append(pb[t])
    d = np.array(a_) - np.array(b_)
    p = stats.wilcoxon(a_, b_).pvalue
    s = f'P1 per-trace, CORAL vs {base}: CORAL wins {int((d > 0).sum())} of {len(d)} test traces, Wilcoxon p = {p:.3g}'
    print(s)
    stat_lines.append(s)
for base in ('ACI', 'OLLA'):
    g = 100 * (mean_metric('CORAL', 'P2 U') / mean_metric(base, 'P2 U') - 1)
    s = f'P2: CORAL commits {g:+.1f}% more assured rate than {base}'
    print(s)
    stat_lines.append(s)
with open(os.path.join(RESULTS, 'stats.txt'), 'w') as f:
    f.write('\n'.join(stat_lines))

def dmean(key, field):
    return np.mean([R['density'][key][field] for R in ALL])
rows = []
for label, key in (('r* (18), static knots', 'static'), ('r* (18), CORAL knots', 'online')):
    row = [label]
    for s in ('P1', 'P2'):
        row += [dmean(f'{s} {key}', 'lnf'), 100 * dmean(f'{s} {key}', 'CE'), 100 * dmean(f'{s} {key}', 'Gamma'), 100 * dmean(f'{s} {key}', 'V')]
    rows.append(row)
for label, key in (('L_t, split CQR', 'Split CQR'), ('L_t, OLLA', 'OLLA'), ('L_t (7), CORAL', 'CORAL')):
    row = [label]
    for s in ('P1', 'P2'):
        row += [np.nan, np.nan, 100 * np.mean([R['density'][key][s]['Gamma'] for R in ALL]), 100 * np.mean([R['density'][key][s]['V'] for R in ALL])]
    rows.append(row)
T2 = pd.DataFrame(rows, columns=['Rate rule', 'P1 ln f', 'P1 CE', 'P1 Gamma', 'P1 V', 'P2 ln f', 'P2 CE', 'P2 Gamma', 'P2 V']).set_index('Rate rule')
print('\nTABLE II  Predictive density and goodput on test data (%; ln f in nats)')
print(T2.round(2).to_string(na_rep='—'))
T2.round(3).to_csv(os.path.join(RESULTS, 'table2.csv'))
lamL = np.mean([R['density']['lamL'] for R in ALL])
lamU = np.mean([R['density']['lamU'] for R in ALL])
print(f'\nTail scales (mean over partitions): lambda_L = {lamL:.3f}, lambda_U = {lamU:.3f}')
TAU = np.array([R['density']['tau_modes'] for R in ALL])
print('CORAL mode time constants (slots), median over partitions:', np.round(np.median(TAU, axis=0), 1))
print('OLLA single mode (slots), median:', round(float(np.median([R['density']['tau_olla'] for R in ALL])), 1))
etas = pd.DataFrame([R['res'][A]['etas'] for R in ALL])
print('Selected steps (eta / gamma) per partition:\n', etas.to_string())
plt.rcParams.update({'font.family': 'serif', 'font.size': 8})
fig = plt.figure(figsize=(3.5, 3.6), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.55, wspace=0.38)
ax = fig.add_subplot(gs[0, :])
groups = [('P1 HO', 'P1\nHO'), ('P1 Worst', 'P1\nWorst'), ('P2 V', 'P2\nOverall'), ('P2 HO', 'P2\nHO'), ('P2 Mobile', 'P2\nMobile'), ('P2 NonNR', 'P2\nNon-NR'), ('P2 Worst', 'P2\nWorst')]
meths = [('Split CQR', '#e8a33d', '///'), ('OLLA', '#7fb07f', '...'), ('ACI', '#c94f4f', ''), ('CORAL', '#3b6fb6', '')]
x = np.arange(len(groups))
wbar = 0.2
for i, (mname, col, hatch) in enumerate(meths):
    vals = [mean_metric(mname, g[0]) / 100 for g in groups]
    shown = np.minimum(vals, 0.35)
    bars = ax.bar(x + (i - 1.5) * wbar, shown, wbar, color=col, hatch=hatch, edgecolor='white', linewidth=0.3, label=mname)
    for xi, v in zip(x + (i - 1.5) * wbar, vals):
        if v > 0.35:
            ax.text(xi, 0.352, f'{v:.2f}', ha='center', va='bottom', fontsize=4.5, rotation=90)
ax.axhline(CFG['ALPHA'], ls='--', lw=0.7, color='k')
ax.set_xticks(x)
ax.set_xticklabels([g[1] for g in groups], fontsize=6)
ax.set_ylabel('Violation rate')
ax.set_ylim(0, 0.42)
ax.legend(ncol=4, fontsize=5.5, frameon=False, loc='upper center', bbox_to_anchor=(0.5, 1.22))
ax.text(-0.08, 1.05, '(a)', transform=ax.transAxes)
ax = fig.add_subplot(gs[1, 0])
for mname, col, mk in (('OLLA', '#7fb07f', 's'), ('ACI', '#c94f4f', '^'), ('CORAL', '#3b6fb6', 'o')):
    vv = [mean_metric(mname, 'P2 V', a) / 100 for a in CFG['ALPHAS_FIG2B']]
    uu = [mean_metric(mname, 'P2 U', a) for a in CFG['ALPHAS_FIG2B']]
    ax.plot(vv, uu, marker=mk, ms=2.5, lw=0.8, color=col, label=mname)
ax.set_xlabel('Achieved violation')
ax.set_ylabel('Utilization U (%)')
ax.legend(fontsize=5, frameon=False)
ax.text(-0.3, 1.04, '(b)', transform=ax.transAxes)
ax = fig.add_subplot(gs[1, 1])
for s, ls in (('P1', '-'), ('P2', '--')):
    for tag, col, mk in (('online', '#3b6fb6', 'o'), ('static', '#999999', 'x')):
        emp = np.mean([R['density'][f'{s} {tag}']['pit'] for R in ALL], axis=0)
        ax.plot(PIT_LEVELS, emp, ls=ls, marker=mk, ms=2, lw=0.7, color=col, label=f'{s} {tag}')
ax.plot([0, 1], [0, 1], color='k', lw=0.5)
ax.set_xlabel('Nominal level p')
ax.set_ylabel('Empirical PIT frequency')
ax.legend(fontsize=4.5, frameon=False)
ax.text(-0.3, 1.04, '(c)', transform=ax.transAxes)
plt.savefig(os.path.join(RESULTS, 'fig2.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(RESULTS, 'fig2.png'), bbox_inches='tight', dpi=300)
plt.show()
summary = dict(config=CFG, dataset=INFO, gbdt_backend=GBDT_BACKEND, table1=T1.to_dict(), table2=T2.round(4).to_dict())
with open(os.path.join(RESULTS, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('Saved:', sorted(os.listdir(RESULTS)))
