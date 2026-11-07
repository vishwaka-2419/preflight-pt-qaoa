"""Ablations and additional statistical analyses of the all-edge probe (reference job and 72.1-hour repeat).

  A. Variants of the probe statistic: cost-weighted (T_E), unweighted mean of the same edge transfers, registered
     statistic, mean transfer over all output pairs (generic aggregate), mean over output pairs that are not cost
     edges, ideal-correlator-weighted statistic, cost-weighted statistic with permuted weights, complete-coverage
     subset. Rank correlation with r_W and partial rank correlation given the gate score, with case-bootstrap
     intervals and paired case-bootstrap intervals for differences from T_E.
  B. Partial rank correlation of T_E given the full score, and given gate and full score jointly.
  C. Graph-level sensitivity: leave-one-graph-out and a bootstrap over the nine graphs.
  D. Probe shot-resampling: multinomial resampling of the counts of every executed probe circuit (shared-shot
     structure preserved), refit of every edge transfer, and recomputation of T_E; intervals for rho from probe
     counts alone and from probe counts together with case resampling.
  E. Matched-cost direct workload sampling: subsamples of the 2,048-shot reference counts at several shot budgets,
     evaluated (i) in the same job against the complementary shots and (ii) against the 72.1-hour outcomes.
  F. Layout-order reversals between t = 0 and 72.1 hours.

usage: python3 -I ablations.py <v3 dir> <canonical dir> <out dir>
"""
import itertools, json, math, os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, pearsonr, spearmanr
from qiskit import qpy, QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli

V3, CAN, OUT = sys.argv[1:4]
R = f"{V3}/results"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261101)
DRAWS = 2000
chk = pd.read_csv(f"{V3}/probe_pairing_check.csv")
pairs = pd.read_csv(f"{V3}/probe_pairing_check_pairs.csv")
rec = json.load(open(f"{V3}/probe_pairing_check_record.json"))
chk["case"] = chk.instance + ":" + chk.depth.astype(str)
chk["graph"] = chk.instance
log = open(f"{OUT}/ablations.log", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n"); log.flush()

# ------------------------------------------------------------------ statistics
def frac_rank(x): return (rankdata(x) - 0.5) / len(x)
def rho(d, pred, y="rW"):
    xs, ys = [], []
    for _, g in d.groupby(["backend", "n"]):
        xs.append(frac_rank(g[pred].values)); ys.append(frac_rank(g[y].values))
    return pearsonr(np.concatenate(xs), np.concatenate(ys))[0]
def partial(d, pred, ctrl, y="rW"):
    """Registered definition: within each stratum (processor x size), regress the fractional ranks of the predictor
    and of the outcome on the ranks of the control(s); correlate the pooled residuals."""
    ctrls = [ctrl] if isinstance(ctrl, str) else list(ctrl)
    xr, yr = [], []
    for _, g in d.groupby(["backend", "n"]):
        x = frac_rank(g[pred].values); yy = frac_rank(g[y].values)
        X = np.column_stack([np.ones(len(g))] + [frac_rank(g[c].values) for c in ctrls])
        xr.append(x - X @ np.linalg.lstsq(X, x, rcond=None)[0]); yr.append(yy - X @ np.linalg.lstsq(X, yy, rcond=None)[0])
    return pearsonr(np.concatenate(xr), np.concatenate(yr))[0]
def case_boot(d, fn, draws=DRAWS, unit="case"):
    units = sorted(d[unit].unique()); idx = {u: d.index[d[unit] == u].values for u in units}
    vals = []
    for _ in range(draws):
        pick = rng.integers(0, len(units), len(units)); ii = np.concatenate([idx[units[i]] for i in pick])
        vals.append(fn(d.loc[ii]))
    return np.array(vals)
def iv(v): return tuple(np.quantile(v, [0.05, 0.95]))

# ------------------------------------------------------------------ A. variants of the statistic
# ideal correlators x_e of the QAOA output state on the readout pairs (statevector simulation on the active qubits)
def ideal_correlators(be, src):
    m = json.load(open(f"{R}/window4_v2/ibm_{be}_window4_v2_manifest.json")); ents = m["entries"]
    with open(f"{R}/window4_v2/ibm_{be}_window4_v2.qpy", "rb") as f: circs = qpy.load(f)
    return m, ents, circs
X_E = {}
for be in ("fez", "kingston"):
    m, ents, circs = ideal_correlators(be, None)
    for src in sorted({e["source_index"] for e in ents}):
        w = [e for e in ents if e["kind"] == "workload" and e["source_index"] == src][0]
        qc = circs[w["circuit_index"]]; r = rec[f"ibm_{be}:{src}"]; active = r["active"]; idx = {q: k for k, q in enumerate(active)}; n = len(active)
        sub = QuantumCircuit(n)
        for inst in qc.data:
            nm = inst.operation.name
            if nm in ("measure", "barrier", "delay"): continue
            qs = [idx[qc.find_bit(q).index] for q in inst.qubits]
            if nm == "rz": sub.rz(float(inst.operation.params[0]), qs[0])
            elif nm == "sx": sub.sx(qs[0])
            elif nm == "x": sub.x(qs[0])
            elif nm == "cz": sub.cz(*qs)
            else: raise ValueError(nm)
        sv = Statevector(sub)
        for (u, v, wt) in r["edges"]:
            a, b = r["readout"][str(u)], r["readout"][str(v)]
            lab = ["I"] * n; lab[a] = "Z"; lab[b] = "Z"
            X_E[(f"ibm_{be}", src, int(u), int(v))] = float(np.real(sv.expectation_value(Pauli("".join(reversed(lab))))))
pairs["x_e"] = [X_E[(b, s, u, v)] for b, s, u, v in zip(pairs.backend, pairs.source, pairs.u, pairs.v)]
pairs.to_csv(f"{OUT}/edge_table_with_ideal_correlators.csv", index=False)

def per_source_stats(pairs_df):
    out = {}
    for (b, s), g in pairs_df.groupby(["backend", "source"]):
        g = g[g.t_true.notna()]
        w = g.weight.values; t = g.t_true.values; x = g.x_e.values
        out[(b, s)] = dict(T_E=np.sum(w * t) / np.sum(w), T_unw=np.mean(t), T_x=np.sum(w * x * t) / np.sum(w * x))
    return out
ps = per_source_stats(pairs)
chk["T_E_ols"] = [ps[(b, s)]["T_E"] for b, s in zip(chk.backend, chk.source)]
assert np.allclose(chk.T_E_ols, chk.T_E_true_pairs), "T_E mismatch between tables"
chk["T_x"] = [ps[(b, s)]["T_x"] for b, s in zip(chk.backend, chk.source)]
chk["T_nonedge"] = np.nan
for key, r in rec.items():
    be, src = key.split(":"); n = r["n"]
    cost = {tuple(sorted((r["readout"][str(u)], r["readout"][str(v)]))) for u, v, _ in r["edges"]}
    vals = []
    for a, b in itertools.combinations(range(n), 2):
        if (a, b) in cost: continue
        S, M = [], []
        for k in r["circuits"]:
            s_ = r["ideal"][f"{k},{a},{b}"]
            if abs(s_) > 0.5: S.append(s_); M.append(r["meas"][f"{k},{a},{b}"])
        if S and min(S) < 0 < max(S): vals.append(np.polyfit(S, M, 1)[0])
    chk.loc[(chk.backend == be) & (chk.source == int(src)), "T_nonedge"] = np.mean(vals) if vals else np.nan
P("x_e: mean %.3f, fraction negative %.2f, |x_e|<0.05: %d of %d" % (pairs.x_e.mean(), (pairs.x_e < 0).mean(), (pairs.x_e.abs() < 0.05).sum(), len(pairs)))

variants = [("T_E_true_pairs", "Cost-weighted T_E (recovered, OLS)"), ("T_E_true_unweighted", "Unweighted mean of the same edge transfers"),
            ("T_E_registered", "Registered statistic (registered pairs)"), ("T_allpairs", "Mean transfer over all output pairs"),
            ("T_nonedge", "Mean transfer over output pairs that are not cost edges"), ("T_x", "Ideal-correlator-weighted statistic")]
rowsA = []
base_boot = case_boot(chk, lambda d: rho(d, "T_E_true_pairs"))
base_pboot = case_boot(chk, lambda d: partial(d, "T_E_true_pairs", "neg_gate"))
for col, lab in variants:
    d = chk[chk[col].notna()]
    r0, p0 = rho(d, col), partial(d, col, "neg_gate")
    bb = case_boot(d, lambda dd: rho(dd, col)); pb = case_boot(d, lambda dd: partial(dd, col, "neg_gate"))
    # paired difference from T_E on the same draws: redo draws jointly
    dif = []; pdif = []
    units = sorted(d.case.unique()); idx = {u: d.index[d.case == u].values for u in units}
    for _ in range(DRAWS):
        pick = rng.integers(0, len(units), len(units)); ii = np.concatenate([idx[units[i]] for i in pick]); dd = d.loc[ii]
        dif.append(rho(dd, col) - rho(dd, "T_E_true_pairs")); pdif.append(partial(dd, col, "neg_gate") - partial(dd, "T_E_true_pairs", "neg_gate"))
    rowsA.append(dict(variant=lab, n=len(d), rho=r0, rho_lo=iv(bb)[0], rho_hi=iv(bb)[1], partial=p0, partial_lo=iv(pb)[0], partial_hi=iv(pb)[1],
                      diff_rho=r0 - rho(d, "T_E_true_pairs"), diff_rho_lo=iv(dif)[0], diff_rho_hi=iv(dif)[1],
                      diff_partial=p0 - partial(d, "T_E_true_pairs", "neg_gate"), diff_partial_lo=iv(pdif)[0], diff_partial_hi=iv(pdif)[1]))
    P(f"{lab:58s} n={len(d)} rho {r0:.3f} [{iv(bb)[0]:.3f},{iv(bb)[1]:.3f}] partial {p0:.3f} [{iv(pb)[0]:.3f},{iv(pb)[1]:.3f}]  diff from T_E: rho {rowsA[-1]['diff_rho']:+.3f} [{iv(dif)[0]:+.3f},{iv(dif)[1]:+.3f}] partial {rowsA[-1]['diff_partial']:+.3f} [{iv(pdif)[0]:+.3f},{iv(pdif)[1]:+.3f}]")
# complete-coverage subset
d63 = chk[chk.weight_fraction_estimable >= 0.999]
P(f"complete-coverage subset: n={len(d63)} rho {rho(d63,'T_E_true_pairs'):.3f} partial {partial(d63,'T_E_true_pairs','neg_gate'):.3f}; registered on same subset rho {rho(d63,'T_E_registered'):.3f}")
rowsA.append(dict(variant="Cost-weighted T_E, complete-coverage circuits only", n=len(d63), rho=rho(d63, "T_E_true_pairs"), partial=partial(d63, "T_E_true_pairs", "neg_gate")))
# permuted weights
perm_rho = []; perm_part = []
grp = {k: g.index.values for k, g in pairs[pairs.t_true.notna()].groupby(["backend", "source"])}
for _ in range(1000):
    te = {}
    for (b, s), ii in grp.items():
        w = pairs.loc[ii, "weight"].values; t = pairs.loc[ii, "t_true"].values; wp = rng.permutation(w)
        te[(b, s)] = np.sum(wp * t) / np.sum(wp)
    dd = chk.copy(); dd["T_perm"] = [te[(b, s)] for b, s in zip(dd.backend, dd.source)]
    perm_rho.append(rho(dd, "T_perm")); perm_part.append(partial(dd, "T_perm", "neg_gate"))
perm_rho = np.array(perm_rho); perm_part = np.array(perm_part)
P(f"permuted weights (1000): rho median {np.median(perm_rho):.3f} [{iv(perm_rho)[0]:.3f},{iv(perm_rho)[1]:.3f}], fraction >= actual {np.mean(perm_rho >= rho(chk,'T_E_true_pairs')):.3f}; partial median {np.median(perm_part):.3f} [{iv(perm_part)[0]:.3f},{iv(perm_part)[1]:.3f}], fraction >= actual {np.mean(perm_part >= partial(chk,'T_E_true_pairs','neg_gate')):.3f}")
rowsA.append(dict(variant="Cost-weighted T_E with permuted weights (median of 1,000 permutations)", n=72, rho=float(np.median(perm_rho)), rho_lo=iv(perm_rho)[0], rho_hi=iv(perm_rho)[1],
                  partial=float(np.median(perm_part)), partial_lo=iv(perm_part)[0], partial_hi=iv(perm_part)[1]))
pd.DataFrame(rowsA).to_csv(f"{OUT}/ablation_variants.csv", index=False)
# paired differences T_E minus gate score and minus single-edge control
for other, lab in [("neg_gate", "gate score"), ("a_PT", "single-edge control"), ("T_E_registered", "registered statistic"), ("neg_full", "full score")]:
    dif = case_boot(chk, lambda d: rho(d, "T_E_true_pairs") - rho(d, other))
    P(f"paired difference rho(T_E) - rho({lab}): {rho(chk,'T_E_true_pairs')-rho(chk,other):+.3f} [{iv(dif)[0]:+.3f},{iv(dif)[1]:+.3f}]")

# ------------------------------------------------------------------ B. partial given full score and both
for ctrl, lab in [("neg_gate", "gate score"), ("neg_full", "full score"), (["neg_gate", "neg_full"], "gate and full scores")]:
    b = case_boot(chk, lambda d: partial(d, "T_E_true_pairs", ctrl))
    P(f"partial rho of T_E given {lab}: {partial(chk,'T_E_true_pairs',ctrl):.3f} [{iv(b)[0]:.3f},{iv(b)[1]:.3f}]")
b = case_boot(chk, lambda d: partial(d, "neg_full", "T_E_true_pairs"))
P(f"partial rho of full score given T_E: {partial(chk,'neg_full','T_E_true_pairs'):.3f} [{iv(b)[0]:.3f},{iv(b)[1]:.3f}]")

# ------------------------------------------------------------------ C. graph-level sensitivity
gb = case_boot(chk, lambda d: rho(d, "T_E_true_pairs"), unit="graph"); gpb = case_boot(chk, lambda d: partial(d, "T_E_true_pairs", "neg_gate"), unit="graph")
P(f"graph-level bootstrap (9 graphs): rho {rho(chk,'T_E_true_pairs'):.3f} [{iv(gb)[0]:.3f},{iv(gb)[1]:.3f}]; partial [{iv(gpb)[0]:.3f},{iv(gpb)[1]:.3f}]")
loo = [(g, rho(chk[chk.graph != g], "T_E_true_pairs"), partial(chk[chk.graph != g], "T_E_true_pairs", "neg_gate")) for g in sorted(chk.graph.unique())]
P("leave-one-graph-out: rho min %.3f max %.3f; partial min %.3f max %.3f" % (min(x[1] for x in loo), max(x[1] for x in loo), min(x[2] for x in loo), max(x[2] for x in loo)))
pd.DataFrame(loo, columns=["graph_left_out", "rho", "partial"]).to_csv(f"{OUT}/leave_one_graph_out.csv", index=False)

# ------------------------------------------------------------------ D. probe shot resampling
probe_counts = {}
for be in ("fez", "kingston"):
    c = json.load(open(f"{R}/window4_v2/runs/ibm_{be}/primary/completed.json"))["counts"]; probe_counts[f"ibm_{be}"] = c
# precompute per source: for each probe circuit k, bitstrings, counts, and parity matrix for the readout pairs of the edges
pre = {}
for key, r in rec.items():
    be, src = key.split(":"); n = r["n"]
    epairs = [tuple(sorted((r["readout"][str(u)], r["readout"][str(v)]))) for u, v, _ in r["edges"]]
    wts = np.array([wt for _, _, wt in r["edges"]])
    circ = []
    for k in r["circuits"]:
        cnt = probe_counts[be][k]; bs = list(cnt.keys()); cvec = np.array([cnt[b_] for b_ in bs], float); tot = cvec.sum()
        par = np.array([[1 if (int(b_[::-1][a]) + int(b_[::-1][b])) % 2 == 0 else -1 for b_ in bs] for a, b in epairs], float)  # edges x bitstrings
        sgn = np.array([r["ideal"][f"{k},{a},{b}"] for a, b in epairs])
        circ.append(dict(p=cvec / tot, tot=int(tot), par=par, sgn=sgn))
    pre[key] = dict(wts=wts, circ=circ, n_e=len(epairs))
def te_from_draw(key, draw):
    """T_E of one source from resampled count vectors (draw: list of count vectors per probe circuit)."""
    pr = pre[key]; n_e = pr["n_e"]; S = [[] for _ in range(n_e)]; M = [[] for _ in range(n_e)]
    for c, cv in zip(pr["circ"], draw):
        m = c["par"] @ cv / c["tot"]
        for e in range(n_e):
            if abs(c["sgn"][e]) > 0.5: S[e].append(c["sgn"][e]); M[e].append(m[e])
    t = np.full(n_e, np.nan)
    for e in range(n_e):
        if S[e] and min(S[e]) < 0 < max(S[e]): t[e] = np.polyfit(S[e], M[e], 1)[0]
    ok = ~np.isnan(t); return np.sum(pr["wts"][ok] * t[ok]) / np.sum(pr["wts"][ok])
# check: observed counts reproduce T_E
for key in list(pre)[:3]:
    obs = te_from_draw(key, [c["p"] * c["tot"] for c in pre[key]["circ"]])
    be, src = key.split(":"); assert abs(obs - chk[(chk.backend == be) & (chk.source == int(src))].T_E_true_pairs.iloc[0]) < 1e-9
NB = 1000
TE_draws = np.zeros((NB, len(chk)))
keys = [f"{b}:{s}" for b, s in zip(chk.backend, chk.source)]
for j, key in enumerate(keys):
    samp = [rng.multinomial(c["tot"], c["p"], size=NB) for c in pre[key]["circ"]]
    for i in range(NB):
        TE_draws[i, j] = te_from_draw(key, [s_[i] for s_ in samp])
te_sd = TE_draws.std(axis=0)
P(f"probe count resampling: median SD of T_E per circuit {np.median(te_sd):.4f} (range {te_sd.min():.4f}-{te_sd.max():.4f}); SD of r_W (count) median {chk_rw_se:.4f}" if False else f"probe count resampling: median SD of T_E per circuit {np.median(te_sd):.4f} (range {te_sd.min():.4f} to {te_sd.max():.4f})")
count_only = []; combined = []
units = sorted(chk.case.unique()); idx = {u: chk.index[chk.case == u].values for u in units}
for i in range(NB):
    dd = chk.copy(); dd["T_E_draw"] = TE_draws[i]; count_only.append(rho(dd, "T_E_draw"))
    pick = rng.integers(0, len(units), len(units)); ii = np.concatenate([idx[units[k]] for k in pick]); combined.append(rho(dd.loc[ii], "T_E_draw"))
P(f"rho(T_E, r_W) with probe-count resampling only: [{iv(count_only)[0]:.3f},{iv(count_only)[1]:.3f}]; probe counts and cases: [{iv(combined)[0]:.3f},{iv(combined)[1]:.3f}]")
pd.DataFrame(dict(backend=chk.backend, source=chk.source, T_E=chk.T_E_true_pairs, T_E_count_sd=te_sd)).to_csv(f"{OUT}/probe_count_sd.csv", index=False)

# ------------------------------------------------------------------ E. matched-cost direct workload sampling
def load_workload(win, run):
    out = {}
    for be in ("fez", "kingston"):
        m = json.load(open(f"{R}/{win}/ibm_{be}_{win}_manifest.json")); ents = m["entries"]
        c = json.load(open(f"{R}/{win}/runs/ibm_{be}/{run}/completed.json"))["counts"]
        for k, e in enumerate(ents):
            if e["kind"] != "workload": continue
            cnt = c[k]; bs = list(cnt.keys()); cv = np.array([cnt[b_] for b_ in bs], float)
            edges = e["edges"]; nq = e["n_qubits"]
            cut = np.array([sum(wt for u, v, wt in edges if b_[::-1][int(u)] != b_[::-1][int(v)]) for b_ in bs]) / e["optimum"]
            out[(f"ibm_{be}", e["source_index"])] = dict(counts=cv, cut=cut, qr=e["random_approximation_ratio"], qi=e["ideal_approximation_ratio"], tot=int(cv.sum()))
    return out
W0 = load_workload("window4_v2", "primary"); W72 = load_workload("window5_v2", "day3_true")
def rw(cv, d): q = (cv @ d["cut"]) / cv.sum(); return (q - d["qr"]) / (d["qi"] - d["qr"]), q
sel = pd.read_csv(f"{CAN}/window4_selector_pairs.csv")
sel["key_top"] = list(zip(sel.backend, sel.instance, sel.depth));
src_of = {(b, i, int(dpt), s): int(src) for b, i, dpt, s, src in zip(chk.backend, chk.instance, chk.depth, chk.selector, chk.source)}
pairs_dec = []
for r_ in sel.itertuples():
    if bool(r_.identical): continue
    pairs_dec.append(dict(backend=r_.backend, instance=r_.instance, depth=int(r_.depth), resolved=bool(r_.resolved),
                          top=src_of[(r_.backend, r_.instance, int(r_.depth), "topology_only")], cal=src_of[(r_.backend, r_.instance, int(r_.depth), "calibration_scored")]))
sel72 = pd.read_csv(f"{V3}/canonical_corrected/day3_selector_pairs.csv")
res72 = {(b, i, int(dpt)): bool(r) for b, i, dpt, r in zip(sel72.backend, sel72.instance, sel72.depth, sel72.resolved)}
for p in pairs_dec: p["resolved72"] = res72[(p["backend"], p["instance"], p["depth"])]
P(f"decision pairs: {len(pairs_dec)} non-identical, {sum(p['resolved'] for p in pairs_dec)} resolved at t0, {sum(p['resolved72'] for p in pairs_dec)} resolved at 72.1 h")
te_of = {(b, s): v for b, s, v in zip(chk.backend, chk.source, chk.T_E_true_pairs)}
gate_of = {(b, s): v for b, s, v in zip(chk.backend, chk.source, chk.neg_gate)}
full_of = {(b, s): v for b, s, v in zip(chk.backend, chk.source, chk.neg_full)}
gate72 = {}; full72 = {}
M72 = pd.read_csv(f"{V3}/canonical_corrected/day3_matched_sources.csv")
for r_ in M72.itertuples(): gate72[(r_.backend, int(r_.source))] = float(r_.w5_neg_gate); full72[(r_.backend, int(r_.source))] = float(r_.w5_neg_full)
assert not any(np.isnan(v) for v in full72.values())
def regret(pick_cal, truth):
    return float(np.mean([max(-t, 0) if pc else max(t, 0) for pc, t in zip(pick_cal, truth)]))
def correct(pick_cal, truth, mask):
    return int(sum((pc == (t > 0)) for pc, t, m in zip(pick_cal, truth, mask) if m)), int(sum(mask))
rowsE = []
budgets = [256, 512, 768, 1024]
NDRAW = 400
for m_shots in budgets:
    acc = {k: [] for k in ["rho_same", "rho_same_TE", "rho_same_gate", "rho_72", "reg_same_sub", "reg_same_TE", "reg_same_gate", "reg_same_cal", "reg_72_sub", "reg_72_TE", "reg_72_gate", "reg_72_full", "reg_72_w2048",
                           "cor_same_sub", "cor_same_TE", "cor_same_gate", "cor_72_sub", "cor_72_TE", "cor_72_gate", "cor_72_w2048"]}
    for _ in range(NDRAW):
        rsub, rcomp, r72 = {}, {}, {}
        for key, d in W0.items():
            cv = d["counts"]; sub = rng.multivariate_hypergeometric(cv.astype(int), m_shots).astype(float); comp = cv - sub
            rsub[key] = rw(sub, d)[0]; rcomp[key] = rw(comp, d)[0]; r72[key] = rw(W72[key]["counts"], W72[key])[0]
        dd = chk.copy(); dd["r_sub"] = [rsub[(b, s)] for b, s in zip(dd.backend, dd.source)]; dd["r_comp"] = [rcomp[(b, s)] for b, s in zip(dd.backend, dd.source)]; dd["r_72"] = [r72[(b, s)] for b, s in zip(dd.backend, dd.source)]
        acc["rho_same"].append(rho(dd, "r_sub", "r_comp")); acc["rho_same_TE"].append(rho(dd, "T_E_true_pairs", "r_comp")); acc["rho_same_gate"].append(rho(dd, "neg_gate", "r_comp")); acc["rho_72"].append(rho(dd, "r_sub", "r_72"))
        truth_same = [rcomp[(p["backend"], p["cal"])] - rcomp[(p["backend"], p["top"])] for p in pairs_dec]
        truth_72 = [r72[(p["backend"], p["cal"])] - r72[(p["backend"], p["top"])] for p in pairs_dec]
        pick_sub = [rsub[(p["backend"], p["cal"])] > rsub[(p["backend"], p["top"])] for p in pairs_dec]
        pick_TE = [te_of[(p["backend"], p["cal"])] > te_of[(p["backend"], p["top"])] for p in pairs_dec]
        pick_gate = [gate_of[(p["backend"], p["cal"])] > gate_of[(p["backend"], p["top"])] for p in pairs_dec]
        pick_gate72 = [gate72[(p["backend"], p["cal"])] > gate72[(p["backend"], p["top"])] for p in pairs_dec]
        pick_full72 = [full72[(p["backend"], p["cal"])] > full72[(p["backend"], p["top"])] for p in pairs_dec]
        pick_w2048 = [rw(W0[(p["backend"], p["cal"])]["counts"], W0[(p["backend"], p["cal"])])[0] > rw(W0[(p["backend"], p["top"])]["counts"], W0[(p["backend"], p["top"])])[0] for p in pairs_dec]
        mask = [p["resolved"] for p in pairs_dec]; mask72 = [p["resolved72"] for p in pairs_dec]
        acc["reg_same_sub"].append(regret(pick_sub, truth_same)); acc["reg_same_TE"].append(regret(pick_TE, truth_same)); acc["reg_same_gate"].append(regret(pick_gate, truth_same)); acc["reg_same_cal"].append(regret([True] * len(pairs_dec), truth_same))
        acc["reg_72_sub"].append(regret(pick_sub, truth_72)); acc["reg_72_TE"].append(regret(pick_TE, truth_72)); acc["reg_72_gate"].append(regret(pick_gate72, truth_72)); acc["reg_72_full"].append(regret(pick_full72, truth_72)); acc["reg_72_w2048"].append(regret(pick_w2048, truth_72))
        acc["cor_same_sub"].append(correct(pick_sub, truth_same, mask)[0]); acc["cor_same_TE"].append(correct(pick_TE, truth_same, mask)[0]); acc["cor_same_gate"].append(correct(pick_gate, truth_same, mask)[0])
        acc["cor_72_sub"].append(correct(pick_sub, truth_72, mask72)[0]); acc["cor_72_TE"].append(correct(pick_TE, truth_72, mask72)[0]); acc["cor_72_gate"].append(correct(pick_gate72, truth_72, mask72)[0]); acc["cor_72_w2048"].append(correct(pick_w2048, truth_72, mask72)[0])
    row = dict(shots=m_shots, **{k: float(np.mean(v)) for k, v in acc.items()}, **{k + "_lo": float(np.quantile(v, 0.05)) for k, v in acc.items() if k.startswith(("rho", "reg"))}, **{k + "_hi": float(np.quantile(v, 0.95)) for k, v in acc.items() if k.startswith(("rho", "reg"))})
    rowsE.append(row)
    P(f"budget {m_shots:4d} shots: same-job rho(sub, complement) {row['rho_same']:.3f} [{row['rho_same_lo']:.3f},{row['rho_same_hi']:.3f}] (T_E vs complement {row['rho_same_TE']:.3f}, gate vs complement {row['rho_same_gate']:.3f}); regret vs complement: sub {row['reg_same_sub']:.4f}, T_E {row['reg_same_TE']:.4f}, gate {row['reg_same_gate']:.4f}, always-cal {row['reg_same_cal']:.4f}; correct/{sum(mask)}: sub {row['cor_same_sub']:.1f}, T_E {row['cor_same_TE']:.1f}, gate {row['cor_same_gate']:.1f}")
    P(f"                   72 h: rho(sub, r_W at 72 h) {row['rho_72']:.3f} [{row['rho_72_lo']:.3f},{row['rho_72_hi']:.3f}]; regret at 72 h: sub {row['reg_72_sub']:.4f}, stale T_E {row['reg_72_TE']:.4f}, current gate {row['reg_72_gate']:.4f}, current full {row['reg_72_full']:.4f}, stale 2048-shot workload {row['reg_72_w2048']:.4f}; correct/{sum(mask72)}: sub {row['cor_72_sub']:.1f}, T_E {row['cor_72_TE']:.1f}, gate {row['cor_72_gate']:.1f}, w2048 {row['cor_72_w2048']:.1f}")
pd.DataFrame(rowsE).to_csv(f"{OUT}/matched_cost_direct_sampling.csv", index=False)
# reference values with full counts (no subsampling) for rho of T_E and gate with r_W at 72 h
dd = chk.copy(); dd["r_72"] = [rw(W72[(b, s)]["counts"], W72[(b, s)])[0] for b, s in zip(dd.backend, dd.source)]
dd["gate72"] = [gate72[(b, s)] for b, s in zip(dd.backend, dd.source)]
P(f"for reference: rho(stale T_E, r_W at 72 h) {rho(dd,'T_E_true_pairs','r_72'):.3f}; rho(current gate, r_W 72 h) {rho(dd,'gate72','r_72'):.3f}; rho(stale 2048-shot r_W, r_W 72 h) {rho(dd,'rW','r_72'):.3f}")

# ------------------------------------------------------------------ F. layout-order reversals between t0 and 72 h
rev = []
for p in pairs_dec:
    d0 = rw(W0[(p["backend"], p["cal"])]["counts"], W0[(p["backend"], p["cal"])])[0] - rw(W0[(p["backend"], p["top"])]["counts"], W0[(p["backend"], p["top"])])[0]
    d72 = rw(W72[(p["backend"], p["cal"])]["counts"], W72[(p["backend"], p["cal"])])[0] - rw(W72[(p["backend"], p["top"])]["counts"], W72[(p["backend"], p["top"])])[0]
    rev.append(dict(backend=p["backend"], instance=p["instance"], depth=p["depth"], resolved_t0=p["resolved"], delta_t0=d0, delta_72=d72, reversed=(d0 > 0) != (d72 > 0),
                    stale_TE_pick_cal=te_of[(p["backend"], p["cal"])] > te_of[(p["backend"], p["top"])], current_gate_pick_cal=gate72[(p["backend"], p["cal"])] > gate72[(p["backend"], p["top"])]))
rev = pd.DataFrame(rev); rev.to_csv(f"{OUT}/layout_order_reversals.csv", index=False)
rr = rev[rev.reversed]
P(f"reversals of the measured layout order between t0 and 72.1 h: {len(rr)} of {len(rev)} non-identical pairs; |delta| at t0 of reversed pairs: {sorted(rr.delta_t0.abs().round(3).tolist())}; resolved at t0 among them: {int(rr.resolved_t0.sum())}")
P(f"on reversed pairs, current gate score correct {int(((rr.current_gate_pick_cal) == (rr.delta_72 > 0)).sum())}, stale T_E correct {int(((rr.stale_TE_pick_cal) == (rr.delta_72 > 0)).sum())}")
log.close()
print("written", OUT)
