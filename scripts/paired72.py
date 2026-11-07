"""Paired case-bootstrap differences at 72.1 hours: rank correlations of the stale probe against the other predictors,
and mean regret of the decision rules.  usage: python3 -I paired72.py <canonical_corrected dir> <out dir>
Same definitions and seed convention as ablations.py (registered stratified rank correlation; case = instance x depth).
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, pearsonr

CAN, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261101)
DRAWS = 2000
log = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)
def frac_rank(x): return (rankdata(x) - 0.5) / len(x)
def rho(d, pred, y):
    xs, ys = [], []
    for _, g in d.groupby(["backend", "n"]):
        xs.append(frac_rank(g[pred].values)); ys.append(frac_rank(g[y].values))
    return pearsonr(np.concatenate(xs), np.concatenate(ys))[0]
def iv(v): return tuple(np.quantile(v, [0.05, 0.95]))
def fmt(point, v): lo, hi = iv(v); return f"{point:+.3f} [{lo:+.3f}, {hi:+.3f}] (bootstrap mean {np.mean(v):+.3f})"

# ---- predictors at 72.1 h
m = pd.read_csv(f"{CAN}/day3_matched_sources.csv")
m["case"] = m.instance + ":" + m.depth.astype(str)
chk = pd.read_csv(f"{CAN}/../probe_pairing_check.csv")
rec = {(b, int(src)): v for b, src, v in zip(chk.backend, chk.source, chk.T_E_true_pairs)}
m["T_E_rec"] = [rec[(b, int(src))] for b, src in zip(m.backend, m.source)]
preds = {"stale probe T_E": "T_E_rec", "stale registered T_E": "w4_T_E", "stale control a_PT": "w4_a_PT", "stale workload r_W": "w4_rW",
         "current gate score": "w5_neg_gate", "current full score": "w5_neg_full"}
P("point estimates, rho with r_W at 72.1 h:")
for k, c in preds.items(): P(f"  {k:22s} {rho(m, c, 'w5_rW'):.3f}")
cases = sorted(m.case.unique()); idx = {u: m.index[m.case == u].values for u in cases}
diffs = {k: [] for k in preds if k != "stale probe T_E"}
for _ in range(DRAWS):
    pick = rng.integers(0, len(cases), len(cases)); d = m.loc[np.concatenate([idx[cases[i]] for i in pick])]
    r0 = rho(d, "T_E_rec", "w5_rW")
    for k, c in preds.items():
        if k != "stale probe T_E": diffs[k].append(r0 - rho(d, c, "w5_rW"))
P("paired case-bootstrap differences, rho(stale probe) - rho(other), 72.1 h; point estimate = observed difference, interval = 5th to 95th percentile of the paired bootstrap differences:")
rows = []
r_obs = rho(m, "T_E_rec", "w5_rW")
for k, v in diffs.items():
    point = r_obs - rho(m, preds[k], "w5_rW")
    P(f"  minus {k:22s} {fmt(point, np.array(v))}"); lo, hi = iv(v); rows.append(dict(comparison=k, diff=point, boot_mean=np.mean(v), lo=lo, hi=hi))
pd.DataFrame(rows).to_csv(f"{OUT}/paired_rho_72h.csv", index=False)

# ---- decision rules at 72.1 h
s = pd.read_csv(f"{CAN}/day3_selector_pairs.csv")
s = s[~s.identical.astype(bool)].copy(); s["case"] = s.instance + ":" + s.depth.astype(str)
truth = s.delta_w5_rW.values
src_of = {(b, i, int(dpt), sel): int(src) for b, i, dpt, sel, src in zip(chk.backend, chk.instance, chk.depth, chk.selector, chk.source)}
d_rec = np.array([rec[(b, src_of[(b, i, int(dpt), "calibration_scored")])] - rec[(b, src_of[(b, i, int(dpt), "topology_only")])] for b, i, dpt in zip(s.backend, s.instance, s.depth)])
chk_reg = {(b, int(src)): v for b, src, v in zip(chk.backend, chk.source, chk.T_E_reg)} if "T_E_reg" in chk.columns else None
if chk_reg is not None:
    d_reg_chk = np.array([chk_reg[(b, src_of[(b, i, int(dpt), "calibration_scored")])] - chk_reg[(b, src_of[(b, i, int(dpt), "topology_only")])] for b, i, dpt in zip(s.backend, s.instance, s.depth)])
    P("check: registered deltas from probe_pairing_check agree with day3_selector_pairs:", np.allclose(d_reg_chk, s.delta_stale_w4_T_E.values, atol=1e-9))
rules = {"stale workload": s.delta_stale_w4_rW.values > 0, "current full score": s.delta_fresh_w5_full.values > 0,
         "always calibration-selected": np.ones(len(s), bool), "stale probe T_E": d_rec > 0, "stale registered T_E": s.delta_stale_w4_T_E.values > 0,
         "current gate score": s.delta_fresh_w5_gate.values > 0, "stale control": s.delta_stale_w4_a_PT.values > 0,
         "always topology-selected": np.zeros(len(s), bool)}
def regret(pick, t): return float(np.mean(np.where(pick, np.maximum(-t, 0), np.maximum(t, 0))))
P(f"decision pairs at 72.1 h: {len(s)} non-identical, {int(s.resolved.sum())} resolved; mean regret and correct among resolved:")
for k, pk in rules.items():
    cor = int(np.sum((pk == (truth > 0)) & s.resolved.values)); P(f"  {k:28s} {regret(pk, truth):.4f}  {cor}/{int(s.resolved.sum())}")
cases = sorted(s.case.unique()); idx = {u: np.where(s.case.values == u)[0] for u in cases}
ref = "stale probe T_E"; rd = {k: [] for k in rules if k != ref}
for _ in range(DRAWS):
    pick = rng.integers(0, len(cases), len(cases)); ii = np.concatenate([idx[cases[i]] for i in pick])
    r0 = regret(rules[ref][ii], truth[ii])
    for k in rd: rd[k].append(regret(rules[k][ii], truth[ii]) - r0)
P("paired case-bootstrap differences of mean regret, rule - stale probe, 72.1 h (negative favours the rule); point estimate = observed difference, interval = percentiles of the paired bootstrap differences:")
rows = []
reg_obs = regret(rules[ref], truth)
for k, v in rd.items():
    v = np.array(v); point = regret(rules[k], truth) - reg_obs
    P(f"  {k:28s} {point:+.4f} [{iv(v)[0]:+.4f}, {iv(v)[1]:+.4f}] (bootstrap mean {np.mean(v):+.4f})"); rows.append(dict(rule=k, diff=point, boot_mean=np.mean(v), lo=iv(v)[0], hi=iv(v)[1]))
pd.DataFrame(rows).to_csv(f"{OUT}/paired_regret_72h.csv", index=False)
open(f"{OUT}/paired72.log", "w").write("\n".join(log) + "\n")

# ---- decision rules in the reference job (t = 0), same convention
s0 = pd.read_csv(f"{CAN}/window4_selector_pairs.csv")
s0 = s0[~s0.identical.astype(bool)].copy(); s0["case"] = s0.instance + ":" + s0.depth.astype(str)
truth0 = s0.delta_rW.values
d_rec0 = np.array([rec[(b, src_of[(b, i, int(dpt), "calibration_scored")])] - rec[(b, src_of[(b, i, int(dpt), "topology_only")])] for b, i, dpt in zip(s0.backend, s0.instance, s0.depth)])
rules0 = {"gate score": s0.delta_neg_gate.values > 0, "all-edge probe T_E": d_rec0 > 0, "registered T_E": s0.delta_T_E.values > 0,
          "always calibration-selected": np.ones(len(s0), bool), "full score": s0.delta_neg_full.values > 0, "single-edge control": s0.delta_a_PT.values > 0,
          "always topology-selected": np.zeros(len(s0), bool)}
P(f"decision pairs in the reference job: {len(s0)} non-identical, {int(s0.resolved.sum())} resolved; mean regret and correct among resolved:")
for k, pk in rules0.items():
    cor = int(np.sum((pk == (truth0 > 0)) & s0.resolved.values)); P(f"  {k:28s} {regret(pk, truth0):.4f}  {cor}/{int(s0.resolved.sum())}")
cases0 = sorted(s0.case.unique()); idx0 = {u: np.where(s0.case.values == u)[0] for u in cases0}
ref0 = "all-edge probe T_E"; rd0 = {k: [] for k in rules0 if k != ref0}
for _ in range(DRAWS):
    pick = rng.integers(0, len(cases0), len(cases0)); ii = np.concatenate([idx0[cases0[i]] for i in pick])
    r0 = regret(rules0[ref0][ii], truth0[ii])
    for k in rd0: rd0[k].append(regret(rules0[k][ii], truth0[ii]) - r0)
P("paired case-bootstrap differences of mean regret, rule - all-edge probe, reference job (negative favours the rule); point estimate = observed difference:")
rows = []
reg_obs0 = regret(rules0[ref0], truth0)
for k, v in rd0.items():
    v = np.array(v); point = regret(rules0[k], truth0) - reg_obs0
    P(f"  {k:28s} {point:+.4f} [{iv(v)[0]:+.4f}, {iv(v)[1]:+.4f}] (bootstrap mean {np.mean(v):+.4f})"); rows.append(dict(rule=k, diff=point, boot_mean=np.mean(v), lo=iv(v)[0], hi=iv(v)[1]))
pd.DataFrame(rows).to_csv(f"{OUT}/paired_regret_t0.csv", index=False)
open(f"{OUT}/paired72.log", "w").write("\n".join(log) + "\n")
