"""Supplementary tables for the ablations, the matched-cost comparison, the sensitivity analyses and the noise-model
simulation.  usage: python3 -I make_extra_tables.py <ablation_out dir> <sim dirs (comma separated)> <out tables dir>
"""
import glob, os, sys
import numpy as np, pandas as pd

AB, SIMS, OUT = sys.argv[1:4]
def num(x, nd=3):
    return ("$-$" + f"{-x:.{nd}f}") if x < 0 else f"{x:.{nd}f}"
def iv(lo, hi, nd=3): return f"[{num(lo, nd)}, {num(hi, nd)}]"
def table(name, colspec, header, rows, caption, label, wide=True):
    out = ["\\begin{table}[H]", "\\centering", "\\small"] + (["\\resizebox{\\textwidth}{!}{%"] if wide else []) + [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    out += [r + " \\\\" for r in rows]
    out += ["\\bottomrule", "\\end{tabular}"] + (["}"] if wide else []) + [f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table}"]
    open(os.path.join(OUT, name), "w").write("\n".join(out) + "\n")

# ---- ablation variants
v = pd.read_csv(f"{AB}/ablation_variants.csv")
names = {"Cost-weighted T_E (recovered, OLS)": "Cost-weighted $\\TE$ (recovered statistic)", "Unweighted mean of the same edge transfers": "Unweighted mean of the same edge transfers",
         "Registered statistic (registered pairs)": "Registered statistic $\\TEreg$ (registered pairs)", "Mean transfer over all output pairs": "Mean transfer over all estimable output pairs of the circuit",
         "Mean transfer over output pairs that are not cost edges": "Mean transfer over estimable output pairs that are not cost edges", "Ideal-correlator-weighted statistic": "Ideal-correlator-weighted statistic",
         "Cost-weighted T_E, complete-coverage circuits only": "$\\TE$, circuits with complete coverage only", "Cost-weighted T_E with permuted weights (median of 1,000 permutations)": "$\\TE$ with permuted weights (median over 1,000 permutations)"}
rows = []
for r in v.itertuples():
    rho = f"{r.rho:.3f}" + (f" {iv(r.rho_lo, r.rho_hi)}" if not np.isnan(r.rho_lo) else "")
    par = f"{r.partial:.3f}" + (f" {iv(r.partial_lo, r.partial_hi)}" if not np.isnan(r.partial_lo) else "")
    ref = r.variant.startswith("Cost-weighted T_E (recovered")
    d1 = "--" if ref or np.isnan(r.diff_rho) else f"{num(r.diff_rho)} {iv(r.diff_rho_lo, r.diff_rho_hi)}"
    d2 = "--" if ref or np.isnan(r.diff_partial) else f"{num(r.diff_partial)} {iv(r.diff_partial_lo, r.diff_partial_hi)}"
    rows.append(f"{names.get(r.variant, r.variant)} & {int(r.n)} & {rho} & {par} & {d1} & {d2}")
table("table_ablation_variants.tex", "p{5.6cm}rllll", "Variant & $n$ & $\\rho$ [90\\%] & Partial given gate score [90\\%] & $\\Delta\\rho$ from $\\TE$ [90\\%] & $\\Delta$ partial [90\\%]", rows,
      "Variants of the probe statistic computed from the same executed circuits of the reference job. Rank correlation with the retained gain within processor and graph size, partial rank correlation given the gate score, and paired case-bootstrap differences from the cost-weighted statistic $\\TE$. For the permuted-weights row, the interval is the 5th to 95th percentile over 1,000 random permutations of the edge weights within each circuit. All intervals in this table come from one set of 2,000 case-bootstrap draws with its own random seed and therefore differ slightly from those of Table~1 of the main text.",
      "tab:ablation")

# ---- matched cost
m = pd.read_csv(f"{AB}/matched_cost_direct_sampling.csv")
rows = []
for r in m.itertuples():
    rows.append(f"{int(r.shots)} & {r.rho_same:.3f} {iv(r.rho_same_lo, r.rho_same_hi)} & {r.rho_same_TE:.3f} & {r.rho_same_gate:.3f} & {r.reg_same_sub:.4f} ({r.cor_same_sub:.1f}) & {r.reg_same_TE:.4f} ({r.cor_same_TE:.0f}) & {r.reg_same_gate:.4f} ({r.cor_same_gate:.0f})")
table("table_matched_cost_same_job.tex", "rllllll", "Shots & $\\rho$(sample, rest) [90\\%] & $\\rho$($\\TE$, rest) & $\\rho$(gate, rest) & Regret, sample & Regret, $\\TE$ & Regret, gate", rows,
      "Comparison with direct sampling at a similar average shot budget in the reference job. A subsample of the stated number of shots was drawn without replacement from the 2,048-shot workload counts of every source circuit and evaluated against the remaining shots of the same measurement (``rest''). $\\rho$ is the rank correlation of the predictor with the retained gain of the remaining shots; the regret of each rule is evaluated against the remaining shots, with the number of correct choices among the 19 resolved pairs in parentheses. Values are means over 400 random partitions of the fixed count record; the interval is the 5th to 95th percentile over partitions, conditional on the executed measurement.",
      "tab:matched-same", wide=False)
rows = []
for r in m.itertuples():
    rows.append(f"{int(r.shots)} & {r.rho_72:.3f} {iv(r.rho_72_lo, r.rho_72_hi)} & {r.reg_72_sub:.4f} ({r.cor_72_sub:.1f}) & {r.reg_72_TE:.4f} ({r.cor_72_TE:.0f}) & {r.reg_72_gate:.4f} ({r.cor_72_gate:.0f}) & {r.reg_72_full:.4f} & {r.reg_72_w2048:.4f} ({r.cor_72_w2048:.0f})")
table("table_matched_cost_72h.tex", "rllllll", "Shots & $\\rho$(sample, 72 h) [90\\%] & Sample & Stale $\\TE$ & Current gate & Current full & Stale 2,048", rows,
      "Comparison with direct sampling at a similar average shot budget, evaluated at 72.1 hours. The subsample of the reference-job workload counts is evaluated against the retained gain measured 72.1 hours later, as are the stale probe, the current gate and full scores and the stale 2,048-shot measurement; the last five columns give the mean regret of each rule, with the number of correct choices among the 20 pairs resolved at 72.1 hours in parentheses (for the subsample, the mean over draws). Values are means over 400 random partitions of the fixed count record of the reference job; the interval is the 5th to 95th percentile over partitions, conditional on the executed measurement. For reference, the stale probe reaches $\\rho = 0.869$, the current gate score 0.813 and the stale 2,048-shot measurement 0.857 against the same outcome.",
      "tab:matched-72", wide=False)

# ---- sensitivity (graph level, probe resampling) from the log
log = open(f"{AB}/ablations.log").read().splitlines()
def grab(prefix):
    return next(l for l in log if l.startswith(prefix))
gl = grab("graph-level bootstrap"); lo = grab("leave-one-graph-out"); pc = grab("probe count resampling"); pr = grab("rho(T_E, r_W) with probe-count")
import re
g_rho = re.search(r"rho ([\d.]+) \[([\d.]+),([\d.]+)\]; partial \[([\d.]+),([\d.]+)\]", gl).groups()
l_vals = re.search(r"rho min ([\d.]+) max ([\d.]+); partial min ([\d.]+) max ([\d.]+)", lo).groups()
p_sd = re.search(r"median SD of T_E per circuit ([\d.]+) \(range ([\d.]+) to ([\d.]+)\)", pc).groups()
p_iv = re.search(r"only: \[([\d.]+),([\d.]+)\]; probe counts and cases: \[([\d.]+),([\d.]+)\]", pr).groups()
rows = [f"Case bootstrap (18 cases), registered scheme & {g_rho[0]} [0.750, 0.934] & [0.387, 0.784]",
        f"Graph bootstrap (9 graphs) & {g_rho[0]} [{g_rho[1]}, {g_rho[2]}] & [{g_rho[3]}, {g_rho[4]}]",
        f"Leave one graph out (range over 9 omissions) & {l_vals[0]} to {l_vals[1]} & {l_vals[2]} to {l_vals[3]}",
        f"Probe shot resampling only & [{p_iv[0]}, {p_iv[1]}] & --",
        f"Probe shot resampling and case bootstrap & [{p_iv[2]}, {p_iv[3]}] & --"]
table("table_sensitivity.tex", "lll", "Resampling scheme & $\\rho(\\TE, \\rW)$ [90\\%] & Partial given gate score [90\\%]", rows,
      f"Sensitivity of the reference-job result to the resampling scheme. Probe shot resampling draws multinomial counts for every executed probe circuit (median standard deviation of $\\TE$ per circuit {p_sd[0]}, range {p_sd[1]} to {p_sd[2]}), refits every edge transfer and recomputes $\\TE$; the shots shared by the edges of a group are preserved.",
      "tab:sensitivity", wide=False)

# ---- simulation
labels = dict(pauli="Pauli (depolarizing $10^{-3}$, $10^{-2}$; readout 2\\%)", damp="Amplitude damping ($2\\times10^{-3}$, $4\\times10^{-3}$; readout 2\\%)",
              coh="Coherent (SX over-rotation 0.05 rad, CZ phase 0.03 rad)", asym="Pauli with asymmetric readout (1\\%, 5\\%)", hw="Hardware-like combination")
rows = []
def simrow(label, d):
    dd = d.t_hat - d.c_e
    g = d.groupby(["backend", "source"]).apply(lambda x: pd.Series(dict(a=np.sum(x.weight * x.t_hat) / x.weight.sum(), b=np.sum(x.weight * x.c_e) / x.weight.sum())), include_groups=False)
    m = round(float(dd.mean()), 4) + 0.0
    return f"{label} & {len(d)} ({d.groupby(['backend','source']).ngroups}) & {num(m, 4)} & {np.sqrt((dd**2).mean()):.4f} & {dd.abs().max():.3f} & {d.t_hat.corr(d.c_e, method='spearman'):.3f} & {g.a.corr(g.b, method='spearman'):.3f}"
for k, dirn in enumerate(SIMS.split(",")):
    for name in ["pauli", "damp", "coh", "asym", "hw"]:
        f = f"{dirn}/sim_{name}.csv"
        if not os.path.exists(f): continue
        d = pd.read_csv(f)
        rows.append(simrow(labels[name] + (", ten-qubit circuits" if k > 0 else ""), d))
table("table_simulation.tex", "p{5.2cm}llllll", "Noise model & Edges (circuits) & Mean $t_e - c_e$ & RMS & Max $|t_e - c_e|$ & Spearman($t_e$, $c_e$) & Spearman($\\TE$, $\\TE^{c}$)", rows,
      "Simulation of the executed Clifford bodies and sign-balanced inputs under noise models that are not Pauli channels (exact density-matrix simulation without shot noise; preparations ideal). $t_e$ is the slope fitted by the registered estimator from the simulated correlators of the executed inputs; $c_e$ is the exact transfer coefficient of the propagated observable under the same noisy body and readout; $\\TE^{c}$ is the cost-weighted mean of $c_e$. Parameters are per gate; the Pauli model is a check of the implementation. Circuits with up to eight qubits were simulated for all models and eight of the 24 ten-qubit circuits for the hardware-like model.",
      "tab:simulation")
print("tables written")
