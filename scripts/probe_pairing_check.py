"""Recover the all-edge probe statistic on the readout qubits of the cost edges.

The workload circuit measures vertex u from the physical qubit meas[u] that holds it at the end of the
circuit; the compiler's layout and routing permute the vertices within the compilation subset, so meas[u]
is in general not the u-th qubit of the sorted subset. The registered probe circuits measure the sorted
active qubits into classical bits (bit k <-> active[k]), so the registered key 'u,v' refers to the pair
(active[u], active[v]), which coincides with the readout pair (meas[u], meas[v]) for only a few edges.

The script (1) counts the coinciding pairs, (2) simulates every executed probe circuit exactly (stabilizer
simulation on the active qubits) to obtain the ideal value of every ZZ pair, (3) refits the edge transfer of
each readout pair (meas[u], meas[v]) by ordinary least squares with intercept over the executed sign assignments
in which that pair has a definite (+1 or -1) ideal value of each sign (the registered estimator; an edge with
both signs present is identifiable), and (4) compares the recovered T_E with the registered T_E and with r_W.
It also writes, for the ablation analyses, the unweighted mean of the recovered edge transfers, the mean transfer
over all output pairs of the circuit (a generic multi-observable aggregate that ignores the cost function), the
standard error of every edge slope, and a JSON record of the ideal values and measured correlators of every
output pair in every executed probe circuit.

usage: python probe_pairing_check.py <results_v3 dir> <out csv>
"""
import json, math, sys, itertools
import numpy as np, pandas as pd
from qiskit import qpy, QuantumCircuit
from qiskit.quantum_info import StabilizerState, Pauli
from scipy.stats import rankdata, pearsonr

R, OUT = sys.argv[1], sys.argv[2]

def frac_rank(x): return (rankdata(x) - 0.5) / len(x)
def strat_rho(d, pred, y, strata):
    d = d.dropna(subset=[pred, y]); rx, ry = [], []
    for _, gg in d.groupby(strata):
        if len(gg) < 3: continue
        rx.append(frac_rank(gg[pred].values)); ry.append(frac_rank(gg[y].values))
    return pearsonr(np.concatenate(rx), np.concatenate(ry))[0]

def ols(S, M):
    """Slope, intercept and standard error of the slope for m = b + t s (both signs of s present)."""
    S = np.asarray(S, float); M = np.asarray(M, float)
    t, b = np.polyfit(S, M, 1)
    resid = M - (b + t * S); dof = len(S) - 2
    se = float(np.sqrt(np.sum(resid ** 2) / dof / np.sum((S - S.mean()) ** 2))) if dof > 0 else np.nan
    return float(t), float(b), se

def zz(counts, i, j):
    tot = sum(counts.values()); s = 0
    for b, c in counts.items():
        bb = b[::-1]; s += c * (1 if (int(bb[i]) + int(bb[j])) % 2 == 0 else -1)
    return s / tot

rows = []; pair_rows = []; record = {}
for be in ("fez", "kingston"):
    m = json.load(open(f"{R}/window4_v2/ibm_{be}_window4_v2_manifest.json")); ents = m["entries"]
    c = json.load(open(f"{R}/window4_v2/runs/ibm_{be}/primary/completed.json")); counts = c["counts"]
    with open(f"{R}/window4_v2/ibm_{be}_window4_v2.qpy", "rb") as f: circs = qpy.load(f)
    per = pd.read_csv(f"{R}/window4_v2_analysis/per_source_v2.csv")
    for src in sorted({e["source_index"] for e in ents}):
        w = [e for e in ents if e["kind"] == "workload" and e["source_index"] == src][0]
        wc = circs[w["circuit_index"]]
        wmeas = {wc.find_bit(i.clbits[0]).index: wc.find_bit(i.qubits[0]).index for i in wc.data if i.operation.name == "measure"}
        probes = [(k, e) for k, e in enumerate(ents) if e["kind"] == "edge_probe" and e["source_index"] == src]
        pc0 = circs[probes[0][1]["circuit_index"]]
        pmeas = {pc0.find_bit(i.clbits[0]).index: pc0.find_bit(i.qubits[0]).index for i in pc0.data if i.operation.name == "measure"}
        active = [pmeas[k] for k in sorted(pmeas)]; idx = {q: k for k, q in enumerate(active)}
        n = len(active)
        # ideal ZZ of every pair in every executed probe circuit (exact stabilizer simulation on active qubits)
        ideal = {}  # (k, a, b) -> value
        meas_corr = {}
        for k, e in probes:
            pc = circs[e["circuit_index"]]
            sub = QuantumCircuit(n)
            for inst in pc.data:
                nm = inst.operation.name
                if nm in ("measure", "barrier", "delay"): continue
                qs = [idx[pc.find_bit(q).index] for q in inst.qubits]
                if nm == "rz":
                    th = float(inst.operation.params[0]); kk = int(round(th / (math.pi / 2))) % 4
                    assert abs(th - round(th / (math.pi / 2)) * math.pi / 2) < 1e-6, ("non-Clifford angle in probe", th)
                    for _ in range(kk): sub.s(qs[0])
                elif nm == "sx": sub.sx(qs[0])
                elif nm == "x": sub.x(qs[0])
                elif nm == "cz": sub.cz(*qs)
                else: raise ValueError(nm)
            st = StabilizerState(sub)
            for a, b in itertools.combinations(range(n), 2):
                lab = ["I"] * n; lab[a] = "Z"; lab[b] = "Z"
                v = st.expectation_value(Pauli("".join(reversed(lab))))
                ideal[(k, a, b)] = float(np.real(v))
                meas_corr[(k, a, b)] = zz(counts[k], a, b)
        record[f"ibm_{be}:{src}"] = dict(active=active, n=n, edges=[[int(u), int(v), float(wt)] for u, v, wt in w["edges"]],
                                        readout={int(cb): idx[q] for cb, q in wmeas.items()},
                                        circuits=[int(k) for k, _ in probes],
                                        ideal={f"{k},{a},{b}": ideal[(k, a, b)] for (k, a, b) in ideal},
                                        meas={f"{k},{a},{b}": meas_corr[(k, a, b)] for (k, a, b) in meas_corr})
        # generic aggregate over all output pairs (unweighted mean of OLS slopes of every pair with both signs present)
        t_all = []
        for a, b in itertools.combinations(range(n), 2):
            vals_p = [(ideal[(k, a, b)], meas_corr[(k, a, b)]) for k, _ in probes if abs(ideal[(k, a, b)]) > 0.5]
            if sum(1 for s_, _ in vals_p if s_ > 0) >= 1 and sum(1 for s_, _ in vals_p if s_ < 0) >= 1:
                t_all.append(ols([s_ for s_, _ in vals_p], [mm for _, mm in vals_p])[0])
        # registered pairs vs true pairs
        reg_ok = 0; t_reg = []; t_true = []; t_se = []; wts = []; n_def = []
        for (u, v, wt) in w["edges"]:
            a_reg, b_reg = sorted((u, v))                      # registered: clbits u, v -> active[u], active[v]
            a_true, b_true = sorted((idx[wmeas[u]], idx[wmeas[v]]))  # true cost pair at the workload output
            reg_ok += (a_reg, b_reg) == (a_true, b_true)
            # registered transfer from the registered group circuits (sign-corrected mean, as in the re-implementation)
            vals = [(ideal[(k, a_reg, b_reg)], meas_corr[(k, a_reg, b_reg)]) for k, _ in probes if abs(ideal[(k, a_reg, b_reg)]) > 0.5]
            if vals and sum(1 for s, _ in vals if s > 0) >= 2 and sum(1 for s, _ in vals if s < 0) >= 2:
                S = np.array([s for s, _ in vals]); M = np.array([mm for _, mm in vals]); treg = float(np.polyfit(S, M, 1)[0])
            else:
                treg = np.mean([s * mm for s, mm in vals]) if vals else np.nan
            # true-pair transfer from every executed probe circuit with a definite ideal value
            vals_t = [(ideal[(k, a_true, b_true)], meas_corr[(k, a_true, b_true)]) for k, _ in probes if abs(ideal[(k, a_true, b_true)]) > 0.5]
            npos = sum(1 for s, _ in vals_t if s > 0); nneg = sum(1 for s, _ in vals_t if s < 0)
            if npos >= 1 and nneg >= 1:   # ordinary least squares slope with intercept, as in the registered estimator
                ttrue, _, tse = ols([s for s, _ in vals_t], [mm for _, mm in vals_t])
            else:
                ttrue, tse = np.nan, np.nan
            t_reg.append(treg); t_true.append(ttrue); t_se.append(tse); wts.append(wt); n_def.append(len(vals_t))
            pair_rows.append(dict(backend=f"ibm_{be}", source=src, u=u, v=v, weight=wt, registered_pair_is_cost_edge=(a_reg, b_reg) == (a_true, b_true),
                                  t_registered=treg, t_true=ttrue, t_true_se=tse, n_circuits_definite_true=len(vals_t), n_pos=npos, n_neg=nneg))
        wts = np.array(wts); t_reg = np.array(t_reg); t_true = np.array(t_true)
        ok = ~np.isnan(t_true)
        TE_reg = np.sum(wts * t_reg) / np.sum(wts)
        TE_true = np.sum(wts[ok] * t_true[ok]) / np.sum(wts[ok]) if ok.any() else np.nan
        TE_true_unw = float(np.mean(t_true[ok])) if ok.any() else np.nan
        T_all = float(np.mean(t_all)) if t_all else np.nan
        p = per[(per.window == f"{be}_w4") & (per.source == src)].iloc[0]
        rows.append(dict(backend=f"ibm_{be}", source=src, instance=w["instance_id"], depth=w["depth"], selector=w["selector"], n=w["n_qubits"],
                         edges=len(wts), registered_pairs_matching_cost_edges=reg_ok, true_pairs_estimable=int(ok.sum()),
                         weight_fraction_estimable=float(np.sum(wts[ok]) / np.sum(wts)), T_E_registered=TE_reg, T_E_canonical=p["T_E"],
                         T_E_true_pairs=TE_true, T_E_true_unweighted=TE_true_unw, T_allpairs=T_all, n_allpairs=len(t_all),
                         rW=p["rW"], neg_gate=p["neg_gate"], neg_full=p["neg_full"], a_PT=p["a_PT"]))
        print(f"ibm_{be} src {src:2d} {w['instance_id']} p{w['depth']} {w['selector'][:4]}: matching pairs {reg_ok}/{len(wts)}; true pairs estimable {ok.sum()}/{len(wts)} "
              f"(weight frac {np.sum(wts[ok])/np.sum(wts):.2f}); T_E registered {TE_reg:.3f} (canonical {p['T_E']:.3f}); T_E true-pairs {TE_true:.3f}; rW {p['rW']:.3f}")
df = pd.DataFrame(rows); df.to_csv(OUT, index=False); pd.DataFrame(pair_rows).to_csv(OUT.replace(".csv", "_pairs.csv"), index=False)
json.dump(record, open(OUT.replace(".csv", "_record.json"), "w"))
print("\nregistered pairs that are cost edges at the output:", int(df.registered_pairs_matching_cost_edges.sum()), "of", int(df.edges.sum()))
print("true pairs estimable from the executed circuits:", int(df.true_pairs_estimable.sum()), "of", int(df.edges.sum()), "; mean weight fraction", round(df.weight_fraction_estimable.mean(), 3))
print("sources with every true pair estimable:", int((df.true_pairs_estimable == df.edges).sum()), "of", len(df))
d = df.copy(); d["window"] = d.backend
print("rank correlation with rW (within processor x size):  registered T_E %.3f | canonical %.3f | true-pair T_E %.3f (n=%d) | gate %.3f | a_PT %.3f" % (
    strat_rho(d, "T_E_registered", "rW", ["window", "n"]), strat_rho(d, "T_E_canonical", "rW", ["window", "n"]),
    strat_rho(d, "T_E_true_pairs", "rW", ["window", "n"]), d.T_E_true_pairs.notna().sum(), strat_rho(d, "neg_gate", "rW", ["window", "n"]), strat_rho(d, "a_PT", "rW", ["window", "n"])))
print("Spearman(registered T_E, true-pair T_E) = %.3f" % pd.Series(d.T_E_registered).corr(d.T_E_true_pairs, method="spearman"))
