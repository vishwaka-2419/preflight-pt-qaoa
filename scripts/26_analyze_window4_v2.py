from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from qiskit import qpy




def unit_to_seconds(value: float, unit: str | None) -> float:
    factors = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "µs": 1e-6, "ns": 1e-9, "ps": 1e-12, "": 1.0}
    key = (unit or "").strip().lower()
    if key not in factors:
        raise ValueError(f"Unsupported time unit: {unit!r}")
    return float(value) * factors[key]


class Snapshot:
    def __init__(self, payload: dict[str, Any]) -> None:
        properties = payload.get("properties", payload)
        self.qubits = []
        for parameter_list in properties["qubits"]:
            item = {}
            for parameter in parameter_list:
                item[str(parameter["name"])] = (float(parameter["value"]), str(parameter.get("unit", "")))
            self.qubits.append(item)
        self.gate_errors = {}
        for gate in properties["gates"]:
            error = None
            for parameter in gate.get("parameters", []):
                if str(parameter.get("name")) == "gate_error":
                    error = float(parameter["value"])
                    break
            if error is not None:
                self.gate_errors[(str(gate["gate"]), tuple(int(q) for q in gate["qubits"]))] = error

    def qubit_parameter(self, qubit: int, name: str):
        return self.qubits[int(qubit)].get(name)

    def readout_error(self, qubit: int) -> float:
        direct = self.qubit_parameter(qubit, "readout_error")
        if direct is not None:
            return float(direct[0])
        p10 = self.qubit_parameter(qubit, "prob_meas0_prep1")
        p01 = self.qubit_parameter(qubit, "prob_meas1_prep0")
        if p10 is None or p01 is None:
            raise KeyError(f"No readout error for qubit {qubit}")
        return 0.5 * (float(p10[0]) + float(p01[0]))

    def coherence(self, qubit: int) -> float:
        values = []
        for name in ("T1", "T2"):
            item = self.qubit_parameter(qubit, name)
            if item is not None:
                values.append(unit_to_seconds(item[0], item[1]))
        values = [value for value in values if value > 0 and np.isfinite(value)]
        if not values:
            raise KeyError(f"No positive T1/T2 for qubit {qubit}")
        return min(values)

    def gate_error(self, name: str, qargs: tuple[int, ...]):
        value = self.gate_errors.get((name, qargs))
        if value is not None:
            return value
        if len(qargs) == 2:
            return self.gate_errors.get((name, tuple(reversed(qargs))))
        return None


def safe_probability(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return min(max(float(value), 0.0), 1.0 - 1e-15)


def score_workload(circuit, entry: dict[str, Any], snapshot: Snapshot) -> dict[str, float]:
    subset = [int(q) for q in entry["physical_qubits"]]
    gate_log = 0.0
    readout_log = 0.0
    missing = 0
    for instruction in circuit.data:
        name = str(instruction.operation.name)
        local = tuple(int(circuit.find_bit(q).index) for q in instruction.qubits)
        qargs = tuple(subset[index] for index in local) if circuit.num_qubits == len(subset) else local
        if name == "measure":
            for qubit in qargs:
                readout_log += -math.log1p(-safe_probability(snapshot.readout_error(qubit)))
            continue
        if name in {"barrier", "delay", "reset", "rz"}:
            continue
        error = snapshot.gate_error(name, qargs)
        if error is None:
            error = 0.0
            missing += 1
        gate_log += -math.log1p(-safe_probability(error))
    coherence = float(np.median([snapshot.coherence(qubit) for qubit in subset]))
    duration = float((entry.get("candidate_metadata") or {}).get("scheduled_duration_s", 0.0))
    coherence_penalty = duration / coherence if coherence > 0 else 0.0
    return {
        "gate_log": gate_log,
        "readout_log": readout_log,
        "coherence_penalty": coherence_penalty,
        "full": gate_log + readout_log + coherence_penalty,
        "missing_gate_error_events": missing,
    }


def cut_value(bits: str, edges: list[list[float]], n: int) -> float:
    clean = str(bits).replace(" ", "")
    logical = [int(clean[n - 1 - i]) for i in range(n)]
    return sum(float(w) for u, v, w in edges if logical[int(u)] != logical[int(v)])


def zz(counts: dict[str, int], i: int, j: int) -> float:
    total = sum(counts.values())
    value = 0.0
    for key, count in counts.items():
        bits = str(key).replace(" ", "")
        bi = int(bits[len(bits) - 1 - i])
        bj = int(bits[len(bits) - 1 - j])
        value += int(count) * (1.0 if bi == bj else -1.0)
    return value / total


def zz_all(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    return sum(int(c) * (1.0 if str(k).replace(" ", "").count("1") % 2 == 0 else -1.0) for k, c in counts.items()) / total


def frac_rank(values: np.ndarray) -> np.ndarray:
    return (rankdata(values, method="average") - 0.5) / len(values)


def stratified_rho(frame: pd.DataFrame, predictor: str, strata: list[str]) -> float:
    frame = frame.dropna(subset=[predictor, "rW"])
    x, y = [], []
    for _, group in frame.groupby(strata):
        if len(group) < 3:
            continue
        x.append(frac_rank(group[predictor].to_numpy(float)))
        y.append(frac_rank(group["rW"].to_numpy(float)))
    if not x:
        return float("nan")
    return float(pearsonr(np.concatenate(x), np.concatenate(y)).statistic)


def residual_rho(frame: pd.DataFrame, predictor: str, control: str, strata: list[str]) -> float:
    frame = frame.dropna(subset=[predictor, control, "rW"])
    xr, yr = [], []
    for _, group in frame.groupby(strata):
        if len(group) < 4:
            continue
        x = frac_rank(group[predictor].to_numpy(float))
        y = frac_rank(group["rW"].to_numpy(float))
        c = frac_rank(group[control].to_numpy(float))
        design = np.column_stack([np.ones(len(group)), c])
        xr.append(x - design @ np.linalg.lstsq(design, x, rcond=None)[0])
        yr.append(y - design @ np.linalg.lstsq(design, y, rcond=None)[0])
    if not xr:
        return float("nan")
    return float(pearsonr(np.concatenate(xr), np.concatenate(yr)).statistic)


def case_bootstrap(frame: pd.DataFrame, statistic, draws: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    cases = list(frame[["instance", "depth"]].drop_duplicates().itertuples(index=False, name=None))
    values = []
    for _ in range(draws):
        pieces = []
        for copy_index, selected_index in enumerate(rng.integers(0, len(cases), len(cases))):
            instance, depth = cases[int(selected_index)]
            piece = frame[(frame.instance == instance) & (frame.depth == depth)].copy()
            piece["case_copy"] = copy_index
            pieces.append(piece)
        try:
            values.append(float(statistic(pd.concat(pieces, ignore_index=True))))
        except Exception:
            continue
    return values


def interval(values: list[float]) -> list[float] | None:
    array = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if not len(array):
        return None
    return [float(x) for x in np.quantile(array, [0.05, 0.95])]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze Window 4/5 V2 results.")
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--completed", action="append", required=True)
    ap.add_argument("--window-id", action="append", default=[])
    ap.add_argument("--bootstrap-draws", type=int, default=100000)
    ap.add_argument("--count-draws", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--out", default="results/window4_v2_analysis")
    args = ap.parse_args()
    if len(args.manifest) != len(args.completed):
        raise SystemExit("Each manifest requires one completed record.")

    rng = np.random.default_rng(args.seed)
    source_rows = []
    edge_rows = []
    workload_payloads = {}

    for index, (manifest_raw, completed_raw) in enumerate(zip(args.manifest, args.completed)):
        manifest_path = Path(manifest_raw)
        completed_path = Path(completed_raw)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        window = args.window_id[index] if index < len(args.window_id) else f"{manifest['backend']}::{manifest['label']}"
        counts_list = completed["counts"]
        qpy_path = Path(manifest["qpy_file"])
        if not qpy_path.exists():
            qpy_path = manifest_path.parent / qpy_path.name
        with qpy_path.open("rb") as handle:
            circuits = list(qpy.load(handle))
        snapshot_path = completed_path.parent / "pre_submit_snapshot.json"
        snapshot = Snapshot(json.loads(snapshot_path.read_text(encoding="utf-8")))
        if len(counts_list) != len(manifest["entries"]) or len(circuits) != len(manifest["entries"]):
            raise RuntimeError("Manifest/result/QPY length mismatch.")

        per = {}
        for entry, counts, circuit in zip(manifest["entries"], counts_list, circuits):
            expected = int(entry["shots"])
            if sum(int(v) for v in counts.values()) != expected:
                raise RuntimeError(f"Shot mismatch at {entry['circuit_index']}")
            source = int(entry["source_index"])
            row = per.setdefault(
                source,
                {
                    "window": window,
                    "backend": manifest["backend"],
                    "source": source,
                    "instance": entry["instance_id"],
                    "depth": int(entry["depth"]),
                    "selector": entry["selector"],
                    "n": int(entry["n_qubits"]),
                    "source_sha": entry["source_qpy_sha256"],
                    "structure_sha": entry["source_structure_sha256"],
                    "physical_qubits": json.dumps(entry["physical_qubits"], separators=(",", ":")),
                    "pt": [],
                    "edge": {},
                    "neg_gate": np.nan,
                    "neg_full": np.nan,
                    "neg_gate_selection_state": -float((entry.get("selector_components") or {}).get("gate_log_score", np.nan)),
                    "neg_full_selection_state": -float((entry.get("selector_components") or {}).get("recomputed_full_score", np.nan)),
                    "neg_2q": -float((entry.get("candidate_metadata") or {}).get("two_qubit_count", np.nan)),
                    "neg_depth": -float((entry.get("candidate_metadata") or {}).get("circuit_depth", np.nan)),
                    "missing_gate_error_events": 0,
                },
            )
            kind = entry["kind"]
            if kind == "workload":
                n = int(entry["n_qubits"])
                edges = entry["edges"]
                optimum = float(entry["optimum"])
                total = expected
                keys = list(counts)
                probabilities = np.asarray([counts[key] for key in keys], dtype=float) / total
                rewards = np.asarray([cut_value(key, edges, n) / optimum for key in keys], dtype=float)
                q = float(probabilities @ rewards)
                random_q = float(entry["random_approximation_ratio"])
                ideal_q = float(entry["ideal_approximation_ratio"])
                r_w = (q - random_q) / (ideal_q - random_q)
                draws = rng.multinomial(total, probabilities, size=min(args.count_draws, 20000)) / total
                r_draws = ((draws @ rewards) - random_q) / (ideal_q - random_q)
                current_score = score_workload(circuit, entry, snapshot)
                row.update({
                    "q": q,
                    "rW": r_w,
                    "rW_se": float(np.std(r_draws, ddof=1)),
                    "neg_gate": -float(current_score["gate_log"]),
                    "neg_full": -float(current_score["full"]),
                    "missing_gate_error_events": int(current_score["missing_gate_error_events"]),
                })
                workload_payloads[(window, source)] = {
                    "shots": total,
                    "probabilities": probabilities,
                    "rewards": rewards,
                    "random_q": random_q,
                    "ideal_q": ideal_q,
                    "q": q,
                    "rW": r_w,
                }
            elif kind in {"pt_plus", "pt_minus"}:
                row["pt"].append((int(entry["replicate"]), kind, zz_all(counts)))
            elif kind == "edge_probe":
                weight_map = {(int(u), int(v)): float(w) for u, v, w in entry["edges"]}
                for key, ideal_sign in entry["ideal_zz"].items():
                    u, v = map(int, key.split(","))
                    row["edge"].setdefault((u, v), []).append(
                        (int(entry["twirl"]), int(ideal_sign), zz(counts, u, v), weight_map[(u, v)])
                    )

        for row in per.values():
            pt = pd.DataFrame(row.pop("pt"), columns=["replicate", "kind", "value"]).pivot(index="replicate", columns="kind", values="value")
            row["a_PT"] = float(((pt.pt_plus - pt.pt_minus) / 2).mean())
            row["beta_PT"] = float(((pt.pt_plus + pt.pt_minus) / 2).mean())
            weighted_num = 0.0
            weighted_den = 0.0
            unweighted = []
            beta_values = []
            for (u, v), values in sorted(row.pop("edge").items()):
                signs = np.asarray([item[1] for item in values], dtype=float)
                measured = np.asarray([item[2] for item in values], dtype=float)
                weight = float(values[0][3])
                if len(set(signs.tolist())) < 2:
                    raise RuntimeError(f"Edge {(u, v)} lacks both ideal signs.")
                design = np.column_stack([np.ones(len(signs)), signs])
                beta, slope = np.linalg.lstsq(design, measured, rcond=None)[0]
                residual = measured - design @ np.asarray([beta, slope])
                dof = max(len(signs) - 2, 1)
                sigma2 = float(residual @ residual / dof)
                covariance = sigma2 * np.linalg.inv(design.T @ design)
                slope_se = float(math.sqrt(max(covariance[1, 1], 0.0)))
                edge_rows.append(
                    {
                        "window": row["window"],
                        "backend": row["backend"],
                        "source": row["source"],
                        "instance": row["instance"],
                        "depth": row["depth"],
                        "selector": row["selector"],
                        "u": u,
                        "v": v,
                        "weight": weight,
                        "transfer": float(slope),
                        "offset": float(beta),
                        "transfer_se": slope_se,
                        "twirls": len(signs),
                        "positive_signs": int(np.sum(signs > 0)),
                    }
                )
                weighted_num += weight * float(slope)
                weighted_den += weight
                unweighted.append(float(slope))
                beta_values.append(float(beta))
            row["T_E"] = weighted_num / weighted_den if weighted_den else np.nan
            row["T_Eu"] = float(np.mean(unweighted)) if unweighted else np.nan
            row["beta_E"] = float(np.average(beta_values)) if beta_values else np.nan
            source_rows.append(row)

    frame = pd.DataFrame(source_rows)
    edge_frame = pd.DataFrame(edge_rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "per_source_v2.csv", index=False)
    edge_frame.to_csv(out_dir / "edge_estimates_v2.csv", index=False)

    predictors = [name for name in ["T_E", "a_PT", "neg_gate", "neg_full", "neg_2q", "neg_depth"] if name in frame and frame[name].notna().any()]
    transfer = {}
    for predictor in predictors:
        point = stratified_rho(frame, predictor, ["window", "n"])
        values = case_bootstrap(
            frame,
            lambda sample, p=predictor: stratified_rho(sample, p, ["window", "n"]),
            args.bootstrap_draws,
            args.seed + len(transfer),
        )
        transfer[predictor] = {
            "rho_window_size": point,
            "rho_window_size_depth": stratified_rho(frame, predictor, ["window", "n", "depth"]),
            "case_cluster_90_interval": interval(values),
        }
    partial = {}
    if "neg_gate" in predictors:
        for predictor in [name for name in ["T_E", "a_PT"] if name in predictors]:
            point = residual_rho(frame, predictor, "neg_gate", ["window", "n"])
            values = case_bootstrap(
                frame,
                lambda sample, p=predictor: residual_rho(sample, p, "neg_gate", ["window", "n"]),
                args.bootstrap_draws,
                args.seed + 100 + len(partial),
            )
            reverse_point = residual_rho(frame, "neg_gate", predictor, ["window", "n"])
            reverse_values = case_bootstrap(
                frame,
                lambda sample, p=predictor: residual_rho(sample, "neg_gate", p, ["window", "n"]),
                args.bootstrap_draws,
                args.seed + 200 + len(partial),
            )
            partial[predictor] = {
                "predictor_given_gate": point,
                "predictor_given_gate_90_interval": interval(values),
                "gate_given_predictor": reverse_point,
                "gate_given_predictor_90_interval": interval(reverse_values),
            }

    pair_rows = []
    for (window, backend, instance, depth), group in frame.groupby(["window", "backend", "instance", "depth"]):
        selected = {row.selector: row for row in group.itertuples(index=False)}
        if set(selected) != {"topology_only", "calibration_scored"}:
            continue
        top = selected["topology_only"]
        cal = selected["calibration_scored"]
        identical = top.structure_sha == cal.structure_sha
        top_payload = workload_payloads[(window, int(top.source))]
        cal_payload = workload_payloads[(window, int(cal.source))]
        top_draws = rng.multinomial(top_payload["shots"], top_payload["probabilities"], size=args.count_draws) / top_payload["shots"]
        cal_draws = rng.multinomial(cal_payload["shots"], cal_payload["probabilities"], size=args.count_draws) / cal_payload["shots"]
        top_q = top_draws @ top_payload["rewards"]
        cal_q = cal_draws @ cal_payload["rewards"]
        delta_q_draws = cal_q - top_q
        top_r = (top_q - top_payload["random_q"]) / (top_payload["ideal_q"] - top_payload["random_q"])
        cal_r = (cal_q - cal_payload["random_q"]) / (cal_payload["ideal_q"] - cal_payload["random_q"])
        delta_r_draws = cal_r - top_r
        directional = float(max(np.mean(delta_q_draws > 0), np.mean(delta_q_draws < 0)))
        delta_q = float(cal.q - top.q)
        delta_r = float(cal.rW - top.rW)
        row = {
            "window": window,
            "backend": backend,
            "instance": instance,
            "depth": int(depth),
            "n": int(top.n),
            "identical": bool(identical),
            "delta_q": delta_q,
            "delta_rW": delta_r,
            "delta_q_90_low": float(np.quantile(delta_q_draws, 0.05)),
            "delta_q_90_high": float(np.quantile(delta_q_draws, 0.95)),
            "directional_probability": directional,
            "resolved": bool((not identical) and directional >= 0.95 and abs(delta_q) >= 0.005),
        }
        for predictor in predictors:
            row[f"delta_{predictor}"] = float(getattr(cal, predictor) - getattr(top, predictor))
        pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out_dir / "selector_pairs_v2.csv", index=False)

    decision = {}
    nonidentical = pairs[~pairs.identical].copy()
    resolved = nonidentical[nonidentical.resolved].copy()
    for predictor in predictors:
        delta = nonidentical[f"delta_{predictor}"].to_numpy(float)
        truth = nonidentical.delta_rW.to_numpy(float)
        pick_cal = delta > 0
        regret = np.where(pick_cal, np.maximum(-truth, 0), np.maximum(truth, 0))
        delta_res = resolved[f"delta_{predictor}"].to_numpy(float)
        truth_res = resolved.delta_rW.to_numpy(float)
        decision[predictor] = {
            "nonidentical_pairs": int(len(nonidentical)),
            "resolved_pairs": int(len(resolved)),
            "rho_all_nonidentical": float(spearmanr(delta, truth).statistic) if len(nonidentical) >= 3 else None,
            "sign_agreement_all_nonidentical": float(np.mean(np.sign(delta) == np.sign(truth))) if len(nonidentical) else None,
            "sign_agreement_resolved": float(np.mean(np.sign(delta_res) == np.sign(truth_res))) if len(resolved) else None,
            "mean_regret_rW": float(np.mean(regret)) if len(regret) else None,
        }
    for name, choose_cal in [("always_calibration", True), ("always_topology", False)]:
        truth = nonidentical.delta_rW.to_numpy(float)
        regret = np.maximum(-truth, 0) if choose_cal else np.maximum(truth, 0)
        decision[name] = {
            "mean_regret_rW": float(np.mean(regret)) if len(regret) else None,
            "sign_agreement_all_nonidentical": float(np.mean(truth > 0)) if choose_cal and len(truth) else float(np.mean(truth < 0)) if len(truth) else None,
            "sign_agreement_resolved": float(np.mean(resolved.delta_rW > 0)) if choose_cal and len(resolved) else float(np.mean(resolved.delta_rW < 0)) if len(resolved) else None,
        }

    summary = {
        "windows": sorted(frame.window.unique().tolist()),
        "source_rows": int(len(frame)),
        "edge_rows": int(len(edge_frame)),
        "transfer": transfer,
        "partial_transfer": partial,
        "decision": decision,
        "identical_pairs": int(pairs.identical.sum()),
        "nonidentical_pairs": int((~pairs.identical).sum()),
        "resolved_nonidentical_pairs": int(pairs.resolved.sum()),
        "registered_notes": {
            "primary_registered_result_is_not_replaced": True,
            "window4_is_prospective_extension": True,
            "edge_probe_name": "balanced sign-twirled edge probe",
        },
    }
    write_json(out_dir / "window4_v2_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print("Saved:", out_dir.resolve())


if __name__ == "__main__":
    main()
