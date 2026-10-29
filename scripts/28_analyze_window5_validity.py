from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from qiskit import qpy


HERE = Path(__file__).resolve().parent


def load_v2_module():
    path = HERE / "26_analyze_window4_v2.py"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Install the Window-4 V2 patch first."
        )
    spec = importlib.util.spec_from_file_location("window4_v2_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def source_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["backend"].astype(str)
        + "::"
        + frame["source_sha"].astype(str)
    )


def build_window5_sources(
    base: Any,
    manifest_path: Path,
    completed_path: Path,
    window_id: str,
    rng: np.random.Generator,
    count_draws: int,
) -> tuple[pd.DataFrame, dict[tuple[str, int], dict[str, Any]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    if bool(completed.get("calibration_changed_during_run_interval")):
        raise RuntimeError(
            f"{manifest['backend']} Window 5 is calibration-flagged and "
            "cannot enter the clean headline analysis."
        )
    qpy_path = Path(manifest["qpy_file"])
    if not qpy_path.exists():
        qpy_path = manifest_path.parent / qpy_path.name
    if sha256_file(qpy_path) != str(manifest["qpy_sha256"]):
        raise RuntimeError(f"Window-5 QPY hash mismatch: {qpy_path}")
    if completed.get("qpy_sha256") not in (None, manifest["qpy_sha256"]):
        raise RuntimeError("Completed record QPY hash mismatch.")
    with qpy_path.open("rb") as handle:
        circuits = list(qpy.load(handle))
    if (
        len(circuits) != len(manifest["entries"])
        or len(completed["counts"]) != len(manifest["entries"])
    ):
        raise RuntimeError("Window-5 QPY/manifest/result length mismatch.")

    snapshot = base.Snapshot(
        json.loads(
            (completed_path.parent / "pre_submit_snapshot.json").read_text(
                encoding="utf-8"
            )
        )
    )
    rows: dict[int, dict[str, Any]] = {}
    payloads: dict[tuple[str, int], dict[str, Any]] = {}

    for entry, counts, circuit in zip(
        manifest["entries"], completed["counts"], circuits
    ):
        expected = int(entry["shots"])
        if sum(int(value) for value in counts.values()) != expected:
            raise RuntimeError(
                f"Shot mismatch at circuit {entry['circuit_index']}"
            )
        source = int(entry["source_index"])
        row = rows.setdefault(
            source,
            {
                "window": window_id,
                "backend": str(manifest["backend"]),
                "source": source,
                "instance": str(entry["instance_id"]),
                "depth": int(entry["depth"]),
                "selector": str(entry["selector"]),
                "n": int(entry["n_qubits"]),
                "source_sha": str(entry["source_qpy_sha256"]),
                "structure_sha": str(entry["source_structure_sha256"]),
                "physical_qubits": json.dumps(
                    entry["physical_qubits"], separators=(",", ":")
                ),
                "pt": [],
            },
        )

        if entry["kind"] == "workload":
            n = int(entry["n_qubits"])
            optimum = float(entry["optimum"])
            keys = list(counts)
            probabilities = (
                np.asarray([counts[key] for key in keys], dtype=float)
                / expected
            )
            rewards = np.asarray(
                [
                    base.cut_value(key, entry["edges"], n) / optimum
                    for key in keys
                ],
                dtype=float,
            )
            q = float(probabilities @ rewards)
            random_q = float(entry["random_approximation_ratio"])
            ideal_q = float(entry["ideal_approximation_ratio"])
            r_w = (q - random_q) / (ideal_q - random_q)

            draws = (
                rng.multinomial(
                    expected,
                    probabilities,
                    size=min(count_draws, 20000),
                )
                / expected
            )
            r_draws = (
                (draws @ rewards) - random_q
            ) / (ideal_q - random_q)
            score = base.score_workload(circuit, entry, snapshot)
            row.update(
                {
                    "q": q,
                    "rW": r_w,
                    "rW_se": float(np.std(r_draws, ddof=1)),
                    "neg_gate": -float(score["gate_log"]),
                    "neg_full": -float(score["full"]),
                }
            )
            payloads[(window_id, source)] = {
                "shots": expected,
                "probabilities": probabilities,
                "rewards": rewards,
                "random_q": random_q,
                "ideal_q": ideal_q,
                "q": q,
                "rW": r_w,
            }
        elif entry["kind"] in {"pt_plus", "pt_minus"}:
            row["pt"].append(
                (
                    int(entry["replicate"]),
                    str(entry["kind"]),
                    base.zz_all(counts),
                )
            )

    output = []
    for row in rows.values():
        pt = (
            pd.DataFrame(
                row.pop("pt"),
                columns=["replicate", "kind", "value"],
            )
            .pivot(index="replicate", columns="kind", values="value")
            .sort_index()
        )
        row["a_PT"] = float(
            ((pt.pt_plus - pt.pt_minus) / 2.0).mean()
        )
        row["beta_PT"] = float(
            ((pt.pt_plus + pt.pt_minus) / 2.0).mean()
        )
        output.append(row)
    return pd.DataFrame(output), payloads


def case_interval(
    base: Any,
    frame: pd.DataFrame,
    statistic,
    draws: int,
    seed: int,
) -> list[float] | None:
    values = base.case_bootstrap(frame, statistic, draws, seed)
    return base.interval(values)


def rank_combo(frame: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby(["backend", "n"]):
        te = (
            rankdata(group["w4_T_E"].to_numpy(float), method="average")
            - 0.5
        ) / len(group)
        gate = (
            rankdata(group["w5_neg_gate"].to_numpy(float), method="average")
            - 0.5
        ) / len(group)
        out.loc[group.index] = (te + 2.0 * gate) / 3.0
    return out


def decision_summary(
    pairs: pd.DataFrame,
    predictor: str,
) -> dict[str, Any]:
    nonidentical = pairs[~pairs["identical"]].copy()
    resolved = nonidentical[nonidentical["resolved"]].copy()
    delta = nonidentical[predictor].to_numpy(float)
    truth = nonidentical["delta_w5_rW"].to_numpy(float)
    pick_cal = delta > 0
    regret = np.where(
        pick_cal,
        np.maximum(-truth, 0),
        np.maximum(truth, 0),
    )
    delta_resolved = resolved[predictor].to_numpy(float)
    truth_resolved = resolved["delta_w5_rW"].to_numpy(float)
    return {
        "nonidentical_pairs": int(len(nonidentical)),
        "resolved_pairs": int(len(resolved)),
        "rho_all_nonidentical": (
            float(spearmanr(delta, truth).statistic)
            if len(nonidentical) >= 3
            else None
        ),
        "sign_agreement_all_nonidentical": (
            float(np.mean(np.sign(delta) == np.sign(truth)))
            if len(nonidentical)
            else None
        ),
        "sign_agreement_resolved": (
            float(
                np.mean(
                    np.sign(delta_resolved)
                    == np.sign(truth_resolved)
                )
            )
            if len(resolved)
            else None
        ),
        "mean_regret_rW": (
            float(np.mean(regret)) if len(regret) else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prospective Window-5 validity analysis."
    )
    parser.add_argument("--w4-sources", required=True)
    parser.add_argument("--w4-pairs", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--completed", action="append", required=True)
    parser.add_argument("--window-id", action="append", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=100000)
    parser.add_argument("--count-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument("--out", default="results/window5_v2_analysis")
    args = parser.parse_args()

    if not (
        len(args.manifest)
        == len(args.completed)
        == len(args.window_id)
    ):
        raise SystemExit(
            "Each Window-5 manifest needs one completed path and window ID."
        )

    base = load_v2_module()
    rng = np.random.default_rng(args.seed)
    w5_frames = []
    payloads: dict[tuple[str, int], dict[str, Any]] = {}
    for manifest, completed, window in zip(
        args.manifest, args.completed, args.window_id
    ):
        frame, current_payloads = build_window5_sources(
            base,
            Path(manifest),
            Path(completed),
            window,
            rng,
            args.count_draws,
        )
        w5_frames.append(frame)
        payloads.update(current_payloads)
    w5 = pd.concat(w5_frames, ignore_index=True)

    w4 = pd.read_csv(args.w4_sources).copy()
    w4["match_key"] = source_key(w4)
    w5["match_key"] = source_key(w5)
    keep_w4 = [
        "match_key",
        "backend",
        "source",
        "instance",
        "depth",
        "selector",
        "n",
        "structure_sha",
        "rW",
        "rW_se",
        "T_E",
        "a_PT",
        "neg_gate",
        "neg_full",
    ]
    keep_w5 = [
        "match_key",
        "window",
        "q",
        "rW",
        "rW_se",
        "a_PT",
        "beta_PT",
        "neg_gate",
        "neg_full",
    ]
    merged = w4[keep_w4].merge(
        w5[keep_w5],
        on="match_key",
        how="inner",
        suffixes=("_w4", "_w5"),
        validate="one_to_one",
    )
    if len(merged) != 72:
        raise RuntimeError(
            f"Expected 72 matched source rows, found {len(merged)}"
        )
    merged = merged.rename(
        columns={
            "rW_w4": "w4_rW",
            "rW_se_w4": "w4_rW_se",
            "T_E": "w4_T_E",
            "a_PT_w4": "w4_a_PT",
            "neg_gate_w4": "w4_neg_gate",
            "neg_full_w4": "w4_neg_full",
            "rW_w5": "w5_rW",
            "rW_se_w5": "w5_rW_se",
            "a_PT_w5": "w5_a_PT",
            "neg_gate_w5": "w5_neg_gate",
            "neg_full_w5": "w5_neg_full",
        }
    )
    merged["instance"] = merged["instance"].astype(str)
    merged["depth"] = merged["depth"].astype(int)
    merged["combo_gate2_TE1"] = rank_combo(merged)
    merged["delta_rW_time"] = merged["w5_rW"] - merged["w4_rW"]
    merged["delta_rW_time_se"] = np.sqrt(
        merged["w4_rW_se"] ** 2 + merged["w5_rW_se"] ** 2
    )

    outcome = merged.rename(columns={"w5_rW": "rW"})
    predictors = {
        "stale_w4_T_E": "w4_T_E",
        "stale_w4_workload": "w4_rW",
        "stale_w4_single_edge_PT": "w4_a_PT",
        "fresh_w5_gate": "w5_neg_gate",
        "fresh_w5_full": "w5_neg_full",
        "fresh_w5_single_edge_PT": "w5_a_PT",
        "exploratory_gate2_TE1": "combo_gate2_TE1",
    }

    transfer: dict[str, Any] = {}
    for name, column in predictors.items():
        point = base.stratified_rho(
            outcome,
            column,
            ["backend", "n"],
        )
        ci = case_interval(
            base,
            outcome,
            lambda sample, col=column: base.stratified_rho(
                sample,
                col,
                ["backend", "n"],
            ),
            args.bootstrap_draws,
            args.seed + len(transfer),
        )
        transfer[name] = {
            "rho_processor_size": point,
            "case_cluster_90_interval": ci,
        }

    partial = {
        "stale_T_E_given_fresh_gate": {
            "point": base.residual_rho(
                outcome,
                "w4_T_E",
                "w5_neg_gate",
                ["backend", "n"],
            ),
            "case_cluster_90_interval": case_interval(
                base,
                outcome,
                lambda sample: base.residual_rho(
                    sample,
                    "w4_T_E",
                    "w5_neg_gate",
                    ["backend", "n"],
                ),
                args.bootstrap_draws,
                args.seed + 100,
            ),
        },
        "fresh_gate_given_stale_T_E": {
            "point": base.residual_rho(
                outcome,
                "w5_neg_gate",
                "w4_T_E",
                ["backend", "n"],
            ),
            "case_cluster_90_interval": case_interval(
                base,
                outcome,
                lambda sample: base.residual_rho(
                    sample,
                    "w5_neg_gate",
                    "w4_T_E",
                    ["backend", "n"],
                ),
                args.bootstrap_draws,
                args.seed + 101,
            ),
        },
    }

    change_summary = {}
    for backend, group in merged.groupby("backend"):
        observed_variance = float(
            np.var(group["delta_rW_time"], ddof=1)
        )
        expected_noise_variance = float(
            np.mean(group["delta_rW_time_se"] ** 2)
        )
        residual_variance = max(
            observed_variance - expected_noise_variance,
            0.0,
        )
        change_summary[str(backend)] = {
            "sources": int(len(group)),
            "mean_change": float(group["delta_rW_time"].mean()),
            "observed_change_variance": observed_variance,
            "expected_count_noise_variance": expected_noise_variance,
            "residual_systematic_sd": float(
                np.sqrt(residual_variance)
            ),
            "fraction_within_two_count_se": float(
                np.mean(
                    np.abs(group["delta_rW_time"])
                    <= 2.0 * group["delta_rW_time_se"]
                )
            ),
            "workload_spearman": float(
                spearmanr(group["w4_rW"], group["w5_rW"]).statistic
            ),
            "single_edge_PT_spearman": float(
                spearmanr(
                    group["w4_a_PT"],
                    group["w5_a_PT"],
                ).statistic
            ),
        }

    pair_rows = []
    for keys, group in merged.groupby(
        ["backend", "instance", "depth"]
    ):
        selected = {
            str(row.selector): row
            for row in group.itertuples(index=False)
        }
        if set(selected) != {
            "topology_only",
            "calibration_scored",
        }:
            continue
        top = selected["topology_only"]
        cal = selected["calibration_scored"]
        identical = top.structure_sha == cal.structure_sha

        top_payload = payloads[(str(top.window), int(top.source))]
        cal_payload = payloads[(str(cal.window), int(cal.source))]
        top_draws = (
            rng.multinomial(
                top_payload["shots"],
                top_payload["probabilities"],
                size=args.count_draws,
            )
            / top_payload["shots"]
        )
        cal_draws = (
            rng.multinomial(
                cal_payload["shots"],
                cal_payload["probabilities"],
                size=args.count_draws,
            )
            / cal_payload["shots"]
        )
        top_q = top_draws @ top_payload["rewards"]
        cal_q = cal_draws @ cal_payload["rewards"]
        delta_q_draws = cal_q - top_q
        directional = float(
            max(
                np.mean(delta_q_draws > 0),
                np.mean(delta_q_draws < 0),
            )
        )
        delta_q = float(cal.q - top.q)
        delta_r = float(cal.w5_rW - top.w5_rW)
        pair_rows.append(
            {
                "backend": keys[0],
                "instance": keys[1],
                "depth": int(keys[2]),
                "n": int(top.n),
                "identical": bool(identical),
                "delta_w5_q": delta_q,
                "delta_w5_rW": delta_r,
                "directional_probability": directional,
                "resolved": bool(
                    (not identical)
                    and directional >= 0.95
                    and abs(delta_q) >= 0.005
                ),
                "delta_stale_w4_T_E": float(
                    cal.w4_T_E - top.w4_T_E
                ),
                "delta_stale_w4_rW": float(
                    cal.w4_rW - top.w4_rW
                ),
                "delta_stale_w4_a_PT": float(
                    cal.w4_a_PT - top.w4_a_PT
                ),
                "delta_fresh_w5_gate": float(
                    cal.w5_neg_gate - top.w5_neg_gate
                ),
                "delta_fresh_w5_full": float(
                    cal.w5_neg_full - top.w5_neg_full
                ),
                "delta_combo_gate2_TE1": float(
                    cal.combo_gate2_TE1 - top.combo_gate2_TE1
                ),
            }
        )
    pairs = pd.DataFrame(pair_rows)

    decision = {}
    for column in [
        "delta_stale_w4_T_E",
        "delta_stale_w4_rW",
        "delta_stale_w4_a_PT",
        "delta_fresh_w5_gate",
        "delta_fresh_w5_full",
        "delta_combo_gate2_TE1",
    ]:
        decision[column] = decision_summary(pairs, column)

    nonidentical = pairs[~pairs["identical"]]
    resolved = nonidentical[nonidentical["resolved"]]
    truth = nonidentical["delta_w5_rW"].to_numpy(float)
    decision["always_calibration"] = {
        "mean_regret_rW": float(np.mean(np.maximum(-truth, 0))),
        "sign_agreement_all_nonidentical": float(
            np.mean(truth > 0)
        ),
        "sign_agreement_resolved": float(
            np.mean(resolved["delta_w5_rW"] > 0)
        )
        if len(resolved)
        else None,
    }
    decision["always_topology"] = {
        "mean_regret_rW": float(np.mean(np.maximum(truth, 0))),
        "sign_agreement_all_nonidentical": float(
            np.mean(truth < 0)
        ),
        "sign_agreement_resolved": float(
            np.mean(resolved["delta_w5_rW"] < 0)
        )
        if len(resolved)
        else None,
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / "window5_matched_sources.csv", index=False)
    pairs.to_csv(out / "window5_selector_pairs.csv", index=False)
    summary = {
        "registered_status": (
            "Prospective Window-5 validity extension. The exploratory "
            "gate2_TE1 policy was frozen after Window 4 and before Window 5."
        ),
        "source_rows": int(len(merged)),
        "transfer_to_window5": transfer,
        "incremental_validity": partial,
        "temporal_change": change_summary,
        "decision": decision,
        "identical_pairs": int(pairs["identical"].sum()),
        "nonidentical_pairs": int((~pairs["identical"]).sum()),
        "resolved_nonidentical_pairs": int(pairs["resolved"].sum()),
    }
    write_json(out / "window5_validity_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print("Saved:", out.resolve())


if __name__ == "__main__":
    main()
