from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def cut_value(bits: str, edges: list[list[float]], n: int) -> float:
    clean = str(bits).replace(" ", "")
    logical = [int(clean[n - 1 - i]) for i in range(n)]
    return sum(
        float(weight)
        for u, v, weight in edges
        if logical[int(u)] != logical[int(v)]
    )


def frac_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (rankdata(values, method="average") - 0.5) / len(values)


def stratified_rho(
    frame: pd.DataFrame,
    predictor: str,
    outcome: str = "w5_rW",
) -> float:
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for _, group in frame.groupby(["backend", "n"]):
        x_parts.append(frac_rank(group[predictor].to_numpy(float)))
        y_parts.append(frac_rank(group[outcome].to_numpy(float)))
    return float(
        pearsonr(
            np.concatenate(x_parts),
            np.concatenate(y_parts),
        ).statistic
    )


def residual_rho(
    frame: pd.DataFrame,
    predictor: str,
    control: str,
    outcome: str = "w5_rW",
) -> float:
    x_residuals: list[np.ndarray] = []
    y_residuals: list[np.ndarray] = []
    for _, group in frame.groupby(["backend", "n"]):
        x = frac_rank(group[predictor].to_numpy(float))
        y = frac_rank(group[outcome].to_numpy(float))
        c = frac_rank(group[control].to_numpy(float))
        design = np.column_stack([np.ones(len(group)), c])
        x_residuals.append(
            x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
        )
        y_residuals.append(
            y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        )
    return float(
        pearsonr(
            np.concatenate(x_residuals),
            np.concatenate(y_residuals),
        ).statistic
    )


def weighted_midrank(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    sorted_weights = weights[order]
    ranks = np.empty(len(values), dtype=float)
    total = float(sorted_weights.sum())
    cumulative = 0.0
    start = 0
    while start < len(values):
        stop = start + 1
        while (
            stop < len(values)
            and sorted_values[stop] == sorted_values[start]
        ):
            stop += 1
        tie_weight = float(sorted_weights[start:stop].sum())
        fractional = (
            (cumulative + tie_weight / 2.0) / total
            if total > 0
            else np.nan
        )
        ranks[order[start:stop]] = fractional
        cumulative += tie_weight
        start = stop
    return ranks


def weighted_correlation(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> float:
    mask = np.asarray(weights) > 0
    x = np.asarray(x, dtype=float)[mask]
    y = np.asarray(y, dtype=float)[mask]
    weights = np.asarray(weights, dtype=float)[mask]
    total = weights.sum()
    x_mean = np.sum(weights * x) / total
    y_mean = np.sum(weights * y) / total
    covariance = np.sum(weights * (x - x_mean) * (y - y_mean)) / total
    x_variance = np.sum(weights * (x - x_mean) ** 2) / total
    y_variance = np.sum(weights * (y - y_mean) ** 2) / total
    return float(covariance / np.sqrt(x_variance * y_variance))


def weighted_stratified_rho(
    frame: pd.DataFrame,
    predictor: str,
    outcome_values: np.ndarray,
    case_weights: np.ndarray,
    case_index: np.ndarray,
    group_indices: list[np.ndarray],
) -> float:
    predictor_values = frame[predictor].to_numpy(float)
    row_weights = case_weights[case_index]
    x_parts, y_parts, weight_parts = [], [], []
    for indices in group_indices:
        weights = row_weights[indices]
        x_parts.append(
            weighted_midrank(predictor_values[indices], weights)
        )
        y_parts.append(
            weighted_midrank(outcome_values[indices], weights)
        )
        weight_parts.append(weights)
    return weighted_correlation(
        np.concatenate(x_parts),
        np.concatenate(y_parts),
        np.concatenate(weight_parts),
    )


def weighted_residual_rho(
    frame: pd.DataFrame,
    predictor: str,
    control: str,
    outcome_values: np.ndarray,
    case_weights: np.ndarray,
    case_index: np.ndarray,
    group_indices: list[np.ndarray],
) -> float:
    predictor_values = frame[predictor].to_numpy(float)
    control_values = frame[control].to_numpy(float)
    row_weights = case_weights[case_index]
    x_parts, y_parts, weight_parts = [], [], []

    for indices in group_indices:
        weights = row_weights[indices]
        x = weighted_midrank(predictor_values[indices], weights)
        y = weighted_midrank(outcome_values[indices], weights)
        c = weighted_midrank(control_values[indices], weights)
        mask = weights > 0
        design = np.column_stack([np.ones(mask.sum()), c[mask]])
        root_weights = np.sqrt(weights[mask])[:, None]
        beta_x = np.linalg.lstsq(
            design * root_weights,
            x[mask] * root_weights[:, 0],
            rcond=None,
        )[0]
        beta_y = np.linalg.lstsq(
            design * root_weights,
            y[mask] * root_weights[:, 0],
            rcond=None,
        )[0]
        x_residual = np.zeros(len(indices))
        y_residual = np.zeros(len(indices))
        x_residual[mask] = x[mask] - design @ beta_x
        y_residual[mask] = y[mask] - design @ beta_y
        x_parts.append(x_residual)
        y_parts.append(y_residual)
        weight_parts.append(weights)

    return weighted_correlation(
        np.concatenate(x_parts),
        np.concatenate(y_parts),
        np.concatenate(weight_parts),
    )


def interval(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return [
        float(value)
        for value in np.quantile(finite, [0.05, 0.95])
    ]


def build_count_draws(
    matched: pd.DataFrame,
    manifest_paths: list[Path],
    completed_paths: list[Path],
    draws: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    payloads: dict[tuple[str, int], tuple] = {}
    integrity: dict[str, Any] = {}
    for manifest_path, completed_path in zip(
        manifest_paths, completed_paths
    ):
        manifest = load_json(manifest_path)
        completed = load_json(completed_path)
        backend = str(manifest["backend"])
        if completed.get("calibration_changed_during_run_interval"):
            raise RuntimeError(f"{backend} is calibration-flagged.")
        if len(manifest["entries"]) != len(completed["counts"]):
            raise RuntimeError(f"{backend}: result/manifest mismatch.")
        shot_total = 0
        workload_count = 0
        for entry, counts in zip(
            manifest["entries"], completed["counts"]
        ):
            observed = sum(int(value) for value in counts.values())
            expected = int(entry["shots"])
            if observed != expected:
                raise RuntimeError(
                    f"{backend}: shot mismatch at "
                    f"{entry['circuit_index']}"
                )
            shot_total += observed
            if entry["kind"] != "workload":
                continue
            workload_count += 1
            aggregated: dict[float, int] = {}
            for bitstring, count in counts.items():
                reward = (
                    cut_value(
                        bitstring,
                        entry["edges"],
                        int(entry["n_qubits"]),
                    )
                    / float(entry["optimum"])
                )
                aggregated[reward] = (
                    aggregated.get(reward, 0) + int(count)
                )
            rewards = np.asarray(sorted(aggregated), dtype=float)
            reward_counts = np.asarray(
                [aggregated[value] for value in rewards],
                dtype=int,
            )
            payloads[(backend, int(entry["source_index"]))] = (
                rewards,
                reward_counts,
                expected,
                float(entry["random_approximation_ratio"]),
                float(entry["ideal_approximation_ratio"]),
            )
        integrity[backend] = {
            "entries": len(manifest["entries"]),
            "workloads": workload_count,
            "observed_circuit_shots": shot_total,
            "registered_circuit_shots": int(
                manifest["total_circuit_shots"]
            ),
            "job_usage": completed.get("job_usage"),
            "remaining_seconds_after": (
                completed.get("service_usage_after") or {}
            ).get("usage_remaining_seconds"),
            "pre_submit_calibration_timestamp": completed.get(
                "pre_submit_calibration_timestamp"
            ),
            "post_completion_calibration_timestamp": completed.get(
                "post_completion_calibration_timestamp"
            ),
        }

    rng = np.random.default_rng(seed)
    output = np.empty((draws, len(matched)), dtype=float)
    for row_index, row in matched.reset_index(drop=True).iterrows():
        rewards, counts, shots, random_q, ideal_q = payloads[
            (str(row.backend), int(row.source))
        ]
        probabilities = counts / shots
        sampled = rng.multinomial(
            shots,
            probabilities,
            size=draws,
        )
        sampled_q = sampled @ rewards / shots
        output[:, row_index] = (
            sampled_q - random_q
        ) / (ideal_q - random_q)
    return output, integrity


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Robust audit of a later validity window. It separates "
            "logical-case, workload-count and combined sensitivities."
        )
    )
    parser.add_argument("--matched-sources", required=True)
    parser.add_argument("--selector-pairs", required=True)
    parser.add_argument("--w4-completed", action="append", required=True)
    parser.add_argument("--w5-manifest", action="append", required=True)
    parser.add_argument("--w5-completed", action="append", required=True)
    parser.add_argument("--case-draws", type=int, default=20000)
    parser.add_argument("--count-draws", type=int, default=10000)
    parser.add_argument("--combined-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument(
        "--out",
        default="results/window5_v2_robust_audit",
    )
    args = parser.parse_args()

    if len(args.w5_manifest) != len(args.w5_completed):
        raise SystemExit(
            "Each later-window manifest needs one completed record."
        )

    matched = pd.read_csv(args.matched_sources).reset_index(drop=True)
    pairs = pd.read_csv(args.selector_pairs)
    predictor_columns = {
        "stale_T_E": "w4_T_E",
        "stale_workload": "w4_rW",
        "stale_single_edge_PT": "w4_a_PT",
        "fresh_gate": "w5_neg_gate",
        "fresh_full": "w5_neg_full",
        "fresh_single_edge_PT": "w5_a_PT",
        "combined_gate2_TE1": "combo_gate2_TE1",
    }

    cases = list(
        matched[["instance", "depth"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    case_map = {case: index for index, case in enumerate(cases)}
    case_index = np.asarray(
        [
            case_map[(row.instance, int(row.depth))]
            for row in matched.itertuples()
        ],
        dtype=int,
    )
    group_indices = [
        group.index.to_numpy()
        for _, group in matched.groupby(["backend", "n"])
    ]
    observed_outcome = matched["w5_rW"].to_numpy(float)

    rng = np.random.default_rng(args.seed)
    case_values: dict[str, list[float]] = {
        name: [] for name in predictor_columns
    }
    case_partial_te: list[float] = []
    case_partial_gate: list[float] = []
    case_difference_combo_gate: list[float] = []

    for _ in range(args.case_draws):
        weights = rng.multinomial(
            len(cases),
            [1.0 / len(cases)] * len(cases),
        )
        current: dict[str, float] = {}
        for name, column in predictor_columns.items():
            current[name] = weighted_stratified_rho(
                matched,
                column,
                observed_outcome,
                weights,
                case_index,
                group_indices,
            )
            case_values[name].append(current[name])
        case_partial_te.append(
            weighted_residual_rho(
                matched,
                "w4_T_E",
                "w5_neg_gate",
                observed_outcome,
                weights,
                case_index,
                group_indices,
            )
        )
        case_partial_gate.append(
            weighted_residual_rho(
                matched,
                "w5_neg_gate",
                "w4_T_E",
                observed_outcome,
                weights,
                case_index,
                group_indices,
            )
        )
        case_difference_combo_gate.append(
            current["combined_gate2_TE1"]
            - current["fresh_gate"]
        )

    count_draws, integrity = build_count_draws(
        matched,
        [Path(value) for value in args.w5_manifest],
        [Path(value) for value in args.w5_completed],
        args.count_draws,
        args.seed + 1,
    )

    # Count-only metrics. Predictor-count uncertainty is not propagated.
    count_values: dict[str, list[float]] = {
        name: [] for name in predictor_columns
    }
    count_partial_te: list[float] = []
    count_partial_gate: list[float] = []
    unit_case_weights = np.ones(len(cases), dtype=int)
    for draw in count_draws:
        for name, column in predictor_columns.items():
            count_values[name].append(
                weighted_stratified_rho(
                    matched,
                    column,
                    draw,
                    unit_case_weights,
                    case_index,
                    group_indices,
                )
            )
        count_partial_te.append(
            weighted_residual_rho(
                matched,
                "w4_T_E",
                "w5_neg_gate",
                draw,
                unit_case_weights,
                case_index,
                group_indices,
            )
        )
        count_partial_gate.append(
            weighted_residual_rho(
                matched,
                "w5_neg_gate",
                "w4_T_E",
                draw,
                unit_case_weights,
                case_index,
                group_indices,
            )
        )

    combined_count = min(
        args.combined_draws,
        len(count_draws),
    )
    combined_values: dict[str, list[float]] = {
        name: [] for name in predictor_columns
    }
    combined_partial_te: list[float] = []
    combined_partial_gate: list[float] = []
    combined_difference_combo_gate: list[float] = []
    for draw_index in range(combined_count):
        weights = rng.multinomial(
            len(cases),
            [1.0 / len(cases)] * len(cases),
        )
        draw = count_draws[draw_index]
        current = {}
        for name, column in predictor_columns.items():
            current[name] = weighted_stratified_rho(
                matched,
                column,
                draw,
                weights,
                case_index,
                group_indices,
            )
            combined_values[name].append(current[name])
        combined_partial_te.append(
            weighted_residual_rho(
                matched,
                "w4_T_E",
                "w5_neg_gate",
                draw,
                weights,
                case_index,
                group_indices,
            )
        )
        combined_partial_gate.append(
            weighted_residual_rho(
                matched,
                "w5_neg_gate",
                "w4_T_E",
                draw,
                weights,
                case_index,
                group_indices,
            )
        )
        combined_difference_combo_gate.append(
            current["combined_gate2_TE1"]
            - current["fresh_gate"]
        )

    # Match decision pairs to source rows and count draws.
    source_lookup = {
        (
            str(row.backend),
            str(row.instance),
            int(row.depth),
            str(row.selector),
        ): index
        for index, row in matched.iterrows()
    }
    pair_records = []
    for row in pairs.itertuples(index=False):
        top_index = source_lookup[
            (
                str(row.backend),
                str(row.instance),
                int(row.depth),
                "topology_only",
            )
        ]
        cal_index = source_lookup[
            (
                str(row.backend),
                str(row.instance),
                int(row.depth),
                "calibration_scored",
            )
        ]
        pair_records.append(
            {
                "backend": str(row.backend),
                "instance": str(row.instance),
                "depth": int(row.depth),
                "identical": bool(row.identical),
                "resolved": bool(row.resolved),
                "top_index": top_index,
                "cal_index": cal_index,
            }
        )
    pair_frame = pd.DataFrame(pair_records)
    pair_frame = pair_frame[~pair_frame["identical"]].reset_index(drop=True)
    pair_case_index = np.asarray(
        [
            case_map[(row.instance, int(row.depth))]
            for row in pair_frame.itertuples()
        ],
        dtype=int,
    )
    truth_point = np.asarray(
        [
            observed_outcome[int(row.cal_index)]
            - observed_outcome[int(row.top_index)]
            for row in pair_frame.itertuples()
        ],
        dtype=float,
    )
    truth_draws = (
        count_draws[:, pair_frame["cal_index"].to_numpy(int)]
        - count_draws[:, pair_frame["top_index"].to_numpy(int)]
    )
    policy_columns = {
        "stale_T_E": "delta_stale_w4_T_E",
        "stale_workload": "delta_stale_w4_rW",
        "stale_single_edge_PT": "delta_stale_w4_a_PT",
        "fresh_gate": "delta_fresh_w5_gate",
        "fresh_full": "delta_fresh_w5_full",
        "combined_gate2_TE1": "delta_combo_gate2_TE1",
    }
    pair_values = pairs[~pairs["identical"]].reset_index(drop=True)
    policy_pick_cal = {
        name: pair_values[column].to_numpy(float) > 0
        for name, column in policy_columns.items()
    }
    policy_pick_cal["always_calibration"] = np.ones(
        len(pair_frame), dtype=bool
    )
    policy_pick_cal["always_topology"] = np.zeros(
        len(pair_frame), dtype=bool
    )

    point_regret = {}
    case_regret_intervals = {}
    count_regret_intervals = {}
    combined_regret_intervals = {}
    for name, pick_cal in policy_pick_cal.items():
        point_pair_regret = np.where(
            pick_cal,
            np.maximum(-truth_point, 0),
            np.maximum(truth_point, 0),
        )
        point_regret[name] = float(point_pair_regret.mean())

        # Case-only.
        values = []
        for _ in range(args.case_draws):
            weights = rng.multinomial(
                len(cases),
                [1.0 / len(cases)] * len(cases),
            )
            pair_weights = weights[pair_case_index]
            values.append(
                float(
                    np.sum(pair_weights * point_pair_regret)
                    / np.sum(pair_weights)
                )
            )
        case_regret_intervals[name] = interval(np.asarray(values))

        # Count-only.
        draw_regret = np.where(
            pick_cal[None, :],
            np.maximum(-truth_draws, 0),
            np.maximum(truth_draws, 0),
        ).mean(axis=1)
        count_regret_intervals[name] = interval(draw_regret)

        # Combined.
        values = []
        for draw_index in range(combined_count):
            weights = rng.multinomial(
                len(cases),
                [1.0 / len(cases)] * len(cases),
            )
            pair_weights = weights[pair_case_index]
            pair_regret = np.where(
                pick_cal,
                np.maximum(-truth_draws[draw_index], 0),
                np.maximum(truth_draws[draw_index], 0),
            )
            values.append(
                float(
                    np.sum(pair_weights * pair_regret)
                    / np.sum(pair_weights)
                )
            )
        combined_regret_intervals[name] = interval(
            np.asarray(values)
        )

    # Elapsed times.
    w4_by_backend = {}
    for raw in args.w4_completed:
        completed = load_json(Path(raw))
        w4_by_backend[str(completed["backend"])] = completed
    horizons = {}
    for raw in args.w5_completed:
        later = load_json(Path(raw))
        backend = str(later["backend"])
        earlier = w4_by_backend[backend]
        elapsed = (
            datetime.fromisoformat(later["submitted_at"])
            - datetime.fromisoformat(earlier["completed_at"])
        )
        horizons[backend] = {
            "w4_completed_at": earlier["completed_at"],
            "later_submitted_at": later["submitted_at"],
            "elapsed_seconds": elapsed.total_seconds(),
            "elapsed_hours": elapsed.total_seconds() / 3600.0,
            "meets_72_hour_rule": (
                elapsed.total_seconds() >= 72 * 3600
            ),
        }

    predictor_summary = {}
    for name, column in predictor_columns.items():
        predictor_summary[name] = {
            "point": stratified_rho(matched, column),
            "case_cluster_90_interval": interval(
                np.asarray(case_values[name])
            ),
            "w5_count_only_90_interval": interval(
                np.asarray(count_values[name])
            ),
            "combined_case_and_w5_count_90_interval": interval(
                np.asarray(combined_values[name])
            ),
        }

    regret_comparisons = {}
    for left, right in [
        ("combined_gate2_TE1", "fresh_gate"),
        ("stale_T_E", "fresh_gate"),
        ("stale_T_E", "always_calibration"),
        ("combined_gate2_TE1", "always_calibration"),
        ("stale_workload", "fresh_gate"),
    ]:
        regret_comparisons[f"{left}_minus_{right}"] = {
            "point": point_regret[left] - point_regret[right],
            "interpretation": (
                "Negative values favour the policy named first. "
                "Use paired case/count resampling before claiming superiority."
            ),
        }

    output = {
        "horizons": horizons,
        "integrity": integrity,
        "predictors": predictor_summary,
        "partial": {
            "stale_T_E_given_fresh_gate": {
                "point": residual_rho(
                    matched,
                    "w4_T_E",
                    "w5_neg_gate",
                ),
                "case_cluster_90_interval": interval(
                    np.asarray(case_partial_te)
                ),
                "w5_count_only_90_interval": interval(
                    np.asarray(count_partial_te)
                ),
                "combined_case_and_w5_count_90_interval": interval(
                    np.asarray(combined_partial_te)
                ),
            },
            "fresh_gate_given_stale_T_E": {
                "point": residual_rho(
                    matched,
                    "w5_neg_gate",
                    "w4_T_E",
                ),
                "case_cluster_90_interval": interval(
                    np.asarray(case_partial_gate)
                ),
                "w5_count_only_90_interval": interval(
                    np.asarray(count_partial_gate)
                ),
                "combined_case_and_w5_count_90_interval": interval(
                    np.asarray(combined_partial_gate)
                ),
            },
            "combined_minus_gate_transfer": {
                "point": (
                    stratified_rho(
                        matched,
                        "combo_gate2_TE1",
                    )
                    - stratified_rho(
                        matched,
                        "w5_neg_gate",
                    )
                ),
                "case_cluster_90_interval": interval(
                    np.asarray(case_difference_combo_gate)
                ),
                "combined_case_and_w5_count_90_interval": interval(
                    np.asarray(combined_difference_combo_gate)
                ),
            },
        },
        "decision_regret": {
            name: {
                "point": point_regret[name],
                "case_cluster_90_interval": (
                    case_regret_intervals[name]
                ),
                "w5_count_only_90_interval": (
                    count_regret_intervals[name]
                ),
                "combined_case_and_w5_count_90_interval": (
                    combined_regret_intervals[name]
                ),
            }
            for name in point_regret
        },
        "decision_regret_comparisons": regret_comparisons,
        "resolved_nonidentical_pairs": int(
            pairs["resolved"].sum()
        ),
        "nonidentical_pairs": int((~pairs["identical"]).sum()),
        "identical_pairs": int(pairs["identical"].sum()),
        "limitations": [
            (
                "The later jobs do not meet the registered 72-hour rule "
                "unless every horizon entry says meets_72_hour_rule=true."
            ),
            (
                "Count sensitivities resample the later workload counts. "
                "They do not propagate measurement uncertainty of stale "
                "Window-4 T_E or the stale workload predictor."
            ),
        ],
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "robust_validity_audit.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "predictor": name,
                **values,
            }
            for name, values in predictor_summary.items()
        ]
    ).to_csv(out / "robust_predictor_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "policy": name,
                **values,
            }
            for name, values in output[
                "decision_regret"
            ].items()
        ]
    ).to_csv(out / "robust_decision_table.csv", index=False)

    print(json.dumps(output, indent=2))
    print("Saved:", out.resolve())


if __name__ == "__main__":
    main()
