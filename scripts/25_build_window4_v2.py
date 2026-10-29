from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Clifford, Pauli, StabilizerState


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def circuit_signature(circuit: QuantumCircuit) -> str:
    rows = []
    for item in circuit.data:
        op = item.operation
        params = []
        for value in getattr(op, "params", []):
            try:
                params.append(round(float(value), 14))
            except Exception:
                params.append(str(value))
        rows.append(
            {
                "name": str(op.name),
                "params": params,
                "q": [int(circuit.find_bit(q).index) for q in item.qubits],
                "c": [int(circuit.find_bit(c).index) for c in item.clbits],
                "duration": str(getattr(op, "duration", None)),
                "unit": str(getattr(op, "unit", None)),
            }
        )
    payload = json.dumps(
        {"num_qubits": circuit.num_qubits, "num_clbits": circuit.num_clbits, "instructions": rows},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def snap_body(workload: QuantumCircuit) -> QuantumCircuit:
    body = QuantumCircuit(*workload.qregs, name="snapped_body")
    for item in workload.data:
        name = str(item.operation.name)
        if name in {"measure", "barrier"}:
            continue
        if name == "rz":
            theta = float(item.operation.params[0])
            k = int(round(theta / (math.pi / 2)))
            body.rz(k * math.pi / 2, item.qubits[0])
        else:
            body.append(item.operation, item.qubits, item.clbits)
    return body


def local_clifford(body: QuantumCircuit, active: list[int]) -> tuple[Clifford, QuantumCircuit]:
    inverse = {int(q): i for i, q in enumerate(active)}
    small = QuantumCircuit(len(active))
    for item in body.data:
        name = str(item.operation.name)
        physical = [int(body.find_bit(q).index) for q in item.qubits]
        try:
            local = [inverse[q] for q in physical]
        except KeyError as exc:
            raise RuntimeError(f"Body uses a qubit outside the declared active subset: {physical}") from exc
        if name == "rz":
            k = int(round(float(item.operation.params[0]) / (math.pi / 2))) % 4
            if k == 1:
                small.s(local[0])
            elif k == 2:
                small.z(local[0])
            elif k == 3:
                small.sdg(local[0])
        elif name == "sx":
            small.sx(local[0])
        elif name == "x":
            small.x(local[0])
        elif name == "id":
            small.id(local[0])
        elif name == "cz":
            small.cz(*local)
        elif name in {"delay", "barrier"}:
            continue
        else:
            raise RuntimeError(f"Unsupported operation in snapped Clifford body: {name}")
    return Clifford(small), small


def pauli_axes(pauli: Pauli) -> dict[int, str]:
    return {
        q: ("Y" if bool(pauli.z[q]) and bool(pauli.x[q]) else "Z" if bool(pauli.z[q]) else "X")
        for q in range(pauli.num_qubits)
        if bool(pauli.z[q]) or bool(pauli.x[q])
    }


def prep_balanced(circuit: QuantumCircuit, qubit: int, axis: str, sign: int) -> None:
    minus = int(sign) < 0
    if axis == "Z":
        if minus:
            circuit.sx(qubit)
            circuit.sx(qubit)
        else:
            circuit.sx(qubit)
            circuit.rz(math.pi, qubit)
            circuit.sx(qubit)
            circuit.rz(-math.pi, qubit)
    elif axis == "X":
        circuit.rz(math.pi / 2, qubit)
        circuit.sx(qubit)
        circuit.rz(math.pi / 2, qubit)
        if minus:
            circuit.rz(math.pi, qubit)
    elif axis == "Y":
        circuit.rz(math.pi / 2, qubit)
        circuit.sx(qubit)
        circuit.rz(0.0 if minus else math.pi, qubit)
    else:
        raise ValueError(axis)


def build_small_prep(n: int, axes: dict[int, str], signs: dict[int, int]) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    for q in range(n):
        prep_balanced(qc, q, axes.get(q, "Z"), signs[q])
    return qc


def find_sign_twirls(
    small_body: QuantumCircuit,
    axes: dict[int, str],
    edges: list[tuple[int, int]],
    twirls: int,
    rng: random.Random,
    max_attempts: int = 2000,
) -> list[tuple[dict[int, int], dict[str, int]]]:
    if twirls < 4:
        raise ValueError("At least four sign twirls are required.")
    n = small_body.num_qubits
    observables = {}
    for u, v in edges:
        label = ["I"] * n
        label[u] = "Z"
        label[v] = "Z"
        observables[(u, v)] = Pauli("".join(reversed(label)))

    for _ in range(max_attempts):
        out = []
        signs_by_edge = {edge: [] for edge in edges}
        for _r in range(twirls):
            signs = {q: rng.choice((-1, 1)) for q in range(n)}
            prep = build_small_prep(n, axes, signs)
            state = StabilizerState(prep.compose(small_body))
            ideal = {}
            for edge, observable in observables.items():
                value = complex(state.expectation_value(observable))
                if abs(value.imag) > 1e-9 or abs(abs(value.real) - 1.0) > 1e-9:
                    raise RuntimeError(f"Non-deterministic ideal edge {edge}: {value}")
                s = 1 if value.real > 0 else -1
                ideal[f"{edge[0]},{edge[1]}"] = s
                signs_by_edge[edge].append(s)
            out.append((signs, ideal))
        if all(2 <= sum(1 for x in values if x > 0) <= twirls - 2 for values in signs_by_edge.values()):
            return out
    raise RuntimeError("Could not find a sign-twirl design with both ideal signs represented for every edge.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build high-shot Window 4/5 jobs with a balanced sign-twirled all-edge Clifford probe.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--qpy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--w-shots", type=int, default=2048)
    ap.add_argument("--pt-shots", type=int, default=256)
    ap.add_argument("--pt-pairs", type=int, default=2)
    ap.add_argument("--edge-twirls", type=int, default=8)
    ap.add_argument("--edge-shots", type=int, default=64)
    ap.add_argument("--no-edge-probe", action="store_true")
    ap.add_argument("--label", default="window4_v2")
    ap.add_argument("--protocol-file", default="protocols/WINDOW4_V2_PROTOCOL.md")
    ap.add_argument("--seconds-per-shot", type=float, default=0.00028)
    ap.add_argument("--job-overhead-s", type=float, default=3.0)
    ap.add_argument("--hard-max-estimated-seconds", type=float, default=55.0)
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    qpy_path = Path(args.qpy).resolve()
    out_dir = Path(args.out).resolve()
    protocol_path = Path(args.protocol_file).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not protocol_path.exists():
        raise SystemExit(f"Protocol file not found: {protocol_path}")

    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(qpy_path) != str(source.get("qpy_sha256")):
        raise SystemExit("Source QPY hash mismatch.")
    with qpy_path.open("rb") as handle:
        circuits = list(qpy.load(handle))
    if len(circuits) != len(source["entries"]):
        raise SystemExit("Source QPY/manifest length mismatch.")

    by_source: dict[int, dict[tuple[str, Any], tuple[dict[str, Any], QuantumCircuit]]] = defaultdict(dict)
    for entry, circuit in zip(source["entries"], circuits):
        by_source[int(entry["source_index"])][(str(entry["kind"]), entry.get("replicate"))] = (entry, circuit)
    if len(by_source) != int(source["source_count"]):
        raise SystemExit("Unexpected source count.")

    rng = random.Random(args.seed)
    pubs: list[tuple[QuantumCircuit, int, dict[str, Any]]] = []
    group_stats = []

    for source_index in sorted(by_source):
        items = by_source[source_index]
        workload_items = [value for key, value in items.items() if key[0] == "workload"]
        if len(workload_items) != 1:
            raise RuntimeError(f"Source {source_index} has {len(workload_items)} workload entries.")
        workload_entry, workload = workload_items[0]
        signature = circuit_signature(workload)
        common = {
            "source_index": source_index,
            "instance_id": workload_entry["instance_id"],
            "depth": int(workload_entry["depth"]),
            "selector": workload_entry["selector"],
            "n_qubits": int(workload_entry["n_qubits"]),
            "edges": workload_entry["edges"],
            "optimum": float(workload_entry["optimum"]),
            "random_approximation_ratio": float(workload_entry["random_approximation_ratio"]),
            "ideal_approximation_ratio": float(workload_entry["ideal_approximation_ratio"]),
            "physical_qubits": [int(x) for x in workload_entry["physical_qubits"]],
            "source_qpy_sha256": workload_entry["source_qpy_sha256"],
            "source_structure_sha256": signature,
            "candidate_metadata": workload_entry.get("candidate_metadata"),
            "selector_components": workload_entry.get("selector_components"),
        }
        pubs.append((workload, args.w_shots, {"kind": "workload", **common}))

        for replicate in range(args.pt_pairs):
            for kind in ("pt_plus", "pt_minus"):
                key = (kind, replicate)
                if key not in items:
                    raise RuntimeError(f"Missing {key} for source {source_index}")
                entry, circuit = items[key]
                pubs.append(
                    (
                        circuit,
                        args.pt_shots,
                        {
                            "kind": kind,
                            "replicate": replicate,
                            "output_edge": entry.get("output_edge"),
                            "transfer_pauli": entry.get("transfer_pauli"),
                            **common,
                        },
                    )
                )

        if args.no_edge_probe:
            group_stats.append({"source_index": source_index, "edge_count": len(common["edges"]), "group_count": 0})
            continue

        active = common["physical_qubits"]
        body = snap_body(workload)
        clifford, small_body = local_clifford(body, active)
        n = len(active)
        propagated = {}
        for u, v, _weight in common["edges"]:
            label = ["I"] * n
            label[int(u)] = "Z"
            label[int(v)] = "Z"
            propagated[(int(u), int(v))] = Pauli("".join(reversed(label))).evolve(clifford, frame="h")

        groups: list[dict[str, Any]] = []
        for edge in sorted(propagated, key=lambda e: -len(pauli_axes(propagated[e]))):
            axes = pauli_axes(propagated[edge])
            for group in groups:
                if all(group["axes"].get(q, axis) == axis for q, axis in axes.items()):
                    group["edges"].append(edge)
                    group["axes"].update(axes)
                    break
            else:
                groups.append({"edges": [edge], "axes": dict(axes)})

        for group_index, group in enumerate(groups):
            designs = find_sign_twirls(
                small_body,
                group["axes"],
                group["edges"],
                args.edge_twirls,
                rng,
            )
            for twirl_index, (signs, ideal) in enumerate(designs):
                probe = QuantumCircuit(workload.num_qubits, n, name=f"edge_probe_{source_index}_{group_index}_{twirl_index}")
                for local_q, physical_q in enumerate(active):
                    prep_balanced(probe, physical_q, group["axes"].get(local_q, "Z"), signs[local_q])
                probe.barrier(*active)
                for item in body.data:
                    qargs = [probe.qubits[int(body.find_bit(q).index)] for q in item.qubits]
                    probe.append(item.operation, qargs, [])
                probe.barrier(*active)
                for local_q, physical_q in enumerate(active):
                    probe.measure(physical_q, local_q)
                if probe.count_ops().get("cz", 0) != workload.count_ops().get("cz", 0):
                    raise RuntimeError("Edge probe body CZ count differs from workload.")
                pubs.append(
                    (
                        probe,
                        args.edge_shots,
                        {
                            "kind": "edge_probe",
                            "group": group_index,
                            "twirl": twirl_index,
                            "group_edges": [list(edge) for edge in group["edges"]],
                            "prep_axes": {str(k): v for k, v in group["axes"].items()},
                            "prep_signs": {str(k): int(v) for k, v in signs.items()},
                            "ideal_zz": ideal,
                            **common,
                        },
                    )
                )
        group_stats.append({"source_index": source_index, "edge_count": len(common["edges"]), "group_count": len(groups)})

    order = list(range(len(pubs)))
    rng.shuffle(order)
    pubs = [pubs[i] for i in order]
    output_qpy = out_dir / f"{source['backend']}_{args.label}.qpy"
    with output_qpy.open("wb") as handle:
        qpy.dump([item[0] for item in pubs], handle)

    total_shots = sum(int(item[1]) for item in pubs)
    estimate = total_shots * args.seconds_per_shot + args.job_overhead_s
    if estimate > args.hard_max_estimated_seconds:
        output_qpy.unlink(missing_ok=True)
        raise SystemExit(
            f"Estimated {estimate:.1f}s exceeds hard build limit {args.hard_max_estimated_seconds:.1f}s."
        )

    counts = Counter(item[2]["kind"] for item in pubs)
    manifest = {
        "schema_version": 2,
        "campaign_name": "preflight_pt_high_shot_edge_probe",
        "label": args.label,
        "backend": source["backend"],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "source_qpy": str(qpy_path),
        "source_qpy_sha256": sha256_file(qpy_path),
        "protocol_file": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "seed": args.seed,
        "qpy_file": str(output_qpy),
        "qpy_sha256": sha256_file(output_qpy),
        "pub_count": len(pubs),
        "total_circuit_shots": total_shots,
        "estimated_quantum_seconds": round(estimate, 3),
        "seconds_per_circuit_shot": args.seconds_per_shot,
        "job_overhead_seconds": args.job_overhead_s,
        "shots": {
            "workload": args.w_shots,
            "pt": args.pt_shots,
            "edge_probe_per_twirl": args.edge_shots,
        },
        "pt_pairs": args.pt_pairs,
        "edge_sign_twirls": 0 if args.no_edge_probe else args.edge_twirls,
        "kind_counts": dict(counts),
        "edge_groups": group_stats,
        "entries": [
            {"circuit_index": index, "shots": int(item[1]), **item[2]}
            for index, item in enumerate(pubs)
        ],
    }
    manifest_path_out = out_dir / f"{source['backend']}_{args.label}_manifest.json"
    write_json(manifest_path_out, manifest)

    print("WINDOW BUILD PASSED")
    print("Backend:", source["backend"])
    print("PUBs:", len(pubs), dict(counts))
    print("Circuit-shots:", total_shots)
    print("Estimated QPU seconds:", round(estimate, 2))
    print("Edge groups total:", sum(item["group_count"] for item in group_stats))
    print("QPY:", output_qpy)
    print("QPY SHA-256:", manifest["qpy_sha256"])
    print("Manifest:", manifest_path_out)
    print("Protocol SHA-256:", manifest["protocol_sha256"])


if __name__ == "__main__":
    main()
