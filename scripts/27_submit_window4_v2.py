from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Any

from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def refreshed_properties(backend: Any) -> Any:
    try:
        return backend.properties(refresh=True)
    except TypeError:
        return backend.properties()


def extract_counts(pub_result: Any, circuit: Any) -> dict[str, int]:
    if not circuit.cregs:
        raise RuntimeError("Circuit has no classical register.")
    name = circuit.cregs[0].name
    register = getattr(pub_result.data, name, None)
    if register is None or not hasattr(register, "get_counts"):
        raise RuntimeError(f"Result lacks count register {name!r}.")
    return {str(k).replace(" ", ""): int(v) for k, v in register.get_counts().items()}


def job_usage(job: Any) -> Any:
    value = getattr(job, "usage", None)
    if callable(value):
        try:
            return value()
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
    return value


def main() -> None:
    ap = argparse.ArgumentParser(description="Guarded variable-shot submission for Window 4/5 V2.")
    ap.add_argument("--account", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--reserve", type=float, default=60.0)
    ap.add_argument("--safety", type=float, default=1.3)
    ap.add_argument("--run-label", default="primary")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--confirm", default="")
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qpy_path = Path(manifest["qpy_file"])
    if not qpy_path.exists():
        qpy_path = manifest_path.parent / qpy_path.name
    if sha256_file(qpy_path) != str(manifest["qpy_sha256"]):
        raise SystemExit("QPY hash mismatch.")
    protocol_path = Path(manifest["protocol_file"])
    if protocol_path.exists() and sha256_file(protocol_path) != str(manifest["protocol_sha256"]):
        raise SystemExit("Protocol hash mismatch.")
    with qpy_path.open("rb") as handle:
        circuits = list(qpy.load(handle))
    if len(circuits) != int(manifest["pub_count"]) or len(circuits) != len(manifest["entries"]):
        raise SystemExit("QPY/manifest circuit-count mismatch.")

    service = QiskitRuntimeService(name=args.account)
    usage_before = service.usage()
    remaining = usage_before.get("usage_remaining_seconds") if isinstance(usage_before, dict) else None
    backend = service.backend(str(manifest["backend"]))
    status = backend.status()
    properties = refreshed_properties(backend)
    if properties is None:
        raise SystemExit("Backend returned no properties.")

    target_ops = {str(name) for name in backend.target.operation_names}
    used_ops = {str(name) for circuit in circuits for name in circuit.count_ops()}
    unsupported = used_ops - target_ops - {"barrier", "measure", "delay"}
    guarded = float(manifest["estimated_quantum_seconds"]) * args.safety
    gate_pass = (
        bool(status.operational)
        and not unsupported
        and remaining is not None
        and float(remaining) - guarded >= args.reserve
    )
    confirmation = f"SUBMIT-W4V2-{manifest['backend']}-{manifest['qpy_sha256'][:10]}"

    print("Backend:", manifest["backend"])
    print("Operational:", status.operational, "pending_jobs:", getattr(status, "pending_jobs", None))
    print("PUBs:", len(circuits), "circuit-shots:", manifest["total_circuit_shots"])
    print("Remaining seconds:", remaining)
    print("Estimated seconds:", manifest["estimated_quantum_seconds"], "guarded:", round(guarded, 2))
    print("Reserve after job:", args.reserve)
    print("Unsupported instructions:", sorted(unsupported) or "none")
    print("Pre-submit calibration:", str(getattr(properties, "last_update_date", None)))
    print("GATE:", "PASS" if gate_pass else "FAIL")
    print("Required confirmation:", confirmation)

    if not args.submit:
        raise SystemExit(0 if gate_pass else 1)
    if not gate_pass or args.confirm != confirmation:
        raise SystemExit("Submission refused: gate failed or confirmation mismatch.")

    run_dir = manifest_path.parent / "runs" / str(manifest["backend"]) / args.run_label
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"Run directory already contains records: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    pre_timestamp = str(getattr(properties, "last_update_date", None))
    write_json(
        run_dir / "pre_submit_snapshot.json",
        {
            "captured_at": utc_now(),
            "calibration_timestamp": pre_timestamp,
            "properties": properties.to_dict(),
        },
    )
    record = {
        "status": "submitting",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "qpy_sha256": manifest["qpy_sha256"],
        "protocol_sha256": manifest["protocol_sha256"],
        "backend": manifest["backend"],
        "run_label": args.run_label,
        "usage_before": usage_before,
        "remaining_seconds_before": remaining,
        "pre_submit_calibration_timestamp": pre_timestamp,
        "created_at": utc_now(),
    }
    write_json(run_dir / "job_record.json", record)

    sampler = SamplerV2(mode=backend)
    try:
        sampler.options.environment.job_tags = ["unige", "preflight", "window4-v2", str(manifest["label"]), str(manifest["backend"])]
    except Exception:
        pass
    try:
        sampler.options.max_execution_time = max(120, int(min(float(remaining) - args.reserve, guarded * 3)))
    except Exception:
        pass

    pubs = [(circuit, None, int(entry["shots"])) for circuit, entry in zip(circuits, manifest["entries"])]
    job = sampler.run(pubs)
    record.update({"status": "submitted", "job_id": job.job_id(), "submitted_at": utc_now()})
    write_json(run_dir / "job_record.json", record)
    print("Runtime job ID:", job.job_id())
    print("Saved immediately:", run_dir / "job_record.json")

    if not args.wait:
        return

    try:
        result = job.result()
        if len(result) != len(circuits):
            raise RuntimeError(f"Result length {len(result)} != circuit count {len(circuits)}")
        count_rows = []
        for index, (pub_result, circuit, entry) in enumerate(zip(result, circuits, manifest["entries"])):
            counts = extract_counts(pub_result, circuit)
            observed = sum(counts.values())
            expected = int(entry["shots"])
            if observed != expected:
                raise RuntimeError(f"Shot mismatch at PUB {index}: {observed} != {expected}")
            count_rows.append(counts)

        post_properties = refreshed_properties(backend)
        post_timestamp = str(getattr(post_properties, "last_update_date", None))
        write_json(
            run_dir / "post_completion_snapshot.json",
            {
                "captured_at": utc_now(),
                "calibration_timestamp": post_timestamp,
                "properties": post_properties.to_dict(),
            },
        )
        try:
            usage_after = service.usage()
        except Exception as exc:
            usage_after = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            metrics = job.metrics()
        except Exception as exc:
            metrics = {"error": f"{type(exc).__name__}: {exc}"}
        completed = {
            **record,
            "status": str(job.status()),
            "completed_at": utc_now(),
            "counts": count_rows,
            "job_usage": job_usage(job),
            "job_metrics": metrics,
            "service_usage_after": usage_after,
            "post_completion_calibration_timestamp": post_timestamp,
            "calibration_changed_during_run_interval": post_timestamp != pre_timestamp,
            "total_circuit_shots": manifest["total_circuit_shots"],
        }
        write_json(run_dir / "completed.json", completed)
        record.update({"status": "completed", "completed_at": completed["completed_at"]})
        write_json(run_dir / "job_record.json", record)
        print("Completed record:", run_dir / "completed.json")
        print("Calibration changed during run:", completed["calibration_changed_during_run_interval"])
        print("Usage after:", usage_after)
    except Exception as exc:
        try:
            metrics = job.metrics()
        except Exception as metrics_exc:
            metrics = {"error": f"{type(metrics_exc).__name__}: {metrics_exc}"}
        try:
            usage_after = service.usage()
        except Exception as usage_exc:
            usage_after = {"error": f"{type(usage_exc).__name__}: {usage_exc}"}
        failure = {
            **record,
            "status": str(job.status()),
            "failed_at": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "job_usage": job_usage(job),
            "job_metrics": metrics,
            "service_usage_after": usage_after,
        }
        write_json(run_dir / "failed.json", failure)
        record.update({"status": "failed", "failed_at": failure["failed_at"]})
        write_json(run_dir / "job_record.json", record)
        raise


if __name__ == "__main__":
    main()
