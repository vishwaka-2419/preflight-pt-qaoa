"""Check the patched builder functions against the executed circuits of the reference job.
usage: python3 -I test_readout_fix.py <v3/results dir> <patched builder path>
"""
import importlib.util, json, sys
import qiskit.qpy
from qiskit.quantum_info import Pauli

R, BUILDER = sys.argv[1:3]
spec = importlib.util.spec_from_file_location("builder", BUILDER); b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
same_total, edges_total = 0, 0
for be in ["ibm_fez", "ibm_kingston"]:
    man = json.load(open(f"{R}/window4_v2/{be}_window4_v2_manifest.json"))
    with open(f"{R}/window4_v2/{be}_window4_v2.qpy", "rb") as fh:
        circs = qiskit.qpy.load(fh)
    same = 0
    for e in man["entries"]:
        if e["kind"] != "workload":
            continue
        w = circs[e["circuit_index"]]; active = [int(x) for x in e["physical_qubits"]]
        ro = b.readout_map(w, active)
        for u, v, _ in e["edges"]:
            edges_total += 1
            same += {ro[int(u)], ro[int(v)]} == {int(u), int(v)}
        if be == "ibm_fez" and e["source_index"] == 1:
            body = b.snap_body(w); cl, small = b.local_clifford(body, active)
            print("source 1 readout (vertex -> physical):", {u: active[l] for u, l in sorted(ro.items())})
            for u, v, _ in e["edges"]:
                lab = ["I"] * len(active); lab[ro[int(u)]] = "Z"; lab[ro[int(v)]] = "Z"
                P = Pauli("".join(reversed(lab))).evolve(cl, frame="h")
                print("  edge", (int(u), int(v)), "propagated", P.to_label())
    print(be, "edges whose registered pair equals the readout pair:", same)
    same_total += same
print("total", same_total, "of", edges_total)
