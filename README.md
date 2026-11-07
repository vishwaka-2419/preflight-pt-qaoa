# PREFLIGHT-PT: code-only research release

Python source code supporting the article:

> **Compiled-circuit Clifford probes predict QAOA performance beyond calibration scores on superconducting processors**  
> Aishwarya Vishwakarma, Department of Quantum Matter Physics, University of Geneva (2026; submitted).

This is a **code-only repository** for the PREFLIGHT-PT experiment on weighted-MaxCut QAOA and compiled-circuit Clifford probes on IBM Heron processors (`ibm_fez` and `ibm_kingston`). It contains the circuit-construction, guarded hardware-submission and registered-analysis scripts of the experiment (version 1.0.0), and, from version 1.0.1, the recovery of the readout-pair statistic reported in the article and the additional analyses of its Supplementary Information.

**Scope:** The repository does **not** contain the experimental datasets (QPY circuit bundles, manifests, count data, calibration snapshots, job records) or the frozen protocol file, so the scripts are **not** runnable on the published measurements from this repository alone. The derived per-edge correlator record and the analysis tables from which every reported statistic is computed accompany the article as Supplementary Data (Zenodo); the raw record is available from the corresponding author on request.

## Files

```text
preflight-pt-qaoa/
├── README.md
├── CITATION.cff
├── LICENSE
├── VERSION
├── requirements.txt
├── .zenodo.json
├── .gitignore
├── scripts/
│   ├── 25_build_window4_v2.py        registered builder; v1.0.1 adds --output-pairing
│   ├── 26_analyze_window4_v2.py
│   ├── 27_submit_window4_v2.py
│   ├── 28_analyze_window5_validity.py
│   ├── 29_audit_validity_window.py
│   ├── probe_pairing_check.py        v1.0.1: recovery of the readout-pair statistic
│   ├── ablations.py                  v1.0.1: variants, resampling, direct-sampling comparison
│   ├── paired72.py                   v1.0.1: paired differences of rank correlation and regret
│   ├── sim_noise_models.py           v1.0.1: estimator under non-Pauli noise models
│   ├── make_extra_tables.py          v1.0.1: Supplementary Tables 11 to 15
│   └── check_readout_fix.py          v1.0.1: check of the patched builder against the executed circuits
└── tests/
    └── test_repository.py
```

| Script | Role |
| --- | --- |
| `25_build_window4_v2.py` | Build workload, single-edge control and all-edge Clifford-probe circuits from source QPY and manifest files. |
| `26_analyze_window4_v2.py` | Analyse the registered same-window workload, probe and calibration comparisons. |
| `27_submit_window4_v2.py` | Checked IBM Quantum Runtime dry runs, submission and collection. |
| `28_analyze_window5_validity.py` | Analyse later exact-circuit workload and control repeats against earlier measurements. |
| `29_audit_validity_window.py` | Case-resampling, shot-count and combined sensitivity of later-window results. |
| `probe_pairing_check.py` | Recover the all-edge probe statistic on the readout qubits of the cost edges from the executed probe circuits (ideal signs by stabilizer simulation; ordinary least squares with intercept per edge); writes the per-edge table with standard errors and the per-input correlator record. |
| `ablations.py` | Variants of the probe statistic with paired case-bootstrap differences, graph-level and leave-one-graph-out resampling, probe-shot resampling, direct workload sampling at a similar average shot budget, layout-order reversals. |
| `paired72.py` | Paired case-bootstrap differences of rank correlation and mean regret in the reference job and at 72.1 hours (point estimates are observed differences). |
| `sim_noise_models.py` | Exact density-matrix simulation of the executed Clifford bodies and inputs under Pauli, amplitude-damping, coherent, asymmetric-readout and hardware-like noise models. |
| `make_extra_tables.py` | LaTeX sources of Supplementary Tables 11 to 15 from the outputs of the two scripts above. |
| `check_readout_fix.py` | Verify `readout_map` and the propagated observables of the patched builder against the executed circuits. |

The script names retain the experiment's original Window 4/5 nomenclature. The files have **not** been renamed or rewritten to make this a general-purpose library.

## Note on the all-edge probe builder (v1.0.1)

The probe circuits executed in the study were built with `25_build_window4_v2.py` as released in v1.0.0. This registered builder places the output observable of cost edge (u, v) on the u-th and v-th qubits of the sorted compilation subset and measures the k-th qubit of the subset into classical bit k. The compiler's layout and routing stages permute the vertices within the subset, so for most edges these are not the qubits that the workload circuit measures into classical bits u and v (the two pairs coincide for 21 of the 872 edges of the study). The executed probe circuits therefore measure the registered statistic described in the article; the statistic reported in the article was recovered from them with `probe_pairing_check.py`.

From v1.0.1 the builder accepts `--output-pairing {registered,readout}`: `registered` (default) reproduces the executed circuits exactly; `readout` places the observable of each edge on the qubits that the workload circuit measures into the classical bits of the two vertices, and should be used for new experiments.

## Installation and static checks

Python 3.12 was used for development. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
```

The repository tests are static source and packaging checks; they do not validate the numerical findings and do not require access to IBM Quantum. Each script prints its arguments with `--help` (registered scripts) or in its module docstring (v1.0.1 scripts).

## Required external research inputs

Re-analysis requires the original data in the formats expected by the scripts: QPY circuit bundles and matching JSON manifests with SHA-256 hashes; IBM Runtime counts and completed-run JSON records; calibration snapshots and derived source and pair tables; and, for building a new bundle, the original frozen protocol file (`--protocol-file`, required because the builder hashes it for provenance; do not substitute a newly written protocol and claim that it was pre-registered).

The v1.0.1 analysis scripts take the directories of the preserved archive as arguments (see the docstrings). `sim_noise_models.py` needs only the per-input correlator record written by `probe_pairing_check.py`, which is part of the Supplementary Data.

## IBM Runtime safety

Do not publish API tokens, saved accounts, project CRNs, backend-access credentials, QPY files or experimental result exports in this repository. Configure any IBM Quantum account **outside** the repository using the official Qiskit IBM Runtime account workflow. The submission script operates in dry-run mode unless `--submit` and an exact printed confirmation are supplied. A dry run may still access IBM Quantum APIs; actual submission can consume charged quantum-processing time.

## Availability and citation

**Code:** this repository; every release is archived on Zenodo. Concept DOI (resolves to the latest version): [10.5281/zenodo.23782910](https://doi.org/10.5281/zenodo.23782910). Version 1.0.0: [10.5281/zenodo.23782911](https://doi.org/10.5281/zenodo.23782911).

**Data:** the per-edge correlator record, the recovered edge transfers with standard errors and the derived analysis tables accompany the article as Supplementary Data. Raw measurements, QPYs, calibration records and frozen protocols are available from the corresponding author (`aishwarya.vishwakarma@unige.ch`) on request.

Use GitHub's **Cite this repository** feature or the `CITATION.cff` metadata.

## Version history

- **1.0.1** (2026-11-07): `--output-pairing` option in the builder (default unchanged); recovery script; ablation, paired-comparison, simulation and table scripts; `check_readout_fix.py`; tests updated.
- **1.0.0** (2026-10-29): registered pipeline as executed in the experiment.

## License

MIT; see [LICENSE](LICENSE).
