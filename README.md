# PREFLIGHT-PT: code-only research release

Python source code supporting the study:

> **Application-structured Clifford probes complement calibration data for QAOA workloads over three days**  
> Aishwarya Vishwakarma, Department of Quantum Matter Physics, University of Geneva.

This is a **minimal, code-only repository** for the PREFLIGHT-PT experiment on weighted-MaxCut QAOA and application-structured Clifford probes on IBM Heron processors (`ibm_fez` and `ibm_kingston`). It makes a core set of circuit-construction, guarded hardware-submission, and analysis scripts public. The study compares application-observable Clifford probes, a single-edge control, and calibration-derived evidence, including repeated measurements through 72.1 hours.

**Scope:** This package is **not** a stand-alone reproduction bundle for all manuscript figures and results. It does **not** contain the experimental datasets, original frozen protocol files, complete independent re-analysis, final corrected output-edge pairing analysis, or figure-generation scripts. Those supporting materials may be requested from the corresponding author, subject to availability and applicable sharing conditions. No dataset DOI is claimed here.

## Files

```text
preflight-pt-qaoa/
├── README.md
├── CITATION.cff
├── LICENSE
├── VERSION
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── 25_build_window4_v2.py
│   ├── 26_analyze_window4_v2.py
│   ├── 27_submit_window4_v2.py
│   ├── 28_analyze_window5_validity.py
│   └── 29_audit_validity_window.py
└── tests/
    └── test_repository.py
```

| Script | Role |
| --- | --- |
| `25_build_window4_v2.py` | Build workload, single-edge control, and all-edge Clifford-probe circuits from pre-existing source QPY and manifest files. |
| `26_analyze_window4_v2.py` | Analyze registered same-window workload, probe and calibration comparisons. |
| `27_submit_window4_v2.py` | Perform checked IBM Quantum Runtime dry runs, submission and collection. |
| `28_analyze_window5_validity.py` | Analyze later exact-circuit workload and control repeats against earlier measurements. |
| `29_audit_validity_window.py` | Assess case-resampling, shot-count and combined sensitivity of later-window results. |

The script names retain the experiment's original Window 4/5 nomenclature. The files have **not** been renamed or rewritten to make this a general-purpose library.

## Installation and static checks

Python 3.12 was used for development. From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
python -m pytest -q
```

The package uses Qiskit, Qiskit Aer, Qiskit IBM Runtime, NumPy, pandas and SciPy. For a description of a script's command-line arguments, run:

```bash
python scripts/25_build_window4_v2.py --help
python scripts/26_analyze_window4_v2.py --help
python scripts/27_submit_window4_v2.py --help
python scripts/28_analyze_window5_validity.py --help
python scripts/29_audit_validity_window.py --help
```

These commands only show the arguments. The repository tests are static source/packaging checks; they do not validate the numerical manuscript findings or require access to IBM Quantum.

## Required external research inputs

The programs are **not directly runnable on the published measurements from this repository alone**. Re-analysis requires the original data in the formats expected by the scripts, including as applicable:

- QPY circuit bundles and matching JSON manifests, with SHA-256 hashes;
- IBM Runtime counts and completed-run JSON records;
- relevant calibration snapshots and derived source/pair tables;
- the original, frozen experiment protocol file for building a new bundle (`--protocol-file`).

For example, to inspect the builder's required inputs:

```bash
python scripts/25_build_window4_v2.py --help
```

The `--protocol-file` flag is **required for a successful build** because the builder hashes the file for provenance. Its default path is historical and is not populated in this code-only release. Supply the actual frozen protocol from your local research records; do not substitute a newly written protocol and claim that it was pre-registered. The analysis scripts likewise require preserved experiment inputs, supplied by their command-line flags.

**Important analysis distinction:** These scripts document the original experimental workflow and registered analysis utilities. The manuscript also describes a later, post-hoc correction of output-observable/cost-edge pairing. The final corrected-statistic and figure-production pipelines are **not** included here; do not claim that this repository alone reproduces every published headline figure or statistic.

## IBM Runtime safety

Do not publish API tokens, saved accounts, project CRNs, backend-access credentials, QPY files or experimental result exports in this repository. Configure any IBM Quantum account **outside** the repository using the official Qiskit IBM Runtime account workflow. The submission script is designed to operate in dry-run mode unless `--submit` and an exact printed confirmation are supplied. A dry run may still access IBM Quantum APIs. Actual submission can consume charged quantum-processing time.

Always inspect the backend, estimated job cost, account allocation and selected circuit bundle before running an experiment. The hardware scripts are research artifacts tied to the particular campaign, not a maintained production service.

## Availability and citation

**Code:** this repository; archive a GitHub release on Zenodo for an immutable **software DOI**. Until a release is archived, no software DOI has been assigned.

**Data and other supporting materials:** raw measurements, QPYs, calibration records, frozen protocols, additional analyses and figure-generation inputs are not included here. Contact the corresponding author at `aishwarya.vishwakarma@unige.ch` to request materials. Sharing is subject to the author retaining and being able to provide the requested files; availability on request should not be taken to imply public deposit.

Use GitHub's **Cite this repository** feature after the repository is created, or the `CITATION.cff` metadata. When a Zenodo software DOI has been assigned, it can be added to `CITATION.cff` and the corresponding manuscript *Code availability* statement. Do **not** invent or pre-fill a DOI.

**Related manuscript:** *Application-structured Clifford probes complement calibration data for QAOA workloads over three days* (manuscript; journal publication details to be added when confirmed).

## License

MIT; see [LICENSE](LICENSE).
