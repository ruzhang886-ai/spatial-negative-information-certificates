# Spatially Redundant Negative-Information Certificates

This repository is the reproducibility package for the manuscript **Spatially
Redundant Negative-Information Certificates for Directional Radio-Emitter
Localization under Bounded Persistent Missed Detections** by Ru Zhang and
Dekuan Fu.

The study isolates the post-detection localization stage for one directional
emitter. It compares a two-point negative-information rule with spatially
redundant probes under bounded persistent missed detections, correlated
shadowing, nonideal antenna gain, and an external Berlin ray-tracing holdout.

## Contents

- `experiments/src`: geometry, certificate, policy, simulator, and external-data adapters.
- `experiments/scripts`: reproducible experiment and figure-generation scripts.
- `experiments/tests`: deterministic certificate tests.
- `experiments/runs`: compact aggregate results, documented experiment configurations, and hash manifests.
- `figures`: manuscript figures in PNG and PDF.
- `CITATION.cff`: author and citation metadata.
- `.zenodo.json`: metadata used when Zenodo archives a GitHub release.
- `LICENSE` and `LICENSE_SCOPE.md`: code and non-code license terms.

Compact result files omit the per-run arrays but preserve the metadata and
aggregate summaries used by Tables 1--8. The full raw outputs are intended for
the separate release asset `Full_Raw_Results_v1.0.0.zip`;
`experiments/runs/raw_results_sha256.tsv` records the expected size and SHA-256
of every recorded local result file. `SHA256SUMS.txt` verifies the files stored in
this repository snapshot.

## Environment

The recorded software environment was Python 3.12.7 with NumPy 2.4.6, Matplotlib
3.10.9, and Pillow 12.2.0. Create an isolated environment and install:

```bash
python -m pip install -r requirements.txt
```

## Fast verification

From the repository root:

```bash
python experiments/run_checks.py
```

This command executes the deterministic unit tests. Before packaging, the
authors separately checked the manuscript reference order, abstract sample
claims, and all entries of Tables 1--8 against the recorded result files.

## Re-running experiments

Experiment scripts are in `experiments/scripts`. The holdout experiment entry point
is:

```bash
cd experiments
python scripts/run_holdout.py
```

Some experiments are computationally expensive. Keep regenerated full outputs
outside normal Git history and verify them against the documented experiment configuration and
hash files before replacing a recorded result.

## External ray-tracing holdout

The 4.5 GB RMDirectionalBerlin archive is deliberately excluded. Download it
from the official Zenodo record:

<https://doi.org/10.5281/zenodo.15721131>

Then, from `experiments`, either read the ZIP directly or prepare the required
subset:

```bash
python scripts/prepare_rmdirectional_dataset.py <path-to-RMBerlinDirectional.zip>
python scripts/run_rmdirectional_external_audit.py <path-to-RMBerlinDirectional.zip> --sample-limit 200 --histories-per-sample 8 --thresholds-db -115 -105 -95 --output runs/rmdirectional_external_audit_200.json
```

The independent 224-tile holdout uses the documented sample list and file hashes in
`experiments/runs/rmdirectional_external_holdout_protocol_v1.json`.

## Scope and interpretation

The simulator is an independent research implementation; it is not the contest
server and does not use a team number or contest interface. A hard deletion is
permitted only after the continuous certificate is satisfied within its stated
model and false-negative budget. Counterexamples and unresolved checks trigger
the conservative fallback. The Berlin experiment is an external stress test,
not a formal safety certificate for arbitrary urban propagation.

## License

Python source code is released under the MIT License. Original figures, results, documented experiment configurations, and documentation are released under CC BY 4.0. See `LICENSE_SCOPE.md` for the exact scope and third-party
exclusions.
