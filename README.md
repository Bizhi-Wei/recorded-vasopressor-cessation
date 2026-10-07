# Recorded vasopressor cessation in electronic health records

Author: Bizhi Wei, Independent Researcher, China. Public source repository: https://github.com/Bizhi-Wei/recorded-vasopressor-cessation. The source archive is also supplied as S2 File for the manuscript.

The scripts reproduce the frozen reconstruction, descriptive dose summaries, Paused/gap sensitivities and supplementary audits. No analysis was executed while assembling this archive. Previously computed inference scripts are included for the supplementary record; their presence is not endorsement of causal exchangeability or of the unreliable CCW effect.

Obtain MIMIC-IV 3.1 separately through https://physionet.org/content/mimiciv/3.1/ with credentialing, required training and a data-use agreement. Never distribute raw inputs, protected caches or participant-level outputs. Only code, protocols and the aggregate 12-row figure source are included here.

Layout: create a project directory beside hosp/ and icu/ raw CSV folders; retain scripts/, docs/ and figures/. Stage 01 creates data_derived/ preparation caches. Stage 29 applies the definitive stricter cardiac-procedure and low-dose definition; the earlier preparatory phenotype is not the final analysis phenotype. Stages 30–31 construct protected extraction/datasets. Stage 43 builds descriptive summaries; stage 50 builds the two requested measurement sensitivities. Stages 45–46 audit record observability. Stages 32–42 retain supplementary comparability/inference diagnostics. Run only stages needed for an independent reproduction, outside the author's immutable frozen snapshot.

Dependencies include Python, numpy, pandas, scipy, statsmodels, patsy, scikit-learn, duckdb, threadpoolctl and matplotlib; rendering scripts additionally use python-docx where indicated. This appendix preserves source code; no fresh installation or full raw-data reproduction was run for this author-information amendment. Intermediate filenames and code dependencies are retained as in the frozen analysis. The clinical comparability and intended-withdrawal limitations remain in the manuscript.

To reproduce Fig 1, run figures/figure1.py with the included figure1_source.csv. It displays medians and IQRs of admission-level two-hour mean doses, not continuous individual trajectories. File hashes are listed in source_manifest.json. Frozen scripts retain their original bytes; the source manifest distinguishes the 2026-09-15 frozen scripts from the later figure script.

## Frozen source provenance

The 31 scripts under scripts/ preserve the exact bytes of the author's 2026-09-15 analysis freeze. Eight protocols accompany that freeze. The figure script and 12-row aggregate figure source were added during the later editorial revision. source_manifest.json records SHA-256 hashes of the public source files. The publication release includes the S2 File source archive as an asset.

The repository covers the cardiac-surgery recorded-cessation measurement study. It does not include the separate septic-shock feasibility screen, protected raw data, patient-level derived records, or manuscript submission materials.

Historical author-workspace provenance checks and manuscript assembly require additional artifacts. See [REPRODUCTION_LIMITS.md](REPRODUCTION_LIMITS.md) before running those stages. This release was checked for source integrity and syntax; full raw-data reproduction was not rerun.
