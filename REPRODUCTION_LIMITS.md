# Reproduction scope and historical checks

This release preserves the original frozen analysis source. It includes eight frozen protocols and the aggregate figure source. The author did not rerun the full raw-data pipeline when preparing this public release.

The analytical scripts require credentialed MIMIC-IV 3.1 inputs and generated protected intermediate files. Create the project beside the hosp/ and icu/ input directories, as described in README.md. Do not place input tables or patient-level outputs in this public repository.

Some historical provenance checks are specific to the original workspace. In particular, script 50 verifies prior manifests from rebuild_v3/ and docs/manuscript_practice_v3/; those manifests include author-workspace paths and unpublished document artifacts. Script 46 assembles a historical manuscript from docs/manuscript_practice_v3_source.md. Protected intermediates and historical manuscript artifacts are not distributed here. These original provenance or manuscript-assembly checks cannot run unmodified from this public source tree alone. The numerical reconstruction and sensitivity routines remain available in the scripts and their core modules. An independent reproduction should establish its own provenance records and document any changes to these historical checks in a separate copy.

Source integrity and Python syntax were checked for this release. These checks do not constitute a fresh end-to-end reproduction, clinical adjudication of withdrawal intent, or validation of causal exchangeability. Earlier inference code is retained as historical supplementary source; the manuscript is a descriptive EHR measurement study.

To reproduce the editorial trajectory figure, run figures/figure1.py using its included 12-row aggregate CSV. This source contains no individual records. The plotted values are group medians and IQRs of admission-level two-hour mean doses.
