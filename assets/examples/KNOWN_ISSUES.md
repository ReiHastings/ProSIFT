# Known issues -- ProSIFT example inputs

**Status as of 2026-09-01: the shipped example does NOT run. Do not use it yet.**

These files were built on 2026-09-01 and reviewed the same day by a fresh-context
`code-reviewer` pass that executed the real pipeline modules against the real
artifacts. The review returned FAIL. The findings are recorded here rather than
fixed, by decision, so the work is not lost and nobody trusts the example
prematurely.

Everything below is [FOLLOW-UP NEEDED].

---

## Blocking: the example cannot complete a run

### E-1. 40 proteins against a hardcoded 50-protein floor

`bin/filter_proteins.py:583` hard-exits when fewer than 50 proteins survive
detection filtering. The example ships 40, so `-profile test` dies at Module 01
step 2 and never reaches Modules 02-05.

Only `bin/validate_inputs.py` was ever run against the example (passed, 0
warnings). Nothing downstream was verified before the review.

**Fix direction:** scale the panel to ~120 proteins. That also resolves the
degenerate enrichment statistics in E-6.

### E-2. The example is not offline, and `databases.enabled: []` does not make it so

`UNIPROT_MAPPING` (`workflows/prosift.nf:148`) is **not** gated by
`databases.enabled`, and it is a hard dependency of `DIFFERENTIAL_ABUNDANCE`
(supplies `--id-mapping`). Running it against the example hits UniProt REST and
then Ensembl BioMart, and fails on the BioMart call.

The "runs fully offline" claim in `nextflow.config` and the README was wrong.
It has been corrected in both files; the underlying limitation stands.

**Open decision for Rei:** does `UNIPROT_MAPPING` get a skip / offline path? No
example can be offline without one. This is a pipeline change beyond the scope
of the examples feature.

### E-3. The example accessions are real UniProt entries, not placeholders

`P00001`-`P00040` is the historical cytochrome c accession block, roughly 40
species. Verified against the UniProt REST API:

    P00001  CYC_HUMAN
    P00009  CYC_MOUSE
    P00040  CYC_SCHGR   (locust)

The generator docstring and README both asserted these were synthetic and would
not resolve. That was false and has been corrected in place.

Consequences beyond the false claim: `UNIPROT_MAPPING` resolves them to
`CYCS`/`CYC`/`Cyct`, so the gene symbols reaching Module 05 overlap the shipped
GMT in **zero** genes, and `prepare_gene_symbols` (`bin/enrichment.py:203-208`)
dedups on `gene_symbol`, collapsing ~13 proteins into a single row.

**Fix direction:** pick a namespace and verify it returns zero UniProt hits
before shipping it. Do not assume an accession pattern is unassigned.

### E-4. `params.yml` hardcodes an absolute path into the author's home directory

**Resolved 2026-09-30.** GMT libraries are now staged as ENRICHMENT inputs: the
workflow (`workflows/prosift.nf`, `resolve_gmts`) reads
`enrichment.gene_set_libraries` from each run's params.yml, resolves relative
entries against the params.yml's directory, and passes the files to
ENRICHMENT, which hands the staged copies to `enrichment.py` via the new
`--gene-set-libraries` option. The example and its generator now write the
relative path `example_gene_sets.gmt`, and re-running the generator after
cloning is no longer required. Verified with a `-profile test -stub` run (GMT
staged under `gmt/1/`), a Nextflow render check of the real process script
(one and three GMTs, a space in a filename, two same-named files), and an
end-to-end `enrichment.py` run in a simulated task directory. Not yet verified
by a full non-stub run: the example still stops at Module 01 (E-1).

Behaviour change: a missing or empty `gene_set_libraries`, or a missing GMT, now
stops the whole run when the input channel is built (every samplesheet row),
not only that run's ENRICHMENT task after Modules 01-04. Follow-up hardening
(2026-09-30 and 2026-10-01, after two fresh code reviews):
- Input paths containing glob characters (`[ ] * ? { }`) are rejected with a
  clear error before any task is staged. This covers the samplesheet itself,
  its entries, and GMT entries. Every row and GMT is validated in the workflow
  body before the input channel is built (2026-10-01; previously the checks
  ran per row alongside task dispatch, so on a long samplesheet tasks could
  start before a bad later row was reached). They are not supported: staged
  inputs are linked with an unquoted `ln -s`, so a bracket name could match a
  sibling such as `x1.gmt`. That re-glob is inferred from the unquoted
  command; it was not reproduced directly, because the rejection now
  prevents the case.
- Because `?` is a glob character, a URL entry with a query string (for
  example a presigned S3 or https link) is also rejected. Supporting such URLs
  needs a check of how Nextflow names and stages downloaded files first.
- Entries are trimmed identically in the workflow and in `enrichment.py`
  (Java `String.trim()` rule: chars <= U+0020).
- A blank entry fails as "not a file".
- Two GMTs that map to the same library short name are rejected when
  ENRICHMENT starts. This check runs inside the task, not at startup.

Original finding: `assets/examples/minimal/params.yml` wrote the GMT path as
`/Users/reina/Library/Mobile Documents/.../example_gene_sets.gmt`.
`bin/enrichment.py:145-151` calls `sys.exit(1)` when the library is missing, so
enrichment fails for every other user and on the cluster.

This is a consequence of `enrichment.py` reading `gene_set_libraries` directly
from params.yml at runtime while Nextflow does not stage those files
(`workflows/prosift.nf:211`). The generator resolves the path at generation
time as a stopgap, and the README documents the "re-run the generator after
cloning" requirement, but that contradicts the one-line quick start.

**Fix direction:** stage the GMT as a process input, or substitute
`${projectDir}`. Either changes ENRICHMENT's signature and deserves its own
review.

---

## Scientific correctness of the synthetic data

### E-5. The simulation violates the median-normalization assumption, and the review checklist predicted it

`.claude/review-domain-checklist.md` lists this exact failure mode: median
normalization assumes most proteins are unchanged, which does not hold under
massive coordinated perturbation.

The example shifts 18 of 40 proteins (45%) in a coordinated direction, in whole
functional blocks with identical magnitude and no within-block variation. This
was written into `generate_examples.py` as a deliberate design feature. It is a
defect.

Measured per-block median log2FC:

| block | planted | pre-norm | post-norm (what the pipeline reports) |
|---|---|---|---|
| ASTROCYTE_MARKERS | +2.5 | +2.75 | **+3.15** |
| MITOCHONDRIAL_RESPIRATION | -2.0 | -1.90 | **-1.47** |
| SYNAPTIC_VESICLE_CYCLE | 0 | -0.11 | **+0.26** |
| CYTOSKELETON_AND_GLYCOLYSIS | 0 | +0.01 | **+0.41** |

Both planted effects are wrong by ~26%, and all 22 null proteins acquire a
+0.3 to +0.4 log2 bias.

**Fix direction:** spike ~15-20% of proteins rather than 45%, balanced up and
down so the per-sample median is preserved. Add within-block variation in
effect size.

### E-6. `ground_truth.csv` is wrong: the pipeline calls 19, not 18

A direct consequence of E-5. The systematic bias pushes one null protein
(`P00034`, `Nefl`) over `fc_threshold: 1.0`:

    no normalization:      logFC = 0.723  -> correctly not called
    median normalization:  logFC = 1.122  -> called significant

Both limma-only and DEqMS agree on 19, so this is not a peptide-count artifact.

A new user comparing pipeline output against `ground_truth.csv` would conclude
ProSIFT has a false positive. A wrong ground truth is worse than none.

**Fix direction:** separate what was *planted* from what the pipeline *calls*,
and generate the latter from an actual run rather than asserting it a priori.

### E-7. The example never exercises MNAR imputation

The restoration loop in `simulate_abundances` guarantees >= 2 detections per
group, so `filter_proteins.py` classifies all 40 proteins as `PASSED` and
`bin/impute.py:279-282` routes every missing value to MAR:

    MNAR: 0 proteins (0 values imputed)
    MAR:  13 proteins (17 values imputed)

The MinProb path, the `PARTIAL` / `SINGLE-GROUP` / `WEAK-ANCHOR` classes, and
the anchor gate are all untouched by the smoke test. The advertised "30%
MNAR-in-tail" collapses to 7.1% total missingness.

Related: the loop always restores the lowest-indexed replicate first, producing
a replicate-index missingness gradient (CTRL 3/2/4, TREAT 1/3/4) that Module 02
sample QC will read as real structure. Restore a randomly chosen cell instead.

Also, the guard's comment claims falling below 2 detections means Module 01
filters the protein out. Not true: `filter_proteins.py:238-243` classifies a
1/3-vs-3/3 protein as `PARTIAL` and retains it if it has an anchor.

### E-8. Enrichment statistics are degenerate at this scale

With a 40-gene background and 8-12 gene sets, the ORA row for
`EXAMPLE_ASTROCYTE_MARKERS` has `overlap_size == gene_set_size == 8`: a fully
saturated hypergeometric. Preranked GSEA over a 40-gene universe reports
`pvalue = 0.000000` exactly rather than `< 1/n_perm`.

Largely resolved by scaling the panel (E-1), but the README should state plainly
that enrichment output from the example is a smoke test, not a meaningful result.

---

## Pipeline defects surfaced by this work (not caused by it)

### E-9. Module 05 reports `gene_set_size` as a character count

**Status (2026-09-29): fixed in code** (`count_matched_genes` in `bin/enrichment.py`;
tests M05-6 and `TestCountMatchedGenes`). Existing cluster outputs still need
regeneration; see the handoff task. The line numbers below refer to the pre-fix code.

`bin/enrichment.py:404-410`:

```python
matched = pre_res_obj.results[term].get("matched_genes", [])
return len(matched) if matched else 0
```

gseapy returns `matched_genes` as a semicolon-joined **string**, so `len()`
counts characters. Observed on the example: gene sets of 12/8/10/10 genes report
`gene_set_size` of 71/46/60/55 in GSEA rows. ORA rows are correct.

`len('Ndufa9;Sdha;Uqcrc1;Cox4i1;Atp5f1a;Cs;Idh3a;Mdh2;Aco2;Slc25a4') == 60`.

**This corrupts a delivered column on every real run**, including the cluster
results already produced. Highest-value item in this file and independent of the
examples feature. Should be split into its own task.

### E-10. `.xlsx` input produces a raw traceback, not a validation error

`infer_separator` (`bin/validate_inputs.py:76-86`) branches on extension and
falls through to reading the file as UTF-8 text. The call at line 184 sits
*outside* the try/except at 185-189, so an `.xlsx` escapes as:

    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc7 in position 15

A two-line extension guard would turn this into a clean "export to CSV first"
message. Users will hit this whether or not the workbook template ships.

---

## Nextflow changes (unverified)

Nextflow is not installed on the development machine, so the `test` profile in
`nextflow.config` and the `resolve_input` closure in `workflows/prosift.nf` were
reviewed statically only and have **never been executed**.

- **E-11.** `p.startsWith('/')` is not an absolute-path test. It misclassifies
  `s3://`, `https://`, `az://`, and `~/` entries as relative and rebases them
  onto the samplesheet directory. The previous bare `file(row.abundance)`
  handled URLs natively. Use `file(p).isAbsolute()` or check for a `://` scheme.
- **E-12.** `file(params.samplesheet)` now runs *before*
  `Channel.fromPath(..., checkIfExists: true)`, so a bare `nextflow run main.nf`
  with no `--samplesheet` hits `file(null)` first. Add an explicit guard.
- **E-13.** A misspelled samplesheet header makes `row.abundance` null, and the
  error says the column is "empty" rather than absent.
- **E-14.** `error(String)` requires Nextflow >= 22.10. There is no
  `manifest.nextflowVersion` in `nextflow.config` and `environment.yml:11` pins
  `nextflow` with no version.
- **E-15.** The `test` profile sets `params.max_cpus` / `max_memory` /
  `max_time`, which are consumed nowhere: `conf/base.config:9-11` uses literals
  and there is no `check_max()` helper. The profile gives false assurance the
  smoke test is resource-capped.

---

## Documentation accuracy

### E-16. The docs describe a pipeline shape that does not currently exist

**Resolved 2026-09-29.** The uncommitted `[stop-before-DB]` edits to
`workflows/prosift.nf` were reverted, so the working tree again matches the
committed pipeline: the five `QUERY_*` calls (Module 06) and the
`RESULTS_ASSEMBLY` join and call (Module 07) are active, and every
`ch_db_*_input` channel is consumed. The README and test-profile comments,
which assume Module 06 runs, are consistent with the pipeline again.

Not yet verified by a run: no stub or test-profile execution has confirmed the
restored Module 06/07 wiring against the current process signatures.

Original finding: `workflows/prosift.nf` had Modules 06 and 07 commented out in
the working tree with `[stop-before-DB]` markers, leaving five `ch_db_*_input`
channels built but never consumed.

### E-17. Vendor mapping caveats

- `vendor/diann/peptide_counts.md` uses `pr.columns[10:]`, a hardcoded
  positional slice. DIA-NN's leading-column set differs between 1.8 and 1.9/2.0.
  Use `pr.columns[pr.columns.get_loc('Precursor.Id')+1:]`. The snippet is
  otherwise correct (verified executable under pandas 2.3.3) but stops before
  joining the counts back to the pg_matrix, so it is not runnable end to end.
- `vendor/maxquant/mapping.yml` documents a `gene_symbol` mapping that **ProSIFT
  cannot consume**: there is no `gene_symbol_column` parameter anywhere in
  `bin/` or `nextflow.config`. Gene symbols come only from `UNIPROT_MAPPING`.
- `vendor/fragpipe/mapping.yml` is docs-only and correctly self-labels as
  unverified. Confirm against a real `combined_protein.tsv`.
- DIA-NN and MaxQuant mappings were verified against real downloaded files
  during authoring, but the reviewer did not re-verify those downloads.

### E-18. Minor

- ~~`openpyxl` is imported by `generate_examples.py` but declared in neither
  `environment.yml` nor `pyproject.toml`.~~ Resolved 2026-10-01: declared in
  `environment.yml` (the runtime manifest; `pyproject.toml` lists no runtime
  dependencies).
- `params.yml` comment claims `knn_k` "must be < the number of samples". False:
  `bin/impute.py:340-369` runs `KNNImputer` over proteins, so `k=3` means three
  neighbouring proteins of 40. The stated constraint does not exist.
- `write_gmt(panel, outdir)` never uses `panel`.
- Generated `params.yml` mixes `"raw"` and `'all'` quote styles against the
  project's single-quote convention.
- No test covers `generate_examples.py` or the shipped example. That absence is
  why E-1 shipped.

---

## Suggested empirical gates

From the review. Every one of these currently FAILS except the last.

1. Integration: run the example through validate -> filter -> normalize ->
   impute -> DA -> enrichment, assert each exits 0. Would have caught E-1 alone.
2. Assert Module 04's significant set equals `ground_truth.csv`'s
   `expected_significant` set.
3. Assert per-block median `log2_fc` is within +/-0.3 of `planted_log2fc`.
4. Assert median `log2_fc` over unspiked blocks is within +/-0.1 of zero. The
   cleanest single detector for E-5.
5. Assert no path in the shipped `params.yml` starts with `/Users/` or `/home/`.
6. Run `-profile test` with network egress blocked, assert success.
7. Determinism: `--seed 42` byte-reproduces the committed CSVs.
   **Currently PASSES** (verified; only `params.yml` differs, by the absolute
   GMT path).
8. Assert the example produces > 0 MNAR-imputed cells.
9. Assert the example accessions return 0 hits from UniProt.
