# Vendor output -> ProSIFT

One directory per search engine, each with a `mapping.yml` giving the column
correspondence. The mappings are machine-readable rather than prose so a future
converter script can consume the same rules without the docs drifting from code.

| Vendor | Source file | Per-sample peptide counts | Verified against |
|---|---|---|---|
| DIA-NN | `report.pg_matrix.tsv` | **no**, derive from `pr_matrix` | real output (PXD028735) |
| MaxQuant | `proteinGroups.txt` | yes, direct | real output (CPTAC lab3) |
| FragPipe | `combined_protein.tsv` | spectral counts only | **docs only, unverified** |

## The one thing to get right per vendor

**DIA-NN.** The run columns are named by the *absolute mzML path* on the machine
that ran DIA-NN. You must rename them to clean sample IDs. Also,
`Protein.Group` and `Protein.Ids` hold the same accessions in *different order*,
so ProSIFT's first-token rule gives a different representative protein depending
on which you map. And there are no peptide counts at all: see
`diann/peptide_counts.md` for the derivation.

**MaxQuant.** Drop rows flagged `+` in `Reverse`, `Potential contaminant`, or
`Only identified by site` before ProSIFT sees them. The first data row of the
reference file is a `CON__` contaminant, so this is not theoretical. Use
`LFQ intensity <sample>`, not the bare `Intensity` column (that is a study-wide
total, not a sample).

**FragPipe.** Peptide counts are global only (`Combined Total Peptides`); the
per-sample columns are *spectral* counts. DEqMS accepts a PSM count as its
variance covariate so this works, but it is a different quantity than the
MaxQuant mapping uses. Say which one you used in your methods.

## Reference files

These are public and were used to verify the mappings above. They are not
vendored into this repo.

- DIA-NN: <https://github.com/fmicompbio/einprot/tree/main/inst/extdata/diann_example>
  (Van Puyvelde et al., *Sci Data* 2022; PXD028735)
- MaxQuant: <https://raw.githubusercontent.com/statOmics/pda/data/quantification/cptacAvsB_lab3/proteinGroups.txt>
  (CPTAC study 6, lab 3)
