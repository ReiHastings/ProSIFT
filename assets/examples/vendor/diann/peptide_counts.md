# Deriving peptide counts for DIA-NN

`report.pg_matrix.tsv` carries no peptide counts, so DEqMS has nothing to weight
by. If you set `peptide_count_prefix` to null and skip this step, ProSIFT logs a
single INFO line and runs DEqMS unweighted, which is a different statistical
model than the one you probably think you are running.

`report.pr_matrix.tsv` has one row per precursor and the same run columns, with
these leading columns (verified against real output):

    Protein.Group  Protein.Ids  Protein.Names  Genes  First.Protein.Description
    Proteotypic  Stripped.Sequence  Modified.Sequence  Precursor.Charge  Precursor.Id

Count the precursors quantified per protein group per run:

```python
import pandas as pd

pr = pd.read_csv('report.pr_matrix.tsv', sep='\t')
run_cols = pr.columns[10:]          # everything after Precursor.Id
counts = (pr.groupby('Protein.Group')[list(run_cols)]
            .apply(lambda block: block.notna().sum()))
counts.columns = [f'peptide_count_{c}' for c in counts.columns]
```

Two caveats worth stating in your methods:

- These are **precursor** counts, not distinct peptide counts. A peptide seen at
  two charge states counts twice. DEqMS accepts either as the count covariate,
  but say which one you used.
- Rename the run columns to your clean sample IDs *before* this step, so the
  counts and the abundances end up with matching suffixes. ProSIFT hard-fails if
  any sample has an abundance column without a matching peptide-count column.
