#!/usr/bin/env python3
# title: test_workflow_stub.py
# project: ProSIFT
# author: Reina Hastings
# contact: reinahastings13@gmail.com
# date created: 2026-10-01
# last modified: 2026-10-01
#
# purpose:
#   Regression tests for the input checks in workflows/prosift.nf, run
#   through `nextflow run -stub` (the repo has no nf-test harness). Every
#   samplesheet row and its GMT libraries are validated in the workflow body
#   before any channel is built, so a bad row must stop the run before ANY
#   task is staged, however late it appears in the samplesheet:
#     - a glob character in a GMT path (x[1].gmt) on the last of 300 rows.
#       The row count matters: with lazy (per-row channel) validation, whether
#       a task is dispatched before the bad row is reached is a race. A
#       41-row run caught a lazy-validation mutant in 1 of 3 runs, 300 rows
#       in 5 of 5 (2026-10-01).
#     - a blank gene_set_libraries entry (resolves to a directory, not a file)
#   Paired control: a valid two-row samplesheet runs and does stage tasks, so
#   the "no task staged" assertion is not vacuous.
#
#   Skipped when Nextflow is not available. The prosift conda env provides
#   nextflow and a JVM next to its Python, which is where they are looked up.
#
# inputs:
#   - workflows/prosift.nf via main.nf; assets/examples/minimal/* (copied)
#
# outputs:
#   - Test results (stdout via pytest); runs write only under tmp_path.
#
# usage example:
#   "$(conda info --base)/envs/prosift/bin/python" -m pytest tests/test_workflow_stub.py -q
#
#   copy/paste: "$(conda info --base)/envs/prosift/bin/python" -m pytest tests/test_workflow_stub.py -q

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / 'assets' / 'examples' / 'minimal'
ENV_BIN = Path(sys.prefix) / 'bin'
NEXTFLOW = ENV_BIN / 'nextflow' if (ENV_BIN / 'nextflow').exists() else shutil.which('nextflow')
JVM = Path(sys.prefix) / 'lib' / 'jvm'


def _java_works() -> bool:
    '''
    True if a usable Java runtime exists: the env's bundled JVM, else JAVA_HOME,
    else java on PATH. Runs `java -version` rather than checking the file,
    because macOS ships a /usr/bin/java stub that exists without a runtime.
    '''
    candidates = [JVM / 'bin' / 'java']
    if os.environ.get('JAVA_HOME'):
        candidates.append(Path(os.environ['JAVA_HOME']) / 'bin' / 'java')
    if shutil.which('java'):
        candidates.append(Path(shutil.which('java')))
    for java in candidates:
        if java.exists():
            try:
                if subprocess.run([str(java), '-version'], capture_output=True,
                                  timeout=30).returncode == 0:
                    return True
            except (OSError, subprocess.TimeoutExpired):
                continue
    return False


HAS_JAVA = _java_works()


def _env() -> dict:
    env = dict(os.environ, NXF_DISABLE_CHECK_LATEST='true')
    if (JVM / 'bin' / 'java').exists():
        env.update(JAVA_HOME=str(JVM), JAVA_CMD=str(JVM / 'bin' / 'java'))
    return env


def _inputs(tmp_path: Path, gmt_entry_for_last_row: str | None, n_rows: int) -> Path:
    '''
    Copy the shipped example into tmp_path and write a samplesheet of n_rows
    valid runs. When gmt_entry_for_last_row is given, the last row instead
    uses a params.yml whose single gene_set_libraries entry is that string.
    '''
    inp = tmp_path / 'in'
    inp.mkdir()
    for name in ('abundance.csv', 'metadata.csv', 'params.yml', 'example_gene_sets.gmt'):
        shutil.copy(EXAMPLE / name, inp / name)
    rows = [f'r{i:03d},abundance.csv,metadata.csv,params.yml' for i in range(n_rows)]
    if gmt_entry_for_last_row is not None:
        text = (inp / 'params.yml').read_text()
        assert text.count('"example_gene_sets.gmt"') == 1
        (inp / 'params_bad.yml').write_text(
            text.replace('"example_gene_sets.gmt"', f'"{gmt_entry_for_last_row}"'))
        (inp / 'x[1].gmt').write_bytes((inp / 'example_gene_sets.gmt').read_bytes())
        (inp / 'x1.gmt').write_text('DECOY\tna\tGeneA\n')
        rows[-1] = f'r{n_rows - 1:03d},abundance.csv,metadata.csv,params_bad.yml'
    sheet = inp / 'samplesheet.csv'
    sheet.write_text('run_id,abundance,metadata,params\n' + '\n'.join(rows) + '\n')
    return sheet


def _run(tmp_path: Path, sheet: Path) -> tuple[int, str, list]:
    '''Run the pipeline in stub mode; return (exit code, output, staged task scripts).'''
    work = tmp_path / 'work'
    proc = subprocess.run(
        [str(NEXTFLOW), '-q', 'run', str(ROOT / 'main.nf'), '-stub',
         '--samplesheet', str(sheet), '--outdir', str(tmp_path / 'results'),
         '-w', str(work)],
        cwd=tmp_path, env=_env(), capture_output=True, text=True, timeout=600,
    )
    staged = list(work.glob('*/*/.command.run')) if work.exists() else []
    return proc.returncode, proc.stdout + proc.stderr, staged


needs_nextflow = pytest.mark.skipif(NEXTFLOW is None or not HAS_JAVA,
                                    reason='nextflow or a Java runtime not available')


@needs_nextflow
def test_glob_gmt_on_last_of_300_rows_stops_before_any_task(tmp_path):
    code, out, staged = _run(tmp_path, _inputs(tmp_path, 'x[1].gmt', n_rows=300))
    assert code != 0
    assert 'glob character' in out, out[-2000:]
    assert staged == [], f'{len(staged)} task(s) staged before the bad row was rejected'


@needs_nextflow
def test_blank_gmt_entry_stops_before_any_task(tmp_path):
    code, out, staged = _run(tmp_path, _inputs(tmp_path, '   ', n_rows=1))
    assert code != 0
    assert 'not a file' in out, out[-2000:]
    assert staged == []


@needs_nextflow
def test_control_valid_samplesheet_runs_and_stages_tasks(tmp_path):
    code, out, staged = _run(tmp_path, _inputs(tmp_path, None, n_rows=2))
    assert code == 0, out[-2000:]
    assert staged, 'valid run staged no tasks: the no-task assertions above would be vacuous'
