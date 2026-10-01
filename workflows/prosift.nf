/*
 * ProSIFT -- workflows/prosift.nf
 * Top-level workflow. Imports and calls module processes in order.
 */

nextflow.enable.dsl = 2

// Module 01: Input Validation & ID Mapping
include { VALIDATE_INPUTS  } from '../modules/local/validate_inputs/main'
include { FILTER_PROTEINS  } from '../modules/local/filter_proteins/main'
include { UNIPROT_MAPPING  } from '../modules/local/uniprot_mapping/main'

// Module 01: Missingness Report (Process 4.10, advisory)
include { MISSINGNESS_REPORT } from '../modules/local/missingness_report/main'

// Module 02: Pre-Normalization QC/EDA
include { PRENORM_QC         } from '../modules/local/prenorm_qc/main'

// Module 03: Normalization & Imputation
include { NORMALIZE          } from '../modules/local/normalize/main'
include { IMPUTE             } from '../modules/local/impute/main'

// Module 04: Differential Abundance
include { DIFFERENTIAL_ABUNDANCE } from '../modules/local/differential_abundance/main'

// Module 05: Enrichment Analysis
include { ENRICHMENT             } from '../modules/local/enrichment/main'

// Module 06: Database Queries (five independent processes)
include { QUERY_UNIPROT          } from '../modules/local/query_uniprot/main'
include { QUERY_PUBMED           } from '../modules/local/query_pubmed/main'
include { QUERY_DISGENET         } from '../modules/local/query_disgenet/main'
include { QUERY_DGIDB            } from '../modules/local/query_dgidb/main'
include { QUERY_CTD              } from '../modules/local/query_ctd/main'

// Module 07: Results Assembly (SQLite convergence point)
include { RESULTS_ASSEMBLY       } from '../modules/local/results_assembly/main'

workflow PROSIFT {

    // --- Build per-run input channel from samplesheet ---
    // Each run tuple: [meta, abundance_matrix, metadata, params_yml, gmts], where
    // gmts is the list of resolved GMT files for ENRICHMENT; multiMap below
    // splits it into per-process branches.
    // meta is a Map; run_id is the only key used by Module 01.
    //
    // ch_input is a queue channel (from Channel.fromList) and can only be
    // consumed once per branch. multiMap is required here. Process output
    // channels (VALIDATE_INPUTS.out.*, FILTER_PROTEINS.out.*) are broadcast
    // in DSL2 and do not need multiMap even with multiple downstream subscribers.
    //
    // Branches:
    //   validate          -> VALIDATE_INPUTS
    //   filter_params     -> FILTER_PROTEINS (joined after validation)
    //   map_params        -> UNIPROT_MAPPING (joined after detection filter)
    //   prenorm_params    -> PRENORM_QC (joined after detection filter)
    //   normalize_params  -> NORMALIZE (joined after QC)
    //   impute_params     -> IMPUTE (joined after NORMALIZE)
    // Path resolution: entries in the samplesheet may be absolute or relative.
    // Relative entries resolve against the SAMPLESHEET's own directory, not the
    // launch directory. This is what nf-core users expect, it lets a samplesheet
    // travel with its data (so the shipped example under assets/examples works
    // from any launch directory), and it removes the need for machine-specific
    // absolute paths that break the moment the inputs move or are rsynced to the
    // cluster.
    // Glob characters in input paths are rejected, not supported. file()
    // globs by default, and even with glob: false Nextflow re-globs each input
    // when it stages it into a task (an unquoted `ln -s` in .command.run), so
    // `x[1].gmt` next to `x1.gmt` would silently stage the wrong file.
    def reject_glob_chars = { String path, String what ->
        if( path =~ /[\[\]*?{}]/ )
            error "${what}: path contains a glob character ([ ] * ? { }), which ProSIFT " +
                  "does not support in input paths: '${path}'. Rename the file or directory."
    }
    reject_glob_chars(params.samplesheet.toString(), 'Samplesheet')

    def sheet_dir = file(params.samplesheet).toAbsolutePath().parent

    def resolve_input = { String raw, String field, String run_id ->
        if( !raw?.trim() )
            error "Samplesheet row '${run_id}' has an empty '${field}' column."
        def p = raw.trim()
        // file() on an absolute path returns it unchanged; a relative path is
        // rebased onto the samplesheet directory before existence checking.
        reject_glob_chars(p, "Samplesheet row '${run_id}' '${field}'")
        def resolved = p.startsWith('/') ? file(p) : file(sheet_dir.resolve(p))
        reject_glob_chars(resolved.toString(), "Samplesheet row '${run_id}' '${field}'")
        if( !resolved.exists() || resolved.isDirectory() )
            error "Samplesheet row '${run_id}': '${field}' not found (or not a file) at ${resolved}\n" +
                  "  (samplesheet value: '${p}'; relative paths resolve against ${sheet_dir})"
        return resolved
    }

    // --- GMT libraries for ENRICHMENT (Module 05) ---
    // enrichment.gene_set_libraries is read from each run's params.yml here and
    // the files are passed to ENRICHMENT as staged path inputs, so the task
    // never opens a host path named inside params.yml. This keeps the shipped
    // example portable (relative GMT path), works in containers, and makes
    // -resume re-run enrichment when a GMT's content changes. Relative entries
    // resolve against the params.yml's own directory, the same rule
    // enrichment.py applies in standalone use.
    def resolve_gmts = { params_yml, String run_id ->
        def cfg  = new org.yaml.snakeyaml.Yaml().load(params_yml.text)
        def libs = (cfg instanceof Map) ? cfg.enrichment?.gene_set_libraries : null
        if( !(libs instanceof List) || libs.isEmpty() )
            error "Run '${run_id}': enrichment.gene_set_libraries in ${params_yml} is missing or empty."
        def params_dir = params_yml.toAbsolutePath().parent
        return libs.collect { raw ->
            // Groovy trim() removes leading/trailing chars <= U+0020;
            // enrichment.py load_params applies the same rule.
            def p = raw.toString().trim()
            // URLs (s3://, https://) and absolute paths are used as given,
            // except that a URL with a query string ('?') is rejected by the
            // glob check (unsupported; see KNOWN_ISSUES E-4).
            reject_glob_chars(p, "Run '${run_id}' gene set library")
            def gmt = ( p.contains('://') || new File(p).isAbsolute() ) ? file(p) : file(params_dir.resolve(p))
            reject_glob_chars(gmt.toString(), "Run '${run_id}' gene set library")
            // A blank entry resolves to the params directory itself: not a file.
            if( !gmt.exists() || gmt.isDirectory() )
                error "Run '${run_id}': gene set library not found (or not a file) at ${gmt}\n" +
                      "  (params.yml value: '${p}'; relative paths resolve against ${params_dir})"
            return gmt
        }
    }

    // --- Resolve and validate every run eagerly ---
    // All rows and their GMT libraries are resolved and checked here, in the
    // workflow body, before any channel exists. With lazy channel operators
    // (.map/.multiMap) the checks ran alongside task dispatch, so on a long
    // samplesheet tasks for earlier rows could be submitted (PBS jobs) before
    // a bad later row stopped the run.
    def sheet_file = file(params.samplesheet)
    if( !sheet_file.exists() )
        error "Samplesheet not found: ${params.samplesheet}"
    def runs = sheet_file.splitCsv(header: true).collect { row ->
        def meta       = [run_id: row.run_id]
        def abund      = resolve_input(row.abundance, 'abundance', row.run_id)
        def meta_csv   = resolve_input(row.metadata,  'metadata',  row.run_id)
        def params_yml = resolve_input(row.params,    'params',    row.run_id)
        [ meta, abund, meta_csv, params_yml, resolve_gmts(params_yml, row.run_id) ]
    }

    Channel
        .fromList(runs)
        .multiMap { meta, abund, meta_csv, params_yml, gmts ->
            // validate branch: all four inputs for VALIDATE_INPUTS
            validate:         [ meta, abund, meta_csv, params_yml ]
            // filter_params: params_yml for the FILTER_PROTEINS join
            filter_params:    [ meta, params_yml ]
            // missingness_params: params_yml for the MISSINGNESS_REPORT join
            missingness_params: [ meta, params_yml ]
            // map_params: params_yml for the UNIPROT_MAPPING join
            map_params:       [ meta, params_yml ]
            // prenorm_params: params_yml for the PRENORM_QC join
            prenorm_params:   [ meta, params_yml ]
            // normalize_params: params_yml for the NORMALIZE join
            normalize_params: [ meta, params_yml ]
            // impute_params: params_yml for the IMPUTE join
            impute_params:    [ meta, params_yml ]
            // da_params: params_yml for the DIFFERENTIAL_ABUNDANCE join
            da_params:        [ meta, params_yml ]
            // enrich_params: params_yml + resolved GMT files for the ENRICHMENT join
            enrich_params:    [ meta, params_yml, gmts ]
            // Module 06: one branch per database process
            db_uniprot_params:  [ meta, params_yml ]
            db_pubmed_params:   [ meta, params_yml ]
            db_disgenet_params: [ meta, params_yml ]
            db_dgidb_params:    [ meta, params_yml ]
            db_ctd_params:      [ meta, params_yml ]
            // assembly_params: params_yml for the RESULTS_ASSEMBLY join (Module 07)
            assembly_params:    [ meta, params_yml ]
        }
        .set { ch_input }

    // --- Module 01, Processes 4.1-4.3: Input validation ---
    VALIDATE_INPUTS(ch_input.validate)

    // --- Module 01, Process 4.4: Detection filter ---
    // Process output channels (VALIDATE_INPUTS.out.*) are broadcast in DSL2 --
    // multiple downstream joins can subscribe without multiMap.
    VALIDATE_INPUTS.out.matrix
        .join(VALIDATE_INPUTS.out.metadata)
        .join(ch_input.filter_params)
        .set { ch_filter_input }

    FILTER_PROTEINS(ch_filter_input)

    // --- Module 01, Process 4.10: Missingness report (advisory) ---
    // Joins: filter_table (FILTER_PROTEINS) + validated_matrix (pre-filter, VALIDATE_INPUTS)
    //        + validated_metadata (VALIDATE_INPUTS) + params_yml (for design.group_column).
    FILTER_PROTEINS.out.filter_table
        .join(VALIDATE_INPUTS.out.matrix)
        .join(VALIDATE_INPUTS.out.metadata)
        .join(ch_input.missingness_params)
        .set { ch_missingness_input }

    MISSINGNESS_REPORT(ch_missingness_input)

    // --- Module 01, Processes 4.5-4.9: UniProt ID mapping ---
    FILTER_PROTEINS.out.matrix
        .join(ch_input.map_params)
        .set { ch_mapping_input }

    UNIPROT_MAPPING(ch_mapping_input)

    // --- Module 06: Database Queries (parallel with Modules 02-05) ---
    // Each database process takes the mapping table + params_yml.
    // Individual databases can be toggled via databases.enabled in params.yml.
    // All five fan out from Module 01 and converge into Module 07.

    // QUERY_UNIPROT
    UNIPROT_MAPPING.out.mapping_table
        .join(ch_input.db_uniprot_params)
        .set { ch_db_uniprot_input }

    QUERY_UNIPROT(ch_db_uniprot_input)

    // QUERY_PUBMED
    UNIPROT_MAPPING.out.mapping_table
        .join(ch_input.db_pubmed_params)
        .set { ch_db_pubmed_input }

    QUERY_PUBMED(ch_db_pubmed_input)

    // QUERY_DISGENET
    UNIPROT_MAPPING.out.mapping_table
        .join(ch_input.db_disgenet_params)
        .set { ch_db_disgenet_input }

    QUERY_DISGENET(ch_db_disgenet_input)

    // QUERY_DGIDB
    UNIPROT_MAPPING.out.mapping_table
        .join(ch_input.db_dgidb_params)
        .set { ch_db_dgidb_input }

    QUERY_DGIDB(ch_db_dgidb_input)

    // QUERY_CTD
    UNIPROT_MAPPING.out.mapping_table
        .join(ch_input.db_ctd_params)
        .set { ch_db_ctd_input }

    QUERY_CTD(ch_db_ctd_input)

    // --- Module 02: Pre-normalization QC/EDA ---
    // Joins: filtered_matrix (post-filter) + validated_metadata + params_yml
    FILTER_PROTEINS.out.matrix
        .join(VALIDATE_INPUTS.out.metadata)
        .join(ch_input.prenorm_params)
        .set { ch_prenorm_input }

    PRENORM_QC(ch_prenorm_input)

    // --- Module 03: Normalization ---
    // Joins: filtered_matrix (post-filter) + validated_metadata + params_yml
    // PRENORM_QC runs in parallel -- NORMALIZE does not depend on its outputs.
    FILTER_PROTEINS.out.matrix
        .join(VALIDATE_INPUTS.out.metadata)
        .join(ch_input.normalize_params)
        .set { ch_normalize_input }

    NORMALIZE(ch_normalize_input)

    // --- Module 03: Imputation ---
    // Joins: normalized_matrix (NORMALIZE) + validated_metadata (VALIDATE_INPUTS)
    //        + filter_table (FILTER_PROTEINS, for MNAR/MAR classification) + params_yml
    NORMALIZE.out.normalized_matrix
        .join(VALIDATE_INPUTS.out.metadata)
        .join(FILTER_PROTEINS.out.filter_table)
        .join(ch_input.impute_params)
        .set { ch_impute_input }

    IMPUTE(ch_impute_input)

    // --- Module 04: Differential Abundance ---
    // Joins: imputed_matrix (IMPUTE) + validated_metadata (VALIDATE_INPUTS)
    //        + mapping_table (UNIPROT_MAPPING) + params_yml
    IMPUTE.out.imputed_matrix
        .join(VALIDATE_INPUTS.out.metadata)
        .join(UNIPROT_MAPPING.out.mapping_table)
        .join(ch_input.da_params)
        .set { ch_da_input }

    DIFFERENTIAL_ABUNDANCE(ch_da_input)

    // --- Module 05: Enrichment Analysis ---
    // Joins: diff_abundance_results (DIFFERENTIAL_ABUNDANCE) + params_yml +
    // GMT files (resolved from params.yml by resolve_gmts, staged as inputs).
    DIFFERENTIAL_ABUNDANCE.out.results_table
        .join(ch_input.enrich_params)
        .set { ch_enrich_input }

    ENRICHMENT(ch_enrich_input)

    // --- Module 07: Results Assembly (SQLite convergence point) ---
    // Collects the analytical spine (Modules 01-05) and the database query
    // layer (Module 06) into a single per-run input, keyed by meta [run_id].
    // All upstream outputs are broadcast channels, so join() barriers here
    // until every producing process has completed for this run. The join order
    // matches the RESULTS_ASSEMBLY process input tuple exactly.
    UNIPROT_MAPPING.out.mapping_table
        .join(FILTER_PROTEINS.out.filter_table)
        .join(PRENORM_QC.out.sample_flags)
        .join(IMPUTE.out.imputed_matrix)
        .join(IMPUTE.out.imputation_mask)
        .join(DIFFERENTIAL_ABUNDANCE.out.results_table)
        .join(ENRICHMENT.out.enrichment_results)
        .join(ENRICHMENT.out.protein_term_mapping)
        .join(QUERY_UNIPROT.out.uniprot_annotations)
        .join(QUERY_PUBMED.out.pubmed_cooccurrence)
        .join(QUERY_DISGENET.out.disgenet_associations)
        .join(QUERY_DGIDB.out.dgidb_interactions)
        .join(QUERY_CTD.out.ctd_interactions)
        .join(ch_input.assembly_params)
        .set { ch_assembly_input }

    RESULTS_ASSEMBLY(ch_assembly_input)

}
