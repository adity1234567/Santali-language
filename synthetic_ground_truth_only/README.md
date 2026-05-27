# Santali Synthetic Data - Ground Truth Only

Generated 10000 synthetic examples from the `ground truth` folder only. The existing `generated` folder was not read or used.

## Outputs

- `santali_synthetic_layoutB_10000.csv`: rich interlinear-style layout with Bangla, English, IPA, normalized segmentation, glosses, and notes.
- `santali_synthetic_layoutA_10000.csv`: compact layout matching `file,utterance_id,start_ms,end_ms,ref,mb,gl,ft`.
- `santali_synthetic_parallel_10000.csv`: parallel `bangla,english,santali` layout.
- `ground_truth_analysis_summary_10000.json`: file profiles, label/domain distributions, extracted pronoun tokens, and validation results.
- `ground_truth_analysis_summary.json`: latest-run copy of the validation summary.

## Validation

- Rows generated: 10000
- Unique normalized IPA rows: 10000
- Exact duplicates against ground truth Santali/ref/mb strings: 0
- Segment/gloss count alignment: True
- Segment length min/avg/max: 2 / 4.45 / 8

## Pattern Distribution

- locative_exist: 1957
- sequence: 1873
- possessive_adjective: 1580
- topic_adjective: 1258
- contrast: 637
- coordination: 617
- transitive_fin: 459
- negative_transitive: 443
- past_transitive: 364
- perfect_transitive: 329
- classifier_exist: 291
- progressive: 103
- question: 50
- imperative: 39

The templates preserve labels and structures observed in the ground truth, including TOP, LOC, GEN, NEG, FIN, PROG, PST, PRF, CVB, IMP, classifier, coordination, contrast, and question patterns.
