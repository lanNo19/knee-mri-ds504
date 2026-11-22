# Run cards (Kaggle, browser)

Every notebook runs on Ablay's account and has two cells. Claude creates it in the
built-in browser, pastes the cells, sets the options, and starts a batch run with
Save Version -> Save & Run All. Claude reads only the summary blocks (between
`=== KNEE SUMMARY BEGIN ===` and `=== KNEE SUMMARY END ===`) and the agg_*.json and
run.json files at the output root. Everything else is under private/.

Claude never opens: the competition Data page, dataset pages, data previews, the input
file tree, the Logs tab, anything under private/, the Output tab of a failed run, or
the montage notebook's pages. Claude reads only tabs it opened itself, by tab ID.
Ablay looks at data (dataset pages, montages, flagged pairs, error logs) only in his own
browser, never in the Claude app's pane.

## Cell 1 (every notebook)

```python
COMMIT = "<commit sha>"
!pip install -q "knee[n0,decode,models] @ git+https://github.com/lanNo19/knee-mri-ds504@{COMMIT}"
import knee.safe_log as sl; sl.install()
```

## Session options and inputs

| Notebook | Accelerator (for the version run) | Internet | Inputs |
|---|---|---|---|
| knee-smoke | GPU T4 x2 | On | Competition; the three label-table datasets (plus dreaddevelopment/knee-raptor-corpus once Ablay has recorded its label file's name) |
| knee-smoke-all | None (CPU) | On | Competition; the three label-table datasets |
| knee-n0 | None (CPU) | On | Competition; the three label-table datasets (plus dreaddevelopment/knee-raptor-corpus if Ablay recorded its file name) |
| knee-n3 | None (CPU) | On | Output of knee-n0 (after the pilot: also the pilot output with pairs.csv) |
| knee-n4-sagittal, knee-n4-coronal, knee-n4-axial | None (CPU) | On | Competition |
| knee-smoke-all-2 | GPU T4 x2 | On | Competition; the three label-table datasets |
| knee-n5 | GPU T4 x2 | On | Outputs of the three N4 notebooks |
| knee-n6p | GPU T4 x2 | On | Outputs of knee-n0, knee-n3 and knee-n5 |
| knee-n8 | None (CPU) | On | Outputs of knee-n0 and knee-n3 |
| knee-montage | None (CPU) | On | Outputs of the three N4 notebooks; after the pilot also knee-n6p (Claude starts it, never opens it) |

Keep the editor on accelerator None while editing; switch to GPU T4 x2 only for the
version run; stop the interactive session after every edit.

Label-table datasets: pilkwang/rsna-knee-llm-labels, stevenleehans/rsna-knee-llm-report-labels,
lixin73/rsna-knee-llm-report-labels-sol56. Ablay records each one's version and licence.

## Cell 2 and pass bands

| Notebook | Cell 2 | Expected time | Pass bands (summary block and agg file) |
|---|---|---|---|
| knee-smoke | `from knee.nb import smoke; smoke.run()` (with dreaddevelopment: `smoke.run(dread_file="<file>.csv")`) | 2-5 min | checks: t4x2, safe_log, competition_attached and pilkwang_verdict_conf_unk all true; train_csv and train_series_csv header_ok true; every label table attached with a recognised layout (with dreaddevelopment: label_layouts.dread recognised too) |
| knee-smoke-all | `from knee.nb import smoke_all; smoke_all.run(limit=20)` | 10-20 min | every stage "OK" |
| knee-n0 | `from knee.nb import n0; n0.run()` (with dreaddevelopment: `n0.run(dread_file="<file>.csv", dread_provenance_ok=<True or False>)`) | 30-60 min | labels.gold.bug_check_passed true (macro AUC at least 0.80, expected about 0.89) and gold.is_58 true; at least 3 tables used, each kept with coverage_non_gold at least 0.98; groups.g_lang_groups at least 6; images.header_read_errors empty; every decode_test entry all_decoded; the summary block shows studies_sha16 (N3 checks it) |
| knee-n3 | `from knee.nb import n3; n3.run(expect_n0="<first 8 characters of studies_sha16 from N0's summary block>")` (after the pilot: add `confirmed_pair_ids=[...]`, pair numbers only) | 2 min | s2.scheme group_5fold (else follow the plan's fold rules); gold_unassigned 0; s1.sizes_match_s2 true |
| knee-n4-* | `from knee.nb import n4; n4.run("sagittal", budget_h=8)` (coronal, axial likewise) | 15-90 min each | checks: shape_ok, failures_under_1pct and blank_under_1pct true; position_order_share at least 0.99 |
| knee-smoke-all-2 | `from knee.nb import smoke_all; smoke_all.run(limit=20, stages=("n0", "n3", "n4", "n5", "n6p", "n8"))` | 20-30 min | every stage "OK" (with 20 studies n6p and n8 report not_testable after a one-epoch or one-model smoke fit) |
| knee-n5 | `from knee.nb import n5; n5.run()` | 20-45 min | checks: study_tables_match, all_finite, missing_rows_zero, features_vary, gpu_agreement and normalisation_as_planned all true; usable_share at least 0.98; the summary block shows index_sha16 |
| knee-n6p | `from knee.nb import n6p; n6p.run(expect={"n0_studies": "<8 chars>", "n3_folds": "<8 chars>", "n5_index": "<8 chars>"})` | 20-40 min | positive_control.all_passed true (else the alignment audit before Stage A); overlap_control.passed true (known_answer_detected false = underpowered, reported, no audit); leakage_assertion_passed true; write half_width_delta1.union into Phase 1 before Stage A; near_duplicates.to_review pairs go to the montage |
| knee-n8 | `from knee.nb import n8; n8.run()` | 20-60 min | leakage_assertion_passed true; n_folds as in N3; s2_macro_auc recorded (H3 is decided in N9) |
| knee-montage | `from knee.nb import montage; montage.run()` | 2 min | (Ablay looks at private/montage/*.png, including pairs.png after the pilot, in his own browser, and passes confirmed pair_ids to N3) |

Counts in every summary: 1-4 show as "<5" and larger counts are rounded to the nearest 5, so a
small count cannot be worked out by subtraction. Pass bands therefore use shares, flags and
structural numbers (folds, groups), never exact counts.

With the CLI fallback, `python scripts/make_notebooks.py --user <name> --commit <sha>` writes the
same cells; add `--dread-file <file>.csv` (and `--dread-provenance-ok` if Ablay recorded that the
labels are LLM output on report text only) to attach and check the dreaddevelopment table.

## If a run fails

The summary block shows `STEP <name>: FAILED - <ErrorType> at <file>:<line> in <function>()`.
That line is all Claude needs for most fixes: Claude fixes the code, runs the tests,
pushes, updates COMMIT, and reruns.

If a fix needs the details, Ablay (not Claude) reads them, in his own browser:
1. The failed version's private/<step>_error.log holds the step's hidden output and the full traceback.
2. Or, in an interactive session only he watches, he reruns cell 2 with `quiet=False` and then runs `sl.show_last()`.
3. He describes the problem to Claude in words, without values.
