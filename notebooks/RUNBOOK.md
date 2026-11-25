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

N10 runs with the internet off, so its cell 1 installs everything from the bundle dataset
(the bundle's project wheel is built from the pinned commit):

```python
import glob, subprocess, sys
W = sorted(glob.glob("/kaggle/input/*/private/bundle/wheels") + glob.glob("/kaggle/input/notebooks/*/*/private/bundle/wheels") + glob.glob("/kaggle/input/datasets/*/*/private/bundle/wheels"))
assert W, "attach the knee bundle dataset"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", W[0], "knee", "timm==1.0.31", "python-gdcm", "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "dicomsdl"], check=True)
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
| knee-n8 | None (CPU) | On | Outputs of knee-n0, knee-n3 and knee-n5 (N5's usable mask) |
| knee-n6a-smoke, knee-n6a, knee-n6b | GPU T4 x2 | On | Outputs of knee-n0, knee-n3 and knee-n5 |
| knee-n7 | GPU T4 x2 (or None (CPU) if no GPU is free: the code runs on either) | On | Outputs of knee-n0, **knee-n3-r** (not knee-n3), knee-n5, knee-n6a and knee-n6b (and a unit rerun such as knee-n6a-r). Never knee-n6a-smoke. Exactly one output per upstream step. While knee-n6b is still queued or running: leave it out (see "N7 and N9 after the pilot reviews") |
| knee-n10-canary | None (CPU) | **Off** | Competition; the bundle dataset; output of knee-n0 |
| knee-n10, knee-n10-convnext | GPU T4 x2 | **Off** | Competition; the bundle dataset; outputs of knee-n0 and knee-n6a |
| knee-bundle | None (CPU) | On | None (public wheels and checkpoints only; with Ablay's OK its output becomes a private versioned dataset) |
| knee-n9 | None (CPU) | On | Outputs of knee-n0, **knee-n3-r** (not knee-n3), knee-n5, knee-n6p, knee-n6a, knee-n6b (and a unit rerun such as knee-n6a-r), knee-n7 and knee-n8. Never knee-n6a-smoke. Exactly one output per upstream step. While knee-n6b is still queued or running: leave it out |
| knee-n9-r3 | None (CPU) | On | Exactly the inputs knee-n9 ran with, in the same versions: knee-n0, knee-n3-r, knee-n5, knee-n6p, knee-n6a, the EXISTING knee-n7, knee-n8 (and knee-n6b only if knee-n9 had it). N7 is not rerun. A new notebook, so knee-n9's freeze file stays as it is |
| knee-n9-b | None (CPU) | On | Reserved for the rerun after knee-n6b has finished: knee-n0, knee-n3-r, knee-n5, knee-n6p, knee-n6a, knee-n6b, knee-n8 and the NEW knee-n7-b in place of knee-n7. A new notebook, so the freeze files of knee-n9 and knee-n9-r3 stay as they are |
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
| knee-n8 | `from knee.nb import n8; n8.run()` | 20-60 min | usable_from_n5 and leakage_assertion_passed true; n_folds as in N3; s2_macro_auc recorded (H3 is decided in N9) |
| knee-n6a-smoke | `from knee.nb import n6; n6.run("n6a_smoke", encoders=("dinov2_vits14",), folds=(0,))` | 15-40 min | all_units_ok and leakage_assertion_passed true; the unit's seconds times 10 / 4 workers is N6a's expected wall time |
| knee-n6a | `from knee.nb import n6; n6.run("n6a", encoders=("dinov2_vits14", "convnext_tiny"), backstop_utc="2026-11-24T03:00:00Z", unit_hours_est=<pilot: projection.unit_hours_one_process of the slower main encoder>)` (if the pilot's projection.n6a_over_6h_gate is true: add `mode="simplified"`) | 1.5-3 h | all_units_ok and leakage_assertion_passed true; fold_modes all "nested" (else the slides label those folds "not nested"); a failed unit: rerun it alone in knee-n6a-r with `n6.run("n6a_r", ..., folds=(k,))`, the same `mode`, and knee-n6a attached (a simplified rerun reuses its fold 0) |
| knee-n6b | `from knee.nb import n6; n6.run("n6b", encoders=("vit_s16",), demo=True, backstop_utc="2026-11-24T03:00:00Z")` | 1-2 h | as N6a; demo.curves has the three arms |
| knee-n7 | `from knee.nb import n7; n7.run(expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | GPU 40-100 min; CPU 45-75 min for two encoders (15 min with `extras=False`) | chain.s2_folds_match_units true (false = the step fails: wrong N3 or unit attached); missing_units empty for every encoder; extras_ok true and extras_errors empty (a failed extra costs only its own block); leakage_assertion_passed true; the consensus configs go to the repo as configs/consensus.json (for KneeMRI S5) |
| knee-n9 | `from knee.nb import n9; n9.run(expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` or, with S4, `from knee.nb import n9; n9.run(expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"}, hidden={"dinov2_vits14": <score>, "convnext_tiny": <score>})` (leave out hidden if S4 did not run) | 15-30 min | chain.s2_folds_match_units true (false = the step fails and no decision is written); main_complete and findings_match_pilot true; secondary_errors empty; agg_results.json is the freeze file; decisions.H1-H4 are read from it, never recomputed by hand |
| knee-n9-r3 | `from knee.nb import n9; n9.run(expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"}, compare_with={"H1": "inconclusive", "H2": "inconclusive", "H3": "not supported"}, expect_primary={"delta1_g_lang": ["-0.0213", "-0.0530", "+0.0105"], "delta1_g_vendor": ["-0.0173", "-0.0562", "+0.0215"], "delta2": ["+0.0142", "-0.0059", "+0.0343"], "h3": ["0.5060", "0.4896", "0.5174"]})` (add `hidden={...}` only if knee-n9 had it) | 20-35 min | as knee-n9; the same inputs must give the same primary results: compare_with_previous.same_h1_h2_h3 true AND primary_matches_previous.all_match true (every number of Delta1 under both groupings, Delta2 and H3 equals knee-n9's, each at the decimals written in the card; pass them as strings so "-0.0530" keeps its 4 decimals). Either one false = stop and report both files; do not pick one. secondary_mutated_primary false. Copy the compare_with strings and the expect_primary numbers from knee-n9's agg_results.json, never from memory |
| knee-n9-b | `from knee.nb import n9; n9.run(expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"}, compare_with={"H1": "inconclusive", "H2": "inconclusive", "H3": "not supported"})` (add `hidden={...}` if S4 ran; no `expect_primary`) | 20-35 min | as knee-n9; compare_with_previous.same_h1_h2_h3 true (false = stop and report both files); secondary_mutated_primary false. Digit-for-digit identity with knee-n9 is NOT expected here: N7 is refit (knee-n7-b), with CPU refit noise of about 0.006 on the numbers N7 feeds (Delta1 under G-vendor, Delta2), so only the categories are compared. Copy the compare_with strings from knee-n9's decisions.H1-H3, never from memory |
| knee-n10-canary | offline cell 1, then `from knee.nb import n10; n10.run(canary=True)` | 1-3 h on the hidden test | the commit run on the public test files passes; submitted after Ablay's OK; scored = at most 1% of hidden studies failed to decode, a scoring error = more than 1% (deliberate) |
| knee-n10, knee-n10-convnext | offline cell 1, then `from knee.nb import n10; n10.run("dinov2_vits14")` (knee-n10-convnext: `from knee.nb import n10; n10.run("convnext_tiny")`) | 2-5 h on the hidden test | submission.layout wide; folds lists every S2 fold from N3; failed_share at most 0.01; then ask Ablay before submitting (at most 5 a day) |
| knee-bundle | `from knee.nb import bundle; bundle.run()` | 5-10 min | wheels_ok and checkpoints_ok true; then ask Ablay before saving the output as a private versioned dataset |
| knee-montage | `from knee.nb import montage; montage.run()` | 2 min | (Ablay looks at private/montage/*.png, including pairs.png after the pilot, in his own browser, and passes confirmed pair_ids to N3) |

Counts in every summary: 1-4 show as "<5" and larger counts are rounded to the nearest 5, so a
small count cannot be worked out by subtraction. Pass bands therefore use shares, flags and
structural numbers (folds, groups), never exact counts.

With the CLI fallback, `python scripts/make_notebooks.py --user <name> --commit <sha>` writes the
same cells; add `--dread-file <file>.csv` (and `--dread-provenance-ok` if Ablay recorded that the
labels are LLM output on report text only) to attach and check the dreaddevelopment table.

## N7 and N9 after the pilot reviews (23 Nov)

Attach rules for both notebooks:
1. Attach knee-n3-r, not knee-n3. knee-n3-r is the N3 rerun with the pilot's pairs.csv: flagged pairs share an S1 fold. Its S2, G-vendor and inner folds are identical to knee-n3; only s1_fold changed.
2. Never attach knee-n6a-smoke. Its fold-0 unit would compete with the real one.
3. Attach exactly one output per upstream step. Two outputs holding the same file name resolve by path order, not by intent.
4. Pass the expect hashes shown in the card. A wrong attached version then fails the step at once.

While knee-n6b is still queued or running (the control encoder ViT-S/16 is then missing):
1. Do not attach knee-n6b to N7 or N9. A queued notebook mounts nothing or an older version.
2. N7 cell 2: `from knee.nb import n7; n7.run(encoder_names=("dinov2_vits14", "convnext_tiny"), expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})`. N9 cell 2 is unchanged.
3. Expected in N9 then: anti_claim_1 is null; tuned_vit_s16 and the b2_default vit rows show complete false; baselines.same_studies_and_folds_for_every_row is false with those rows in incomplete_rows. H1, H2 and H3 do not use ViT-S/16.
4. When knee-n6b has finished: run N7 and N9 again in NEW notebooks (knee-n7-b, knee-n9-b) with knee-n6b attached and all three encoders. Never save a new version over tonight's knee-n7 or knee-n9. In knee-n9-b pass tonight's decisions: `n9.run(expect={...}, compare_with={"H1": "<tonight>", "H2": "<tonight>", "H3": "<tonight>"})`.
5. Read compare_with_previous.same_h1_h2_h3 in the summary block. True = use the new file for the control rows only. False = stop and report both files; do not pick one.
6. Two different reruns, two names. knee-n9-r3 = N9 again on the EXISTING knee-n7 output with the round-3 code (same inputs, so the primary numbers must be identical: pass `expect_primary` and `compare_with`). knee-n9-b = N9 after knee-n6b with the new knee-n7-b (N7 is refit, so only the categories are compared: pass `compare_with` only). Never use one name for the other.
With the CLI fallback: `python scripts/make_notebooks.py ... --without-n6b` (and `--n7-cpu` for N7 without a GPU).

If no GPU is free for N7: set the accelerator to None (CPU). Nothing in N7 needs CUDA; it picks the device itself and records it as device. On CPU expect about 15 min for S1 and G-vendor of two encoders and 45-75 min with all extras. To save time use `extras=("seeds", "defaults")` (no triage fits, no probe) or `extras=False`. A CPU refit is not equal to Stage A's T4 refit, even with seed 0: weights and batch order are the same, but dropout draws come from the device's own generator, so the CPU refit behaves like one more seed. seed_sensitivity.seed0_refit_check.reproduces_stage_a will then be false. That is not a chain problem (the chain block is the chain check). Read delta1_gap against delta1_sd (delta1_gap_within_seed_sd true = fine).

Known and intended: Stage A ran before the rerun, so agg_n6a.json and agg_n6b.json record the old N3 hash (a21b342fec48d2b6), while N7 and N9 record the new one (670ddcc8c1927355). chain.n3_hash_same_as_stage_a is therefore false. That is by design: the chain guard checks the S2 folds themselves.

What the chain guard proves: every unit's test studies and gold studies are exactly S2 fold k of the attached folds.csv, so (with all folds present and the N0 and N5 hashes checked) every unit's training set is the same too. What it does not prove: that the inner folds are the same (units do not store them).

Everything below is secondary and descriptive. No field changes H1, H2, H3, the primary metric or a chosen setting. Decisions use seed 0.

New fields in agg_n7.json (the summary block shows chain, device, leakage_assertion_passed, extras_ok, extras_errors and near_duplicate_probe):

| Field | Meaning |
|---|---|
| chain.s2_folds_match_units | true = every attached Stage A unit's test fold and gold studies equal the attached S2 fold; false fails the step; null = no unit attached (nothing was checked) |
| chain.units_checked | number of units the guard checked (rounded) |
| chain.mismatched_units | names (encoder_fK) of units that do not match; empty when the guard passes |
| chain.mismatch_counts | per failing unit: studies of its fold it lacks (missing) and studies it has beyond its fold (extra), rounded |
| device | cuda or cpu: where N7's fits ran |
| base_seconds, extras_seconds | time of the base pass, and of each extra block |
| extras_requested | the extra blocks asked for (seeds, defaults, triage, probe) |
| extras_errors | empty when every requested block ran; else block name with the error type and place; the other blocks still ran |
| chain.n3_folds_sha16 | hash of the attached folds.csv (expect 670ddcc8...) |
| chain.s2_sha16 | hash of the attached folds.csv without its s1_fold column; equal to agg_n3.json's s2_sha16 of knee-n3-r |
| chain.stage_a_n3_folds_sha16 | the N3 hash each Stage A notebook recorded (the old one, by design) |
| chain.n3_hash_same_as_stage_a | false is expected here (only s1_fold changed) |
| extras.<encoder>.seeds | seeds refit for S2 (main encoders only): 0 is a reproduction check, 1-5 the sensitivity seeds |
| extras.<encoder>.default_l2_defaults | epochs E per fold, and their source, for the default head = the fold's L1 winner at the L2 defaults |
| extras.<encoder>.default_fixed | the same for the default head = gated attention over all planes at the L2 defaults (no selection at all) |
| extras.<encoder>.triage | true = this encoder's triage thresholds were written |
| extras_ok | true = no extra block failed; null = no extras were run |
| near_duplicate_probe.cosine_above_T.pairs | study pairs with cosine above T (0.90, 0.95, 0.98) between the pilot's DINOv2 study vectors |
| near_duplicate_probe.cosine_above_T.across_language_groups, across_vendor_groups | how many of those pairs have different groups |
| near_duplicate_probe.cosine_above_T.pairs_both_training | pairs whose two studies are both training studies |
| near_duplicate_probe.cosine_above_T.across_s2_folds, across_vendor_folds, across_s1_folds | of the both-training pairs, how many sit in different folds |
| leakage_assertion_passed | true = no fit received a test-fold or gold study; in the base pass a breach fails the step, in an extra it is recorded and turns this false |

N7 also writes, under private/ only: s2seeds_<encoder>.csv, default_<encoder>.csv, default_fixed_<encoder>.csv and triage_thresholds.csv. N9 reads them.

New fields in agg_results.json (the summary block shows chain, seed_sensitive, seed0_refit, sign_agrees_with_primary, compare_with_previous and secondary_errors):

| Field | Meaning |
|---|---|
| chain.* | as in agg_n7.json; a false s2_folds_match_units writes status "chain mismatch", no decision, and fails the step |
| h1.secondary.per_fold_delta1 | Delta1 (DINOv2 minus ConvNeXt) inside each S2 test fold, in fold order |
| h1.secondary.pair_share | each fold's share of the primary metric's pair weight, in fold order |
| h1.secondary.findings_without_pairs | per fold, the findings with no positive (or no negative) there; they are left out of that fold's Delta1 |
| h1.secondary.unweighted_mean_of_fold_delta1 | the plain mean of the per-fold Delta1 values (every fold counts the same) |
| h1.secondary.primary_delta1 | the primary, pair-weighted Delta1, repeated for comparison |
| h1.secondary.sign_agrees_with_primary | true = the unweighted mean has the same sign as the primary Delta1 |
| h1.g_lang.interval_within_fold_study_bootstrap | second interval for Delta1: studies resampled inside each test fold (2,000 draws, percentile); it shows sampling noise inside folds with sites fixed; used_for_decisions is false |
| average_precision.definition | how macro AP is computed: AP inside each test fold, folds weighted by positives, mean over the 12 findings |
| average_precision.chance | the chance level of that macro AP: the exact expected AP of a random ranking per fold, weighted like the AP; use this for "times chance" |
| average_precision.prevalence_weighted | the same weighting of each fold's plain prevalence (lower than chance for rare findings; reference only) |
| average_precision.chance_per_finding | the chance level per finding |
| average_precision.mean_prevalence | mean over findings of the pooled share of positives |
| average_precision.models.<model> | macro_ap and per_finding AP for each encoder and for floor_a |
| seed_sensitivity.delta1_per_seed | Delta1 per refit seed; seed_0 is Stage A's value, the one decisions use |
| seed_sensitivity.delta1_sd | standard deviation of Delta1 over the six seeds |
| seed_sensitivity.delta1_low_high | lowest and highest Delta1 over the six seeds |
| seed_sensitivity.interval_per_seed | the G-lang decision interval of Delta1 for seeds 1-5 |
| seed_sensitivity.decision_per_seed | the unchanged decision rule applied per seed (G-lang category only) |
| seed_sensitivity.seed_sensitive | true = some seed's category differs from seed 0's; report H1 as seed-sensitive |
| seed_sensitivity.share_of_extra_seeds_same_category | share of seeds 1-5 whose category equals seed 0's (the "k of 5" on the slide) |
| seed_sensitivity.sign_same_all_seeds | true = Delta1 has the same sign in all six seeds |
| seed_sensitivity.macro_auc_per_seed | S2 macro AUC of each main encoder per seed |
| seed_sensitivity.seed0_refit_check | N7's seed-0 refit against Stage A's stored predictions: refit_device, tolerance, max_abs_probability_gap, delta1_of_refit, delta1_gap, reproduces_stage_a (gap within tolerance; expected on a GPU), delta1_gap_within_seed_sd (on CPU this is false in about half of healthy runs, because the gap is a difference of two draws; worry only if delta1_gap is above 3 times delta1_sd) |
| compare_with_previous | only with compare_with: the earlier H1-H3, same_h1_h2_h3 and the list that differs |
| triage.models.<encoder>.confusion | pooled confusion matrix (tp, fp, fn, tn; rounded to 5) for "any finding" at the 95%-sensitivity threshold from inner out-of-fold predictions |
| triage.models.<encoder>.sensitivity, specificity, precision, f1 | rates from the exact counts (2 decimals; the rounded matrix will not reproduce them exactly); null when a cell the rate uses holds 1 to 4 studies |
| triage.models.<encoder>.normal_studies_moved_back_share | share of studies without a finding that fall below the threshold (= specificity) |
| triage.models.<encoder>.inner_sensitivity_mean | mean sensitivity reached on the inner predictions the thresholds were set on |
| baselines.rows | one table: b0_chance, b1_scanner_tags, b2_default_<encoder>, b2_default_fixed_<encoder>, tuned_<encoder> |
| baselines.same_studies_and_folds_for_every_row | true = every row is complete, so all rows use the same studies and folds; false = see incomplete_rows |
| baselines.incomplete_rows | rows without numbers (model or its N7 file missing) |
| baselines.rows.*.s2_macro_auc | the primary metric with its group-level decision interval |
| baselines.rows.*.mean_of_fold_macros | mean over folds of each fold's macro AUC (use this beside published fold means) |
| baselines.rows.*.gain_over_b1 | S2 macro AUC minus floor A's |
| baselines.rows.tuned_*.gain_over_own_default | tuned minus the same encoder's b2_default row, with the decision interval |
| baselines.rows.*.macro_ap | macro average precision; its chance level is b0_chance.macro_ap |
| baselines.rows.*.gold58_macro_auc | macro AUC on the gold studies |
| baselines.rows.b0_chance.measured_permuted_label_null | the pilot's permuted-label macro AUC, if the pilot is attached |
| baselines.reference_row | says the published-numbers row is not computed; add it by hand |
| hpo_evidence.boundary.<setting> | share of units whose chosen value, and of top-5 trials, lies in the outer 5% of the search range (low end, high end, both) |
| hpo_evidence.best_so_far.<encoder>.mean_best_so_far | best inner score after each of the 30 trials, mean over units |
| hpo_evidence.best_so_far.<encoder>.first_trial_of_final_best_median | trial number (the logged number, from 0) that first reached the final best, median over units |
| hpo_evidence.best_so_far.<encoder>.share_units_best_is_trial_0 | share of units where the defaults were never beaten |
| hpo_evidence.best_so_far.<encoder>.mean_gain_after_startup_trials | final best minus the best after the 10 start-up trials (what the GP phase added) |
| hpo_evidence.hierarchy.<encoder>.per_unit | per fold: l1_margin (L1 winner minus runner-up) and l2_gain (best after BO minus the winner at the defaults) |
| hpo_evidence.hierarchy.<encoder>.share_units_l2_gain_above_l1_margin | share of units where L2 tuning gained more than the L1 choice |
| hpo_evidence.profiles.<encoder>.<setting> | inner score in 6 equal-width bins of the (log) range, from the start-up trials only (trial number below 10: the defaults and 9 random draws): mean, mean_centred (each unit's mean removed), trial_share |
| hpo_evidence.profiles_from | says which trials the profiles use |
| hpo_evidence.units_without_own_search | units tuned in simplified mode; they are left out of the evidence |
| near_duplicate_probe | copied from agg_n7.json |
| secondary_errors | empty when every secondary block ran; else the block name with the error type and place |

Blocks added after the results critique (23 Nov, late; knee.nb.n9c). N7 is not rerun: each block reads only files N7 and Stage A already wrote. Each has descriptive true and a one-line scope. The summary block shows a few of their numbers under one key, descriptive_only, whose label is "descriptive, not a test": same_device_gain, refit_gap (the flag and the two median gaps), any_finding_auc, threshold_overshoot and unseen_site_cost (estimate and se). None of them is a decision.

How N9 protects the primary results from these blocks:
1. agg_results.json is written with the primary results before any secondary block runs, and again after them. A crash or an import error in the secondary code leaves the first file in place.
2. Every block works on copies of the primary results.
3. After the blocks, H1, H2, H3, the decisions and every other key written before the blocks are compared with a copy taken before them. On a difference the copy is put back and secondary_mutated_primary is true. Every block also gets a fresh copy of H1 and H2, so one block cannot change the H1 and H2 the next block reads.

| Field | Meaning |
|---|---|
| secondary_mutated_primary | false = no secondary block changed h1, h2, h3, decisions or any other key written before the blocks; true = one did, the earlier values were put back, and the code needs a fix before the file is used; null = unknown, only possible after a failed second write (secondary_errors.final_write) |
| secondary_errors.n9c, .n9x, .final_write | the round-3 module could not be imported (its blocks are missing), the whole secondary step crashed, or the second write failed (the file then holds the primary results only) |
| primary_matches_previous | only with expect_primary: numbers.<name>.estimate, lo, hi are true where this run's value equals the supplied one at the supplied number of decimals (within half a unit of the last one; at most 4); complete_card = all four names were supplied, each as exactly three values (estimate, lo, hi); all_match = complete_card, every one is true and no unknown name was passed; unknown_keys_present = a name outside the four was passed (its text is not written); compared_at_decimals says the precision used. Names: delta1_g_lang, delta1_g_vendor, delta2, h3. It reports only; it never fails the step |

| Field | Meaning |
|---|---|
| baselines.same_device_gain.device, stage_a_device | where N7's refits ran, and where Stage A's units were fit (from Stage A's run.json; "unknown" if not recorded) |
| baselines.same_device_gain.gain_over_own_default_mixes_devices | true = the older gain_over_own_default subtracts fits from two devices; quote the same-device gain instead |
| baselines.same_device_gain.gain_over_own_default_note | one sentence saying what the older gain is |
| baselines.same_device_gain.encoders.<encoder>.seed0_minus_default | tuned (N7 refit, seed 0) minus default head (N7 refit, seed 0), paired, with the registered group-level interval |
| baselines.same_device_gain.encoders.<encoder>.mean_of_seeds_minus_default | mean S2 macro AUC of the tuned refits over seeds 0-5, minus the default's (the default has one seed) |
| baselines.same_device_gain.encoders.<encoder>.tuned_seed_sd | SD of the tuned S2 macro AUC over those seeds: the yardstick for the gain |
| baselines.same_device_gain.not_available | encoders without seed refits from N7 (the control encoder) |
| refit_agreement.encoders.<encoder>.n7_seed0_vs_stage_a | absolute probability gap between N7's seed-0 refit and Stage A's stored predictions: median, p90, p99, max |
| refit_agreement.encoders.<encoder>.n7_seed_pairs_median | the yardstick: the same statistics between two seeds of N7, each one the median over all pairs of N7's seeds (15 pairs with six seeds) |
| refit_agreement.gap_like_another_seed | true = for both encoders the median and p99 of the first are at most 1.5 times the yardstick's. A rule of thumb (flag_is says so; the factor was chosen after the results), not a test: read the statistics themselves |
| refit_agreement.fits | Delta1 of the seven fits (Stage A's decision fit and N7 seeds 0-5), each with its device |
| refit_agreement.delta1_low_high_all_fits, delta1_low_high_n7_fits, delta1_sd_n7_fits, sign_same_all_fits | range over the seven fits; range and SD over N7's six same-device fits; whether all seven have one sign |
| triage_secondary.any_finding_prevalence | share of evaluation studies with at least one finding |
| triage_secondary.models.<encoder>.any_finding_auc | AUC of the max-of-12 score against "any finding", pairs inside a test fold, pair-weighted |
| triage_secondary.models.<encoder>.specificity_interval | Wilson 95% interval of the triage specificity from exact counts, 2 decimals. Binomial only: studies count as independent, thresholds as fixed, sites are ignored (interval_scope). Null with fewer than 20 normal studies (interval_min_studies) or when a cell it uses holds 1 to 4 studies |
| triage_secondary.models.<encoder>.threshold_overshoot | achieved test sensitivity minus the 0.95 target, 2 decimals (positive = the inner threshold was too low for the new site) |
| largest_site_decomposition.group, fold | the largest site group and its S2 fold k |
| largest_site_decomposition.training_sets | size (rounded) and the group's share of the training set for a (S2 fold k), b (S1 fold k: the same size) and c (the other S1 folds, mean) |
| largest_site_decomposition.encoders.<encoder>.a_unseen | S2 macro AUC on the group's studies (its models never saw the group) |
| largest_site_decomposition.encoders.<encoder>.b_seen_equal_training_size | S1 macro AUC on the group's studies in S1 fold k (same training-set size and config as a, but the group is in training) |
| largest_site_decomposition.encoders.<encoder>.c_seen_larger_training_size | S1 macro AUC on the group's studies in the other S1 folds (larger training sets) |
| largest_site_decomposition.encoders.<encoder>.a_on_b_studies, a_on_c_studies | the S2 predictions scored on exactly b's and c's studies |
| largest_site_decomposition.encoders.<encoder>.unseen_site_cost_at_equal_training_size | b minus a_on_b_studies: same studies, size and config, so this is an estimate of the cost of the site being unseen (an upper bound). One site, one fitted model per side, one seed; the two models also differ in their mix of sites. Not H2, not a test |
| largest_site_decomposition.encoders.<encoder>.unseen_site_cost_se, unseen_site_cost_se_valid_share | paired study-bootstrap standard error of that estimate (2,000 draws of b's studies, both models on the same draw, fixed seed), and the share of draws that were defined. It covers which studies were drawn, not sites, seeds or refits |
| largest_site_decomposition.encoders.<encoder>.larger_training_set_seen_site_rough | c minus b: different studies and configs, so only a rough view of training-set size |
| largest_site_decomposition.encoders.<encoder>.larger_training_set_view_paired | (c minus a_on_c_studies) minus (b minus a_on_b_studies): the same view with each term paired on its own studies; still rough |
| largest_site_decomposition.caveat, caveats | "one site, one model pair, descriptive", and the limits of this view one by one |
| best_epoch_spread.encoders.<encoder>.chosen_config | inner best epochs of each unit's best trial: lowest, median, p90, share at or below 6, share at the 30-epoch cap |
| best_epoch_spread.encoders.<encoder>.all_trials | the same over every completed trial |
| best_epoch_spread.encoders.<encoder>.refit_epochs, median_E | each unit's E, and their median |
| best_epoch_spread.encoders.<encoder>.lr_fraction_of_peak_at_median_E | learning rate at the end of epoch E as a fraction of its peak (from heads.schedule_factor); near 1 = the cosine had barely started |
| best_epoch_spread.encoders.<encoder>.warmup_share_of_training | the 1-epoch warm-up as a share of E |
| gold_uncertainty.macro_auc.<encoder> | gold macro AUC (estimate), its bootstrap standard error (se), and the fold_stratified version (pairs inside one fold model) |
| gold_uncertainty.delta1 | gold Delta1 with its paired bootstrap standard error |
| gold_uncertainty.studies, exact_count_file | the rounded gold count; the exact one is only in private/n9/gold_count.json (Ablay reads it) |
| h2_without_lookalikes | post hoc (post_hoc true): Delta2 again without every study in a look-alike pair above cosine 0.90; pairs, studies_removed, share_removed, delta2_registered, delta2_without_lookalikes, change |
| h2_without_lookalikes.pair_encoder, caveats | the pairs are found with one encoder's study vectors, DINOv2 (dinov2_vits14), which can favour one side of Delta2; 0.90 is one of the probe's three thresholds and was chosen after the results |
| h2_without_lookalikes.pair_source, rounded_pair_count_matches_n7_probe, pair_count_check | the pairs are found again in N9 with the pilot's code (N7 wrote counts only, rounded to 5); true = the count found here equals N7's probe after the same rounding to buckets of 5, not exactly (counts up to 4 apart can match, counts 1 apart can differ; "<5" matches any 1 to 4) |
| h1.per_finding_exploratory_note | the 12 per-finding intervals are unadjusted: intervals_excluding_zero ("k of 12") and the rule "all 12 or none" |

New fields in other summaries:

| File | Field | Meaning |
|---|---|---|
| agg_n3.json | pairs.pairs_applied | flagged pairs with both studies in the training index; only these shape the S1 folds |
| agg_n3.json | pairs.pairs_not_applied | flagged pairs that were skipped (a study outside the training index) |
| agg_n3.json | pairs.pairs_together | true = every applied pair shares an S1 fold |
| agg_n3.json | pairs.pairs_across_vendor_folds | flagged pairs whose two studies sit in different G-vendor folds |
| agg_n3.json | pairs.pairs_across_s2_folds | flagged pairs whose two studies sit in different S2 folds |
| agg_n3.json | s2_sha16 | hash of folds.csv without s1_fold; it must not change in a rerun that only regroups S1 |
| agg_n8.json | features.missing_share | now also holds manufacturer and model; 1.0 for a feature means it is dead |
| agg_n8.json | features.columns_absent | feature columns that tags.csv does not have (should be empty) |
| agg_pilot.json | projection.corrected | the Stage A time projection with the review's fixes: real inner-fit size, x1.26 for batch size, 30 epochs per fit, divided by 2; the old keys are unchanged |

## Iteration 2 (plan r2)

Plan: iteration_plan_r2.md, sections 2, 5 and 7, with the plan r3 changes (the iteration plan,
section 9, written after code review r1). Code: knee.nb.n13 (tests: tests/test_iteration2.py).
Everything in this section is "added after the results" (R10). Iteration-1 notebooks are not rerun
with this code; with its defaults the code is the registered code bit for bit.

Rules for every N13 notebook:
1. Cell 1 is the usual one (section "Cell 1 (every notebook)", the pip install from GitHub), so Internet is On.
   COMMIT is the final reviewed commit that holds knee/nb/n13.py (branch it2-a1 until it is merged).
2. Inputs of the arms: knee-n0, knee-n3-r and knee-n5. Nothing else: no knee-n6a and no other N13 output.
3. Never knee-n3: a GPU run with knee-n3 attached stays queued. knee-n3-r has the same S2 and inner folds.
4. Pass the expect hashes of the card. A wrong attached version then fails the step at once.
5. One arm per notebook, each with its own output. Never save one arm over another arm's notebook.
6. The arms a1, base0 and null run on one device type, one Kaggle image and one commit. First choice:
   GPU T4 x2. Two GPU sessions run at a time: a1 and null first, then base0.
   To keep one image, create the arm notebooks as forks of one notebook and keep "Pin to original
   environment": a fork keeps the environment pin (issue_log.md:382). knee-n13-a1, -null and -base0
   are forks of knee-n6b-g v2. Create every rerun as a fork of its arm's notebook.
7. If no GPU session is free: run all three on None (CPU). Never mix: decide() gives no decision when
   the arms' environments differ in any field.
8. Never attach an N13 output to N7 or N9 (R9). Its units look like Stage A's, and N7 and N9 take every unit they find.

Each arm records its environment in run.json (config.environment) and in its summary block
(environment): the Kaggle image (GIT_COMMIT and BUILD_DATE), python, torch, CUDA, cuDNN and the device name.

Expect hashes for every card: `{"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"}`.

| Notebook | Accelerator | Cell 2 | Expected time | Pass |
|---|---|---|---|---|
| knee-n13-a1 | GPU T4 x2 (None (CPU) under rule 7) | `from knee.nb import n13; n13.run("a1", expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | T4: about 20 min (setup included); CPU: 1-2.5 h | STEP n13_a1: OK and STEP n13_a1_inner: OK; all_units_ok, all_nested and leakage_assertion_passed true; units holds 10 entries (dinov2_vits14 and convnext_tiny, folds 0-4), each ok with schedule_epochs 8, patience 3, seed 0 and E from 1 to 8; device_names holds one name (Tesla T4, or cpu); trial0_inner_scores and heads_sha16 have 5 folds per encoder; environment present; run.json config: test_overrides true, test_overrides_reason "iteration 2, plan r3: defaults only, see the iteration plan, section 9", schedule_epochs 8, patience 3, arm "a1", environment |
| knee-n13-base0 | the same as knee-n13-a1 | `from knee.nb import n13; n13.run("base0", expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | T4: 20-30 min; CPU: 1.5-3 h | as knee-n13-a1, but schedule_epochs 30, patience 5, seed 0 and E from 1 to 30; run.json config has no schedule_epochs or patience (the registered values) and arm "base0" |
| knee-n13-null | the same as knee-n13-a1 | `from knee.nb import n13; n13.run("null", expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | T4: 20-30 min; CPU: 1.5-3 h | as knee-n13-base0, but seed 100 and arm "null" |
| knee-n13-a7 | GPU T4 x2 | `from knee.nb import n13; n13.run("a7", expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"}, schedule_epochs=<S>, patience=<P>)`, with S and P written in the plan before the run: 8 and 3 if A1 was kept for ConvNeXt-T, else 30 and 5 | T4: 30-90 min (5 units) | STEP n13_a7: OK and STEP n13_a7_inner: OK; descriptive true; keep_decision "none (plan r3): A7 is descriptive only; no keep decision and no submission"; encoders ["convnext_tiny"]; units holds 5 entries (folds 0-4), each ok with schedule_epochs S and patience P; settings.n_trials 30 and settings.space_overrides {"batch_size": [8, 128], "weight_decay": [1e-06, 1.0]}; best_inner_scores (a maximum over 30 trials) and boundary_widened_ranges present. Report best_minus_trial0 beside Stage A's selection-only gain for ConvNeXt-T (best of 30 minus trial 0: 0.0048, n9_r3_numbers.md:50-51): a gain of that size is what selection alone gives. Runs only if the A1 decision is written by Tue 09:00 and a GPU slot is free |

Reruns (failed units only):
1. A unit that failed shows ok false in units. Rerun only the failed units, in a new notebook
   knee-n13-<arm>-r: the same cell 2 plus `rerun=1, folds=(<k>,), encoders=("<encoder>",)`, on the
   same device and image. Its step is n13_<arm>_r. A second rerun is knee-n13-<arm>-r2 with `rerun=2`.
2. A rerun's unit replaces the original's only when the original failed. Never rerun a unit that
   finished: decide() and check_heads() refuse two finished units for one encoder and fold.
3. Wherever the arm is used, use the original with its reruns: decide() and check_heads() take the
   list [original summary, rerun summary, ...]; knee-n13-dev and N10 get the rerun outputs attached too.

What Claude reads: the blocks of each run (n6's STEP line and summary, then N13's STEP line and
summary; N13's summary is also agg_n13_<arm>_inner.json) and run.json. Per unit, N13's summary gives
trial0_inner_score (the mean over the inner folds of the validation macro AUC at each inner fit's
best epoch: Stage A's objective), E, schedule_epochs, patience, seed, device, device_name and head_sha16.

The A1 decision (Claude, outside Kaggle, from inner scores only, before any S2 number of A1 is read, R5):
1. Download agg_n13_a1_inner.json, agg_n13_base0_inner.json and agg_n13_null_inner.json (and each rerun's).
2. Run `decision = n13.decide(a1, base0, null)`. Each argument is the arm's summary dict, or the list
   [original, rerun, ...].
3. decide() refuses (ValueError) unless each arm has the plan's settings and exactly one finished unit
   per main encoder and outer fold 0-4.
4. If any environment field differs, the status is "env_mismatch": no numbers, neither encoder kept.
   Report it; there is no A1 decision on these arms.
5. Otherwise the status is "decided", with per encoder gain, threshold, folds_up, keep and s, and also
   s_divisor, n_null, null_mean and null_mean_se. Every comparison is made at 4 decimals; a tie is not kept.
6. same_knee_commit must be true. If it is false, report it, and use the decision only if
   `git diff --stat <commit> <commit> -- src/knee/heads.py src/knee/nb/n6.py src/knee/stage.py src/knee/metrics.py`
   between the summaries' knee_commit values shows no change.
7. Save the output as a1_decision.json. Write its numbers and the decision into the plan document with the time.
8. Descriptive only, never in the rule: `n13.stage_a_check(base0, [knee-n6a's ten agg_n6a_<encoder>_f<k>.json dicts])`
   gives base0 minus Stage A's logged trial 0 per unit. Report it beside the decision.
9. Only then run knee-n13-dev, and only with a "decided" record.

A1's S2 development numbers (only after the decision is written; R5, R6):

| Notebook | Accelerator | Internet | Inputs | Cell 2 | Expected time | Pass |
|---|---|---|---|---|---|---|
| knee-n13-dev | None (CPU) | On | knee-n0, knee-n3-r, knee-n5, knee-n13-a1 and knee-n13-base0 (with their reruns, if any). Nothing else | `from knee.nb import n13; n13.dev("""<the JSON text of a1_decision.json>""", expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | 5-10 min | STEP n13_dev: OK; status ok; chain.a1 and chain.base0 s2_folds_match_units true; label "development: these studies were test studies in iteration 1; likely optimistic"; descriptive and added_after_results true; s2.<encoder>.a1, .base0 and .a1_minus_base0 each hold fold_stratified_macro_auc and mean_of_fold_macros. dev refuses, before anything runs, without a decision record and on any status other than "decided" (an env_mismatch record holds no A1 decision). The step fails when the attached units are not the ones the record was written on (head hashes), or when the record's numbers do not follow from them (per-fold differences, gain, folds_up, threshold, keep) |

These numbers never go in a slide title and are never shown as a gain over iteration 1 (R6).

N10 with N13's heads (only if A1 is kept for ConvNeXt-T; Ablay presses Submit):

| Notebook | Accelerator | Internet | Inputs | Cell 2 | Expected time | Pass |
|---|---|---|---|---|---|---|
| knee-n10-n13 | GPU T4 x2 | **Off** | Competition; the bundle dataset; knee-n0; knee-n13-a1 and every rerun output of the a1 arm (knee-n13-a1-r, -r2). Never knee-n6a | offline cell 1, then `from knee.nb import n10; n10.run("convnext_tiny")` | 2-5 h on the hidden test | as knee-n10-convnext; then the heads check below. Only then ask Ablay to submit |

The heads check (Claude, outside Kaggle, before asking Ablay to submit):
1. N10 runs the bundle's wheel (commit 47fdcbf), which has no N13 code. So the check runs outside
   Kaggle, on two downloaded files.
2. N10 side: heads_sha16 in agg_n10.json, {"fold_k": sha16 of head.pt} (n10.py:126, written at :201).
   N10's summary block does not show it.
3. N13 side: heads_sha16.convnext_tiny in agg_n13_a1_inner.json (finished units only; a rerun's
   summary lists the units it replaced). Both are the same 16-character hash of head.pt.
4. `n13.check_heads(<agg_n10.json>, [<agg_n13_a1_inner.json>, <rerun summaries>])["match"]` must be
   true: the same five folds and equal hashes. Comparing the five strings by eye is the same check.

N10 runs the bundle's wheel. Its n10.py and the head code it calls (heads.make_head, config_from,
predict) are the same in this branch, so it reads N13's units unchanged.

## Batch-floor check (it2b; added after the results)

Declaration: it2b_batch_floor_r2.md in the planning folder (r2, after review r1), written before any of
these runs. Code: knee.nb.n13, the section "the batch-floor check" (tests: tests/test_n13_batch.py).
The three arms run at commit 39c3c6f. Later commits change only the local decide step.
Finding behind it: the head search chose a batch size on its floor (16) in 6 of 10 units and the floor
was never extended. Question: does a batch size below the floor score higher on inner validation?

Each arm fits, per unit, the unit's chosen Stage A configuration on its inner folds, as n6's inner
fits do (seed 0, 30 epochs, patience 5). Only the batch size differs: bc keeps the chosen one (the
base), b8 uses 8, b4 uses 4. base0 is not the base: it fits the default configuration at batch 48.
No arm refits on an outer training set, predicts a test fold or writes a unit folder.

Rules for the three notebooks:
1. Cell 1 is the usual one (the pip install from GitHub), so Internet is On. Its first line, the same
   in all three: `COMMIT = "39c3c6fc7626e047ed643d5253e8ce9e662302af"`. A fork keeps the old COMMIT
   line: replace it. Check after the run: knee_commit in each summary equals it.
2. Create each notebook as a fork of knee-n13-base0 (CPU, image 20261117-122508). Keep "Pin to
   original environment". Then add the input knee-n6a, replace the COMMIT line and replace cell 2.
   The three then share iteration 2's image, the one s was measured on.
3. Accelerator None (CPU). A GPU session is refused by the code.
4. Inputs, exactly these four: knee-n0, knee-n5, knee-n3-r and knee-n6a. Nothing else. From knee-n6a
   the code reads only agg_n6a_<encoder>_f<k>.json (the chosen configurations) and checks each unit's
   batch size and best inner score against the values read on 25 Nov. Never knee-n6a-smoke, never a
   second knee-n6a. Without knee-n6a the code stops before any fit.
5. Cell 2 is the exact text of the table below. "bc" and "b4" look alike: read the arm name twice.
6. The three run at the same time. budget_h is the soft stop of a session: after it no inner fit
   starts, a running fit ends with its epoch, and the session ends with status "partial" and every
   finished unit in its summary. A session that starts later than 15:30 gets a smaller budget_h, so
   that it ends by 22:00 (budget_h = hours until 21:30).
7. A worker crash is not retried inside the session. Every unit still queued is then listed in
   units_missing with error "worker crashed". Do not restart the session. Start knee-n13-<arm>-r at
   once (see below) for the missing units.
8. Never attach an output of these arms to N7, N9 or N10. They hold no unit folder.

| Notebook | Accelerator | Inputs | Cell 2 | Expected time | Pass |
|---|---|---|---|---|---|
| knee-n13-bc | None (CPU) | knee-n0, knee-n5, knee-n3-r, knee-n6a | `from knee.nb import n13; n13.run("bc", budget_h=6.0, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | about 40 min (estimate from the code; at most about 3 h if every fit ran all 30 epochs) | STEP n13_bc: OK; status "ok"; all_units_ok and leakage_assertion_passed true; added_after_results, test_overrides and chosen_ref_checked true; units_finished holds 10 units and units_missing none; settings.config.batch_size "chosen"; each unit ok with config.batch_size equal to batch_chosen (33, 16, 16, 16, 16 for DINOv2 folds 0-4; 33, 16, 16, 27, 23 for ConvNeXt-T), step_ratio 1.0, schedule_epochs 30, patience 5, seed 0, device cpu; environment present; no head.pt and no predictions.csv in the output |
| knee-n13-b8 | None (CPU) | the same | `from knee.nb import n13; n13.run("b8", budget_h=6.0, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | about 45-60 min (at most about 3 h) | as knee-n13-bc, but STEP n13_b8, settings.config.batch_size 8, each unit's config.batch_size 8 and step_ratio 2 to 4.1 (chosen batch / 8) |
| knee-n13-b4 | None (CPU) | the same | `from knee.nb import n13; n13.run("b4", budget_h=6.0, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | about 1 h to 1 h 15 min (at most about 3.5 h) | as knee-n13-bc, but STEP n13_b4, settings.config.batch_size 4, each unit's config.batch_size 4 and step_ratio 4 to 8.3 |

A session that ends "partial": report it. Its missing units (stopped, failed or "worker crashed") can
run in knee-n13-<arm>-r: a fork of the arm's notebook, the same COMMIT, the same cell 2 plus
`rerun=1, folds=(<k>,), encoders=("<encoder>",)`, on the same image, only if it ends by 22:00.
An encoder with fewer than 5 finished folds in an arm (or in bc) gets no verdict for that arm.

The decision (Claude, outside Kaggle, from inner scores only; no test-fold number exists for these arms):
1. Download agg_n13_bc_inner.json, agg_n13_b8_inner.json and agg_n13_b4_inner.json (and each rerun's)
   into a new directory. Iteration 2's agg_n13_base0_inner.json and agg_n13_null_inner.json give s.
2. `n13.decide_batch(bc, b8, b4, base0, null)`: each argument a summary dict or the list
   [original, rerun, ...]; b8 or b4 may be None. It is a pure function. The planning folder's
   it2b_decision/run_decide_batch.py calls it and writes the record.
3. Write its output with `agg.write("n13_batch_decide", record, folder=<dir>)`: agg_n13_batch_decide.json.
   The checker runs on it: the batch sizes used sit under "config" keys.
4. A status with no verdict at all, checked in this order on every arm summary given:
   "no_base" (bc did not run); "leakage_check_failed" (a summary's leakage_assertion_passed is not
   true); "chosen_ref_not_checked" (a summary's chosen_ref_checked is not true); "commit_mismatch"
   (more than one knee_commit, or not 39c3c6f); "env_mismatch" (the arms ran on different images).
5. status "partial": at least one encoder and arm has "no verdict". status "decided": four verdicts,
   each "the floor mattered" or "no support", by iteration 2's rule with bc as the base (gain above
   0.005 at 4 decimals and higher in at least 4 of 5 folds; s = 0.0030 from base0 and null).
6. After "env_mismatch": which arm is rerun is decided from the environment fields only. The rerun
   replaces the arm whole: give the rerun's summary in the arm's place and the first run as
   `replaced={"<arm>": <summary>}` (the script: move its file into <dir>/replaced/). Both are reported.
7. same_environment_as_iteration2 false is reported, not a stop: s then comes from another image.
8. bc against Stage A (bc_minus_stage_a, never in the rule): a unit reproduces when |bc minus Stage A's
   best inner score| is at most 0.006. Units outside are listed, the verdict is still given, and the
   slide sentence says so (bc_minus_stage_a.words).
9. The slide sentences: `n13.batch_slide_sentences(record)`. One per encoder and arm, from words,
   words_margin and words_limits.
10. Report beside it, as descriptive: units (per-unit scores, E, steps per epoch) and by_chosen_batch
    (units on the floor and above it). The graph and both tables go on a backup slide only.
11. No outcome changes a model, reruns the head search or computes a test-fold number.

## Slice-count check (it2c; added after the results)

Declaration: it2c_slice_count_r3.md in the planning folder (r3; the arm summaries carry the label of
r2, frozen in the arm commit), written before any of these runs. Code: knee.nb.n13s, a new module (tests: tests/test_n13_slices.py). n13.py is unchanged.
The five arms run at commit ecf264f. The later commit changes only the local decide step.
Finding behind it: the proposal promised a search over the slice count and it was never searched; every
model reads 32 slices per plane. Question: does the head, at its chosen settings, score lower on inner
validation when it reads fewer of the 32 cached slices per plane?

Each arm fits, per unit, the unit's chosen Stage A configuration at its chosen batch size on its inner
folds, as arm bc does (seed 0, 30 epochs, patience 5). Only the slices the head reads differ: s32 reads
all 32 (the base; it repeats bc), s16 positions 1, 3, ..., 31, s8 positions 2, 6, ..., 30, s4 positions
4, 12, 20, 28, and s16b the 16 positions s16 leaves out (0, 2, ..., 30; descriptive only, no verdict).
No arm refits on an outer training set, predicts a test fold or writes a unit folder.

Rules for the five notebooks:
1. Create each notebook as a fork of knee-n13-bc (CPU, image 20261117-122508). Keep "Pin to
   original environment". The inputs stay as they are. Then replace the COMMIT line and replace cell 2.
2. Cell 1 is the usual one (the pip install from GitHub), so Internet is On. Its first line, the same
   in all five: `COMMIT = "ecf264fcc11b780e53940b29de86e5a0fc512ae2"`. A fork keeps the old COMMIT
   line: replace it. Check after the run: knee_commit in each summary equals it.
3. Accelerator None (CPU). A GPU session is refused by the code.
4. Inputs, exactly these four, as knee-n13-bc: knee-n0, knee-n5, knee-n3-r and knee-n6a. Nothing else.
5. Cell 2 is the exact text of the table below. "s16" and "s16b" look alike: read the arm name twice.
6. The five run at the same time. budget_h is the soft stop of a session: after it no inner fit
   starts, a running fit ends with its epoch, and the session ends with status "partial" and every
   finished unit in its summary. Latest start 19:30; a session that starts late gets a smaller budget_h.
7. A worker crash is not retried inside the session. Every unit still queued is then listed in
   units_missing with error "worker crashed". Do not restart the session.
8. Never attach an output of these arms to N7, N9 or N10. They hold no unit folder.

| Notebook | Accelerator | Inputs | Cell 2 | Expected time | Pass |
|---|---|---|---|---|---|
| knee-n13-s32 | None (CPU) | knee-n0, knee-n5, knee-n3-r, knee-n6a | `from knee.nb import n13s; n13s.run_slices("s32", budget_h=4.0, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | about 40 min (bc took 36 min) | STEP n13_s32: OK; status "ok"; all_units_ok and leakage_assertion_passed true; added_after_results, test_overrides and chosen_ref_checked true; units_finished holds 10 units and units_missing none; settings.config.slices 32 and slice_positions "0,1,...,31"; each unit ok with config.batch_size equal to batch_chosen, step_ratio 1.0, schedule_epochs 30, patience 5, seed 0, device cpu, slices.config.slices 32; environment present; no head.pt and no predictions.csv in the output |
| knee-n13-s16 | None (CPU) | the same | `from knee.nb import n13s; n13s.run_slices("s16", budget_h=4.0, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | under 40 min | as knee-n13-s32, but STEP n13_s16, slices 16 and slice_positions "1,3,...,31" |
| knee-n13-s16b | None (CPU) | the same | `from knee.nb import n13s; n13s.run_slices("s16b", budget_h=4.0, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | under 40 min | as knee-n13-s32, but STEP n13_s16b, slices 16 and slice_positions "0,2,...,30" |
| knee-n13-s8 | None (CPU) | the same | `from knee.nb import n13s; n13s.run_slices("s8", budget_h=4.0, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | under 40 min | as knee-n13-s32, but STEP n13_s8, slices 8 and slice_positions "2,6,...,30" |
| knee-n13-s4 | None (CPU) | the same | `from knee.nb import n13s; n13s.run_slices("s4", budget_h=4.0, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | under 40 min | as knee-n13-s32, but STEP n13_s4, slices 4 and slice_positions "4,12,...,28" |

A session that ends "partial": report it. Its missing units can run in knee-n13-<arm>-r: a fork of the
arm's notebook, the same COMMIT, the same cell 2 plus `rerun=1, folds=(<k>,), encoders=("<encoder>",)`,
on the same image. An encoder with fewer than 5 finished folds in an arm (or in s32) gets no verdict
for that arm.

The decision (Claude, outside Kaggle, from inner scores only; no test-fold number exists for these arms):
1. Download the five agg_n13_<arm>_inner.json files (and each rerun's) into a new directory. The
   batch-floor check's agg_n13_bc_inner.json gives the rerun check; iteration 2's
   agg_n13_base0_inner.json and agg_n13_null_inner.json give s.
2. `n13s.decide_slices(s32, s16, s8, s4, bc, base0, null, s16b=s16b)`: each argument a summary dict or
   the list [original, rerun, ...]; s16, s8, s4, s16b and bc may be None. It is a pure function. The
   planning folder's it2c_decision/run_decide_slices.py calls it and writes agg_n13_slices_decide.json
   through the aggregate checker: the slices read sit under "config" keys.
3. A status with no verdict at all, checked in this order on every arm summary given:
   "no_base" (s32 did not run); "leakage_check_failed"; "chosen_ref_not_checked"; "commit_mismatch"
   (more than one knee_commit, or not ecf264f); "env_mismatch" (the arms ran on different images).
4. status "partial": at least one encoder and arm has "no verdict". status "decided": six verdicts
   (s16, s8, s4 for each encoder), each "fewer slices cost", "fewer slices scored higher" or
   "no difference shown": diff = mean over the 5 folds of (arm minus s32) at 4 decimals; cost when
   diff < -0.005 and lower in at least 4 of 5 folds; higher when diff > 0.005 and higher in at least
   4 of 5 folds; a fold at 0.0000 is neither; a tie at 0.0050 is inside "no difference shown".
5. Descriptive, never in a verdict: rerun_check (s32 minus bc per unit; a unit above 0.0005 is listed
   and the slide says so), position_effect (s16b minus s16 and s16b minus s32 per encoder), units,
   mean_inner_score and median_best_epoch.
6. The slide sentences: `n13s.slice_slide_sentences(record)`. One per encoder and arm (a no-verdict
   sentence says what finished), then the summary lines (the counts that cost, per encoder; open
   points; the number of comparisons without a verdict), then the rerun-noise sentence if any, then
   the common ending once, as the last line. No sentence names a count as the one to use.
7. No outcome changes a model or computes a test-fold number. No submission.

## Dropout-ceiling check (it2d; added after the results)

Declaration: it2d_dropout_r2.md in the planning folder (r2, after review r1), written before any of these
runs. Code: knee.nb.n13d, a new module (tests: tests/test_n13_dropout.py). n13.py, n13s.py, n6.py and
heads.py are unchanged. The four arms run at commit a886594. The later commit changes only the local decide step.
Finding behind it: the head search drew dropout from 0 to 0.5 and the range end was never extended; two
units chose 0.500. Question: does dropout above the searched range (0.6, 0.7) score higher on inner
validation than each unit's chosen dropout?

Each arm fits, per unit, the unit's chosen Stage A configuration at its chosen batch size on its inner
folds, as arm bc does (seed 0, 30 epochs, patience 5, all 32 slices). Only the dropout of the head differs
(one dropout on the 12 per-finding pooled vectors, in training only): dc keeps the chosen one (the base;
it repeats bc), p5 uses 0.5 in every unit (the range end; descriptive and for the words only), p6 uses
0.6, p7 uses 0.7. Each unit's chosen dropout is checked against the values read on 25 Nov (3 decimals)
with its batch size and best inner score, before any fit.
No arm refits on an outer training set, predicts a test fold or writes a unit folder.

Rules for the four notebooks:
1. Create each notebook as a fork of knee-n13-bc (CPU, image 20261117-122508). Keep "Pin to
   original environment". The inputs stay as they are. Then replace the COMMIT line and replace cell 2.
2. Cell 1 is the usual one (the pip install from GitHub), so Internet is On. Its first line, the same
   in all four: `COMMIT = "a88659486b5ba3bb4aeba189fb605899b57aeddd"`. A fork keeps the old COMMIT
   line: replace it. Check after the run: knee_commit in each summary equals it.
3. Accelerator None (CPU). A GPU session is refused by the code.
4. Inputs, exactly these four, as knee-n13-bc: knee-n0, knee-n5, knee-n3-r and knee-n6a. Nothing else.
5. Cell 2 is the exact text of the table below. "p6" and "p5" look alike: read the arm name twice.
6. The four run at the same time (with one GPU session that is the five-session limit). budget_h is the
   soft stop of a session: after it no inner fit starts, a running fit ends with its epoch, and the
   session ends with status "partial" and every finished unit in its summary. Latest start 20:30; a
   session that starts late gets a smaller budget_h.
7. A worker crash is not retried inside the session. Every unit still queued is then listed in
   units_missing with error "worker crashed". Do not restart the session.
8. Never attach an output of these arms to N7, N9 or N10. They hold no unit folder.

| Notebook | Accelerator | Inputs | Cell 2 | Expected time | Pass |
|---|---|---|---|---|---|
| knee-n13-dc | None (CPU) | knee-n0, knee-n5, knee-n3-r, knee-n6a | `from knee.nb import n13d; n13d.run_dropout("dc", budget_h=3.0, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | about 35-45 min (bc took 36 min) | STEP n13_dc: OK; status "ok"; all_units_ok and leakage_assertion_passed true; added_after_results, test_overrides and chosen_ref_checked true; units_finished holds 10 units and units_missing none; settings.config.dropout "chosen"; each unit ok with config.batch_size equal to batch_chosen, dropout.config.dropout_used equal to dropout_chosen (0.472, 0.0, 0.5, 0.053, 0.0 for DINOv2 folds 0-4; 0.472, 0.5, 0.386, 0.0, 0.222 for ConvNeXt-T, at 3 decimals), step_ratio 1.0, schedule_epochs 30, patience 5, seed 0, device cpu; environment present; no head.pt and no predictions.csv in the output |
| knee-n13-p5 | None (CPU) | the same | `from knee.nb import n13d; n13d.run_dropout("p5", budget_h=3.0, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | about 35-45 min | as knee-n13-dc, but STEP n13_p5, settings.config.dropout 0.5 and each unit's dropout_used 0.5 |
| knee-n13-p6 | None (CPU) | the same | `from knee.nb import n13d; n13d.run_dropout("p6", budget_h=3.0, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | about 35-45 min | as knee-n13-dc, but STEP n13_p6, settings.config.dropout 0.6 and each unit's dropout_used 0.6 |
| knee-n13-p7 | None (CPU) | the same | `from knee.nb import n13d; n13d.run_dropout("p7", budget_h=3.0, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})` | about 35-45 min | as knee-n13-dc, but STEP n13_p7, settings.config.dropout 0.7 and each unit's dropout_used 0.7 |

A session that ends "partial": report it. Its missing units can run in knee-n13-<arm>-r: a fork of the
arm's notebook, the same COMMIT, the same cell 2 plus `rerun=1, folds=(<k>,), encoders=("<encoder>",)`,
on the same image. An encoder with fewer than 5 finished folds in an arm (or in dc) gets no verdict
for that arm.

The decision (Claude, outside Kaggle, from inner scores only; no test-fold number exists for these arms):
1. Download the four agg_n13_<arm>_inner.json files (and each rerun's) into a new directory. The
   batch-floor check's agg_n13_bc_inner.json gives the rerun check; iteration 2's
   agg_n13_base0_inner.json and agg_n13_null_inner.json give s.
2. `n13d.decide_dropout(dc, p6, p7, bc, base0, null, p5=p5)`: each argument a summary dict or the list
   [original, rerun, ...]; p5, p6, p7 and bc may be None. It is a pure function. The planning folder's
   it2d_decision/run_decide_dropout.py calls it and writes agg_n13_dropout_decide.json through the
   aggregate checker: the dropout values sit under "config" keys.
3. A status with no verdict at all, checked in this order on every arm summary given (p5 too):
   "no_base" (dc did not run); "leakage_check_failed"; "chosen_ref_not_checked"; "commit_mismatch"
   (more than one knee_commit, or not a886594); "env_mismatch" (the arms ran on different images).
4. status "partial": at least one of the four comparisons (p6 and p7 against dc, per encoder) has
   "no verdict"; a missing p5 never blocks one. status "decided": four verdicts, each "higher dropout
   scored higher", "higher dropout scored lower" or "no difference shown": gain = mean over the 5 folds
   of (arm minus dc) at 4 decimals; higher when gain > 0.005 and higher in at least 4 of 5 folds; lower
   when gain < -0.005 and lower in at least 4 of 5 folds; a fold at 0.0000 is neither; a tie at 0.0050 is
   inside "no difference shown".
5. A "scored higher" is compared with p5 by the same rule: "the ceiling mattered" only when arm minus p5
   meets it too; "inside the searched range" when it does not; without a whole p5 the open point says
   the cause is not shown. These words change no verdict.
6. Descriptive, never in a verdict: rerun_check (dc minus bc per unit; a unit above 0.0005 is listed and
   the slide says so), p5_minus_dc, by_chosen_dropout (units with chosen dropout at least 0.4 and the
   others), units, mean_inner_score and median_best_epoch.
7. The slide sentences: `n13d.dropout_slide_sentences(record)`. One per encoder and arm (a no-verdict
   sentence says what finished), then the summary lines, then the rerun-noise sentence if any, then the
   common ending once, as the last line. With no verdict in all four comparisons: only the summary line,
   no common ending. No sentence names a dropout as the one to use.
8. No outcome changes a model or computes a test-fold number. No submission.

## If a run fails

The summary block shows `STEP <name>: FAILED - <ErrorType> at <file>:<line> in <function>()`.
That line is all Claude needs for most fixes: Claude fixes the code, runs the tests,
pushes, updates COMMIT, and reruns.

If a fix needs the details, Ablay (not Claude) reads them, in his own browser:
1. The failed version's private/<step>_error.log holds the step's hidden output and the full traceback.
2. Or, in an interactive session only he watches, he reruns cell 2 with `quiet=False` and then runs `sl.show_last()`.
3. He describes the problem to Claude in words, without values.

## Error analysis N12 (plan r2)

N12 is descriptive. It reads existing outputs only. It fits nothing, decides nothing and switches nothing on (plan r2, section 4 and R8). It may run before Ablay signs off readings r3, but nothing it writes (summary block, agg_n12.json, run.json) is opened before that sign-off (R1, as agreed with Ablay on 24 Nov at 00:23). Read only a version run at 89a9058 or later: check run.json's commit first. knee-n12 version 1 ran at d7c14bf, whose per-rate rule leaks small cells (review r2, R2-B1); it is never read. Never attach its output to N7, N9, N10 or N13.

| Notebook | Accelerator | Internet | Inputs |
|---|---|---|---|
| knee-n12 | None (CPU) | On | Outputs of knee-n0, **knee-n3-r** (not knee-n3), knee-n5, knee-n6a, knee-n6b-g and **knee-n7-b** (not knee-n7). Exactly one output per upstream step. Never knee-n6a-smoke, a knee-n9 output or an iteration-2 output |

Cell 1 (commit 89a9058 holds the N12 code after reviews r1 and r2; the commit that moved COMMIT here changes only this file):

```python
COMMIT = "89a9058177e73f1d79a936caac3513aa7e29fd23"
!pip install -q "knee[n0,decode,models] @ git+https://github.com/lanNo19/knee-mri-ds504@{COMMIT}"
import knee.safe_log as sl; sl.install()
```

Cell 2:

```python
from knee.nb import n12; n12.run(expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})
```

Expected time: 5-15 min. Most of it is the pip install. The study bootstraps (2,000 draws each) take about 1-3 min on CPU.

Pass criteria (summary block and agg_n12.json):
1. STEP n12: OK. status "ok", errors empty, chain_ok true, main_complete true. A chain mismatch fails the step, as in N9.
2. Every block is present with descriptive true and added_after_results true: gold_paired_check, gold_label_quality, fm5b_attenuation_check, per_finding_table, english_group_by_language, and development with a2_seed_ensemble, a3_encoder_ensemble and a4_calibration. Each development block carries the label "development: these studies were test studies in iteration 1; likely optimistic".
3. N12 must read the same data as N9. Compare each number with knee-n9-b's agg_results.json, file against file (both hold 4 decimals; per-finding AUCs 3). The values in brackets are readings for orientation. Any difference = stop and report both files.
   - per_finding_table.rows.<finding>.s2_auc.<encoder> equals s2.<encoder>.per_finding.<finding> (for example ACL: convnext_tiny 0.679, dinov2_vits14 0.638).
   - gold_paired_check.encoders.<encoder>.expert_labels_on_n9_findings equals s3_gold.macro_auc.<encoder> (convnext_tiny 0.7515, dinov2_vits14 0.7317). This holds always: it uses N9's own gold studies and findings.
   - For each encoder in development.a4_calibration.encoders: ece.before equals s2.<encoder>.ece (convnext_tiny 0.256, dinov2_vits14 0.255), and any_finding_auc.before equals triage_secondary.models.<encoder>.any_finding_auc (convnext_tiny 0.765, dinov2_vits14 0.734).
4. gold_paired_check.studies_text is "about 60", or "about 55" when gold_studies_without_report_labels is not 0 (the gold studies with a report label). gold_paired_check.gold_studies_without_report_labels is 0 or not: if not, studies_without_report_labels_left_out is true and those studies are left out of every comparison that needs the report label (expected and correct; quote the count as written).
5. english_group_by_language.group is "ENGLISH" and fold is 0 (both found from the English reports, not hard-coded); training_studies is about 1,350. The finding list is set by the named rows (named_rows: ENGLISH, SPANISH, TURKISH) that have 50 studies; list_fixed_on names those rows, which on the real data are all three, and languages_reported holds them. FRENCH and LATIN appear in languages_reported only if they meet that whole list; otherwise languages_suppressed gives the reason.
6. In development.a4_calibration, each encoder is either in encoders or in skipped with its reason. In encoders: every check named in gate_must_be_true is true, none named in gate_must_not_be_false is false (null = not recorded), fold_stratified_macro_auc_unchanged is true. info.stage_a_n5_index_same never stops A4 (the weights do not use N5); if it is false, say so beside the A4 numbers. A4 is optional, so a skip is not a failure.
7. fm5b_attenuation_check.encoders.<encoder>.report_minus_if_independent is there for each encoder. per_finding_blank and gold_label_quality.rates_blank list findings whose rates are blank by the privacy rule: expected, not a failure. Expect most per-finding rates to be blank: on about 60 gold studies most findings have a cell of 1-4. The soft-label AUC, item 3's Spearman and the fm5b macro do not depend on them, so FM5b stays measurable.

Rules for reading and quoting N12:
1. Nothing in agg_n12.json is a test or a decision.
2. The development numbers (A2, A3, A4) keep their label. They are never shown as a gain over iteration 1 and never go in a slide title (R6). members_mean is a reference for reading an ensemble, never a gain.
3. Per-finding numbers: show all 12 or none.
4. Name the metric beside every number (FM10). Fold-stratified S2 numbers are mostly the English group.
5. Gold per-finding numbers have 2 decimals, and a rate is blank when its 2x2 table holds a cell of 1-4 studies. That is the privacy rule (review r1, B1): never work a blank back out, and never quote more decimals than the file holds.
6. fm5b_attenuation_check: read the macro. Above 0 = the model's errors line up with the report labels' errors; the relation assumes they are independent given the expert label.

| Field | Meaning |
|---|---|
| gold_paired_check.encoders.<encoder>.expert_labels_on_n9_findings | macro AUC on N9's own gold studies and N9's gold findings: N9's s3_gold number, recomputed (the cross-check) |
| gold_paired_check.encoders.<encoder>.expert_labels, report_labels | macro AUC on the gold studies with a report label, fold models pooled, against the expert labels and against the binary report labels (uncertain = negative); the same findings for both (5 positives and 5 negatives in both versions: findings, findings_left_out; findings_same_as_n9_s3 says whether they are N9's list) |
| gold_paired_check.encoders.<encoder>.expert_labels_interval, report_labels_interval, expert_minus_report | study-bootstrap percentile intervals of the two macros, and their difference with its paired interval (FM5b) |
| gold_*.gold_studies_without_report_labels, studies_without_report_labels_left_out | gold studies with no report label at all (rounded), and whether any were left out; they are never read as "absent" |
| gold_label_quality.per_finding.<finding> | soft_label_auc (the label mean against the expert label, on the studies with a soft label) with its interval; ppv and npv of the binary report label and ppv_plus_npv_minus_1 (the FM5b factor). 2 decimals; all three are null when any cell of the finding's 2x2 table holds 1-4 studies (the gold total is exact, so one shown rate would carry the other cells); suppressed with under 5 expert positives or negatives |
| gold_label_quality.rates_blank, soft_label_macro_auc | findings whose rates are blank by that rule (no cell of 1-4 comes back; exact counts of 5 or more are not hidden); the mean soft-label AUC over the shown findings (N0's 0.892 used every gold study) |
| fm5b_attenuation_check.encoders.<encoder>.report_labels, report_labels_if_independent, report_minus_if_independent | macro over item 1's findings: the report-label AUC, what the FM5b relation predicts from the expert-label AUC and PPV + NPV - 1, and their difference with a paired study-bootstrap interval |
| fm5b_attenuation_check.encoders.<encoder>.per_finding.<finding> | the same per finding at 2 decimals (expert_labels_auc, report_labels_auc; the predicted value and the difference blank where per_finding_blank lists the finding) |
| per_finding_table.rows.<finding> | s2_auc per encoder (fold-stratified, N9's numbers), training_prevalence (N0, over the training studies), label_quality (as in gold_label_quality) |
| per_finding_table.spearman_label_quality_vs_s2_auc | per encoder: Spearman across findings between the soft label's gold AUC and the S2 AUC; no_association_band (about 0.58 for 12 findings: a smaller absolute rho fits no association); descriptive |
| english_group_by_language.rows.<language> | macro AUC of the English group's own fold model on the group's studies of one report language (at least 50 studies), on the list set by the named rows with 50 studies (list_fixed_on), study-bootstrap interval |
| english_group_by_language.languages_suppressed, findings_left_out, training_studies | per suppressed language its reason (under 50 studies, or a finding of the list without 5 positives and 5 negatives, named); findings left out of the list, with the named rows that lack them; the fold model's training set (rounded) |
| english_group_by_language.reference_whole_group | the same for the whole group (all its languages), on the same list |
| development.a2_seed_ensemble.encoders.<encoder> | mean probability over N7's seed refits (seeds listed): fold-stratified macro AUC and mean of fold macros; members_mean = the same for the single seed refits |
| development.a3_encoder_ensemble | equal-weight mean of the three encoders' Stage A probabilities: the same two metrics; members_mean = the three encoders alone on the same studies; not evidence about H1 |
| development.a4_calibration.encoders.<encoder> | ECE and any-finding AUC (max of 12) before and after subtracting log(positive weight) from each logit; checks (the gate) and info (never gates); log_positive_weight_mean_over_folds |
| development.a4_calibration.skipped.<encoder> | why A4 is not computed for that encoder (not attached, incomplete predictions, or a failed check), with the checks |
| not_run | the analyses plan r2 leaves out, each with its reason |

## Error analysis N14 (error hunt, 24 Nov)

N14 is error analysis (n14_spec_r3.md in the planning folder). It reads existing outputs and fits only C1's three all-data heads. Its numbers keep nothing and switch nothing on (R8): the next change is chosen from the failure-mode report that uses them, written into the plan before it runs (R3) and kept or dropped on inner validation only (R4). Ablay set this order on 24 Nov at 11:15. Never attach its output to N7, N9, N10 or N13.

| Notebook | Accelerator | Internet | Inputs |
|---|---|---|---|
| knee-n14 | None (CPU) | On | The competition data (C5 reads one DICOM header per chosen series), and the outputs of knee-n0, **knee-n3-r**, knee-n5, knee-n6a, knee-n6b-g and **knee-n7-b**. Exactly one output per upstream step. Never knee-n6a-smoke, knee-n3, knee-n7 or an iteration-2 output |

Cell 1: COMMIT = the commit named in the planning log for N14 (the N14 code after its review).

Cell 2:

```python
from knee.nb import n14; n14.run(expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})
```

Expected time: 20-45 min on CPU (pip 3-5 min; C1's three refits 5-15 min; C5's header pass 5-10 min; bootstraps a few min).

Pass criteria (summary block and agg_n14.json):
1. STEP n14: OK. status "ok", errors empty, chain_ok true, main_complete true. A chain mismatch fails the step.
2. All six blocks present with descriptive true and added_after_results true: c1_gold_all_data (label "development: ..."), c2_side_discrimination, c3_sagittal_attention, c4_fallback_contrast, c5_centre_crop, c6_s1_level.
3. Same data as N9 (knee-n9-b's agg_results.json, 4 decimals): c1_gold_all_data.encoders.<encoder>.s2_pooled equals s3_gold.macro_auc.<encoder> (convnext_tiny 0.7515, dinov2_vits14 0.7317); s2_fold_stratified equals the gold fold-stratified values (convnext_tiny 0.7272, dinov2_vits14 0.7099); c6_s1_level.encoders.<encoder>.s2_mean_of_fold_macros equals s2.<encoder>.mean_of_fold_macros (convnext_tiny 0.7502, dinov2_vits14 0.7422) when same_studies_as_evaluation_set is true. A difference = stop and report both.
4. c1: E per encoder is N7's consensus E (dinov2_vits14 4, convnext_tiny 5, vit_s16 10); gold_per_s2_fold, lookalikes_removed and training_studies present.
5. c3: skipped_units empty (every main unit is gated over all planes, and max_abs_gap_vs_predictions is at most 0.001 for each: the rebuilt head reproduces the unit's predictions); otherwise say which and why.
6. c5: series_read about 4,405 per plane; unread 0 or small.

Rules for reading N14:
1. Every number is descriptive. C1 carries the development label (R6): never a gain over iteration 1, never in a slide title.
2. Each block's reading follows the rule written in n14_spec_r3.md before the run; quote the rule with the number.

## Stage B-lite N11 (iteration 3, all five folds)

N11 is Stage B-lite on all five S2 folds (Iteration plan, section 12 = it3_amendment_r3.md, which replaces section 11 where they differ; code knee.nb.n11, tests tests/test_n11.py). Everything is "added after the results" (R10). The text of section 12 must be in the plan before the first run (R3). N11's outputs are attached only to its own decide and dev steps and, if the change is kept, to N10. Never to N7 or N9 (R9).

Order: 1. knee-n11-smoke. 2. knee-n11-a and knee-n11-b at the same time. 3. knee-n11-decide. 4. knee-n11-dev.

| Notebook | Accelerator | Internet | Inputs |
|---|---|---|---|
| knee-n11-smoke, knee-n11-a, knee-n11-b (and knee-n11-c, knee-n11-d after a partial session) | GPU T4 x2 | On (pip, and the encoder weights as in N5) | Outputs of knee-n0, **knee-n3-r**, knee-n4-sagittal, knee-n4-coronal, knee-n4-axial, knee-n5 and knee-n6a. Exactly one output per upstream step. Never knee-n3, knee-n6a-smoke, a unit rerun or an iteration-2 output (the code refuses a unit that is not knee-n6a's). No competition data |
| knee-n11-decide | None (CPU) | On | Outputs of knee-n11-a and knee-n11-b (and knee-n11-c, knee-n11-d if they ran). Nothing else |
| knee-n11-dev | None (CPU) | On | Outputs of knee-n11-a, knee-n11-b (and knee-n11-c, knee-n11-d if they ran) and knee-n11-decide. Nothing else |

Create the three GPU notebooks as forks of knee-n5 and keep "Pin to original environment": check (b) compares fp16 features with N5's, and decide() refuses two sessions with different environments. Create knee-n11-decide and knee-n11-dev as forks of one of them.

Cell 1 (all five): COMMIT = the commit named in the planning log for N11 (the same commit in every notebook).

### 1. knee-n11-smoke

```python
from knee.nb import n11; n11.run(smoke=True, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})
```

Expected time: 5-10 min. It runs 200 micro-batches per encoder on fold 0 and reads no score.

Pass criteria (summary block and agg_n11_smoke.json):
1. STEP n11_smoke: OK. status "ok (smoke: no score was read)".
2. Per unit: smoke.reads_no_score true; smoke.peak_memory_gb under 14; smoke.micro_batch_slices 192 (96 = 2 studies did not fit and 1 is used: say so).
3. smoke.late_code_paths: head_only_step, flipped_forward_finite, checkpoint_round_trip and run_json_lock all true; feature_store float16.
4. trainable_parameters: dinov2_vits14 5.2-5.4M, convnext_tiny 4.7-4.9M (the code refuses other counts).
5. environment (torch, python, CUDA, image) equals knee-n5's run.json versions. If not, stop.
6. validation_table lists all five folds with the plan's groups (names and rounded sizes). "FAILED - AssertionError" at inner_split = the groups differ from the plan's table: correct the table in the plan first, then the code.
7. suggested_schedule is "full" or "halved" (the slower encoder under 300 slices/s = halved). Write this value into cell 2 of BOTH sessions. steady_slices_per_s is reported only. Hand projection for session a, per encoder (review r2, MS2; forward assumed at 1.5 x the training speed, not measured): 300 slices/s = 12-15 h on the full schedule (about 9 h halved); 400 = 9.3-11.4 h; 450 = 8.3-10.1 h; 500 = 7.4-9.1 h. The registered rule says "full" at 300 or more. Which schedule to use at 300-450 slices/s is Ablay's decision, taken before the sessions start and written here. A session that passes its soft stop ends "partial" and loses only its unfinished units.
8. cache_copy.where is "local copy" (the three caches are copied once to local disk and checked against the source; copy_seconds and copy_mb_per_s say how fast). "source" = no room or a failed copy: the input files are read in whole-study blocks; say so. time_breakdown per encoder (read, to_gpu, trunk, tail_and_backward, optimiser) and read_share show where the time goes: a read share above about 0.2 means the data path is still the limit.
9. units_loaded lists all five folds for each encoder (every fold's Stage A unit loads).

### 2. knee-n11-a and knee-n11-b

knee-n11-a:

```python
from knee.nb import n11; n11.run(session="a", folds=(0, 1, 2), expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"}, schedule="full")
```

knee-n11-b:

```python
from knee.nb import n11; n11.run(session="b", folds=(3, 4), expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"}, schedule="full")
```

If the smoke run said "halved", write `schedule="halved"` in both. Plan triggers: not started by 20:00 = halved in every fold; not started by 23:00 = halved and `encoders=("convnext_tiny",)` in both (the two GPUs then take different folds). Never fewer folds.

Expected time (estimate, not measured): session a 6.5-8 h, session b 4.5-5.5 h at 500 slices/s. The watchdog stops a session at 11.5 h.

Inside a session: frozen0 checks of every fold, then all inner work of every fold, then the final fits of every fold. The notebook computes no test-fold metric of a trained model.

Pass criteria per session (summary block, agg_n11_a.json or agg_n11_b.json, run.json):
1. STEP n11_a (n11_b): OK. status "ok", errors empty. status "partial" (a unit failed, or the session passed its 10.5 h soft stop: out_of_time true) = the units in units_complete are good; rerun the units in units_missing in knee-n11-c (and knee-n11-d for a second call). "FAILED - InvalidRun" = frozen0 check (b) failed for a unit: the run is invalid, nothing else is read, log it. "FAILED - AssertionError" at inner_split = the printed groups differ from the plan's table: stop and correct the table in the plan first. After status "failed": Ablay reads private/n11_<session>/<encoder>_f<fold>/error.log in his own browser; the agg still lists every inner run that finished.
2. All units present: 3 folds x 2 encoders in a, 2 x 2 in b. Each with frozen0_valid true, inner_complete true, final_complete true.
3. Validation table, per unit: test_groups and inner_validation_groups are the plan's (12.1 item 3: fold 0 English / Croatian; 1 Croatian / Greek; 2 Greek / German; 3 Bosnian+German / Bulgarian+Dutch; 4 Bulgarian+Dutch / Greek), sizes 250-500 (the code refuses anything else).
4. frozen0 per unit: check_b.abs_difference at most 0.0005; max_abs_probability_gap and check_a.min_cosine reported.
5. config.epochs: 12 for fold 0 and 4 for folds 1-4 under "full" (6 and 2 under "halved"). schedule equal in both sessions and equal to run.json config.schedule.
6. Inner work per unit: inner.rung0 has the three rates (curves of 3 points); inner.to_the_end has the survivor and 2e-05 (curves of 12 points with train_loss, val_loss, val_macro_auc); inner.head_only has seed_0 and seed_100; inner.keep_rule_arms has start, head_only_seed_0, head_only_seed_100 and b_lite_2e-05.
7. Chosen rate recorded: run.json n11_chosen_encoder_lr.<encoder>_f<fold>.lr equals units.<unit>.inner.chosen_lr (the code refuses a test pass of a trained model otherwise).
8. No test-fold metric: the only keys with "auc" outside inner are frozen0.check_b's. Do not compute anything from the private prediction files.

Cut order (plan 12.5; arguments, recorded in cuts): the B-lite S4 submission; the side AUC (`n11.dev(side_auc=False)`); `head_only_seeds=(0,)`; `rung1_tie=False`. Never cut: any fold, the inner head-only arm, the null seed, the 2e-5 run, the frozen0 checks, the training-and-validation graph.

### 2b. knee-n11-c and knee-n11-d (only after a session ended "partial")

Rerun units whole, with the same schedule and the same commit. Name only units from units_missing: a rerun of a unit that is already complete is refused later ("equally far"). One notebook per call: the first call is knee-n11-c with session="c", a second call (the other encoder, or other folds) is knee-n11-d with session="d". Never two calls with one session name: the second would delete the first call's folder. Edit folds and encoders:

```python
from knee.nb import n11; n11.run(session="c", folds=(0,), encoders=("convnext_tiny",), expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"}, schedule="full")
```

knee-n11-d:

```python
from knee.nb import n11; n11.run(session="d", folds=(0,), encoders=("convnext_tiny",), expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"}, schedule="full")
```

Pass criteria: as a session, for the units it names. decide and dev take, per unit, the folder that got further (final fits complete, else inner work complete); two folders that got equally far are refused.

### 3. knee-n11-decide (after both sessions have ended)

```python
from knee.nb import n11; n11.decide()
```

Expected time: 3-6 min. It reads inner validation files only and opens no test-fold file.

Pass criteria (summary block = agg_n11_decide.json, the decision record):
1. STEP n11_decide: OK. status "decided". "env_mismatch" = the two sessions ran in different environments: there is no decision; say so and stop (no dev step).
2. knee_commits has one value (two values or "None" give env_mismatch). sessions ["a", "b"] (plus "c" after a rerun); encoders_run lists both encoders (or convnext_tiny only after the 23:00 trigger; then n_null is 5 and development_delta1_reported false); n_null 10; written_at_utc and sha256 present.
3. Per encoder: keep (true or false), gain, threshold = max(0.005, 2 s / sqrt(5)), folds_up, interval, baseline (start or head_only), per_fold. Copy them; never recompute by hand. Kept needs all three conditions.
4. If one encoder cannot be completed: `n11.decide(encoders=("<the complete encoder>",))` reads only that encoder (n_null 5, encoders_left_out names the other). This needs Ablay's OK first: the plan defines the one-encoder case only for the 23:00 trigger.
5. Record the file in the planning folder with its sha256 before the dev step starts (R5).

### 4. knee-n11-dev (only after the decision is recorded)

```python
from knee.nb import n11; n11.dev()
```

Expected time: 3-6 min. It refuses to run without a "decided" record, after an edited record (sha256) and when a prediction file is not the one the record lists.

Pass criteria (summary block and agg_n11_dev.json):
1. STEP n11_dev: OK. status "ok". decision.sha256 equals the recorded decision's.
2. Per encoder: frozen0, head_only_seed_0 and b_lite, each with mean_of_fold_macros, fold_stratified_macro_auc and fold_stratified_interval (2,000 study draws inside each test fold, paired); both differences with their intervals; head_only_seed_values (5).
3. frozen0 equals iteration 1's S2 values for that encoder (knee-n9-b: mean of fold macros convnext_tiny 0.7502, dinov2_vits14 0.7422) to about 0.001. A larger gap = stop and report.
4. delta1 present with both encoders, with the paired interval of its change (change_interval_excludes_0) ("not reported" with ConvNeXt-T only); side_discrimination has frozen0 and b_lite.

### 4b. knee-n11-dev-findings (optional; after knee-n11-dev; the per-finding description)

CPU, internet On. Inputs: the same as knee-n11-dev (outputs of knee-n11-a, knee-n11-b, and knee-n11-c, knee-n11-d if they ran, and knee-n11-decide). Nothing else (knee-n11-c and knee-n11-d: add by hand if they ran). Fork of knee-n11-dev. COMMIT = the reviewed commit that holds n11.dev_findings (named in the planning log); dev()'s code path is unchanged from 095c91f.

```python
from knee.nb import n11; n11.dev_findings()
```

Expected time: 3-6 min. The gate is dev()'s: it refuses to run without a "decided" record, after an edited record (sha256) and when a prediction file is not the one the record lists. It reads the same prediction files as dev() and trains nothing.

What to read (summary block and agg_n11_dev_findings.json):
1. STEP n11_dev_findings: OK. status "ok". decision.sha256 equals the recorded decision's.
2. Check first: per encoder and arm, the mean of the 12 findings' fold_stratified_auc equals fold_stratified_macro_auc in agg_n11_dev.json (to 0.0002; both sides are rounded to 4 decimals). A gap = stop and report.
3. encoders.<encoder>.findings.<finding>: positives (rounded to 5; "<5" = too few to show), folds and n_folds (the folds where the finding has both classes), then per arm fold_stratified_auc and mean_of_fold_aucs, then b_lite_minus_frozen0 and b_lite_minus_head_only with fold_stratified_interval and interval_side (above_0, below_0, includes_0, undefined). undefined = valid_share under 1 (a draw had no pair) or no interval. A finding with 1 to 4 positive or negative studies has suppressed "too few to show": its AUCs, differences, intervals, folds and n_folds are null.
4. encoders.<encoder>.per_fold_macro.fold_<k>: each arm's macro AUC in that fold and b_lite_minus_frozen0. null = a finding lacks a class in that fold. The mean of the values that are not null is dev()'s mean_of_fold_macros.
5. encoders.<encoder>.summary: all_findings and groups (meniscus_pair, oa_pair, all_other), each with the lists of findings whose interval of b_lite minus frozen0 lies above 0, below 0 or includes 0. The number of findings is the length of the list.
6. Read a finding with few positives or n_folds under 5 with care: its interval is wide.

Reading rule: every number here carries the development label (R6) and is descriptive. The intervals (12 findings x 2 differences per encoder) have no multiplicity correction; they describe, they decide nothing (R8). The decision stays the record of knee-n11-decide.

Rules for reading N11:
1. The decision is the record's keep per encoder, from inner validation only (R4). Limits said with it (record.limits).
2. Every dev number carries the development label (R6): grey, never in a title, never as a gain over iteration 1, never on slide 12. They are reported whatever the decision (R10). A development Delta1 is not a new H1 result.
3. The graph for the instructor: units.<unit>.inner.to_the_end.<2e-05 and the survivor>.curve (training and validation, 12 points; the training curve has dropout on, the validation curve has it off). Fold 0 on the slide, all folds in the appendix.
4. If kept for an encoder: the S4 submission needs N10 with tails (private/n11_<session>/<encoder>_f<fold>/b_lite_seed_0.pt) and Ablay's OK (plan 12.4). N10 reads these .pt files, each unit's state.json and each session's run.json (aggregates only); the per-study files (labels, UIDs) in the same folders are not read there. Card: "N10 with N11's B-lite models" below.

## N10 with N11's B-lite models (iteration 3; only if the decision keeps the change; Ablay presses Submit)

Code knee.nb.n10 (tails="n11"), tests tests/test_n10_tails.py. The submitted model for one encoder is the mean of five fold models. A fold model is the shared frozen trunk, the fold's fine-tuned tail and the fold's head, from N11's final fits. Per study the trunk runs once and the five tails run on its output (a study of unresolved laterality gets one more, mirrored, trunk pass). The forward is N11's own code (fp16 on the GPU, flip averaging, z-scores with the fold's Stage A statistics). knee-n10, knee-n10-convnext and knee-n10-canary are unchanged.

| Notebook | Accelerator | Internet | Inputs | Cell 2 | Expected time |
|---|---|---|---|---|---|
| knee-n10-blite | GPU T4 x2 | **Off** | Competition; the bundle dataset; outputs of knee-n0, knee-n6a, knee-n11-a and knee-n11-b (and knee-n11-c, knee-n11-d if they ran). Never a unit rerun of Stage A, never knee-n11-smoke's output alone | offline cell 1, then `from knee.nb import n10; n10.run("dinov2_vits14", tails="n11")` | 3-6 h on the hidden test |
| knee-n10-convnext-blite | GPU T4 x2 | **Off** | the same | offline cell 1, then `from knee.nb import n10; n10.run("convnext_tiny", tails="n11")` | 3-6 h on the hidden test |

Before the run:
1. The bundle dataset must be a new version built at a commit that holds this code (the bundle's wheel is what runs; 47fdcbf has no tails). bundle_commit in agg_n10.json shows which one ran.
2. knee-n6a stays attached. N11 saves the tail and the head, not the fold's z-score statistics or head config. N10 reads both from knee-n6a's unit, as N11 did. Attach the same version of knee-n6a that knee-n11-a had attached (open knee-n11-a, Input, and read the version). N11 logged no hash of stats.pt, so the code cannot check this; the replay check below does.

The replay check (run it first; it must pass for the encoder that will be submitted):

| Notebook | Accelerator | Internet | Inputs | Cell 2 | Expected time | Pass |
|---|---|---|---|---|---|---|
| knee-n10-tails-check | GPU T4 x2 | **Off** | The bundle dataset (the version the submission will use); outputs of knee-n0, knee-n3-r, knee-n4-sagittal, knee-n4-coronal, knee-n4-axial, knee-n6a, knee-n11-a and knee-n11-b (and knee-n11-c, knee-n11-d if they ran). No competition data | offline cell 1, then `from knee.nb import n10; n10.check_tails("dinov2_vits14", fold=0, n=200)` (ConvNeXt-T: `"convnext_tiny"`; another fold: fold=k) | 5-10 min | agg_n10_tails_check.json: passed true (max_abs_probability_gap at most 0.01 and mean_abs_probability_gap at most 0.001); studies 200; amp_fp16 true. Keep trunk_sha16 |

What the check does: it runs the fold's b-lite model (bundle checkpoint, the fold's tail and head, knee-n6a's statistics) on the first 200 studies of N11's own pred_b_lite.csv for that fold, from the N4 caches, and compares the probabilities. It reads the UID, "resolved" and the 12 probability columns of that private file. It reads no label, computes no metric and writes aggregates only. A fail (STEP n10_tails_check: FAILED; the agg file is still written, with passed false) means the trunk, the knee-n6a version, the tail or the head is not what N11 ran: stop and report. studies_unresolved_laterality is the rounded number of replayed studies that took the mirrored pass.

Order: knee-n10-tails-check runs only after knee-n11-decide has written the decision record. It opens test-fold predictions of a trained model (rule R5). The code does not enforce this order.

Only one GPU is used by the b-lite path.

The step refuses to run (STEP n10: FAILED, before any decoding, no submission.csv) unless all of these hold for the encoder: exactly folds 0-4; every unit complete (frozen0 valid, inner work and final fits complete); each b_lite_seed_0.pt is the file its state.json records; one known commit over the sessions; one schedule; one learning-rate rule (rung1_tie and passes of the sessions, and every final fit at its unit's chosen rate); no session with test overrides; knee-n6a's config of the fold equals the one N11 logged.

Check in agg_n10.json before asking Ablay to submit:
1. mode "b-lite". folds [0, 1, 2, 3, 4]. failed_share 0 (the step itself allows at most 0.01). submission.layout wide.
2. b_lite.n11_commit = the commit of the N11 sessions. b_lite.schedule = the sessions' schedule.
3. Per fold, b_lite.units.fold_k.weights_file_sha16 equals units.<encoder>_f<k>.files_sha16["b_lite_seed_0.pt"] in agg_n11_<session>.json (the same string is in the decision record's prediction_files_sha16). This one hash covers the tail and the head: N11 saves both in one file.
4. Per fold, tail_sha16 and head_sha16 are present and differ between folds. They are hashes of the loaded tensors. N11's outputs hold no separate tail or head hash, so these two are a record of what was submitted, not a comparison.
5. b_lite.units.fold_k.encoder_lr equals units.<unit>.inner.chosen_lr of the N11 summary.
6. b_lite.amp_fp16 true, equal to amp_fp16_in_training of every fold. b_lite.test_overrides false.
7. b_lite.trunk_passes_per_study between 1.0 and 2.0 (1.0 + the share of studies with unresolved laterality).
8. series_tiers (all three N10 modes): per plane, how many test studies got a fluid-sensitive fat-suppressed series, a fluid-sensitive one, any other series, or none. Rounded counts. Compare the shares with training (N0). null = the count itself failed; the submission is not affected.
9. Against agg_n10_tails_check.json of the passed check run for the same encoder (fold 0): b_lite.trunk_sha16 equals trunk_sha16; bundle_commit is equal; and b_lite.units.fold_0's stage_a_stats_sha16, weights_file_sha16, tail_sha16 and head_sha16 equal the check's fields of the same names. Equal = the submission ran the trunk, the knee-n6a statistics, the tail and the head that the check replayed.

## Learning-rate extension N11x (iteration 3; it3_lr_extension_r3)

N11x extends the encoder learning-rate search of DINOv2 B-lite by two grid points, 3.2e-4 and 1.28e-3 (code knee.nb.n11: run_lrx, select_lrx; tests tests/test_n11x.py). Every number is "added after the results" (R10). The text of it3_lr_extension_r3.md must be in the plan before the first run (R3). Every score here is an inner validation score. No test-fold file is read or written.

Order: 1. knee-n11x-smoke (and knee-n11x-smoke-select). 2. knee-n11x-inner. 3. knee-n11x-select. 4. knee-n11x-refine, only if the selection and the quota rule allow it.

| Notebook | Accelerator | Internet | Inputs |
|---|---|---|---|
| knee-n11x-smoke, knee-n11x-inner, knee-n11x-refine | GPU T4 x2 | On | Outputs of knee-n0, **knee-n3-r**, knee-n4-sagittal, knee-n4-coronal, knee-n4-axial, knee-n5 and knee-n6a: the same as knee-n11-a. NOT knee-n11-a or knee-n11-b (R9: iteration 3's numbers are constants in the code, EXT_REF) |
| knee-n11x-smoke-select | None (CPU) | On | Outputs of knee-n11x-smoke, knee-n11-a and knee-n11-b. Nothing else |
| knee-n11x-select | None (CPU) | On | Outputs of knee-n11x-inner, knee-n11-a and knee-n11-b (and knee-n11x-refine once it ran). Nothing else |

Create the GPU notebooks as forks of knee-n11-a and keep "Pin to original environment": the selection compares the environment with iteration 3's field by field. Cell 1: COMMIT = the reviewed commit of branch it3-lrx named in the planning log (the same in all N11x notebooks).

budget_h has no default. Use the declaration's value (section 9): smoke 0.5; inner session (option A) the smaller of 5.8 and (quota minus 0.6); option C and the refinement point the smaller of 3.4 and (quota minus 0.4). Read the quota before every session start.

### 1. knee-n11x-smoke

```python
from knee.nb import n11; n11.run_lrx("smoke", smoke=True, budget_h=0.5, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})
```

Expected time: about 0.4 h. It runs both processes on all five folds with fits of 12 optimiser steps, validation on the first 32 validation studies, one forced invalid run (fold 2 of the first process, twelfth 2), the replay without a stop flag, the end-of-session write, and the selection on its own output. The start checks are the real ones. No score of it is read.

What to read (summary block and agg_n11x_smoke.json):
1. STEP n11x_smoke: OK. status "ok (smoke: no score is meaningful)". smoke_passed true.
2. smoke_checks, the declaration's pass criteria as booleans: c1 both processes list five folds (the page's "Successful" is read by eye); c2 the start check passed in every fold of both processes; c3 the replay stored three gaps; c4 the forced run is invalid with its twelfth and reason; c5 every other run has 12 records and a finite end score; c6 micro-batches of 2 studies; c7 the selection ran in smoke mode.
3. Skipped optimiser steps do not invalidate a smoke run and do not fail c5 (a smoke fit has 12 steps, so one skipped step is already 0.083): they are reported in smoke_skipped_step_share. The inner session keeps the 0.05 rule. A smoke run that is invalid by a non-finite loss at 1.28e-3 only is information about that rate, not a code fault: read overview.0.00128.
3b. c7_environment_equals_iteration3 is null here (nothing of iteration 3 is attached). Run knee-n11x-smoke-select and read environment.differences: it must be an empty list.

```python
from knee.nb import n11; n11.select_lrx(smoke=True)
```

4. start_check: per rate and fold, abs_difference at most 0.0005. These are real numbers: a fail here is a fail of the inner session.

If it does not pass: fix, test locally, one new smoke run. After the second smoke: start check failing in one or two folds = the inner session runs and those folds keep 8e-5 for that rate; in three or more = no inner session; environment still different = the inner session runs, labelled "across environments", and no fold can move. No third smoke run.

### 2. knee-n11x-inner

```python
from knee.nb import n11; n11.run_lrx("inner", budget_h=5.8, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})
```

Expected time: 5.1 h planned, 5.6 h bound (up to 0.4 h more if the early-stop rule fires). The step refuses to start, before the cache copy, without two GPUs. GPU 0 runs 3.2e-4, GPU 1 runs 1.28e-3, each on folds 0-4. The watchdog stops the step at 1.1 x budget_h.

Option C (quota from 3.8 to 6.2 h): `n11.run_lrx("inner", rates=(3.2e-4,), split_folds=True, budget_h=<the smaller of 3.4 and quota minus 0.4>, expect=...)`.

Inside the session, in this order:
1. Every process, every fold: the head refit and the start check (the refit head's validation score against iteration 3's, tolerance 0.0005). Nothing is trained before this.
2. The second process: the replay (fold 0 at 8e-5 to the quarter mark).
3. The runs, one twelfth at a time.

What to read (summary block, agg_n11x_inner.json, agg_n11x_inner_0.00032.json, agg_n11x_inner_0.00128.json):
1. STEP n11x_inner: OK, whatever the status. The status says what happened.
2. start_check.<lr key>.fold_k: val_macro_auc, reference, abs_difference, passed.
3. replay: val_macro_auc, reference_val_macro_auc, max_abs_gap, passed, above_half_the_band.
4. overview.<lr key>.fold_k: status, valid, val_macro_auc (end of schedule), quarter_val_macro_auc, best (twelfth k and its score: reported, never used), invalid (twelfth k and reason), skipped_step_share, train_seconds.
5. Curves: units.<lr key>.dinov2_vits14_f<k>.ext.run.curve (train_loss, val_loss, val_macro_auc, 12 points) and per_twelfth (skipped share, non-finite flag, train seconds).
6. Do not compute anything from the private files (inner_start.csv, inner_b_lite_<lr key>.csv).

Stop rules (the code applies them; the status names them):
1. Three or more failed start checks in a process: the session stop flag (private/n11x_STOP) is written and nothing trains. status "stopped: the start check failed in three or more folds". The cause is looked for once; one new session may start under the declaration's section 9.
2. One or two failed start checks: the runs go on; that fold cannot pick that rate, and the rate has no five-fold mean.
3. Replay gap above 0.005, or the replay process ending before its replay record exists (flag reason "replay: no record"): the stop flag is written and every process stops at its next twelfth. status "inconclusive: reused and new scores cannot be compared at 0.005". No fold moves, no refinement point, no further GPU run.
4. An invalid run (a non-finite training or validation loss, a skipped-step share above 0.05, a non-finite end score) stops at that twelfth and is reported. Only that fold and rate are affected.
5. Early-stop rule: 1.28e-3 invalid in folds 0 and 1, both at or before twelfth 3: that process runs 6.4e-4 on all five folds (agg_n11x_inner_0.00064.json; swapped in the summary).
6. Soft stop at budget_h: status "partial", out_of_time true, everything written. runs_missing lists what did not finish. The selection accepts a partial session.

### 3. knee-n11x-select (after the inner session has ended)

```python
from knee.nb import n11; n11.select_lrx()
```

What to read (summary block = agg_n11x_select.json, the signed selection record):
1. STEP n11x_select: OK. status "selected". Keep sha256 and written_at_utc: the record exists before any final fit (R5).
2. status "inconclusive" (inconclusive_because names why: replay_gap_above_tolerance, no_replay_record, across_environments, environment_not_compared, reference_mismatch) or "stopped_at_start_check": all five picks are 8e-5 and refinement.may_run is false.
3. iteration3.constants_equal_the_states true. False = a constant in EXT_REF differs from iteration 3's state: stop and report.
4. picks: one rate per fold (Rule S). folds.fold_k: scores of all rates, runs (valid, eligible, why_not_eligible), steps, best_twelfth.
5. rule_d: means, c1, c2, cur, r_star, line, words (words_as_shown carries the label after a stop), words_swap (the sentence every line adds if 6.4e-4 replaced 1.28e-3), reasons, compare.convention (how d is computed), schedule_sentence, mean_curve. Rule D describes; it never overrides a pick.
6. missing: the runs that did not finish. refinement: r_star, neighbour, rate, may_run.
7. graphs.g1 and graphs.g2: the data of the two graphs. intervals: descriptive only.

### 4. knee-n11x-refine (only if refinement.may_run is true, no fold moved, quota at least 3.7 h, start by Wed 17:00)

```python
from knee.nb import n11; n11.run_lrx("refine", rates=(1.6e-4,), split_folds=True, replay=False, budget_h=3.4, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})
```

Write refinement.rate of the selection record into rates. Then rerun the selection with knee-n11x-refine attached: `n11.select_lrx(describe=("refine",))`. A refinement point is never eligible: the picks do not change.

Rules for reading N11x:
1. A fold's pick is the highest of several scores on one site. Its validation score is never reported as a performance number.
2. Nothing of N11x is attached to N7 or N9 (R9).

Delivery D2 (the three `TODO D2` lines of delivery D1 are sections 5 to 8 below; code knee.nb.n11: run_final_lrx, dev_lrx; knee.nb.n10: tails="n11x"; tests tests/test_n11x_final.py, tests/test_n10_tails.py). Cell 1 of sections 5 and 6: COMMIT = the reviewed commit that holds delivery D2 (named in the planning log). The inner session and the final fits run at two commits; the loop's files are byte-equal to 095c91f at both (tests/test_n11x.py).

Order after the selection: 5. knee-n11x-final, only if a fold moved and the gates hold. 6. knee-n11x-dev. 7. knee-n10-tails-check-x. 8. knee-n10-blite-x (the commit run; the submission itself waits for Ablay's OK, which D25 does not cover).

| Notebook | Accelerator | Internet | Inputs |
|---|---|---|---|
| knee-n11x-final | GPU T4 x2 | On | Outputs of knee-n0, **knee-n3-r**, knee-n4-sagittal, knee-n4-coronal, knee-n4-axial, knee-n5, knee-n6a and knee-n11x-select. NOT knee-n11-a or knee-n11-b (R9) |
| knee-n11x-dev | None (CPU) | On | Outputs of knee-n11x-final, knee-n11x-select, knee-n11-a, knee-n11-b and knee-n11-decide. Nothing else |
| knee-n10-tails-check-x | GPU T4 x2 | **Off** | The bundle dataset (rebuilt, see section 7); outputs of knee-n0, knee-n3-r, knee-n4-sagittal, knee-n4-coronal, knee-n4-axial, knee-n6a, knee-n11x-final and knee-n11x-select. No competition data. NOT knee-n11-a or knee-n11-b |
| knee-n10-blite-x | GPU T4 x2 | **Off** | Competition; the bundle dataset (the same version); outputs of knee-n0, knee-n6a, knee-n11x-final and knee-n11x-select. NOT knee-n11-a or knee-n11-b |

Why knee-n11x-final attaches nothing of iteration 3: frozen0 is computed again there (check b needs only knee-n6a's saved predictions), and no head-only seed is fitted. Iteration 3's frozen0 and head-only predictions are read only by the CPU step knee-n11x-dev.

Attach the version of knee-n11x-select whose sha256 was recorded before the final fits. Every final fit stores that sha256; a later version of the record (for example after `select_lrx(describe=("refine",))`) is refused by knee-n11x-dev and by N10.

### 5. knee-n11x-final (only if any_fold_moves is true; start by Wed 19:00)

Gates, read first (declaration, sections 6 and 9):
1. The selection record has status "selected" and any_fold_moves true. No fold moved = nothing is refit; skip sections 5 to 8.
2. Read the quota. At least 4.2 h left: all five folds. Under 4.2 h: only the moved folds (the fallback call below), and only if the quota left is at least that call's bound plus 0.5 h (1.8 h for one or two folds, 3.1 h for three or four); then no S4 model is built.
3. Create the notebook as a fork of knee-n11-a ("Pin to original environment").
4. Speed, read before the start: train_seconds of the finished inner runs in agg_n11x_inner.json (overview.<lr key>.fold_k). Plan: about 2,020 s for fold 0 and about 3,100 s for the other folds. More than 10% above the plan = the last fit on GPU 0 (folds 0, 1, 4) is at risk of the soft stop (slack about 16%). The code already fits the moved folds first (the moved-folds-first order), so the fit at risk is a fold that keeps 8e-5, if GPU 0 has one. Say the risk in the log before the start.

```python
from knee.nb import n11; n11.run_final_lrx(budget_h=BUDGET_H, expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"})
```

BUDGET_H is not defined on purpose: replace it by a number. No default exists anywhere; the cell fails at once if it is left.
1. Five folds: budget_h = the smaller of 3.8 and (quota minus 0.4).
2. Fallback: add `folds=(<the moved folds>)` before budget_h. Three or four folds: the smaller of 2.7 and (quota minus 0.4). One or two: the smaller of 1.4 and (quota minus 0.4).
3. The code refuses a value above the bound plus 0.1 (3.8, 2.7, 1.4), a folds argument that is neither all five nor exactly the moved folds, and a session with two processes without two GPUs (before the cache copy).

Expected time: 3.3 h planned, 3.7 h bound (five folds). GPU 0: folds 0, 1, 4. GPU 1: folds 2, 3. Fewer folds alternate over the two GPUs. The watchdog stops the step at 1.1 x budget_h.

Inside a process, in this order:
1. frozen0 with check (b) for each of its folds. A failed check: the run is invalid, STEP n11x_final: FAILED, no fit is kept (as in iteration 3; the other GPU may have trained up to two twelfths by then).
2. Per fold, the moved folds of the process first, then the folds that keep 8e-5: the pick is written to run.json (n11_chosen_encoder_lr); then the b-lite fit from Stage A's head on all training studies, seed 0, one twelfth at a time; then pred_b_lite.csv and b_lite_seed_0.pt. No head-only seed. No test-fold metric.

What to read (summary block and agg_n11x_final.json):
1. STEP n11x_final: OK. status "ok" = every fit finished and is valid. "partial" = read units_missing, fits_invalid, fits_unfinished, out_of_time, errors.
2. selection.sha256 equals the recorded selection's. overview.<unit>.pick equals the record's pick for the fold.
3. overview.<unit>: frozen0_valid true, check_b_abs_difference at most 0.0005, status "finished", valid true, skipped_step_share, train_seconds, weights_file_sha16.
3b. units.<unit>.config.micro_batch_studies is 2 in all five final-fit units (as in iteration 3 and in D1's smoke run). If it is 1: the fit is valid, but report it beside the same-seed noise; knee-n11x-dev then writes micro_batch.label "micro-batch differs from iteration 3".
4. A fit is valid with a finite training loss at every twelfth and a skipped-step share of at most 0.05 (units.<unit>.final.b_lite.seed_0.per_twelfth). An invalid fit stops at that twelfth and writes no prediction and no weights. An invalid or unfinished fit is reported and not repeated, also not at another rate.
5. refit_model.complete true = all five fits finished and valid. False = per-fold development numbers only, no section 7 or 8.
6. Do not compute anything from the private files (pred_frozen0.csv, pred_b_lite.csv).

### 6. knee-n11x-dev (after the final fits)

```python
from knee.nb import n11; n11.dev_lrx()
```

Expected time: 3-8 min. It refuses to run without the keep record (sha256 0a8cce4c..., checked against the constant in the code), without the signed selection record, after an edited record, when a prediction file is not the one its record or state lists, and when a final fit ran under another selection record or rate.

What to read (summary block and agg_n11x_dev.json):
1. STEP n11x_dev: OK. decision.sha256 = 0a8cce4c... selection.sha256 = the recorded selection's.
2. label on every block: "development: these studies were test studies in iteration 1; likely optimistic; rate chosen after iteration 3's results were read". Grey, never in a title, never a gain over iteration 1, never on slide 12 (R6).
3. status "ok": model holds frozen0, head_only_seed_0, b_lite (the refit model) and b_lite_iteration3 (8e-5 in every fold), each with mean_of_fold_macros, fold_stratified_macro_auc and fold_stratified_interval; b_lite_minus_frozen0, b_lite_minus_head_only and b_lite_minus_iteration3_b_lite with the paired study-bootstrap interval. Check: frozen0, head_only_seed_0 and b_lite_iteration3 equal agg_n11_dev.json (the same files, folds, draws and seed). A gap = stop and report.
4. status "ok (per-fold numbers only: the refit model is incomplete)": fewer than five refit folds, or an invalid or unfinished one. model.status "per-fold numbers only", model.why. No five-fold number exists.
5. per_fold.fold_<k>: pick, moved, iteration3_lr, final_fit, each arm's macro AUC in that fold and the three differences with an interval. null = a finding lacks a class in that fold.
6. same_seed_noise and per_fold.fold_<k>.same_seed_noise_of_a_final_fit: for each fold refit at the rate iteration 3 used (8e-5), the gap between the two final fits: macro AUC difference and the largest and mean probability gap. Say it as "same-seed noise of a final fit" (development, labelled). environment_label "across environments" = the two fits ran in different environments. micro_batch (and per_fold.fold_<k>.config): each refit unit's micro-batch against iteration 3's unit of the same fold; label "micro-batch differs from iteration 3" on a mismatch.
7. report_final_model.model: "refit" only if all five final fits are finished and valid; else "iteration3_8e-5" with the sentence for the report. This is fixed before any second S4 score exists.

### 7. knee-n10-tails-check-x (only if refit_model.complete; at least 1.2 h of quota; before Wed 22:30)

The offline bundle must be rebuilt first, at the commit that holds delivery D2 (knee-bundle with that COMMIT, then a new version of the bundle dataset). The bundle's wheel is what runs: an older bundle has no tails="n11x". bundle_commit in the summary shows which one ran.

```python
from knee.nb import n10; n10.check_tails("dinov2_vits14", fold=None, n=200, tails="n11x")
```

fold=None = the lowest fold that moved in the selection record (a model at a new rate). Another fold: fold=k. Offline cell 1. Expected time: 5-10 min. Pass: agg_n10_tails_check.json has passed true (max_abs_probability_gap at most 0.01, mean_abs_probability_gap at most 0.001), studies 200, amp_fp16 true, source "n11x", selection_sha256 = the recorded selection's, encoder_lr = that fold's pick. Keep trunk_sha16. A fail = stop and report.

### 8. knee-n10-blite-x (the commit run of the submission notebook; Ablay presses Submit)

```python
from knee.nb import n10; n10.run("dinov2_vits14", tails="n11x")
```

Offline cell 1, the same bundle version as section 7, the same version of knee-n6a that knee-n11x-final had attached. Expected time: 3-6 h on the hidden test.

The step refuses to run (STEP n10: FAILED, before any decoding, no submission.csv) unless all of these hold: the signed selection record is attached (status "selected", intact sha256); exactly folds 0-4 of private/n11x_final; every unit's state holds that record's sha256; every final fit is finished and valid, with finite test-fold predictions and frozen0 valid; every fit ran at the record's pick for its fold; each b_lite_seed_0.pt is the file its state records; one known commit; one schedule; no test overrides; knee-n6a's config of the fold equals the logged one.

Check in agg_n10.json before asking Ablay: the list of the card "N10 with N11's B-lite models" (items 1 to 9), with these changes. Item 2: b_lite.n11_commit = the commit of knee-n11x-final. Item 3: weights_file_sha16 against overview.<unit>.weights_file_sha16 of agg_n11x_final.json. Item 5: b_lite.units.fold_k.encoder_lr = the selection record's pick for fold k. Item 9: against the check of section 7, for the fold it replayed. New: b_lite.source "n11x" and b_lite.selection_sha256 = the recorded selection's.

Reading the second S4 score: against 0.863 as "higher", "lower" or "about the same" (under 0.01). No choice is made from it (R7).

### Graphs G1 and G2 (scripts/plot_lrx.py; matplotlib; aggregates only)

Download the aggregate files (never a private file) and run on the Mac:

```
python scripts/plot_lrx.py --it3 agg_n11_a.json agg_n11_b.json --inner agg_n11x_inner.json --select agg_n11x_select.json --out graphs
```

Add `--refine agg_n11x_refine.json` if the refinement point ran. Written: g1_steps_to_the_optimum.png, g2_fold0.png, g2_all_folds.png. G1: quarter mark, end of schedule and (dashed, "reported, not used") the best twelfth of the mean curve; thin lines per fold, the thick mean only over rates with a mean by Rule D; filled markers = each fold's pick; crosses = invalid runs. G2: training and validation loss per rate for fold 0's pick and its grid neighbours, the validation AUC per twelfth under each panel.
