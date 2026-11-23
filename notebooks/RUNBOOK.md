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
| knee-n9-b | None (CPU) | On | The same inputs as knee-n9 (knee-n0, knee-n3-r, knee-n5, knee-n6p, knee-n6a, knee-n7, knee-n8; knee-n6b once it has finished, then with knee-n7-b in place of knee-n7). A new notebook, so knee-n9's freeze file stays as it is |
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
| knee-n9-b | `from knee.nb import n9; n9.run(expect={"n0_studies": "371cbe04", "n3_folds": "670ddcc8", "n5_index": "4ccecd01"}, compare_with={"H1": "inconclusive", "H2": "inconclusive", "H3": "not supported"})` (add `hidden={...}` if S4 ran) | 20-35 min | as knee-n9; compare_with_previous.same_h1_h2_h3 true (false = stop and report both files); the compare_with strings are tonight's decisions.H1-H3 exactly as agg_results.json of knee-n9 wrote them |
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

Blocks added after the results critique (23 Nov, late; knee.nb.n9c). N7 is not rerun: each block reads only files N7 and Stage A already wrote. Each has descriptive true and a one-line scope. The summary block also shows same_device_gain, gap_like_another_seed, any_finding_auc and threshold_overshoot.

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
| refit_agreement.encoders.<encoder>.n7_seed0_vs_n7_seed1 | the yardstick: the same statistics between two seeds of N7 |
| refit_agreement.gap_like_another_seed | true = for both encoders the median and p99 of the first are at most 1.5 times the second's: the device gap behaves like one more seed |
| refit_agreement.fits | Delta1 of the seven fits (Stage A's decision fit and N7 seeds 0-5), each with its device |
| refit_agreement.delta1_low_high_all_fits, delta1_low_high_n7_fits, delta1_sd_n7_fits, sign_same_all_fits | range over the seven fits; range and SD over N7's six same-device fits; whether all seven have one sign |
| triage_secondary.any_finding_prevalence | share of evaluation studies with at least one finding |
| triage_secondary.models.<encoder>.any_finding_auc | AUC of the max-of-12 score against "any finding", pairs inside a test fold, pair-weighted |
| triage_secondary.models.<encoder>.specificity_interval | Wilson 95% interval of the triage specificity from exact counts; null when a cell it uses holds 1 to 4 studies |
| triage_secondary.models.<encoder>.threshold_overshoot | achieved test sensitivity minus the 0.95 target (positive = the inner threshold was too low for the new site) |
| largest_site_decomposition.group, fold | the largest site group and its S2 fold k |
| largest_site_decomposition.training_sets | size (rounded) and the group's share of the training set for a (S2 fold k), b (S1 fold k: the same size) and c (the other S1 folds, mean) |
| largest_site_decomposition.encoders.<encoder>.a_unseen | S2 macro AUC on the group's studies (its models never saw the group) |
| largest_site_decomposition.encoders.<encoder>.b_seen_equal_training_size | S1 macro AUC on the group's studies in S1 fold k (same training-set size and config as a, but the group is in training) |
| largest_site_decomposition.encoders.<encoder>.c_seen_larger_training_size | S1 macro AUC on the group's studies in the other S1 folds (larger training sets) |
| largest_site_decomposition.encoders.<encoder>.a_on_b_studies, a_on_c_studies | the S2 predictions scored on exactly b's and c's studies |
| largest_site_decomposition.encoders.<encoder>.unseen_site_cost_at_equal_training_size | b minus a_on_b_studies: same studies, size and config, so this is the cost of the site being unseen (an upper bound) |
| largest_site_decomposition.encoders.<encoder>.larger_training_set_seen_site_rough | c minus b: different studies and configs, so only a rough view of training-set size |
| largest_site_decomposition.caveats | the two limits of this view |
| best_epoch_spread.encoders.<encoder>.chosen_config | inner best epochs of each unit's best trial: lowest, median, p90, share at or below 6, share at the 30-epoch cap |
| best_epoch_spread.encoders.<encoder>.all_trials | the same over every completed trial |
| best_epoch_spread.encoders.<encoder>.refit_epochs, median_E | each unit's E, and their median |
| best_epoch_spread.encoders.<encoder>.lr_fraction_of_peak_at_median_E | learning rate at the end of epoch E as a fraction of its peak (from heads.schedule_factor); near 1 = the cosine had barely started |
| best_epoch_spread.encoders.<encoder>.warmup_share_of_training | the 1-epoch warm-up as a share of E |
| gold_uncertainty.macro_auc.<encoder> | gold macro AUC (estimate), its bootstrap standard error (se), and the fold_stratified version (pairs inside one fold model) |
| gold_uncertainty.delta1 | gold Delta1 with its paired bootstrap standard error |
| gold_uncertainty.studies, exact_count_file | the rounded gold count; the exact one is only in private/n9/gold_count.json (Ablay reads it) |
| h2_without_lookalikes | post hoc (post_hoc true): Delta2 again without every study in a look-alike pair above cosine 0.90; pairs, studies_removed, share_removed, delta2_registered, delta2_without_lookalikes, change |
| h2_without_lookalikes.pair_source, pairs_match_n7_probe | the pairs are found again in N9 with the pilot's code (N7 wrote counts only); true = their count equals N7's probe |
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

## If a run fails

The summary block shows `STEP <name>: FAILED - <ErrorType> at <file>:<line> in <function>()`.
That line is all Claude needs for most fixes: Claude fixes the code, runs the tests,
pushes, updates COMMIT, and reruns.

If a fix needs the details, Ablay (not Claude) reads them, in his own browser:
1. The failed version's private/<step>_error.log holds the step's hidden output and the full traceback.
2. Or, in an interactive session only he watches, he reruns cell 2 with `quiet=False` and then runs `sl.show_last()`.
3. He describes the problem to Claude in words, without values.
