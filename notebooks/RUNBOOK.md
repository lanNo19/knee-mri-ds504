# Run cards (Kaggle, browser)

Every notebook runs on Ablay's account and has two cells. Claude creates it in
the browser, pastes the cells, sets the options, and starts a batch run with
Save Version -> Save & Run All. Claude reads only the cell's printed summary
and the agg_*.json / run.json files. Never open data previews, the input file
tree or the Logs tab.

## Cell 1 (every notebook)

```python
COMMIT = "<commit sha>"
!pip install -q "knee[n0] @ git+https://github.com/lanNo19/knee-mri-ds504@{COMMIT}"
import knee.safe_log as sl; sl.install()
```

## Session options (right-hand panel)

| Notebook | Accelerator | Internet | Inputs to add |
|---|---|---|---|
| knee-smoke | GPU T4 x2 | On | Competition; the three label-table datasets |
| knee-n0 | None (CPU) | On | Competition; the three label-table datasets |
| knee-n3 | None (CPU) | On | Output of knee-n0 |
| knee-n4-sagittal, knee-n4-coronal, knee-n4-axial | None (CPU) | On | Competition |

Label-table datasets: pilkwang/rsna-knee-llm-labels, stevenleehans/rsna-knee-llm-report-labels,
lixin73/rsna-knee-llm-report-labels-sol56. Record each dataset's version and licence in the tracker.

Smoke versions of real notebooks live in their own notebooks (for example knee-n0-smoke with
`limit=40`), so a downstream notebook never attaches a smoke output.

## Cell 2 and pass bands

| Notebook | Cell 2 | Expected time | Pass bands (from the printed summary and agg file) |
|---|---|---|---|
| knee-smoke | `from knee.nb import smoke; smoke.run()` | 2-5 min | checks: t4x2, safe_log and competition_attached all true; each label table attached with a recognised layout |
| knee-n0-smoke | `from knee.nb import n0; n0.run(limit=40)` | 5 min | STEP n0: OK |
| knee-n0 | `from knee.nb import n0; n0.run()` | 30-60 min | labels.gold.bug_check_passed true (macro AUC at least 0.80, expected about 0.89); at least 3 tables used, each with coverage_non_gold at least 0.98; groups.g_lang_groups at least 6; images.header_read_errors empty; every decode_test entry decoded = tried |
| knee-n3 | `from knee.nb import n3; n3.run()` | 2 min | s2.scheme group_5fold (else follow the plan's fold rules); gold_unassigned 0; s1.sizes_match_s2 true |
| knee-n4-* | `from knee.nb import n4; n4.run("sagittal", budget_h=8)` (coronal, axial likewise) | 15-90 min each | checks.no_failures, shape_ok and blank_under_1pct true; position_order_share at least 0.99 |

## If a run fails

The cell prints `STEP <name>: FAILED - <ErrorType> at <file>:<line> in <function>()`.
That line is all Claude needs for most fixes: Claude fixes the code, pushes, updates COMMIT,
and reruns. If a fix needs more detail, Ablay reruns with `quiet=False` in an interactive
session, reads the error himself, and describes it in words without values.
