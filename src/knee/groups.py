"""Report language and site groups (plan, "Splits").

G-lang: report language; two languages are merged when they share a scanner
fingerprint seen in at most two languages (union-find over languages).
Why: a hospital writes in one language, and the merge catches a bilingual
hospital that uses an uncommon scanner. A fingerprint seen in three or more
languages is a common model, so it says nothing about the hospital.

Unsure language IDs (confidence below 0.80, or a report under 20 characters)
join the group that most often shares their scanner. They are trained on but
not scored. When no sure study shares their scanner, they are left out
entirely (not trained, not scored, counted), because a guessed group could
put one hospital on both sides of a fold. (The plan is silent on this case.)

G-vendor: G-lang group x manufacturer (the sensitivity grouping).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

MIN_CONF = 0.80
MIN_CHARS = 20


def detect_languages(reports: pd.Series) -> pd.DataFrame:
    """Top language and its confidence per report (lingua, all languages)."""
    from lingua import LanguageDetectorBuilder

    det = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    langs, confs, lens = [], [], []
    for txt in reports.fillna("").astype(str):
        lens.append(len(txt.strip()))
        if not txt.strip():
            langs.append(None)
            confs.append(0.0)
            continue
        vals = det.compute_language_confidence_values(txt)
        if not vals:
            langs.append(None)
            confs.append(0.0)
            continue
        top = vals[0]
        langs.append(top.language.name)
        confs.append(float(top.value))
    return pd.DataFrame({"language": langs, "lang_conf": confs, "report_len": lens}, index=reports.index)


class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # deterministic: the alphabetically smaller root wins
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra


def build_groups(studies: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """studies: index UID; columns language, lang_conf, report_len, fingerprint, manufacturer, has_report.

    Returns the table with g_lang, g_vendor, sure, scored and in_training,
    plus a summary dict (no per-study values).
    """
    s = studies.copy()
    s["sure"] = s["has_report"] & s["language"].notna() & (s["lang_conf"] >= MIN_CONF) & (s["report_len"] >= MIN_CHARS)
    sure = s[s["sure"]]
    langs = sorted(sure["language"].unique())
    uf = _UF(langs)
    merges = 0
    fp_langs = sure.dropna(subset=["fingerprint"]).groupby("fingerprint")["language"].agg(lambda x: sorted(set(x)))
    for fp, ls in fp_langs.items():
        if len(ls) == 2 and uf.find(ls[0]) != uf.find(ls[1]):
            uf.union(ls[0], ls[1])
            merges += 1
    group_of_lang = {lg: uf.find(lg) for lg in langs}
    s["g_lang"] = s["language"].map(group_of_lang).where(s["sure"])

    # Unsure studies: the group that most often shares their scanner.
    fp_group = (
        s[s["sure"]].dropna(subset=["fingerprint"]).groupby("fingerprint")["g_lang"]
        .agg(lambda x: Counter(x).most_common(1)[0][0])
    )
    unsure = s["has_report"] & ~s["sure"]
    guess = s.loc[unsure, "fingerprint"].map(fp_group)
    s.loc[unsure, "g_lang"] = guess
    s["scored"] = s["sure"]
    s["in_training"] = s["has_report"] & s["g_lang"].notna()
    man = s["manufacturer"].fillna("UNKNOWN").astype(str)
    s["g_vendor"] = np.where(s["g_lang"].notna(), s["g_lang"].astype(str) + "|" + man, None)
    sizes = s.loc[s["in_training"], "g_lang"].value_counts()
    summary = {
        "languages": int(len(langs)),
        "language_merges": int(merges),
        "g_lang_groups": int(sizes.shape[0]),
        "g_vendor_groups": int(s.loc[s["in_training"], "g_vendor"].nunique()),
        "largest_group_share": round(float(sizes.max() / sizes.sum()), 4) if len(sizes) else None,
        "unsure_studies": int(unsure.sum()),
        "unsure_assigned": int(guess.notna().sum()),
        "unsure_left_out": int(guess.isna().sum()),
        "no_report": int((~s["has_report"]).sum()),
    }
    return s, summary
