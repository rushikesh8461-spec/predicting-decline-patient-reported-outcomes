"""
Merge breast-cancer PRO workbooks into a clean CSV with a transparent audit trail.

Folder layout:
  raw/: original Excel workbooks
  data/: this script, merged CSV, surgery mismatch audit CSV

Run:
  python data/merge_breast_pro.py

Install:
  pip install pandas openpyxl numpy

Caution:
  breast_cppo.xlsx has no patient ID. The script aligns rows by order and checks
  follow-up and baseline scores. These checks support alignment but do not prove
  patient identity. Surgery conflicts are preserved and flagged, not guessed.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "raw"
OUT_FILE = SCRIPT_DIR / "breast_pro_clean.csv"
AUDIT_FILE = SCRIPT_DIR / "surgery_mismatch_audit.csv"

qlq_path = RAW_DIR / "breast_qlq_c30.xlsx"
cppo_path = RAW_DIR / "breast_cppo.xlsx"
scores_path = RAW_DIR / "breast_scores.xlsx"
for path in (qlq_path, cppo_path, scores_path):
    if not path.exists():
        raise FileNotFoundError(f"Missing source workbook: {path}")

qlq = pd.read_excel(qlq_path)
cppo = pd.read_excel(cppo_path)
scores = pd.read_excel(scores_path)

for df in (qlq, cppo, scores):
    df.columns = df.columns.str.strip()

qlq = qlq.rename(columns={
    "Nau_baseline": "NA_baseline",
    "Overall_EORTC_baseline": "Overall_baseline",
    "Overall_EORTC_end": "Overall_end",
    "Surgery": "Surgery_qlq",
})
cppo = cppo.rename(columns={
    "Surgery": "Surgery_clin",
    "EORTC_baseline": "Overall_baseline_clin",
    "Health_status_end": "HE_end_clin",
})

for df in (qlq, cppo):
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA})

# Format cleanup only. Do NOT assume BSC means BCS; preserve source category.
cppo["Children"] = cppo["Children"].replace({"> 2 children": "3"})
cppo["Children"] = pd.to_numeric(cppo["Children"], errors="raise")
cppo["Chemotherapy_regimen"] = cppo["Chemotherapy_regimen"].str.replace(
    "-based", "", regex=False
)

# Verify the scores workbook duplicates columns from the QLQ workbook.
pairs = {
    "HE_BASELINE": "HE_baseline", "PH_BASELINE": "PH_baseline",
    "RO_BASELINE": "RO_baseline", "EF_BASELINE": "EF_baseline",
    "SO_BASELINE": "SO_baseline", "COG_BASELINE": "COG_baseline",
    "HE6": "HE_end", "PH6": "PH_end", "RO6": "RO_end",
    "EF6": "EF_end", "SO6": "SO_end", "COG6": "COG_end",
}
for source_col, qlq_col in pairs.items():
    assert np.allclose(
        pd.to_numeric(scores[source_col], errors="coerce"),
        pd.to_numeric(qlq[qlq_col], errors="coerce"),
        equal_nan=True,
    ), f"scores workbook differs from QLQ workbook in {source_col}"

# The clinical workbook has no ID. Align by row order and validate score columns.
followup = qlq.loc[qlq["HE_end"].notna()].reset_index(drop=True)
assert len(followup) == len(cppo), (
    f"Cannot align rows: {len(followup)} QLQ follow-up rows vs {len(cppo)} clinical rows"
)
assert np.allclose(
    pd.to_numeric(followup["HE_end"], errors="coerce"),
    pd.to_numeric(cppo["HE_end_clin"], errors="coerce"),
    equal_nan=True,
), "Follow-up health scores do not line up; refusing to merge"

base_ok = (
    followup["Overall_baseline"].isna()
    | np.isclose(
        pd.to_numeric(followup["Overall_baseline"], errors="coerce"),
        pd.to_numeric(cppo["Overall_baseline_clin"], errors="coerce"),
        equal_nan=True,
    )
)
assert base_ok.all(), "Baseline overall scores do not line up; refusing to merge"

clin = cppo.drop(columns=["HE_end_clin"])
merged = pd.concat([followup, clin], axis=1)
merged["Overall_baseline"] = merged["Overall_baseline"].fillna(
    merged["Overall_baseline_clin"]
)

# Preserve original clinical label and provide a separate normalized column.
# BSC is treated as a likely spelling/code variant of BCS for comparison only;
# the original Surgery_clin column is not overwritten.
merged["Surgery_clin_normalized"] = merged["Surgery_clin"].replace({"BSC": "BCS"})
surgery_a = merged["Surgery_qlq"].astype("string").str.strip()
surgery_b = merged["Surgery_clin_normalized"].astype("string").str.strip()
both_missing = surgery_a.isna() & surgery_b.isna()
merged["Surgery_mismatch"] = ((surgery_a != surgery_b) & ~both_missing).fillna(True).astype(int)
merged["Surgery_review_status"] = np.where(
    merged["Surgery_mismatch"].eq(1), "REVIEW_REQUIRED", "MATCH"
)

audit = merged.loc[merged["Surgery_mismatch"].eq(1), [
    "ID", "Surgery_qlq", "Surgery_clin", "Surgery_clin_normalized", "HE_end",
    "Overall_baseline", "Overall_baseline_clin",
]].copy()
audit.insert(0, "merged_row_1based", audit.index + 1)
audit["note"] = (
    "Conflicting source labels; no shared patient ID or authoritative mapping "
    "provided. Do not infer a correction."
)
audit.to_csv(AUDIT_FILE, index=False)

merged["Overall_change"] = (
    pd.to_numeric(merged["Overall_end"], errors="coerce")
    - pd.to_numeric(merged["Overall_baseline"], errors="coerce")
)
merged["Decline_10"] = np.where(
    merged["Overall_change"].isna(), np.nan,
    (merged["Overall_change"] <= -10).astype(float),
)
merged["HE_change"] = (
    pd.to_numeric(merged["HE_end"], errors="coerce")
    - pd.to_numeric(merged["HE_baseline"], errors="coerce")
)
merged["HE_decline_10"] = np.where(
    merged["HE_change"].isna(), np.nan,
    (merged["HE_change"] <= -10).astype(float),
)

merged.to_csv(OUT_FILE, index=False)
print(f"Saved {OUT_FILE}: {merged.shape[0]} patients x {merged.shape[1]} columns")
print("Target Decline_10:", merged["Decline_10"].value_counts(dropna=False).to_dict())
print("Surgery conflicts flagged:", int(merged["Surgery_mismatch"].sum()))
print(f"Audit written to {AUDIT_FILE}")
