# Preprocessing Decisions

## Dataset Summary

- Rows: 307,511
- Columns: 122
- Numerical Features: 106
- Categorical Features: 16

---

## Missing Values

Several building-related features contain more than 60% missing values.

Examples:

- COMMONAREA_*
- NONLIVINGAPARTMENTS_*
- FLOORSMIN_*
- YEARS_BUILD_*

Decision:
- Do not drop these columns yet.
- Investigate feature importance before deciding.

---

## Duplicate Records

Duplicate Rows: 0

Decision:
- No duplicate removal required.

---

## Target Variable

Distribution:

- Class 0: 91.93%
- Class 1: 8.07%

Decision:

- Use StratifiedKFold during cross-validation.
- Evaluate with ROC-AUC, Recall and F1-score.
- Avoid using Accuracy as the primary metric.
---

# Executive Summary

## Dataset Overview

- Dataset: Home Credit Default Risk
- Rows: 307,511
- Columns: 122
- Numerical Features: 106
- Categorical Features: 16

---

## Data Quality

### Missing Values

Several building/property-related features contain over 60% missing values.

Examples:

- COMMONAREA_*
- FLOORSMIN_*
- YEARS_BUILD_*
- LIVINGAPARTMENTS_*
- NONLIVINGAPARTMENTS_*

Decision:

These features will not be dropped immediately.
Their predictive importance will be evaluated before removal.

---

### Duplicate Records

Duplicate Rows: 0

Decision:

No duplicate removal is required.

---

## Target Variable

Class Distribution

- Class 0 (No Default): 91.93%
- Class 1 (Default): 8.07%

Decision:

The dataset is highly imbalanced.

Model evaluation will prioritize:

- ROC-AUC
- Precision
- Recall
- F1-score

Cross-validation will use StratifiedKFold.

---

## Categorical Features

Binary Features: 4

Low Cardinality: 10

Medium Cardinality: 1

High Cardinality: 1

Decision:

Encoding strategy will be selected individually for each feature group instead of applying One-Hot Encoding universally.

---

## Next Phase

Implement a reusable preprocessing pipeline based on these findings.