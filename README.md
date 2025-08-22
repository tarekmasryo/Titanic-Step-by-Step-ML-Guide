# 🚢 Titanic Survival — A Data Story

> *Behind every survivor is a pattern — of class, gender, age, deck, and
> whether someone stood alone at the rail or with family at their side.*

In this notebook, we uncover the **drivers of survival on the Titanic**
and build a clean, reproducible pipeline that turns passenger records
into **defensible predictions**.

---

## 🚀 Project Highlights

- **Data Cleaning**
  - Handle missing values: `Embarked` (mode), `Fare` (median by `Pclass`),
    `Age` (median by `(Title, Sex, Pclass)` with global fallback).
  - Sanity checks for duplicates/outliers.

- **EDA**
  - Survival by **Sex** and **Pclass** (bar charts).
  - **Sex × Class** heatmap (mean survival).
  - Age distributions by outcome; Fare and Embarked effects.
  - Survival counts (balance check).

- **Feature Engineering (Leak-Safe)**
  - `Title` from `Name` (rare titles grouped).
  - `FamilySize = SibSp + Parch + 1`, and `IsAlone`.
  - `CabinDeck` from first letter of `Cabin` (unknown → `U`).
  - **Leak-safe encoding**: train-only mappings; unknowns in test → `-1`.

- **Modeling Arena**
  - **Logistic Regression** as a simple, interpretable baseline.
  - **Random Forest** as the main model (`class_weight="balanced"`).
  - Metrics @ 0.5: Accuracy, **F1**, ROC-AUC + confusion matrices.
  - **F1-tuned threshold** on validation, then reused for test.

- **Interpretability & Capacity**
  - RandomForest feature importances + **permutation importance**.
  - **Learning curve** (data sufficiency) & **validation curve** (`n_estimators`).

- **Reproducibility**
  - Fixed seed `RNG = 42` and a `FAST_MODE` switch for quick runs.
