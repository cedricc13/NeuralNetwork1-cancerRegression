import pandas as pd
import numpy as np



def analyze_dataset(data, target_col="TARGET_deathRate", max_corr_pairs=12):
    """
    Analyze the dataset for HW1 Step 1 and preview a preprocessing plan 
    (function taken from a previous project and adapted here)
    Allow me to analyze the dataset and answer the questions of STEP 1.

    Parameters
    ----------
    data : (str)
        CSV path 
    target_col : str
        Name of the label column (default: 'TARGET_deathRate').
    max_corr_pairs : int
        Max number of highly correlated feature pairs to print.

    Returns
    -------
    dict
        Small dictionary of key stats (also printed to stdout).
    """
    # --- Load ---------------------------------------------------------------
    df = pd.read_csv(data, encoding='latin1')
    source = str(data)

    n_rows, n_cols = df.shape
    print("═" * 78)
    print("DATASET OVERVIEW")
    print("═" * 78)
    print(f"Source: {source}")
    print(f"Samples (rows): {n_rows}")
    print(f"Columns: {n_cols}")

    # --- Types & basic structure -------------------------------------------
    dtype_counts = df.dtypes.value_counts()
    print("\nDtype counts:", dict(dtype_counts))

    has_target = target_col in df.columns
    if has_target:
        tgt_dtype = df[target_col].dtype
        problem = "regression" if np.issubdtype(tgt_dtype, np.number) else "classification"
    else:
        tgt_dtype = "—"
        problem = "unknown (target not found)"

    print(f'Problem type (inferred): {problem}')
    print(f'Label column (expected): "{target_col}" — found: {has_target}')

    # --- Features per sample -----------------------------------------------
    features_per_sample = n_cols - (1 if has_target else 0)
    print(f"Features per sample (excluding target if present): {features_per_sample}")

    # --- Missingness --------------------------------------------------------
    missing_counts = df.isna().sum()
    total_missing = int(missing_counts.sum())
    pct_missing = 100 * total_missing / (n_rows * n_cols) if n_rows and n_cols else 0.0
    print("\nMISSINGNESS")
    print(f"Total missing cells: {total_missing} ({pct_missing:.2f}%)")

    cols_with_na = missing_counts[missing_counts > 0].sort_values(ascending=False)
    if not cols_with_na.empty:
        top_na = cols_with_na.head(15).to_dict()
        print("Top columns with missing values (up to 15):")
        for k, v in top_na.items():
            print(f"  - {k}: {v}")
        rows_with_any_na = int(df.isna().any(axis=1).sum())
        print(f"Rows with ≥1 missing value: {rows_with_any_na} ({rows_with_any_na / n_rows * 100:.2f}%)")
    else:
        print("No missing values detected.")

    # --- Duplicates & constants --------------------------------------------
    dup_rows = int(df.duplicated().sum())
    nunique = df.nunique(dropna=False)
    constant_cols = nunique[nunique <= 1].index.tolist()
    print("\nDATA QUALITY")
    print(f"Duplicate rows: {dup_rows}")
    if constant_cols:
        preview = ", ".join(constant_cols[:12]) + (" ..." if len(constant_cols) > 12 else "")
        print(f"Constant columns ({len(constant_cols)}): {preview}")
    else:
        print("No constant columns detected.")

    # --- Numeric / Categorical split ---------------------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols + ([target_col] if has_target else [])]
    print("\nFEATURE TYPES")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical/other features: {len(cat_cols)}")

    if cat_cols:
        card = {c: int(df[c].nunique(dropna=True)) for c in cat_cols}
        top_card = sorted(card.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("Categorical cardinalities (top 10):")
        for c, k in top_card:
            print(f"  - {c}: {k} unique")

    # --- Global min/max & numeric ranges -----------------------------------
    print("\nRANGES")
    if numeric_cols:
        global_min = df[numeric_cols].min(numeric_only=True).min()
        global_max = df[numeric_cols].max(numeric_only=True).max()
        print(f"Global numeric min / max across dataset: {global_min} / {global_max}")

        desc = df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T
        rng = desc[["min", "1%", "5%", "50%", "95%", "99%", "max"]].round(6)
        print("\nNumeric range summary (first 12 columns):")
        print(rng.head(12).to_string())
    else:
        print("No numeric columns found.")

    # --- Target stats & histogram ------------------------------------------
    if has_target:
        tgt = df[target_col]
        print("\nTARGET STATS")
        print(
            f'{target_col} → count={tgt.notna().sum()}, '
            f'mean={tgt.mean():.6f}, std={tgt.std():.6f}, '
            f'min={tgt.min()}, max={tgt.max()}'
        )
        if np.issubdtype(tgt.dtype, np.number):
            hist, edges = np.histogram(tgt.dropna(), bins=10)
            print("Histogram (counts per bin):")
            for i, (l, r) in enumerate(zip(edges[:-1], edges[1:])):
                print(f"  bin {i+1:02d}: [{l:.4f}, {r:.4f}) -> {int(hist[i])}")

    # --- Correlations -------------------------------------------------------
    if has_target and np.issubdtype(df[target_col].dtype, np.number):
            # On calcule corr(feature, target) directement → Series, pas de DataFrame piégeux
            feats = [c for c in numeric_cols if c != target_col]
            if len(feats) == 0:
                print("\nNo numeric features (besides target) to correlate with.")
            else:
                corr = df[feats].corrwith(df[target_col])  # Series
                corr = corr.dropna()
                corr_sorted = corr.abs().sort_values(ascending=False)

                print("\nTop 12 |correlation| with target:")
                print(corr_sorted.head(12).round(4).to_string())
    else:
        print("\nTarget is missing or not numeric — skipping target correlation.")




    # --- Multicollinearity (highly correlated feature pairs) ----------------
    if len(numeric_cols) >= 2:
        corrmat = df[numeric_cols].corr(numeric_only=True).abs()
        upper = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(bool))
        pairs = [
            (i, j, upper.loc[i, j])
            for i in upper.index
            for j in upper.columns
            if pd.notna(upper.loc[i, j]) and upper.loc[i, j] > 0.95
        ]
        pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:max_corr_pairs]
        if pairs:
            print("\nHighly correlated feature pairs (|r| > 0.95):")
            for i, j, r in pairs:
                print(f"  {i} ~ {j}: r={r:.3f}")


    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "has_target": has_target,
        "numeric_cols": numeric_cols,
        "categorical_cols": cat_cols,
        "constant_cols": constant_cols,
        "duplicate_rows": dup_rows,
        "total_missing_cells": total_missing,
    }

analyze_dataset('../cancer_reg-1.csv')