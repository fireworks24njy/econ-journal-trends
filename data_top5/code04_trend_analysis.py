"""Reproducible exploratory trend analysis for the Top-5 journal classifications.

The script implements:
1. overall multinomial-logit likelihood-ratio tests;
2. field-specific linear logit trends with Benjamini-Hochberg correction;
3. permutation-corrected change-point and two-year low-window scans;
4. stratified multinomial bootstrap stability analysis;
5. sensitivity analysis across all classification scopes available in the input.

All stochastic procedures use fixed seeds. The source classification CSV is
read-only; every result is written to ``trend_analysis_outputs``.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import chi2, chi2_contingency, norm


DEFAULT_SEED = 20260810
DEFAULT_PERMUTATIONS = 10_000
DEFAULT_BOOTSTRAPS = 5_000
HIGH_SCORE = 0.020
HIGH_MARGIN = 0.003
BERT_HIGH_SIMILARITY = 0.35

FIELD_ORDER = [
    "Development Economics",
    "Economic History",
    "Finance",
    "Industrial Organization",
    "International Economics",
    "Labor Economics",
    "Macroeconomics",
    "Microeconomics",
    "Public Finance",
    "Miscellaneous & Methods",
]
UNCLASSIFIED_LABEL = "Unclassified"

KEY_LABOR = "Labor Economics"
KEY_PUBLIC = "Public Finance"
KEY_IO = "Industrial Organization"


@dataclass(frozen=True)
class ScopeData:
    name: str
    label_column: str
    frame: pd.DataFrame


def resolve_input(path: Path | None, script_dir: Path) -> Path:
    """Resolve an explicit path or find a code02/code03 result beside this script."""
    if path is not None:
        candidates = [path, script_dir / path.name]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        raise FileNotFoundError(f"Input CSV not found: {path}")

    filenames = [
        "Classified_Top5_Journals_Result.csv",
        "Classified_Top5_Journals_ModelScope.csv",
    ]
    for filename in filenames:
        candidate = script_dir / filename
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "No classified CSV found beside the script. Expected one of: "
        + ", ".join(filenames)
    )


def load_input(path: Path) -> pd.DataFrame:
    """Load one classified CSV and validate the common analysis fields."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = {"Year", "Title", "Predicted_Field"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    if df["Year"].isna().any():
        raise ValueError("Year contains missing or non-numeric values.")
    df["Year"] = df["Year"].astype(int)
    invalid_years = sorted(set(df["Year"]) - set(range(2020, 2026)))
    if invalid_years:
        raise ValueError(f"Year must be within 2020-2025; found: {invalid_years}")

    for column in ["Predicted_Field", "Raw_Predicted_Field"]:
        if column in df.columns:
            df[column] = df[column].fillna("").astype(str).str.strip()
            if (df[column] == "").any():
                raise ValueError(f"{column} contains empty labels.")
    return df


def prepare_scopes(
    df: pd.DataFrame,
) -> tuple[list[ScopeData], str, str, dict[str, str], pd.Series | None]:
    """Adapt either code02 TF-IDF output or code03 Sentence-BERT output."""
    allowed = set(FIELD_ORDER) | {UNCLASSIFIED_LABEL}
    unknown = sorted(set(df["Predicted_Field"]) - allowed)
    if unknown:
        raise ValueError(f"Unknown Predicted_Field labels: {unknown}")

    tfidf_columns = {
        "Raw_Predicted_Field", "Classification_Score", "Classification_Margin"
    }
    if tfidf_columns.issubset(df.columns):
        raw_unknown = sorted(set(df["Raw_Predicted_Field"]) - set(FIELD_ORDER))
        if raw_unknown:
            raise ValueError(f"Unknown Raw_Predicted_Field labels: {raw_unknown}")

        for column in ["Classification_Score", "Classification_Margin"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
            if df[column].isna().any():
                raise ValueError(f"{column} contains missing or non-numeric values.")

        final = df[df["Predicted_Field"].isin(FIELD_ORDER)].copy()
        if "Evidence_Band" in df.columns:
            high_mask = df["Evidence_Band"].fillna("").astype(str).str.casefold().eq("high")
            high_rule = "Evidence_Band equals High"
        else:
            high_mask = (
                (df["Classification_Score"] >= HIGH_SCORE)
                & (df["Classification_Margin"] >= HIGH_MARGIN)
            )
            high_rule = (
                f"Classification_Score >= {HIGH_SCORE:.3f} and "
                f"Classification_Margin >= {HIGH_MARGIN:.3f}"
            )
        scopes = [
            ScopeData("Final classification", "Predicted_Field", final),
            ScopeData("Raw classification", "Raw_Predicted_Field", df.copy()),
            ScopeData(
                "High-confidence classification",
                "Raw_Predicted_Field",
                df[high_mask].copy(),
            ),
        ]
        notes = {
            "Final classification": (
                f"Uses Predicted_Field and excludes {len(df) - len(final):,} "
                "records labelled Unclassified."
            ),
            "Raw classification": "Uses the best-scoring field for every record.",
            "High-confidence classification": f"Uses raw labels where {high_rule}.",
        }
        return scopes, "Final classification", "TF-IDF keyword classifier", notes, high_mask

    if "Similarity" in df.columns:
        df["Similarity"] = pd.to_numeric(df["Similarity"], errors="coerce")
        if df["Similarity"].isna().any():
            raise ValueError("Similarity contains missing or non-numeric values.")
        high_mask = df["Similarity"] >= BERT_HIGH_SIMILARITY
        scopes = [ScopeData("Sentence-BERT classification", "Predicted_Field", df.copy())]
        if high_mask.any():
            scopes.append(ScopeData(
                "High-similarity classification",
                "Predicted_Field",
                df[high_mask].copy(),
            ))
        notes = {
            "Sentence-BERT classification": "Uses all Sentence-BERT predictions.",
            "High-similarity classification": (
                f"Keeps Similarity >= {BERT_HIGH_SIMILARITY:.2f}."
            ),
        }
        return (
            scopes,
            "Sentence-BERT classification",
            "Sentence-BERT semantic classifier",
            notes,
            high_mask,
        )

    final = df[df["Predicted_Field"].isin(FIELD_ORDER)].copy()
    return (
        [ScopeData("Final classification", "Predicted_Field", final)],
        "Final classification",
        "generic classified dataset",
        {"Final classification": "Uses recognized ten-field Predicted_Field labels."},
        None,
    )


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg false-discovery-rate adjustment."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0.0, 1.0)
    return out


def softmax_with_reference(theta: np.ndarray, x: np.ndarray, n_classes: int) -> np.ndarray:
    """Multinomial probabilities with the last category as reference."""
    beta = theta.reshape(x.shape[1], n_classes - 1)
    eta = np.column_stack([x @ beta, np.zeros(x.shape[0])])
    return np.exp(eta - logsumexp(eta, axis=1, keepdims=True))


def multinomial_year_test(counts: np.ndarray) -> dict[str, float | int | bool]:
    """Compare intercept-only and linear-year multinomial models."""
    n_years, n_classes = counts.shape
    if n_years < 4 or n_classes < 2:
        raise ValueError("Overall test requires at least four years and two fields.")
    if (counts.sum(axis=1) == 0).any() or (counts.sum(axis=0) == 0).any():
        raise ValueError("Every analyzed year and field must contain at least one record.")
    year_centered = np.arange(n_years, dtype=float)
    x = np.column_stack([np.ones(n_years), year_centered])
    totals = counts.sum(axis=1)
    grand = counts.sum(axis=0)
    grand_total = grand.sum()
    base_probs = np.clip(grand / grand_total, 1e-15, 1.0)
    ll_null = float(np.sum(grand * np.log(base_probs)))

    reference = grand[-1]
    initial = np.zeros((2, n_classes - 1), dtype=float)
    initial[0] = np.log(np.clip(grand[:-1], 1e-12, None) / max(reference, 1e-12))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        probs = softmax_with_reference(theta, x, n_classes)
        nll = -float(np.sum(counts * np.log(np.clip(probs, 1e-15, 1.0))))
        diff = totals[:, None] * probs[:, :-1] - counts[:, :-1]
        grad = (x.T @ diff).ravel()
        return nll, grad

    result = minimize(
        lambda t: objective(t)[0],
        initial.ravel(),
        jac=lambda t: objective(t)[1],
        method="L-BFGS-B",
        options={"maxiter": 10_000, "ftol": 1e-12, "gtol": 1e-9},
    )
    ll_year = -float(result.fun)
    statistic = max(0.0, 2.0 * (ll_year - ll_null))
    df = n_classes - 1
    p_value = float(chi2.sf(statistic, df))
    pearson_chi2, pearson_p, pearson_df, expected = chi2_contingency(counts)
    cramers_v = math.sqrt(pearson_chi2 / (grand_total * min(n_years - 1, n_classes - 1)))
    return {
        "N": int(grand_total),
        "LogLik_Null": ll_null,
        "LogLik_Year": ll_year,
        "LRT_Statistic": statistic,
        "LRT_df": df,
        "LRT_p": p_value,
        "McFadden_R2": 1.0 - ll_year / ll_null,
        "Pearson_Chi2": float(pearson_chi2),
        "Pearson_df": int(pearson_df),
        "Pearson_p": float(pearson_p),
        "Cramers_V": float(cramers_v),
        "Minimum_Expected_Frequency": float(expected.min()),
        "Converged": bool(result.success),
        "Maximum_Absolute_Gradient": float(np.max(np.abs(result.jac))),
    }


def fit_binomial_trend(successes: np.ndarray, totals: np.ndarray) -> dict[str, float]:
    """Grouped-binomial linear logit using Newton scoring."""
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    x = np.column_stack([np.ones(len(totals)), np.arange(len(totals), dtype=float)])
    overall = np.clip(successes.sum() / totals.sum(), 1e-8, 1 - 1e-8)
    beta = np.array([math.log(overall / (1 - overall)), 0.0])

    for _ in range(100):
        probs = expit(x @ beta)
        weights = np.clip(totals * probs * (1.0 - probs), 1e-10, None)
        information = x.T @ (weights[:, None] * x)
        score = x.T @ (successes - totals * probs)
        step = np.linalg.solve(information, score)
        beta_new = beta + step
        if np.max(np.abs(step)) < 1e-11:
            beta = beta_new
            break
        beta = beta_new

    probs = expit(x @ beta)
    weights = np.clip(totals * probs * (1.0 - probs), 1e-10, None)
    information = x.T @ (weights[:, None] * x)
    covariance = np.linalg.inv(information)
    slope = float(beta[1])
    slope_se = float(math.sqrt(covariance[1, 1]))
    z_value = slope / slope_se
    p_value = float(2.0 * norm.sf(abs(z_value)))
    return {
        "Slope": slope,
        "Slope_SE": slope_se,
        "OR_per_year": math.exp(slope),
        "OR_CI_Low": math.exp(slope - 1.96 * slope_se),
        "OR_CI_High": math.exp(slope + 1.96 * slope_se),
        "Wald_z": z_value,
        "p_value": p_value,
    }


def difference_z(success_a: np.ndarray, total_a: float, success_b: np.ndarray, total_b: float) -> np.ndarray:
    """Two-sample pooled z statistics for proportions: group B minus group A."""
    p_a = success_a / total_a
    p_b = success_b / total_b
    pooled = (success_a + success_b) / (total_a + total_b)
    se = np.sqrt(np.clip(pooled * (1.0 - pooled) * (1.0 / total_a + 1.0 / total_b), 1e-15, None))
    return (p_b - p_a) / se


def change_scan(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scan all breakpoints leaving at least two years on each side."""
    candidates = np.arange(2, counts.shape[0] - 1)  # 2, 3, 4 for six years
    totals = counts.sum(axis=1)
    stats = []
    differences = []
    for split in candidates:
        earlier = counts[:split].sum(axis=0)
        later = counts[split:].sum(axis=0)
        n_earlier = totals[:split].sum()
        n_later = totals[split:].sum()
        stats.append(difference_z(earlier, n_earlier, later, n_later))
        differences.append(later / n_later - earlier / n_earlier)
    stats_array = np.vstack(stats)
    differences_array = np.vstack(differences)
    best_rows = np.argmax(np.abs(stats_array), axis=0)
    cols = np.arange(counts.shape[1])
    return (
        candidates[best_rows],
        stats_array[best_rows, cols],
        differences_array[best_rows, cols],
        np.max(np.abs(stats_array), axis=0),
    )


def low_window_scan(counts: np.ndarray, width: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scan all contiguous windows for a lower inside proportion than outside."""
    starts = np.arange(0, counts.shape[0] - width + 1)
    totals = counts.sum(axis=1)
    grand = counts.sum(axis=0)
    grand_n = totals.sum()
    stats = []
    differences = []
    for start in starts:
        inside = counts[start : start + width].sum(axis=0)
        n_inside = totals[start : start + width].sum()
        outside = grand - inside
        n_outside = grand_n - n_inside
        # Positive statistic means the scanned window is lower than its complement.
        stats.append(difference_z(inside, n_inside, outside, n_outside))
        differences.append(outside / n_outside - inside / n_inside)
    stats_array = np.vstack(stats)
    differences_array = np.vstack(differences)
    best_rows = np.argmax(stats_array, axis=0)
    cols = np.arange(counts.shape[1])
    return (
        starts[best_rows],
        stats_array[best_rows, cols],
        differences_array[best_rows, cols],
        np.max(stats_array, axis=0),
    )


def permutation_scans(
    year_codes: np.ndarray,
    labels: np.ndarray,
    n_years: int,
    n_classes: int,
    observed_change: np.ndarray,
    observed_low: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Permutation p-values that repeat the full scan in every permutation."""
    rng = np.random.default_rng(seed)
    change_exceed = np.zeros(n_classes, dtype=int)
    low_exceed = np.zeros(n_classes, dtype=int)
    for _ in range(repetitions):
        shuffled = rng.permutation(labels)
        perm_counts = np.bincount(
            year_codes * n_classes + shuffled,
            minlength=n_years * n_classes,
        ).reshape(n_years, n_classes)
        perm_change = change_scan(perm_counts)[3]
        perm_low = low_window_scan(perm_counts)[3]
        change_exceed += perm_change >= (observed_change - 1e-12)
        low_exceed += perm_low >= (observed_low - 1e-12)
    return (
        (change_exceed + 1.0) / (repetitions + 1.0),
        (low_exceed + 1.0) / (repetitions + 1.0),
    )


def fixed_change_difference(counts: np.ndarray, split: int, field_index: int) -> float:
    totals = counts.sum(axis=1)
    earlier = counts[:split, field_index].sum() / totals[:split].sum()
    later = counts[split:, field_index].sum() / totals[split:].sum()
    return float(later - earlier)


def fixed_low_difference(counts: np.ndarray, start: int, field_index: int, width: int = 2) -> float:
    totals = counts.sum(axis=1)
    grand_n = totals.sum()
    inside_n = totals[start : start + width].sum()
    inside_count = counts[start : start + width, field_index].sum()
    outside_count = counts[:, field_index].sum() - inside_count
    outside_n = grand_n - inside_n
    return float(outside_count / outside_n - inside_count / inside_n)


def bootstrap_scope(
    counts: np.ndarray,
    fields: list[str],
    years: list[int],
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Year-stratified multinomial bootstrap for trends and scan stability."""
    rng = np.random.default_rng(seed)
    totals = counts.sum(axis=1).astype(int)
    probabilities = counts / totals[:, None]
    n_fields = len(fields)
    ors = np.empty((repetitions, n_fields), dtype=float)

    public_index = fields.index(KEY_PUBLIC)
    io_index = fields.index(KEY_IO)
    observed_public_split = int(change_scan(counts)[0][public_index])
    observed_io_start = int(low_window_scan(counts)[0][io_index])
    public_best_split = np.empty(repetitions, dtype=int)
    io_best_start = np.empty(repetitions, dtype=int)
    public_fixed_difference = np.empty(repetitions, dtype=float)
    io_fixed_difference = np.empty(repetitions, dtype=float)

    for rep in range(repetitions):
        boot_counts = np.vstack([
            rng.multinomial(int(total), prob)
            for total, prob in zip(totals, probabilities)
        ])
        for field_index in range(n_fields):
            ors[rep, field_index] = fit_binomial_trend(
                boot_counts[:, field_index], totals
            )["OR_per_year"]
        public_best_split[rep] = change_scan(boot_counts)[0][public_index]
        io_best_start[rep] = low_window_scan(boot_counts)[0][io_index]
        public_fixed_difference[rep] = fixed_change_difference(
            boot_counts, observed_public_split, public_index
        )
        io_fixed_difference[rep] = fixed_low_difference(
            boot_counts, observed_io_start, io_index
        )

    trend_rows = []
    for field_index, field in enumerate(fields):
        trend_rows.append({
            "Field": field,
            "Bootstrap_OR_Median": float(np.median(ors[:, field_index])),
            "Bootstrap_OR_CI_Low": float(np.quantile(ors[:, field_index], 0.025)),
            "Bootstrap_OR_CI_High": float(np.quantile(ors[:, field_index], 0.975)),
            "Bootstrap_P_OR_gt_1": float(np.mean(ors[:, field_index] > 1.0)),
        })

    public_rows = []
    for split in range(2, counts.shape[0] - 1):
        public_rows.append({
            "Record_Type": "Breakpoint selection",
            "Candidate": f"{years[0]}-{years[split - 1]} vs {years[split]}-{years[-1]}",
            "Selection_Frequency": float(np.mean(public_best_split == split)),
            "Is_Observed_Best": split == observed_public_split,
        })
    public_rows.append({
        "Record_Type": "Fixed-split effect",
        "Candidate": "Fixed observed split: later-minus-earlier difference (pp)",
        "Selection_Frequency": np.nan,
        "Is_Observed_Best": True,
        "Effect_Median_pp": float(np.median(public_fixed_difference) * 100),
        "Effect_CI_Low_pp": float(np.quantile(public_fixed_difference, 0.025) * 100),
        "Effect_CI_High_pp": float(np.quantile(public_fixed_difference, 0.975) * 100),
    })

    io_rows = []
    for start in range(0, counts.shape[0] - 1):
        io_rows.append({
            "Record_Type": "Low-window selection",
            "Candidate": f"{years[start]}-{years[start + 1]}",
            "Selection_Frequency": float(np.mean(io_best_start == start)),
            "Is_Observed_Best": start == observed_io_start,
        })
    io_rows.append({
        "Record_Type": "Fixed-window effect",
        "Candidate": "Fixed observed window: outside-minus-inside difference (pp)",
        "Selection_Frequency": np.nan,
        "Is_Observed_Best": True,
        "Effect_Median_pp": float(np.median(io_fixed_difference) * 100),
        "Effect_CI_Low_pp": float(np.quantile(io_fixed_difference, 0.025) * 100),
        "Effect_CI_High_pp": float(np.quantile(io_fixed_difference, 0.975) * 100),
    })
    return pd.DataFrame(trend_rows), pd.DataFrame(public_rows), pd.DataFrame(io_rows)


def build_counts(scope: ScopeData, years: list[int], fields: list[str]) -> np.ndarray:
    table = pd.crosstab(scope.frame["Year"], scope.frame[scope.label_column])
    return table.reindex(index=years, columns=fields, fill_value=0).to_numpy(dtype=int)


def make_scope_tables(
    scope: ScopeData,
    years: list[int],
    fields: list[str],
    repetitions_perm: int,
    repetitions_boot: int,
    seed: int,
) -> dict[str, pd.DataFrame | dict]:
    counts = build_counts(scope, years, fields)
    totals = counts.sum(axis=1)
    if len(scope.frame) == 0:
        raise ValueError(f"{scope.name} is empty.")
    if (totals == 0).any():
        missing_years = [year for year, total in zip(years, totals) if total == 0]
        raise ValueError(f"{scope.name} has no records for years: {missing_years}")
    if int(counts.sum()) != len(scope.frame):
        accepted = set(fields)
        invalid = sorted(set(scope.frame[scope.label_column]) - accepted)
        raise ValueError(
            f"{scope.name} contains labels outside the ten-field scheme: {invalid}"
        )
    if (counts.sum(axis=0) == 0).any():
        empty_fields = [field for field, total in zip(fields, counts.sum(axis=0)) if total == 0]
        raise ValueError(f"{scope.name} has empty fields: {empty_fields}")
    overall = multinomial_year_test(counts)

    annual_rows = []
    for year_index, year in enumerate(years):
        for field_index, field in enumerate(fields):
            annual_rows.append({
                "Scope": scope.name,
                "Year": year,
                "Field": field,
                "Count": int(counts[year_index, field_index]),
                "Year_Total": int(totals[year_index]),
                "Proportion": counts[year_index, field_index] / totals[year_index],
            })

    trend_rows = []
    for field_index, field in enumerate(fields):
        result = fit_binomial_trend(counts[:, field_index], totals)
        trend_rows.append({
            "Scope": scope.name,
            "Field": field,
            "Field_Count": int(counts[:, field_index].sum()),
            "N": int(totals.sum()),
            **result,
        })
    trend_table = pd.DataFrame(trend_rows)
    trend_table["BH_q"] = bh_adjust(trend_table["p_value"].to_numpy())

    change_split, change_z, change_diff, change_max = change_scan(counts)
    low_start, low_z, low_diff, low_max = low_window_scan(counts)
    year_codes = pd.Categorical(scope.frame["Year"], categories=years, ordered=True).codes
    label_codes = pd.Categorical(scope.frame[scope.label_column], categories=fields).codes
    if (year_codes < 0).any() or (label_codes < 0).any():
        raise ValueError(f"{scope.name} contains a year or field outside the analysis scheme.")
    perm_change_p, perm_low_p = permutation_scans(
        year_codes,
        label_codes,
        len(years),
        len(fields),
        change_max,
        low_max,
        repetitions_perm,
        seed,
    )

    change_table = pd.DataFrame({
        "Scope": scope.name,
        "Field": fields,
        "Best_Split_After": [years[int(split) - 1] for split in change_split],
        "Earlier_Period": [f"{years[0]}-{years[int(split) - 1]}" for split in change_split],
        "Later_Period": [f"{years[int(split)]}-{years[-1]}" for split in change_split],
        "Later_Minus_Earlier_pp": change_diff * 100,
        "Selected_Signed_z": change_z,
        "Max_Abs_z": change_max,
        "Permutation_p_Scan_Adjusted": perm_change_p,
    })
    change_table["BH_q_Across_Fields"] = bh_adjust(perm_change_p)

    low_table = pd.DataFrame({
        "Scope": scope.name,
        "Field": fields,
        "Best_Low_Window": [f"{years[int(start)]}-{years[int(start) + 1]}" for start in low_start],
        "Outside_Minus_Inside_pp": low_diff * 100,
        "Selected_z": low_z,
        "Max_Low_z": low_max,
        "Permutation_p_Scan_Adjusted": perm_low_p,
    })
    low_table["BH_q_Across_Fields"] = bh_adjust(perm_low_p)

    boot_trends, boot_public, boot_io = bootstrap_scope(
        counts, fields, years, repetitions_boot, seed + 100_000
    )
    boot_trends.insert(0, "Scope", scope.name)
    boot_public.insert(0, "Scope", scope.name)
    boot_io.insert(0, "Scope", scope.name)
    trend_table = trend_table.merge(boot_trends, on=["Scope", "Field"], how="left")

    overall["Scope"] = scope.name
    overall["Permutation_Repetitions"] = repetitions_perm
    overall["Bootstrap_Repetitions"] = repetitions_boot
    return {
        "counts": counts,
        "annual": pd.DataFrame(annual_rows),
        "overall": overall,
        "trends": trend_table,
        "changes": change_table,
        "lows": low_table,
        "boot_public": boot_public,
        "boot_io": boot_io,
    }


def key_sensitivity(
    trends: pd.DataFrame,
    changes: pd.DataFrame,
    lows: pd.DataFrame,
    boot_public: pd.DataFrame,
    boot_io: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for scope in trends["Scope"].unique():
        labor = trends[(trends["Scope"] == scope) & (trends["Field"] == KEY_LABOR)].iloc[0]
        public = changes[(changes["Scope"] == scope) & (changes["Field"] == KEY_PUBLIC)].iloc[0]
        io = lows[(lows["Scope"] == scope) & (lows["Field"] == KEY_IO)].iloc[0]

        pub_stability = boot_public[
            (boot_public["Scope"] == scope)
            & (boot_public["Candidate"] == f"{public['Earlier_Period']} vs {public['Later_Period']}")
        ]["Selection_Frequency"]
        io_stability = boot_io[
            (boot_io["Scope"] == scope)
            & (boot_io["Candidate"] == io["Best_Low_Window"])
        ]["Selection_Frequency"]

        rows.extend([
            {
                "Scope": scope,
                "Signal": "Labor Economics linear trend",
                "Effect": labor["OR_per_year"],
                "Effect_Unit": "odds ratio per year",
                "Raw_or_Permutation_p": labor["p_value"],
                "Multiple_Test_q": labor["BH_q"],
                "Bootstrap_Stability": labor["Bootstrap_P_OR_gt_1"],
                "Selected_Time_Structure": "linear 2020-2025",
            },
            {
                "Scope": scope,
                "Signal": "Public Finance change point",
                "Effect": public["Later_Minus_Earlier_pp"],
                "Effect_Unit": "later minus earlier, percentage points",
                "Raw_or_Permutation_p": public["Permutation_p_Scan_Adjusted"],
                "Multiple_Test_q": public["BH_q_Across_Fields"],
                "Bootstrap_Stability": float(pub_stability.iloc[0]) if len(pub_stability) else np.nan,
                "Selected_Time_Structure": f"split after {int(public['Best_Split_After'])}",
            },
            {
                "Scope": scope,
                "Signal": "Industrial Organization low window",
                "Effect": io["Outside_Minus_Inside_pp"],
                "Effect_Unit": "outside minus low window, percentage points",
                "Raw_or_Permutation_p": io["Permutation_p_Scan_Adjusted"],
                "Multiple_Test_q": io["BH_q_Across_Fields"],
                "Bootstrap_Stability": float(io_stability.iloc[0]) if len(io_stability) else np.nan,
                "Selected_Time_Structure": str(io["Best_Low_Window"]),
            },
        ])
    return pd.DataFrame(rows)


def create_key_trend_figure(
    annual: pd.DataFrame,
    primary_scope: str,
    output_path: Path,
) -> None:
    final = annual[annual["Scope"] == primary_scope]
    colors = {
        KEY_LABOR: "#355C7D",
        KEY_PUBLIC: "#6C8EAD",
        KEY_IO: "#A8BFD3",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), dpi=220, sharex=True)
    for ax, field in zip(axes, [KEY_LABOR, KEY_PUBLIC, KEY_IO]):
        block = final[final["Field"] == field].sort_values("Year")
        y = block["Proportion"].to_numpy() * 100
        ax.plot(block["Year"], y, color=colors[field], linewidth=2.4, marker="o", markersize=5)
        ax.fill_between(block["Year"], 0, y, color=colors[field], alpha=0.16)
        ax.set_title(field, fontsize=11, fontweight="bold")
        ax.set_xticks(block["Year"])
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
        for x_value, y_value in zip(block["Year"], y):
            ax.annotate(f"{y_value:.1f}%", (x_value, y_value), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=7.5, color="#263238")
    axes[0].set_ylabel("Share of publications (%)", fontsize=9)
    fig.suptitle("Selected Top-5 Economics Field Trends, 2020-2025", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def fmt_p(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def create_report(
    source_path: Path,
    output_path: Path,
    annual: pd.DataFrame,
    overall: pd.DataFrame,
    trends: pd.DataFrame,
    changes: pd.DataFrame,
    lows: pd.DataFrame,
    boot_public: pd.DataFrame,
    boot_io: pd.DataFrame,
    sensitivity: pd.DataFrame,
    permutations: int,
    bootstraps: int,
    seed: int,
    primary_scope: str,
    classifier_name: str,
    scope_notes: dict[str, str],
    has_journal: bool,
) -> None:
    final_annual = annual[annual["Scope"] == primary_scope]
    final_overall = overall[overall["Scope"] == primary_scope].iloc[0]
    final_labor = trends[(trends["Scope"] == primary_scope) & (trends["Field"] == KEY_LABOR)].iloc[0]
    final_public = changes[(changes["Scope"] == primary_scope) & (changes["Field"] == KEY_PUBLIC)].iloc[0]
    final_io = lows[(lows["Scope"] == primary_scope) & (lows["Field"] == KEY_IO)].iloc[0]

    def annual_pct(field: str) -> str:
        block = final_annual[final_annual["Field"] == field].sort_values("Year")
        return " → ".join(f"{int(row.Year)}: {row.Proportion * 100:.2f}%" for row in block.itertuples())

    public_stability_row = boot_public[
        (boot_public["Scope"] == primary_scope)
        & (boot_public["Candidate"] == f"{final_public['Earlier_Period']} vs {final_public['Later_Period']}")
    ].iloc[0]
    io_stability_row = boot_io[
        (boot_io["Scope"] == primary_scope)
        & (boot_io["Candidate"] == final_io["Best_Low_Window"])
    ].iloc[0]

    labor_rows = sensitivity[sensitivity["Signal"] == "Labor Economics linear trend"]
    labor_effects = "、".join(
        f"{row.Scope} {row.Effect:.3f}" for row in labor_rows.itertuples()
    )
    scope_lines = [
        f"- **{row.Scope}**：{int(row.N):,} 篇；{scope_notes.get(row.Scope, '')}"
        for row in overall.itertuples()
    ]
    journal_note = (
        "数据包含期刊字段，但当前模型将五种期刊合并，未控制期刊固定效应。"
        if has_journal
        else "输入不含期刊字段，无法分解期刊构成变化。"
    )
    overall_interpretation = (
        "整体构成检验达到 5% 显著性水平，但实际差异幅度仍应结合 Cramér's V 判断。"
        if min(final_overall["LRT_p"], final_overall["Pearson_p"]) < 0.05
        else "两项整体检验均未显示强年度构成差异，领域结构总体较稳定。"
    )
    labor_interpretation = (
        "劳动经济学线性趋势通过十领域 BH 校正。"
        if final_labor["BH_q"] < 0.05
        else "劳动经济学线性趋势未通过十领域 BH 校正，只宜视为探索性信号。"
    )
    public_interpretation = (
        "公共财政阶段差异通过扫描置换和十领域 BH 校正。"
        if final_public["BH_q_Across_Fields"] < 0.05
        else "公共财政阶段差异未通过十领域 BH 校正，不宜称为确定的结构突变。"
    )
    io_interpretation = (
        "产业组织低谷窗口通过扫描置换和十领域 BH 校正。"
        if final_io["BH_q_Across_Fields"] < 0.05
        else "产业组织低谷窗口未通过十领域 BH 校正，宜视为阶段波动。"
    )
    lines = [
        "# Top-5 经济学期刊领域趋势：改进后的探索性统计分析",
        "",
        "## 一、分析定位与数据",
        "",
        f"本分析使用 `{source_path.name}`，分类来源为 {classifier_name}。主口径“{primary_scope}”共 {int(final_overall['N']):,} 篇论文。",
        "",
        "可用分类口径：",
        "",
        *scope_lines,
        "",
        f"置换检验重复 {permutations:,} 次，Bootstrap 重复 {bootstraps:,} 次，随机种子为 {seed}。",
        "",
        "## 二、整体构成：仍然以稳定为主",
        "",
        f"最终分类的多项 Logit 整体似然比检验为：$LR({int(final_overall['LRT_df'])})={final_overall['LRT_Statistic']:.3f}$，$p={fmt_p(final_overall['LRT_p'])}$，McFadden $R^2={final_overall['McFadden_R2']:.4f}$。传统列联表结果为 $\\chi^2({int(final_overall['Pearson_df'])})={final_overall['Pearson_Chi2']:.3f}$，$p={fmt_p(final_overall['Pearson_p'])}$，Cramér's $V={final_overall['Cramers_V']:.3f}$。",
        "",
        overall_interpretation,
        "",
        "## 三、三条重点信号",
        "",
        "### 1. 劳动经济学：方向稳定，但多重校正后仍属于中等证据",
        "",
        annual_pct(KEY_LABOR),
        "",
        f"最终分类的年度优势比为 $OR={final_labor['OR_per_year']:.3f}$，95% CI $[{final_labor['OR_CI_Low']:.3f}, {final_labor['OR_CI_High']:.3f}]$，原始 $p={fmt_p(final_labor['p_value'])}$，十领域 BH 校正后 $q={fmt_p(final_labor['BH_q'])}$。Bootstrap 中年度优势比大于 1 的比例为 {final_labor['Bootstrap_P_OR_gt_1'] * 100:.1f}%，百分位区间为 $[{final_labor['Bootstrap_OR_CI_Low']:.3f}, {final_labor['Bootstrap_OR_CI_High']:.3f}]$。",
        "",
        f"各可用口径的效应估计为：{labor_effects}。是否稳健应结合方向、置信区间与 BH 校正结果共同判断。",
        "",
        f"**建议表述：** {labor_interpretation}",
        "",
        "### 2. 公共财政：阶段扫描",
        "",
        annual_pct(KEY_PUBLIC),
        "",
        f"在三个允许的断点中，最终分类的最大差异出现在 {final_public['Earlier_Period']} 与 {final_public['Later_Period']} 之间，后期高 {final_public['Later_Minus_Earlier_pp']:.2f} 个百分点。因为每次置换都重新搜索断点，扫描校正后的 $p={fmt_p(final_public['Permutation_p_Scan_Adjusted'])}$；再对十个领域做 BH 校正后 $q={fmt_p(final_public['BH_q_Across_Fields'])}$。Bootstrap 中同一断点再次被选中的频率为 {public_stability_row['Selection_Frequency'] * 100:.1f}%。",
        "",
        f"**建议表述：** {public_interpretation}",
        "",
        "### 3. 产业组织：两年低谷窗口扫描",
        "",
        annual_pct(KEY_IO),
        "",
        f"扫描全部五个连续两年窗口后，最终分类选择 {final_io['Best_Low_Window']} 为最低窗口，窗口外占比高 {final_io['Outside_Minus_Inside_pp']:.2f} 个百分点。全流程置换 $p={fmt_p(final_io['Permutation_p_Scan_Adjusted'])}$，十领域 BH 校正 $q={fmt_p(final_io['BH_q_Across_Fields'])}$。Bootstrap 中同一窗口再次被选中的频率为 {io_stability_row['Selection_Frequency'] * 100:.1f}%。",
        "",
        f"**建议表述：** {io_interpretation}",
        "",
        "## 四、方法为什么比原方案更严谨",
        "",
        "1. **构成约束**：多项 Logit 在一个模型中处理十个互斥领域，不再把十个比例当成彼此无关。",
        "2. **断点选择**：公共财政不是固定看图后选出的 2023 年，而是扫描所有保留两年端点的合理断点；置换时重复完整扫描。",
        "3. **低谷选择**：产业组织扫描所有连续两年窗口；置换零分布包含“挑选最深低谷”的过程。",
        "4. **多重检验**：线性趋势、断点扫描和低谷扫描分别对十个领域进行 BH 校正。",
        "5. **稳定性**：按年份分层的多项 Bootstrap 保持各年样本量和领域互斥关系，用于评估方向、断点和低谷窗口能否重复出现。",
        "6. **分类敏感性**：最终、原始、高置信三种口径并列报告，不把某一种阈值下的结果当成唯一答案。",
        "",
        "## 五、仍不能由方法消除的边界",
        "",
        "- 分类标签没有独立人工真值，因此推断对象首先是“预测标签的年度变化”；",
        f"- {journal_note}",
        "- 只有六个年度位置，复杂曲线和持久性拐点的识别能力有限；",
        "- Bootstrap 反映的是在当前经验分布附近的抽样稳定性，不能修复系统性分类偏差。",
        "",
        "## 六、可直接放入论文的总述",
        "",
        f"> 主口径的整体检验结果为：多项 Logit $p={fmt_p(final_overall['LRT_p'])}$，Pearson $p={fmt_p(final_overall['Pearson_p'])}$，Cramér's $V={final_overall['Cramers_V']:.3f}$。单领域结论应以 BH 校正结果和 Bootstrap 稳定性为准，不作因果解释。",
        "",
        "## 七、复现说明",
        "",
        "运行：",
        "",
        "```bash",
        "python code04_trend_analysis.py --input Classified_Top5_Journals_Result.csv",
        "```",
        "",
        "脚本会生成全部 CSV 结果表、JSON 元数据和趋势图。随机种子、阈值和重复次数均写入结果元数据。",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Classified CSV; auto-detected beside the script when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "trend_analysis_outputs",
    )
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--bootstraps", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.permutations < 1 or args.bootstraps < 1:
        raise ValueError("--permutations and --bootstraps must be positive integers.")

    input_path = resolve_input(args.input, script_dir)
    df = load_input(input_path)
    years = list(range(2020, 2026))
    fields = FIELD_ORDER.copy()
    scopes, primary_scope, classifier_name, scope_notes, high_mask = prepare_scopes(df)
    print(f"Input: {input_path}")
    print(f"Classifier: {classifier_name}")

    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scope_results = []
    valid_scopes = []
    for index, scope in enumerate(scopes):
        absent = [
            field for field in fields
            if not (scope.frame[scope.label_column] == field).any()
        ]
        if absent:
            print(f"Skipping {scope.name}: empty fields ({', '.join(absent)}).")
            continue
        print(f"Analyzing {scope.name} ({len(scope.frame):,} records)...")
        scope_results.append(make_scope_tables(
            scope,
            years,
            fields,
            args.permutations,
            args.bootstraps,
            args.seed + index * 10_000,
        ))
        valid_scopes.append(scope)
    if not scope_results:
        raise ValueError("No classification scope has enough data for analysis.")
    if primary_scope not in {scope.name for scope in valid_scopes}:
        raise ValueError(f"Primary scope could not be analyzed: {primary_scope}")

    annual = pd.concat([r["annual"] for r in scope_results], ignore_index=True)
    overall = pd.DataFrame([r["overall"] for r in scope_results])
    trends = pd.concat([r["trends"] for r in scope_results], ignore_index=True)
    changes = pd.concat([r["changes"] for r in scope_results], ignore_index=True)
    lows = pd.concat([r["lows"] for r in scope_results], ignore_index=True)
    boot_public = pd.concat([r["boot_public"] for r in scope_results], ignore_index=True)
    boot_io = pd.concat([r["boot_io"] for r in scope_results], ignore_index=True)
    sensitivity = key_sensitivity(trends, changes, lows, boot_public, boot_io)

    preferred_columns = [
        "Record_ID", "Year", "Journal", "Title", "Similarity",
        "Classification_Score", "Classification_Margin", "Raw_Predicted_Field",
        "Predicted_Field", "Evidence_Band", "Assignment_Basis", "Is_Ambiguous",
    ]
    analysis_data = df[[c for c in preferred_columns if c in df.columns]].copy()
    primary_indices = next(
        scope.frame.index for scope in valid_scopes if scope.name == primary_scope
    )
    analysis_data["Included_In_Primary_Scope"] = analysis_data.index.isin(primary_indices)
    if high_mask is not None:
        analysis_data["High_Confidence_or_Similarity"] = high_mask.to_numpy()

    tables = {
        "annual_proportions.csv": annual,
        "multinomial_overall_tests.csv": overall,
        "linear_trends.csv": trends,
        "change_point_scans.csv": changes,
        "low_window_scans.csv": lows,
        "bootstrap_public_breakpoints.csv": boot_public,
        "bootstrap_io_windows.csv": boot_io,
        "key_sensitivity.csv": sensitivity,
        "analysis_dataset.csv": analysis_data,
    }
    for filename, table in tables.items():
        table.to_csv(args.output_dir / filename, index=False, encoding="utf-8-sig")

    metadata = {
        "source_file": str(input_path),
        "classifier": classifier_name,
        "records_total": int(len(df)),
        "records_by_scope": {scope.name: int(len(scope.frame)) for scope in valid_scopes},
        "primary_scope": primary_scope,
        "scope_notes": {scope.name: scope_notes[scope.name] for scope in valid_scopes},
        "years": years,
        "fields": fields,
        "permutations": args.permutations,
        "bootstraps": args.bootstraps,
        "seed": args.seed,
        "permutation_design": "shuffle field labels across fixed year slots; repeat full scan",
        "bootstrap_design": "year-stratified multinomial resampling with fixed annual totals",
    }
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    create_key_trend_figure(
        annual, primary_scope, args.output_dir / "key_field_trends.png"
    )
    create_report(
        input_path,
        args.output_dir / "trend_analysis_report_zh.md",
        annual,
        overall,
        trends,
        changes,
        lows,
        boot_public,
        boot_io,
        sensitivity,
        args.permutations,
        args.bootstraps,
        args.seed,
        primary_scope,
        classifier_name,
        scope_notes,
        "Journal" in df.columns,
    )
    print(f"Saved analysis outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
