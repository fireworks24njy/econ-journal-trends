# 用于做中英文趋势比较分析
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import chi2, chi2_contingency, norm


YEARS = list(range(2020, 2026))
FIELDS = [
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
SUBSTANTIVE_FIELDS = FIELDS[:-1]
CN_TO_EN = {
    "发展经济学": "Development Economics",
    "经济史": "Economic History",
    "金融学": "Finance",
    "产业组织": "Industrial Organization",
    "国际经济学": "International Economics",
    "劳动经济学": "Labor Economics",
    "宏观经济学": "Macroeconomics",
    "微观经济学": "Microeconomics",
    "公共财政": "Public Finance",
    "方法与杂项": "Miscellaneous & Methods",
}
FIELD_CN = {value: key for key, value in CN_TO_EN.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether field-composition trends differ between Chinese and English journals."
    )
    parser.add_argument(
        "--cn",
        type=Path,
        default=Path("Cleaned_Custom_Dataset_Classified.csv"),
    )
    parser.add_argument(
        "--en",
        type=Path,
        default=Path("Classified_Top5_Journals_Result.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cn_en_group_comparison_output"),
    )
    return parser.parse_args()


def read_csv_robust(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Unable to decode CSV: {path}")


def prepare_data(cn_path: Path, en_path: Path) -> pd.DataFrame:
    cn = read_csv_robust(cn_path)
    en = read_csv_robust(en_path)
    required = {
        "Year",
        "Predicted_Field",
        "Raw_Predicted_Field",
        "Classification_Score",
        "Classification_Margin",
    }
    for name, frame in (("Chinese", cn), ("English", en)):
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"{name} data missing columns: {sorted(missing)}")

    cn = cn.copy()
    en = en.copy()
    cn["Group"] = 1
    cn["Group_Name"] = "Chinese journals"
    en["Group"] = 0
    en["Group_Name"] = "English Top 5"

    for frame in (cn, en):
        frame["Final_Field"] = frame["Predicted_Field"].replace(CN_TO_EN)
        frame["Raw_Field"] = frame["Raw_Predicted_Field"].replace(CN_TO_EN)
        frame["High_Confidence"] = (
            (frame["Classification_Score"] >= 0.020)
            & (frame["Classification_Margin"] >= 0.003)
        )
        frame["t"] = frame["Year"] - 2020

    data = pd.concat([en, cn], ignore_index=True)
    data = data[data["Year"].isin(YEARS)].copy()
    for column in ("Final_Field", "Raw_Field"):
        invalid = sorted(set(data[column].dropna()) - set(FIELDS))
        if invalid:
            raise ValueError(f"Unexpected values in {column}: {invalid}")
    return data.reset_index(drop=True)


def design_with_constant(*columns: np.ndarray | pd.Series) -> np.ndarray:
    n = len(columns[0])
    return np.column_stack([np.ones(n)] + [np.asarray(x, dtype=float) for x in columns])


def multinomial_fit(y: np.ndarray, x: np.ndarray, n_categories: int) -> dict[str, float | int | bool]:
    """Reference-category multinomial logit estimated by maximum likelihood."""
    y = np.asarray(y, dtype=int)
    x = np.asarray(x, dtype=float)
    n_obs, n_predictors = x.shape

    def objective(flat: np.ndarray) -> float:
        beta = flat.reshape(n_predictors, n_categories - 1)
        eta = np.column_stack([x @ beta, np.zeros(n_obs)])
        return float(np.sum(logsumexp(eta, axis=1) - eta[np.arange(n_obs), y]))

    def gradient(flat: np.ndarray) -> np.ndarray:
        beta = flat.reshape(n_predictors, n_categories - 1)
        eta = np.column_stack([x @ beta, np.zeros(n_obs)])
        probabilities = np.exp(eta - logsumexp(eta, axis=1)[:, None])
        indicators = np.eye(n_categories)[y]
        return (x.T @ (probabilities[:, :-1] - indicators[:, :-1])).ravel()

    result = minimize(
        objective,
        np.zeros(n_predictors * (n_categories - 1)),
        jac=gradient,
        method="L-BFGS-B",
        options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-8, "maxls": 100},
    )
    if not result.success:
        raise RuntimeError(f"Multinomial logit did not converge: {result.message}")
    return {
        "Log_Likelihood": -float(result.fun),
        "N_Parameters": int(n_predictors * (n_categories - 1)),
        "Iterations": int(result.nit),
        "Converged": bool(result.success),
    }


def likelihood_ratio_test(reduced: dict, full: dict) -> tuple[float, int, float]:
    statistic = 2.0 * (full["Log_Likelihood"] - reduced["Log_Likelihood"])
    degrees_freedom = int(full["N_Parameters"] - reduced["N_Parameters"])
    p_value = float(chi2.sf(statistic, degrees_freedom))
    return float(statistic), degrees_freedom, p_value


def year_dummies(year: pd.Series) -> np.ndarray:
    return pd.get_dummies(year.astype(str), drop_first=True, dtype=float).to_numpy()


def encode_field(field: pd.Series) -> np.ndarray:
    codes = pd.Categorical(field, categories=FIELDS).codes
    if (codes < 0).any():
        raise ValueError("Missing or invalid field labels after mapping")
    return codes


def overall_tests(data: pd.DataFrame, field_column: str, label: str) -> dict[str, float | int | str]:
    field = data[field_column]
    y = encode_field(field)
    group = data["Group"].to_numpy(dtype=float)
    time = data["t"].to_numpy(dtype=float)
    dummies = year_dummies(data["Year"])

    # Average composition difference, controlling flexibly for year.
    level_reduced = multinomial_fit(
        y,
        np.column_stack([np.ones(len(data)), dummies]),
        len(FIELDS),
    )
    level_full = multinomial_fit(
        y,
        np.column_stack([np.ones(len(data)), dummies, group]),
        len(FIELDS),
    )
    level_lr, level_df, level_p = likelihood_ratio_test(level_reduced, level_full)

    # Primary group difference in approximately linear time trends.
    linear_reduced = multinomial_fit(y, design_with_constant(group, time), len(FIELDS))
    linear_full = multinomial_fit(
        y,
        design_with_constant(group, time, group * time),
        len(FIELDS),
    )
    linear_lr, linear_df, linear_p = likelihood_ratio_test(linear_reduced, linear_full)

    # General group-by-year interaction: year is categorical, so no linearity assumption.
    categorical_reduced_x = np.column_stack([np.ones(len(data)), group, dummies])
    categorical_full_x = np.column_stack(
        [categorical_reduced_x, dummies * group[:, None]]
    )
    categorical_reduced = multinomial_fit(y, categorical_reduced_x, len(FIELDS))
    categorical_full = multinomial_fit(y, categorical_full_x, len(FIELDS))
    categorical_lr, categorical_df, categorical_p = likelihood_ratio_test(
        categorical_reduced, categorical_full
    )

    # Descriptive pooled 2 x 10 table, included only as a transparent effect-size check.
    contingency = pd.crosstab(data["Group"], field).reindex(columns=FIELDS, fill_value=0)
    pearson, pearson_p, pearson_df, expected = chi2_contingency(contingency)
    cramers_v = np.sqrt(
        pearson
        / (
            contingency.to_numpy().sum()
            * min(contingency.shape[0] - 1, contingency.shape[1] - 1)
        )
    )

    return {
        "Specification": label,
        "N_English": int((data["Group"] == 0).sum()),
        "N_Chinese": int((data["Group"] == 1).sum()),
        "Adjusted_Group_LR": level_lr,
        "Adjusted_Group_DF": level_df,
        "Adjusted_Group_P": level_p,
        "Linear_Interaction_LR": linear_lr,
        "Linear_Interaction_DF": linear_df,
        "Linear_Interaction_P": linear_p,
        "Categorical_Interaction_LR": categorical_lr,
        "Categorical_Interaction_DF": categorical_df,
        "Categorical_Interaction_P": categorical_p,
        "Pooled_Pearson_Chi2": float(pearson),
        "Pooled_Pearson_DF": int(pearson_df),
        "Pooled_Pearson_P": float(pearson_p),
        "Pooled_Cramers_V": float(cramers_v),
        "Pooled_Min_Expected": float(expected.min()),
    }


def binary_logit_irls(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Binomial logit by Newton/IRLS, returning coefficients and standard errors."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    beta = np.zeros(x.shape[1])
    for _ in range(200):
        probability = expit(x @ beta)
        weight = np.clip(probability * (1.0 - probability), 1e-12, None)
        gradient = x.T @ (y - probability)
        information = x.T @ (x * weight[:, None])
        step = np.linalg.solve(information, gradient)
        beta_new = beta + step
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    else:
        raise RuntimeError("Binary logit IRLS did not converge")

    probability = expit(x @ beta)
    weight = np.clip(probability * (1.0 - probability), 1e-12, None)
    information = x.T @ (x * weight[:, None])
    covariance = np.linalg.inv(information)
    return beta, np.sqrt(np.diag(covariance))


def bh_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    n = len(values)
    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * n / np.arange(1, n + 1))[::-1]
    )[::-1]
    adjusted = np.empty(n)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def field_interaction_tests(
    data: pd.DataFrame,
    field_column: str,
    label: str,
) -> pd.DataFrame:
    group = data["Group"].to_numpy(dtype=float)
    time = data["t"].to_numpy(dtype=float)
    x = design_with_constant(group, time, group * time)
    rows = []
    for field in SUBSTANTIVE_FIELDS:
        outcome = (data[field_column] == field).astype(float).to_numpy()
        beta, standard_error = binary_logit_irls(outcome, x)
        interaction = beta[3]
        interaction_se = standard_error[3]
        z_value = interaction / interaction_se
        p_value = 2.0 * norm.sf(abs(z_value))
        english_or = np.exp(beta[2])
        chinese_or = np.exp(beta[2] + interaction)
        ratio = np.exp(interaction)
        rows.append(
            {
                "Specification": label,
                "Field": field,
                "Field_CN": FIELD_CN[field],
                "Interaction_Coefficient": interaction,
                "Interaction_SE": interaction_se,
                "Z": z_value,
                "P": p_value,
                "English_Annual_OR": english_or,
                "Chinese_Annual_OR": chinese_or,
                "Annual_OR_Ratio_CN_to_EN": ratio,
                "Ratio_CI_Low": np.exp(interaction - 1.96 * interaction_se),
                "Ratio_CI_High": np.exp(interaction + 1.96 * interaction_se),
            }
        )
    result = pd.DataFrame(rows)
    result["BH_Q"] = bh_adjust(result["P"])
    return result


def annual_shares(data: pd.DataFrame, field_column: str, label: str) -> pd.DataFrame:
    rows = []
    for (year, group, group_name), subset_data in data.groupby(
        ["Year", "Group", "Group_Name"], sort=True
    ):
        counts = subset_data[field_column].value_counts()
        total = len(subset_data)
        for field in FIELDS:
            count = int(counts.get(field, 0))
            rows.append(
                {
                    "Specification": label,
                    "Year": int(year),
                    "Group": int(group),
                    "Group_Name": group_name,
                    "Field": field,
                    "Field_CN": FIELD_CN[field],
                    "Count": count,
                    "Total": total,
                    "Share": count / total,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    data = prepare_data(args.cn.resolve(), args.en.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specifications: list[tuple[str, pd.DataFrame, str]] = [
        ("Final classification", data, "Final_Field"),
        ("Raw classification", data, "Raw_Field"),
        (
            "High-confidence classification",
            data[data["High_Confidence"]].copy(),
            "Final_Field",
        ),
        (
            "Final classification, 2020-2024",
            data[data["Year"] <= 2024].copy(),
            "Final_Field",
        ),
    ]

    overall_rows = []
    field_tables = []
    share_tables = []
    for label, subset_data, field_column in specifications:
        overall_rows.append(overall_tests(subset_data, field_column, label))
        field_tables.append(field_interaction_tests(subset_data, field_column, label))
        share_tables.append(annual_shares(subset_data, field_column, label))

    overall = pd.DataFrame(overall_rows)
    field_tests = pd.concat(field_tables, ignore_index=True)
    shares = pd.concat(share_tables, ignore_index=True)

    overall.to_csv(
        args.output_dir / "overall_group_tests.csv", index=False, encoding="utf-8-sig"
    )
    field_tests.to_csv(
        args.output_dir / "field_group_trend_interactions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    shares.to_csv(
        args.output_dir / "annual_group_field_shares.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nOVERALL GROUP TESTS")
    print(overall.round(6).to_string(index=False))
    print("\nFINAL CLASSIFICATION: FIELD-SPECIFIC INTERACTIONS")
    print(
        field_tests[field_tests["Specification"] == "Final classification"]
        .round(6)
        .to_string(index=False)
    )
    print(f"\nOutputs written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
