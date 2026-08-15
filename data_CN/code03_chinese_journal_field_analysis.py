"""中文期刊经济学领域年度变化分析。
1. 使用 Predicted_Field 作为最终分类；
2. 保留“方法与杂项”在全部样本的整体检验中；
3. 整体检验始终使用完整十领域；
4. 比较最终分类、原始分类和高置信分类三种口径。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from scipy.stats import chi2, chi2_contingency, norm


# ============================ 可调整参数 ============================

INPUT_FILE = Path("Cleaned_Custom_Dataset_Classified.csv")  # 引用量数据为0是因为原始数据即没有引用量数据
OUTPUT_DIR = Path("chinese_field_analysis_output")

YEAR_START = 2020
YEAR_END = 2025
FINAL_FIELD_COLUMN = "Predicted_Field"
RAW_FIELD_COLUMN = "Raw_Predicted_Field"
HIGH_CONFIDENCE_COLUMN = "Is_High_Confidence"

# 该类别不进入单领域趋势检验、阶段扫描及对应的多重检验。
EXCLUDED_FOCUS_FIELDS = {"方法与杂项"}

# 置换次数越多，阶段扫描的 p 值越稳定，但运行时间也越长。
N_PERMUTATIONS = 10_000
RANDOM_SEED = 20260814

# 候选分界点分别代表 2021/2022、2022/2023、2023/2024。
# 每个分界点两侧至少各保留两个年份。
CANDIDATE_CUTS = (2021, 2022, 2023)


FIELD_ORDER = [
    "发展经济学",
    "经济史",
    "金融学",
    "产业组织",
    "国际经济学",
    "劳动经济学",
    "宏观经济学",
    "微观经济学",
    "公共财政",
    "方法与杂项",
]

# 淡雅的莫兰迪色系。
FIELD_COLORS = [
    "#7393B3", "#D9A273", "#7FA487", "#C77C78", "#9A86B8",
    "#A78978", "#C58DB5", "#8F9699", "#B7B85A", "#65B8C4",
]


def configure_plot_style() -> None:
    """设置中文字体和图形样式。"""
    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "Noto Sans CJK SC",
        "Source Han Sans SC", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 15
    plt.rcParams["axes.labelsize"] = 12


def to_bool(series: pd.Series) -> pd.Series:
    """兼容布尔值和字符串形式的 True/False。"""
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_and_clean_data(path: Path) -> pd.DataFrame:
    required = {"Year", FINAL_FIELD_COLUMN}
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"数据缺少必要字段：{sorted(missing)}")

    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].between(YEAR_START, YEAR_END)].copy()
    df["Year"] = df["Year"].astype(int)
    df = df[df[FINAL_FIELD_COLUMN].notna()].copy()

    if df.empty:
        raise ValueError("筛选年份和有效分类后没有可分析数据。")
    return df


def available_field_order(series: pd.Series) -> list[str]:
    present = set(series.dropna().astype(str))
    ordered = [field for field in FIELD_ORDER if field in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def build_crosstab(
    df: pd.DataFrame,
    field_column: str,
    fields: list[str] | None = None,
) -> pd.DataFrame:
    years = list(range(YEAR_START, YEAR_END + 1))
    if fields is None:
        fields = available_field_order(df[field_column])
    return pd.crosstab(df["Year"], df[field_column]).reindex(
        index=years, columns=fields, fill_value=0
    )


def pearson_test(counts: pd.DataFrame) -> dict[str, float]:
    statistic, p_value, degrees_freedom, expected = chi2_contingency(counts.values)
    sample_size = counts.values.sum()
    denominator = min(counts.shape[0] - 1, counts.shape[1] - 1)
    cramers_v = np.sqrt(statistic / (sample_size * denominator))
    return {
        "Pearson_chi2": float(statistic),
        "Pearson_df": int(degrees_freedom),
        "Pearson_p": float(p_value),
        "Cramers_V": float(cramers_v),
        "Minimum_expected_frequency": float(expected.min()),
    }


def multinomial_logit_lr(counts: pd.DataFrame) -> dict[str, float | bool]:
    """检验所有领域相对份额是否存在共同的线性年份趋势。

    使用最后一个领域作为参照，仅检验年份斜率；因此 LR 检验自由度为 K-1。
    模型直接使用“年份×领域”汇总频数，和逐篇论文拟合等价。
    """
    c = counts.to_numpy(dtype=float)
    years = counts.index.to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(years)), years - YEAR_START])
    category_count = c.shape[1]

    if category_count < 2:
        raise ValueError("多项Logit至少需要两个领域。")

    def objective(flat_beta: np.ndarray) -> tuple[float, np.ndarray]:
        beta = flat_beta.reshape(2, category_count - 1)
        eta = x @ beta
        logits = np.column_stack([eta, np.zeros(len(years))])
        normalizer = logsumexp(logits, axis=1)
        log_likelihood = (
            (c[:, :-1] * eta).sum()
            - (c.sum(axis=1) * normalizer).sum()
        )
        probabilities = np.exp(logits - normalizer[:, None])
        gradient = x.T @ (
            c[:, :-1]
            - c.sum(axis=1)[:, None] * probabilities[:, :-1]
        )
        return -float(log_likelihood), -gradient.ravel()

    initial = np.zeros(2 * (category_count - 1))
    result = minimize(
        lambda b: objective(b),
        initial,
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 10_000},
    )

    fitted_log_likelihood = -float(result.fun)
    total_by_category = c.sum(axis=0)
    null_probabilities = total_by_category / total_by_category.sum()
    null_log_likelihood = float(
        np.sum(total_by_category * np.log(null_probabilities))
    )
    lr_statistic = 2 * (fitted_log_likelihood - null_log_likelihood)
    degrees_freedom = category_count - 1

    return {
        "Multinomial_LR": float(lr_statistic),
        "Multinomial_df": int(degrees_freedom),
        "Multinomial_p": float(chi2.sf(lr_statistic, degrees_freedom)),
        "Optimizer_converged": bool(result.success),
        "Maximum_absolute_gradient": float(np.max(np.abs(result.jac))),
    }


def fit_binomial_logit(
    successes: np.ndarray,
    totals: np.ndarray,
    years: np.ndarray,
) -> dict[str, float]:
    """用IRLS拟合 logit(p_y)=alpha+beta*(year-YEAR_START)。"""
    successes = np.asarray(successes, dtype=float)
    totals = np.asarray(totals, dtype=float)
    years = np.asarray(years, dtype=float)
    x = np.column_stack([np.ones(len(years)), years - YEAR_START])
    beta = np.zeros(2)

    for _ in range(100):
        probability = expit(x @ beta)
        weights = totals * probability * (1 - probability)
        information = x.T @ (weights[:, None] * x)
        score = x.T @ (successes - totals * probability)
        step = np.linalg.solve(information, score)
        beta += step
        if np.max(np.abs(step)) < 1e-12:
            break

    covariance = np.linalg.inv(information)
    standard_error = float(np.sqrt(covariance[1, 1]))
    slope = float(beta[1])
    z_value = slope / standard_error
    p_value = float(2 * norm.sf(abs(z_value)))
    odds_ratio = float(np.exp(slope))

    return {
        "Beta": slope,
        "SE": standard_error,
        "z": float(z_value),
        "p_value": p_value,
        "OR": odds_ratio,
        "OR_CI_lower": float(np.exp(slope - 1.96 * standard_error)),
        "OR_CI_upper": float(np.exp(slope + 1.96 * standard_error)),
        "Annual_odds_change_percent": float((odds_ratio - 1) * 100),
    }


def benjamini_hochberg(p_values: np.ndarray | list[float]) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted_sorted = p_values[order] * count / np.arange(1, count + 1)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted = np.empty(count)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def field_trend_tests(
    counts: pd.DataFrame,
    excluded_fields: set[str] = EXCLUDED_FOCUS_FIELDS,
) -> pd.DataFrame:
    totals = counts.sum(axis=1).to_numpy(dtype=float)
    years = counts.index.to_numpy(dtype=float)
    records: list[dict[str, float | str | int]] = []

    for field in counts.columns:
        if field in excluded_fields:
            continue
        result = fit_binomial_logit(
            counts[field].to_numpy(dtype=float), totals, years
        )
        records.append(
            {
                "Field": field,
                "Paper_count": int(counts[field].sum()),
                **result,
            }
        )

    output = pd.DataFrame(records)
    output["BH_q_value"] = benjamini_hochberg(output["p_value"].to_numpy())
    output["Significant_after_BH_0.05"] = output["BH_q_value"] < 0.05
    return output.sort_values("p_value").reset_index(drop=True)


def overall_tests_for_scope(
    df: pd.DataFrame,
    field_column: str,
    scope_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """使用完整十领域进行整体构成检验。"""
    all_fields = available_field_order(df[field_column])
    full_counts = build_crosstab(df, field_column, all_fields)
    result = pd.DataFrame(
        [
            {
                "Scope": scope_name,
                "Analysis": "十领域整体检验",
                "N": int(full_counts.values.sum()),
                "Number_of_fields": int(full_counts.shape[1]),
                **pearson_test(full_counts),
                **multinomial_logit_lr(full_counts),
            }
        ]
    )
    return result, full_counts


def stage_difference(
    labels: np.ndarray,
    years: np.ndarray,
    field: str,
    cut: int,
) -> float:
    early = years <= cut
    late = years > cut
    return float((labels[late] == field).mean() - (labels[early] == field).mean())


def stage_scan_permutation(
    df: pd.DataFrame,
    field_column: str,
    tested_fields: list[str],
    permutations: int = N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """扫描候选断点，并用最大绝对阶段差的置换分布校正断点选择。

    随后对所有被检验领域的置换 p 值再进行 BH 校正。
    """
    labels = df[field_column].astype(str).to_numpy()
    years = df["Year"].to_numpy(dtype=int)
    observed: dict[str, np.ndarray] = {}

    for field in tested_fields:
        observed[field] = np.array(
            [stage_difference(labels, years, field, cut) for cut in CANDIDATE_CUTS]
        )

    exceedances = {field: 0 for field in tested_fields}
    rng = np.random.default_rng(seed)

    for _ in range(permutations):
        permuted = rng.permutation(labels)
        for field in tested_fields:
            permutation_differences = np.array(
                [
                    stage_difference(permuted, years, field, cut)
                    for cut in CANDIDATE_CUTS
                ]
            )
            if (
                np.max(np.abs(permutation_differences))
                >= np.max(np.abs(observed[field])) - 1e-15
            ):
                exceedances[field] += 1

    records = []
    for field in tested_fields:
        differences = observed[field]
        best_index = int(np.argmax(np.abs(differences)))
        best_cut = CANDIDATE_CUTS[best_index]
        early = years <= best_cut
        late = years > best_cut
        p_value = (exceedances[field] + 1) / (permutations + 1)
        records.append(
            {
                "Field": field,
                "Best_cut": f"{best_cut}/{best_cut + 1}",
                "Early_count": int(np.sum(labels[early] == field)),
                "Early_total": int(early.sum()),
                "Early_share_percent": float(100 * (labels[early] == field).mean()),
                "Late_count": int(np.sum(labels[late] == field)),
                "Late_total": int(late.sum()),
                "Late_share_percent": float(100 * (labels[late] == field).mean()),
                "Difference_pp": float(100 * differences[best_index]),
                "Permutation_p": float(p_value),
            }
        )

    output = pd.DataFrame(records)
    output["BH_q_value"] = benjamini_hochberg(output["Permutation_p"].to_numpy())
    output["Significant_after_BH_0.05"] = output["BH_q_value"] < 0.05
    return output.sort_values("Permutation_p").reset_index(drop=True)


def leave_one_year_out_tests(
    df: pd.DataFrame,
    field_column: str,
    fields: list[str],
) -> pd.DataFrame:
    records = []
    all_years = list(range(YEAR_START, YEAR_END + 1))

    for field in fields:
        for omitted_year in all_years:
            retained_years = [year for year in all_years if year != omitted_year]
            subset = df[df["Year"].isin(retained_years)]
            counts = build_crosstab(
                subset,
                field_column,
                available_field_order(subset[field_column]),
            )
            if field not in counts.columns:
                continue
            result = fit_binomial_logit(
                counts[field].to_numpy(dtype=float),
                counts.sum(axis=1).to_numpy(dtype=float),
                counts.index.to_numpy(dtype=float),
            )
            records.append(
                {"Field": field, "Omitted_year": omitted_year, **result}
            )
    return pd.DataFrame(records)


def sensitivity_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scopes: list[tuple[str, pd.DataFrame, str]] = [
        ("最终分类", df, FINAL_FIELD_COLUMN),
    ]

    if RAW_FIELD_COLUMN in df.columns:
        raw = df[df[RAW_FIELD_COLUMN].notna()].copy()
        scopes.append(("原始分类", raw, RAW_FIELD_COLUMN))

    if HIGH_CONFIDENCE_COLUMN in df.columns:
        high_confidence = df[to_bool(df[HIGH_CONFIDENCE_COLUMN])].copy()
        scopes.append(("高置信分类", high_confidence, FINAL_FIELD_COLUMN))

    overall_outputs = []
    trend_outputs = []

    for scope_name, subset, field_column in scopes:
        overall, full_counts = overall_tests_for_scope(
            subset, field_column, scope_name
        )
        trends = field_trend_tests(full_counts)
        trends.insert(0, "Scope", scope_name)
        trends.insert(1, "N", len(subset))
        overall_outputs.append(overall)
        trend_outputs.append(trends)

    return (
        pd.concat(overall_outputs, ignore_index=True),
        pd.concat(trend_outputs, ignore_index=True),
    )


def plot_overall_distribution(counts: pd.DataFrame, output_path: Path) -> None:
    shares = 100 * counts.sum(axis=0) / counts.values.sum()
    colors = FIELD_COLORS[: len(shares)]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(shares.index, shares.values, color=colors, edgecolor="white")
    ax.set_title(f"{YEAR_START}—{YEAR_END}年中文期刊经济学领域总体分布")
    ax.set_xlabel("研究领域")
    ax.set_ylabel("论文占比（%）")
    ax.grid(axis="y", alpha=0.20)
    ax.tick_params(axis="x", rotation=35)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_annual_trend(counts: pd.DataFrame, output_path: Path) -> None:
    shares = counts.div(counts.sum(axis=1), axis=0)
    colors = FIELD_COLORS[: len(shares.columns)]

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.stackplot(
        shares.index,
        shares.to_numpy().T,
        labels=shares.columns,
        colors=colors,
        alpha=0.92,
    )
    ax.set_title("中文期刊十大经济学领域研究占比演变趋势")
    ax.set_xlabel("出版年份")
    ax.set_ylabel("论文占比")
    ax.set_xlim(YEAR_START, YEAR_END)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(YEAR_START, YEAR_END + 1))
    ax.grid(alpha=0.18)
    ax.legend(title="研究领域", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_key_results(
    annual_counts: pd.Series,
    overall_tests: pd.DataFrame,
    trends: pd.DataFrame,
    stage_scan: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> None:
    print("\n========== 样本年度分布 ==========")
    print(annual_counts.to_string())

    print("\n========== 整体构成检验 ==========")
    display_columns = [
        "Analysis", "N", "Number_of_fields", "Pearson_chi2", "Pearson_df",
        "Pearson_p", "Cramers_V", "Minimum_expected_frequency",
        "Multinomial_LR", "Multinomial_df", "Multinomial_p",
    ]
    print(overall_tests[display_columns].round(6).to_string(index=False))

    print("\n========== 单领域线性趋势（不含方法与杂项） ==========")
    trend_columns = [
        "Field", "Paper_count", "Beta", "SE", "p_value", "BH_q_value",
        "OR", "OR_CI_lower", "OR_CI_upper", "Annual_odds_change_percent",
    ]
    print(trends[trend_columns].round(6).to_string(index=False))

    print("\n========== 阶段扫描置换检验（不含方法与杂项） ==========")
    print(stage_scan.round(6).to_string(index=False))

    if not leave_one_out.empty:
        print("\n========== 显著领域留一年检验 ==========")
        print(
            leave_one_out[
                ["Field", "Omitted_year", "OR", "p_value", "OR_CI_lower", "OR_CI_upper"]
            ].round(6).to_string(index=False)
        )


def main() -> None:
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")
    configure_plot_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_path = INPUT_FILE
    if not input_path.exists():
        uploaded_path = Path("upload") / INPUT_FILE.name
        if uploaded_path.exists():
            input_path = uploaded_path
        else:
            raise FileNotFoundError(
                f"找不到数据文件：{INPUT_FILE.resolve()}。请修改 INPUT_FILE。"
            )

    df = load_and_clean_data(input_path)
    fields = available_field_order(df[FINAL_FIELD_COLUMN])
    full_counts = build_crosstab(df, FINAL_FIELD_COLUMN, fields)
    annual_counts = full_counts.sum(axis=1)
    annual_shares = full_counts.div(annual_counts, axis=0)

    final_overall, _ = overall_tests_for_scope(
        df, FINAL_FIELD_COLUMN, "最终分类"
    )
    final_trends = field_trend_tests(full_counts)

    tested_fields = [
        field for field in full_counts.columns
        if field not in EXCLUDED_FOCUS_FIELDS
    ]
    stage_scan = stage_scan_permutation(
        df,
        FINAL_FIELD_COLUMN,
        tested_fields,
        permutations=N_PERMUTATIONS,
        seed=RANDOM_SEED,
    )

    significant_fields = final_trends.loc[
        final_trends["Significant_after_BH_0.05"], "Field"
    ].tolist()
    leave_one_out = leave_one_year_out_tests(
        df, FINAL_FIELD_COLUMN, significant_fields
    )

    sensitivity_overall, sensitivity_trends = sensitivity_analysis(df)

    # 保存表格结果。
    annual_counts.rename("Sample_size").to_csv(
        OUTPUT_DIR / "annual_sample_sizes.csv", encoding="utf-8-sig"
    )
    full_counts.to_csv(
        OUTPUT_DIR / "annual_field_counts.csv", encoding="utf-8-sig"
    )
    annual_shares.to_csv(
        OUTPUT_DIR / "annual_field_shares.csv", encoding="utf-8-sig"
    )
    final_overall.to_csv(
        OUTPUT_DIR / "overall_tests.csv", index=False, encoding="utf-8-sig"
    )
    final_trends.to_csv(
        OUTPUT_DIR / "field_linear_trends.csv", index=False, encoding="utf-8-sig"
    )
    stage_scan.to_csv(
        OUTPUT_DIR / "stage_scan_permutation.csv", index=False, encoding="utf-8-sig"
    )
    leave_one_out.to_csv(
        OUTPUT_DIR / "leave_one_year_out.csv", index=False, encoding="utf-8-sig"
    )
    sensitivity_overall.to_csv(
        OUTPUT_DIR / "classification_sensitivity_overall.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sensitivity_trends.to_csv(
        OUTPUT_DIR / "classification_sensitivity_trends.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 保存图片。
    plot_overall_distribution(
        full_counts, OUTPUT_DIR / "field_distribution.png"
    )
    plot_annual_trend(full_counts, OUTPUT_DIR / "field_trend.png")

    print_key_results(
        annual_counts,
        final_overall,
        final_trends,
        stage_scan,
        leave_one_out,
    )
    print(f"\n全部结果已保存到：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
