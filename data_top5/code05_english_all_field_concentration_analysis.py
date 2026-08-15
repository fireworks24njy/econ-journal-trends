from __future__ import annotations

import argparse
import math
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import normalize


# 本文件为独立版本，不依赖同目录下的其他 Python 文件。
GENERIC_STOPWORDS = set(ENGLISH_STOP_WORDS).union(
    {
        "paper", "papers", "article", "articles", "study", "studies",
        "result", "results", "finding", "findings", "show", "shows", "shown",
        "use", "uses", "using", "used", "based", "data", "dataset", "datasets",
        "evidence", "empirical", "analysis", "analyses", "estimate", "estimates",
        "estimated", "estimation", "effect", "effects", "economic", "economics",
        "approach", "method", "methods", "model", "models", "new", "different",
        "large", "small", "important", "provide", "provides", "including",
        "respectively", "however", "also", "among", "across", "within", "et",
        "al", "doi", "journal", "review", "university", "press", "vol", "volume",
        "issue", "abstract", "keywords", "keyword", "copyright", "author", "authors",
        "united", "january", "february", "march", "april", "june", "july",
        "august", "september", "october", "november", "december",
    }
)

METRICS = [
    "top10_share",
    "top20_share",
    "hhi",
    "normalized_entropy",
    "max_document_coverage",
]


def read_csv_robust(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Could not decode: {path}")


def normalize_keyword_phrase(value: str) -> list[str]:
    phrases = re.split(r"[;,|]", value.lower())
    cleaned: list[str] = []
    for phrase in phrases:
        phrase = re.sub(r"[^a-z0-9\- ]+", " ", phrase)
        phrase = re.sub(r"\s+", " ", phrase).strip()
        if len(phrase) >= 3 and phrase not in GENERIC_STOPWORDS:
            cleaned.append(phrase)
    return sorted(set(cleaned))


def vectorize_keyword_phrases(
    df: pd.DataFrame,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    documents = [normalize_keyword_phrase(x) for x in df["Keywords"]]
    document_frequency: dict[str, int] = {}
    for phrases in documents:
        for phrase in phrases:
            document_frequency[phrase] = document_frequency.get(phrase, 0) + 1

    vocabulary = sorted(term for term, count in document_frequency.items() if count >= 3)
    term_index = {term: j for j, term in enumerate(vocabulary)}
    rows: list[int] = []
    cols: list[int] = []
    for i, phrases in enumerate(documents):
        for phrase in phrases:
            if phrase in term_index:
                rows.append(i)
                cols.append(term_index[phrase])

    values = np.ones(len(rows), dtype=np.int8)
    matrix = sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(df), len(vocabulary))
    )
    return matrix, np.asarray(vocabulary), df["Predicted_Field"].to_numpy()


def concentration_metrics(matrix: sparse.csr_matrix) -> dict[str, float]:
    n_docs, vocabulary_size = matrix.shape
    frequencies = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
    total = frequencies.sum()
    if total <= 0 or n_docs == 0:
        return {metric: np.nan for metric in METRICS}

    shares = frequencies / total
    sorted_shares = np.sort(shares)[::-1]
    positive_shares = shares[shares > 0]
    entropy = -np.sum(positive_shares * np.log(positive_shares))
    normalized_entropy = (
        entropy / math.log(vocabulary_size) if vocabulary_size > 1 else 0.0
    )
    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    return {
        "top10_share": float(sorted_shares[:10].sum()),
        "top20_share": float(sorted_shares[:20].sum()),
        "hhi": float(np.sum(shares**2)),
        "normalized_entropy": float(normalized_entropy),
        "max_document_coverage": float(document_frequency.max() / n_docs),
    }


def top_terms(
    matrix: sparse.csr_matrix,
    terms: np.ndarray,
    field: str,
    top_n: int = 20,
) -> pd.DataFrame:
    frequencies = np.asarray(matrix.sum(axis=0)).ravel().astype(float)
    document_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel().astype(int)
    total = frequencies.sum()
    order = np.argsort(frequencies)[::-1][:top_n]
    return pd.DataFrame(
        {
            "Field": field,
            "Rank": np.arange(1, len(order) + 1),
            "Term": terms[order],
            "Frequency": frequencies[order].astype(int),
            "Term_Frequency_Share": frequencies[order] / total,
            "Document_Count": document_frequency[order],
            "Document_Coverage": document_frequency[order] / matrix.shape[0],
        }
    )


def permutation_comparison(
    matrix_a: sparse.csr_matrix,
    matrix_b: sparse.csr_matrix,
    field_a: str,
    field_b: str,
    n_resamples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    combined = sparse.vstack([matrix_a, matrix_b], format="csr")
    n_a = matrix_a.shape[0]
    n_total = combined.shape[0]
    observed_a = concentration_metrics(matrix_a)
    observed_b = concentration_metrics(matrix_b)
    observed_difference = {
        metric: observed_a[metric] - observed_b[metric] for metric in METRICS
    }
    null_differences = {
        metric: np.empty(n_resamples, dtype=float) for metric in METRICS
    }

    for draw in range(n_resamples):
        permutation = rng.permutation(n_total)
        values_a = concentration_metrics(combined[permutation[:n_a]])
        values_b = concentration_metrics(combined[permutation[n_a:]])
        for metric in METRICS:
            null_differences[metric][draw] = values_a[metric] - values_b[metric]

    rows = []
    for metric in METRICS:
        raw_difference = observed_difference[metric]
        advantage_a = (
            -raw_difference if metric == "normalized_entropy" else raw_difference
        )
        if advantage_a > 0:
            more_concentrated = field_a
        elif advantage_a < 0:
            more_concentrated = field_b
        else:
            more_concentrated = "Tie"

        null_values = null_differences[metric]
        p_upper = (np.sum(null_values >= raw_difference) + 1) / (n_resamples + 1)
        p_lower = (np.sum(null_values <= raw_difference) + 1) / (n_resamples + 1)
        rows.append(
            {
                "Field_A": field_a,
                "Field_B": field_b,
                "Metric": metric,
                "Field_A_Value": observed_a[metric],
                "Field_B_Value": observed_b[metric],
                "Raw_Difference_A_minus_B": raw_difference,
                "Concentration_Advantage_A": advantage_a,
                "More_Concentrated": more_concentrated,
                "Permutation_Null_Mean_Difference": float(null_values.mean()),
                "Permutation_P_Two_Sided": min(1.0, 2 * min(p_upper, p_lower)),
                "Resamples": n_resamples,
            }
        )
    return pd.DataFrame(rows)


def bh_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=p_values.index)


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
]

ADDITIONAL_PAIRS = [
    ("Finance", "Public Finance"),
    ("Industrial Organization", "International Economics"),
    ("Finance", "Industrial Organization"),
]

ALL_INTERPRETIVE_PAIRS = [
    ("Microeconomics", "Macroeconomics"),
    ("Labor Economics", "Development Economics"),
] + ADDITIONAL_PAIRS

COLORS = {
    "Development Economics": "#C3A477",
    "Economic History": "#A88F7A",
    "Finance": "#4F7896",
    "Industrial Organization": "#6D8A9E",
    "International Economics": "#8B78A6",
    "Labor Economics": "#708F7D",
    "Macroeconomics": "#A9B7C6",
    "Microeconomics": "#5F7F9F",
    "Public Finance": "#B6A95C",
}

FIELD_LABEL_STOPWORDS = {
    "development", "developing", "history", "historical", "finance", "financial",
    "industrial", "organization", "international", "labor", "labour", "macroeconomics",
    "macroeconomic", "microeconomics", "microeconomic", "public",
}


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.maximum.accumulate((len(values) - np.arange(len(values))) * ranked)
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=p_values.index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Systematic within-field hot-word concentration analysis for nine English fields."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "Classified_Top5_Journals_Result.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "english_all_field_concentration_output",
    )
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--scan-resamples", type=int, default=5000)
    parser.add_argument("--rarefaction-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20250814)
    return parser.parse_args()


def prepare_data(path: Path) -> pd.DataFrame:
    df = read_csv_robust(path)
    required = {"Predicted_Field", "Full_Text", "Keywords"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {sorted(missing)}")
    df = df[df["Predicted_Field"].isin(FIELDS)].copy()
    df["Full_Text"] = df["Full_Text"].fillna("").astype(str)
    df["Keywords"] = df["Keywords"].fillna("").astype(str)
    return df.reset_index(drop=True)


def vectorize(
    df: pd.DataFrame,
    ngram_range: tuple[int, int] = (1, 2),
    extra_stopwords: set[str] | None = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    stopwords = set(GENERIC_STOPWORDS)
    if extra_stopwords:
        stopwords.update(extra_stopwords)
    vectorizer = CountVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words=sorted(stopwords),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{1,}\b",
        ngram_range=ngram_range,
        min_df=5,
        max_df=0.85,
    )
    matrix = vectorizer.fit_transform(df["Full_Text"]).tocsr()
    return matrix, vectorizer.get_feature_names_out(), df["Predicted_Field"].to_numpy()


def subset(matrix: sparse.csr_matrix, labels: np.ndarray, field: str) -> sparse.csr_matrix:
    return matrix[labels == field].tocsr()


def specification_table(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    name: str,
) -> pd.DataFrame:
    rows = []
    for field in FIELDS:
        field_data = subset(matrix, labels, field)
        row = {
            "Specification": name,
            "Field": field,
            "N_Papers": field_data.shape[0],
            "Vocabulary_Size": field_data.shape[1],
        }
        row.update(concentration_metrics(field_data))
        rows.append(row)
    return pd.DataFrame(rows)


def systematic_top20_scan(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One pre-specified primary statistic, separately tested for all 36 pairs.

    For each pair, labels are permuted only inside the pooled papers from those two
    fields. This directly tests the pair-specific exchangeability null. Two-sided
    p-values use both permutation tails, allowing the null distribution of this
    nonlinear statistic to have a nonzero mean when group sizes differ.
    """
    pairs = list(combinations(FIELDS, 2))
    rows = []
    for field_a, field_b in pairs:
        data_a = subset(matrix, labels, field_a)
        data_b = subset(matrix, labels, field_b)
        pooled = sparse.vstack([data_a, data_b], format="csr")
        n_a = data_a.shape[0]
        n_total = pooled.shape[0]
        pooled_frequency = np.asarray(pooled.sum(axis=0)).ravel().astype(float)

        observed_a = concentration_metrics(data_a)["top20_share"]
        observed_b = concentration_metrics(data_b)["top20_share"]
        observed_difference = observed_a - observed_b
        null = np.empty(n_resamples, dtype=np.float32)

        for b in range(n_resamples):
            permutation = rng.permutation(n_total)
            frequency_a = np.asarray(pooled[permutation[:n_a]].sum(axis=0)).ravel().astype(float)
            frequency_b = pooled_frequency - frequency_a
            top_a = np.partition(frequency_a, -20)[-20:].sum() / frequency_a.sum()
            top_b = np.partition(frequency_b, -20)[-20:].sum() / frequency_b.sum()
            null[b] = top_a - top_b

        p_upper = (np.sum(null >= observed_difference) + 1) / (n_resamples + 1)
        p_lower = (np.sum(null <= observed_difference) + 1) / (n_resamples + 1)
        rows.append(
            {
                "Field_A": field_a,
                "Field_B": field_b,
                "Field_A_Top20": observed_a,
                "Field_B_Top20": observed_b,
                "Difference_A_Minus_B": observed_difference,
                "Permutation_Null_Mean": float(null.mean()),
                "Permutation_P_Two_Sided": min(1.0, 2 * min(p_upper, p_lower)),
                "Resamples": n_resamples,
            }
        )
    result = pd.DataFrame(rows)
    result["BH_Q_Across_36_Pairs"] = bh_adjust(result["Permutation_P_Two_Sided"])
    result["Holm_P_Across_36_Pairs"] = holm_adjust(result["Permutation_P_Two_Sided"])
    result["Bonferroni_P_Across_36_Pairs"] = np.minimum(
        result["Permutation_P_Two_Sided"] * len(result), 1.0
    )
    result["Higher_Concentration"] = np.where(
        result["Difference_A_Minus_B"] > 0, result["Field_A"], result["Field_B"]
    )
    return result.sort_values(["BH_Q_Across_36_Pairs", "Permutation_P_Two_Sided"])


def rarefaction_difference(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    field_a: str,
    field_b: str,
    n_resamples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    data_a = subset(matrix, labels, field_a)
    data_b = subset(matrix, labels, field_b)
    common_n = min(data_a.shape[0], data_b.shape[0])
    draws = {metric: np.empty(n_resamples) for metric in METRICS}
    for b in range(n_resamples):
        idx_a = rng.choice(data_a.shape[0], common_n, replace=False)
        idx_b = rng.choice(data_b.shape[0], common_n, replace=False)
        metrics_a = concentration_metrics(data_a[idx_a])
        metrics_b = concentration_metrics(data_b[idx_b])
        for metric in METRICS:
            draws[metric][b] = metrics_a[metric] - metrics_b[metric]
    rows = []
    for metric, values in draws.items():
        oriented = -values if metric == "normalized_entropy" else values
        rows.append(
            {
                "Field_A": field_a,
                "Field_B": field_b,
                "Metric": metric,
                "Common_N_Per_Field": common_n,
                "Median_Raw_Difference_A_Minus_B": float(np.median(values)),
                "P2_5": float(np.quantile(values, 0.025)),
                "P97_5": float(np.quantile(values, 0.975)),
                "Share_Resamples_A_More_Concentrated": float(np.mean(oriented > 0)),
                "Resamples": n_resamples,
            }
        )
    return pd.DataFrame(rows)


def text_length_table(matrix: sparse.csr_matrix, labels: np.ndarray) -> pd.DataFrame:
    lengths = np.asarray(matrix.sum(axis=1)).ravel()
    rows = []
    for field in FIELDS:
        values = lengths[labels == field]
        rows.append(
            {
                "Field": field,
                "N_Papers": len(values),
                "Mean_Filtered_Tokens": float(np.mean(values)),
                "Median_Filtered_Tokens": float(np.median(values)),
                "P25": float(np.quantile(values, 0.25)),
                "P75": float(np.quantile(values, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def pair_robustness_table(specifications: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for specification in specifications["Specification"].unique():
        block = specifications[specifications["Specification"] == specification].set_index("Field")
        for field_a, field_b in ALL_INTERPRETIVE_PAIRS:
            row = {
                "Specification": specification,
                "Field_A": field_a,
                "Field_B": field_b,
            }
            for metric in METRICS:
                raw = block.loc[field_a, metric] - block.loc[field_b, metric]
                row[f"{metric}_Raw_Difference_A_Minus_B"] = raw
                row[f"{metric}_Concentration_Advantage_A"] = -raw if metric == "normalized_entropy" else raw
            rows.append(row)
    return pd.DataFrame(rows)


def plot_all_field_ranking(metrics: pd.DataFrame, output: Path) -> None:
    data = metrics.sort_values("top20_share")
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    ax.barh(data["Field"], data["top20_share"] * 100, color=[COLORS[f] for f in data["Field"]])
    ax.set_xlabel("Top-20 share of within-field term frequency (%)")
    ax.set_title("Systematic Scan of Hot-word Concentration across Nine Fields", fontsize=16, weight="bold")
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for y, value in enumerate(data["top20_share"] * 100):
        ax.text(value + 0.18, y, f"{value:.1f}", va="center", fontsize=9)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_additional_curves(
    matrix: sparse.csr_matrix,
    labels: np.ndarray,
    output: Path,
    max_rank: int = 50,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.3), constrained_layout=True)
    for ax, (field_a, field_b) in zip(axes, ADDITIONAL_PAIRS):
        for field in (field_a, field_b):
            frequencies = np.asarray(subset(matrix, labels, field).sum(axis=0)).ravel().astype(float)
            shares = np.sort(frequencies / frequencies.sum())[::-1]
            cumulative = np.cumsum(shares[:max_rank]) * 100
            ax.plot(np.arange(1, max_rank + 1), cumulative, color=COLORS[field], linewidth=2.5, label=field)
        ax.set_title(f"{field_a}\nvs {field_b}", fontsize=12, weight="bold")
        ax.set_xlabel("Term rank")
        ax.set_ylabel("Cumulative frequency share (%)")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Additional Pairwise Concentration Comparisons", fontsize=17, weight="bold")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_additional_phrases(top_phrases: pd.DataFrame, output: Path) -> None:
    fields = [field for pair in ADDITIONAL_PAIRS[:2] for field in pair]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, field in zip(axes.flat, fields):
        data = (
            top_phrases[top_phrases["Field"] == field]
            .nlargest(10, "Document_Coverage")
            .sort_values("Document_Coverage")
        )
        ax.barh(data["Term"], data["Document_Coverage"] * 100, color=COLORS[field])
        ax.set_title(field, fontsize=13, weight="bold")
        ax.set_xlabel("Share of papers containing phrase (%)")
        ax.grid(axis="x", alpha=0.22)
        ax.spines[["top", "right", "left"]].set_visible(False)
        for y, value in enumerate(data["Document_Coverage"] * 100):
            ax.text(value + 0.25, y, f"{value:.1f}", va="center", fontsize=9)
    fig.suptitle("Topic-phrase Coverage for Additional Field Pairs", fontsize=17, weight="bold")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_robustness(robustness: pd.DataFrame, output: Path) -> None:
    display_specs = [
        "Token-weighted full text",
        "Each paper equal weight",
        "Each year and paper equal weight",
        "Document presence only",
        "Bigrams only",
        "Exact keyword phrases",
        "Field labels removed",
        "Non-ambiguous papers only",
    ]
    display_specs = [spec for spec in display_specs if spec in set(robustness["Specification"])]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.2), constrained_layout=True)
    for ax, (field_a, field_b) in zip(axes, ADDITIONAL_PAIRS):
        data = robustness[(robustness["Field_A"] == field_a) & (robustness["Field_B"] == field_b)].copy()
        data["Specification"] = pd.Categorical(data["Specification"], display_specs, ordered=True)
        data = data.sort_values("Specification", ascending=False)
        values = data["top20_share_Concentration_Advantage_A"] * 100
        colors = ["#5F7F9F" if value >= 0 else "#C3A477" for value in values]
        ax.barh(data["Specification"].astype(str), values, color=colors)
        ax.axvline(0, color="#333333", linewidth=1)
        ax.set_title(f"{field_a}\nminus {field_b}", fontsize=12, weight="bold")
        ax.set_xlabel("Top-20 concentration advantage (pp)")
        ax.grid(axis="x", alpha=0.22)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Robustness of Additional Pairwise Findings", fontsize=17, weight="bold")
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        script_dir = Path(__file__).resolve().parent
        candidates = [
            script_dir / "Classified_Top5_Journals_Result.csv",
            script_dir / "upload" / "Classified_Top5_Journals_Result.csv",
        ]
        candidates.extend(sorted(script_dir.glob("Classified_Top5_Journals_Result*.csv")))
        input_path = next((path.resolve() for path in candidates if path.exists()), input_path)
    if not input_path.exists():
        raise FileNotFoundError(
            "找不到 Classified_Top5_Journals_Result.csv。"
            "请把CSV与本脚本放在同一文件夹，或使用 "
            "--input \"D:\\\\你的路径\\\\Classified_Top5_Journals_Result.csv\" 指定路径。"
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    rng = np.random.default_rng(args.seed)
    df = prepare_data(input_path)
    matrix, terms, labels = vectorize(df, (1, 2))
    bigram_matrix, bigram_terms, bigram_labels = vectorize(df, (2, 2))
    label_free_matrix, _, label_free_labels = vectorize(df, (1, 2), FIELD_LABEL_STOPWORDS)
    keyword_matrix, _, keyword_labels = vectorize_keyword_phrases(df)

    equal_paper_matrix = normalize(matrix, norm="l1", axis=1, copy=True)
    if "Year" in df.columns:
        group_sizes = df.groupby(["Predicted_Field", "Year"])["Year"].transform("size").to_numpy(dtype=float)
        year_paper_equal_matrix = equal_paper_matrix.multiply((1.0 / group_sizes)[:, None]).tocsr()
    else:
        year_paper_equal_matrix = equal_paper_matrix
    document_presence_matrix = matrix.copy()
    document_presence_matrix.data = np.ones_like(document_presence_matrix.data)

    specification_tables = [
        specification_table(matrix, labels, "Token-weighted full text"),
        specification_table(equal_paper_matrix, labels, "Each paper equal weight"),
        specification_table(year_paper_equal_matrix, labels, "Each year and paper equal weight"),
        specification_table(document_presence_matrix, labels, "Document presence only"),
        specification_table(bigram_matrix, bigram_labels, "Bigrams only"),
        specification_table(keyword_matrix, keyword_labels, "Exact keyword phrases"),
        specification_table(label_free_matrix, label_free_labels, "Field labels removed"),
    ]

    if "Is_Ambiguous" in df.columns:
        is_ambiguous = df["Is_Ambiguous"].astype(str).str.lower().isin({"true", "1", "yes"}).to_numpy()
        if is_ambiguous.any():
            specification_tables.append(
                specification_table(matrix[~is_ambiguous], labels[~is_ambiguous], "Non-ambiguous papers only")
            )

    specifications = pd.concat(specification_tables, ignore_index=True)
    primary_metrics = specifications[specifications["Specification"] == "Token-weighted full text"].copy()
    scan = systematic_top20_scan(matrix, labels, args.scan_resamples, rng)

    interpretive_tests = []
    for field_a, field_b in ALL_INTERPRETIVE_PAIRS:
        interpretive_tests.append(
            permutation_comparison(
                subset(matrix, labels, field_a),
                subset(matrix, labels, field_b),
                field_a,
                field_b,
                args.resamples,
                rng,
            )
        )
    interpretive_tests = pd.concat(interpretive_tests, ignore_index=True)
    interpretive_tests["BH_Q_Within_Pair"] = interpretive_tests.groupby(
        ["Field_A", "Field_B"], group_keys=False
    )["Permutation_P_Two_Sided"].apply(bh_adjust)

    rarefaction_tables = []
    for field_a, field_b in ALL_INTERPRETIVE_PAIRS:
        rarefaction_tables.append(
            rarefaction_difference(
                matrix,
                labels,
                field_a,
                field_b,
                args.rarefaction_resamples,
                rng,
            )
        )
    rarefaction = pd.concat(rarefaction_tables, ignore_index=True)
    robustness = pair_robustness_table(specifications)

    top_term_tables = []
    top_phrase_tables = []
    for field in FIELDS:
        top_term_tables.append(top_terms(subset(matrix, labels, field), terms, field, 30))
        top_phrase_tables.append(top_terms(subset(bigram_matrix, bigram_labels, field), bigram_terms, field, 30))
    top_term_table = pd.concat(top_term_tables, ignore_index=True)
    top_phrase_table = pd.concat(top_phrase_tables, ignore_index=True)

    primary_metrics.to_csv(output_dir / "all_field_primary_metrics.csv", index=False, encoding="utf-8-sig")
    scan.to_csv(output_dir / "all_36_pair_top20_scan.csv", index=False, encoding="utf-8-sig")
    interpretive_tests.to_csv(output_dir / "interpretive_pair_permutation_tests.csv", index=False, encoding="utf-8-sig")
    specifications.to_csv(output_dir / "all_specification_metrics.csv", index=False, encoding="utf-8-sig")
    robustness.to_csv(output_dir / "pair_robustness.csv", index=False, encoding="utf-8-sig")
    rarefaction.to_csv(output_dir / "equal_sample_rarefaction.csv", index=False, encoding="utf-8-sig")
    text_length_table(matrix, labels).to_csv(output_dir / "field_text_lengths.csv", index=False, encoding="utf-8-sig")
    top_term_table.to_csv(output_dir / "all_field_top_terms.csv", index=False, encoding="utf-8-sig")
    top_phrase_table.to_csv(output_dir / "all_field_top_phrases.csv", index=False, encoding="utf-8-sig")

    plot_all_field_ranking(primary_metrics, output_dir / "all_field_top20_ranking.png")
    plot_additional_curves(matrix, labels, output_dir / "additional_pair_concentration_curves.png")
    plot_additional_phrases(top_phrase_table, output_dir / "additional_pair_topic_phrases.png")
    plot_robustness(robustness, output_dir / "additional_pair_robustness.png")

    print(f"Input: {input_path}")
    print(f"Substantive-field papers: {len(df)}")
    print(f"Primary shared vocabulary: {matrix.shape[1]}")
    print(f"Systematic pair tests: {len(scan)}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
