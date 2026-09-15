import os
import zipfile
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

TARGET_YEARS = range(2020, 2026)
ALLOWED_DOCUMENT_TYPES = {"Article", "Article; Early Access"}
LOW_SCORE, LOW_MARGIN = 0.015, 0.002
MEDIUM_SCORE, MEDIUM_MARGIN = 0.020, 0.003
NON_RESEARCH_TITLE_REGEX = (
    r"(?i)(?:^|:\s*)(?:a\s+)?comments?\s+(?:on|to)\b|:\s*comments?\s*$"
    r"|^\s*(?:a\s+)?discussion\s+(?:of|on)\b|^\s*commentary\b"
    r"|^\s*(?:a\s+)?(?:reply|response|rejoinder)\s+(?:to|on)\b"
    r"|:\s*(?:reply|response|rejoinder)\s*$"
    r"|^\s*(?:erratum|corrigendum|correction\s+(?:to|of)|retraction|withdrawal|addendum)\b"
    r"|:\s*(?:erratum|corrigendum|correction|retraction)\s*$"
    r"|^\s*(?:notice|announcement|call for papers|editorial|editor['’]?s note|"
    r"publisher['’]?s note|in memoriam|obituary)\b"
)

# ------------------ 数据加载 ------------------
def load_ready_csv(file_path="Cleaned_Top5_Journals_Dataset.csv"):
    print(f">> Loading pre-formatted CSV data from {file_path}...")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, encoding="utf-8-sig")
    else:
        archive = os.path.join(os.path.dirname(os.path.dirname(file_path)), "data_top5.zip")
        member = "data_top5/Cleaned_Top5_Journals_Dataset.csv"
        if not os.path.exists(archive):
            raise FileNotFoundError(f"Neither {file_path} nor {archive} exists")
        with zipfile.ZipFile(archive) as zf, zf.open(member) as source:
            df = pd.read_csv(source, encoding="utf-8-sig")
    for col in ["Title", "Abstract", "Keywords", "Year"]:
        if col not in df.columns:
            raise ValueError(f"Missing essential column in CSV: {col}")
    df["Title"] = df["Title"].fillna("").astype(str)
    df["Abstract"] = df["Abstract"].fillna("").astype(str)
    df["Keywords"] = df["Keywords"].fillna("").astype(str)
    year = pd.to_numeric(df["Year"], errors="coerce")
    if year.isna().any() or not year.isin(TARGET_YEARS).all():
        raise ValueError("Year contains missing/invalid/out-of-range values; rerun code01 cleaning first")
    df["Year"] = year.astype(int)
    if df["Title"].str.strip().eq("").any() or df["Abstract"].str.strip().eq("").any():
        raise ValueError("Title or Abstract is empty; rerun code01 cleaning first")
    non_research = df["Title"].str.contains(NON_RESEARCH_TITLE_REGEX, regex=True, na=False)
    if non_research.any():
        raise ValueError(
            f"{int(non_research.sum())} comment/reply/correction/notice records remain; "
            "rerun code01 cleaning first"
        )
    if "Document_Type" in df and not df["Document_Type"].isin(ALLOWED_DOCUMENT_TYPES).all():
        raise ValueError("Non-article records remain; rerun code01 cleaning first")
    if "DOI" in df:
        doi = df["DOI"].fillna("").astype(str).str.strip().str.lower()
        if (doi.ne("") & doi.duplicated(keep=False)).any():
            raise ValueError("Duplicate DOI remains; rerun code01 cleaning first")
    if "Record_ID" not in df:
        df["Record_ID"] = [f"EN{i:05d}" for i in range(1, len(df) + 1)]
    elif df["Record_ID"].duplicated().any():
        raise ValueError("Record_ID is not unique; rerun code01 cleaning first")
    df["Full_Text"] = (df["Title"] + ". ") * 2 + (df["Keywords"] + ". ") * 2 + df["Abstract"]
    print(f">> Successfully loaded {len(df)} valid records.")
    return df

# ------------------ 分类函数 ------------------
def classify_economics_fields(clean_df):
    print(">> Performing zero-shot economic field classification (TF-IDF + cosine similarity)...")
    # Terms are selected from corpus hot words/phrases, but generic hot words such
    # as "effect", "policy", "model", "market" and "firm" are excluded because
    # they occur across fields and reduce discriminative power.
    fields_dict = {
        "Development Economics": (
            "randomized controlled trial field experiment development aid poverty trap "
            "agricultural productivity health outcomes education intervention sanitation "
            "microfinance rural development structural transformation household welfare low income countries"
        ),
        "Economic History": (
            "economic history cliometrics historical data archival records historical census "
            "nineteenth century demographic transition industrial revolution slavery colonialism "
            "historical institutions path dependence historical persistence"
        ),
        "Finance": (
            "asset pricing risk premium portfolio choice stock returns bond yields market liquidity "
            "return volatility dividend policy option pricing corporate finance corporate debt "
            "bank credit financial intermediation systemic risk"
        ),
        "Industrial Organization": (
            "industrial organization antitrust market concentration entry barriers monopoly collusion "
            "oligopoly vertical restraints firm productivity price discrimination markup dispersion "
            "network effects digital platform product market competition"
        ),
        "International Economics": (
            "international trade gravity model import competition export participation trade costs "
            "tariff pass through exchange rate multinational production foreign direct investment "
            "global value chains current account balance of payments trade agreement"
        ),
        "Labor Economics": (
            "labor market wage inequality employment unemployment worker reallocation returns to schooling "
            "job training occupational choice internal migration immigration earnings inequality "
            "labor supply labor demand minimum wage gender wage gap"
        ),
        "Macroeconomics": (
            "macroeconomics monetary policy interest rates inflation expectations fiscal multiplier "
            "sovereign debt consumption dynamics business cycle aggregate fluctuations output gap "
            "economic growth productivity growth recession central bank"
        ),
        "Microeconomics": (
            "microeconomic theory utility maximization general equilibrium game theory nash equilibrium "
            "mechanism design auction design contract theory adverse selection moral hazard "
            "information design social preferences strategic interaction"
        ),
        "Public Finance": (
            "public finance income taxation corporate tax value added tax tax enforcement "
            "social insurance unemployment insurance disability insurance public goods environmental externality "
            "transfer payments redistribution optimal taxation deadweight loss government spending"
        ),
        "Miscellaneous & Methods": (
            "econometric theory instrumental variables two stage least squares difference in differences "
            "regression discontinuity synthetic control panel data fixed effects generalized method moments "
            "asymptotic distribution bayesian estimation causal identification machine learning"
        )
    }

    field_names = list(fields_dict.keys())
    field_texts = list(fields_dict.values())
    corpus = clean_df["Full_Text"].tolist()

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=30000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        strip_accents="unicode",
        min_df=1,
        norm="l2",
    )
    all_texts = corpus + field_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    doc_vectors = tfidf_matrix[:len(corpus)]
    field_vectors = tfidf_matrix[len(corpus):]
    similarity_matrix = cosine_similarity(doc_vectors, field_vectors)

    best_field_indices = similarity_matrix.argmax(axis=1)
    sorted_scores = np.sort(similarity_matrix, axis=1)
    best_scores = sorted_scores[:, -1]
    second_scores = sorted_scores[:, -2]
    raw_fields = np.array([field_names[idx] for idx in best_field_indices], dtype=object)
    clean_df["Classification_Score"] = best_scores
    clean_df["Classification_Margin"] = best_scores - second_scores
    # 分数仅表示与字段词典的相似证据强弱，不能在人工验证前称为“准确概率”。
    margins = best_scores - second_scores
    high_risk = (best_scores < LOW_SCORE) | (margins < LOW_MARGIN)
    medium_risk = (~high_risk) & ((best_scores < MEDIUM_SCORE) | (margins < MEDIUM_MARGIN))
    clean_df["Raw_Predicted_Field"] = raw_fields
    # 不将低证据论文强行归入具体领域，与中文样本保持同一口径。
    clean_df["Predicted_Field"] = np.where(
        high_risk, "Miscellaneous & Methods", raw_fields
    )
    clean_df["Evidence_Band"] = np.select(
        [high_risk, medium_risk], ["Low", "Medium"], default="High"
    )
    clean_df["Assignment_Basis"] = np.select(
        [high_risk, raw_fields == "Miscellaneous & Methods"],
        [
            "Insufficient field evidence; assigned to methods/miscellaneous",
            "Closest match: methods/miscellaneous",
        ],
        default="Closest substantive-field match",
    )
    clean_df["Is_Ambiguous"] = high_risk
    clean_df["Needs_Review"] = high_risk | medium_risk
    clean_df["Review_Priority"] = np.select(
        [high_risk, medium_risk], ["High", "Medium"], default="Low"
    )

    print("\n[Diagnostic] Field Classification Distribution:")
    print("[Raw field labels]")
    print(clean_df["Raw_Predicted_Field"].value_counts().to_string())
    print("\n[Evidence bands]")
    print(clean_df["Evidence_Band"].value_counts().to_string())
    print("-" * 65)
    return clean_df

# ------------------ 趋势图 ------------------
def generate_field_trend_stacked_chart(clean_df, output_path="Field_Trends_Stacked_Area.png",field_col="Raw_Predicted_Field", title_suffix="Raw labels"):
    print(">> Generating field trend stacked area chart...")
    min_year = int(clean_df["Year"].min())
    max_year = int(clean_df["Year"].max())
    all_years = range(min_year, max_year + 1)
    trend_pivot = clean_df.pivot_table(index="Year", columns=field_col, values="Title", aggfunc="count", fill_value=0)
    trend_pivot = trend_pivot.reindex(all_years, fill_value=0)
    trend_proportion = trend_pivot.div(trend_pivot.sum(axis=1), axis=0)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    trend_proportion.plot.area(ax=ax, cmap="tab10", alpha=0.85, linewidth=0.5)
    ax.set_title(f"Evolution of Research Field Proportions ({title_suffix})", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Publication Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("Proportion of Publications", fontsize=12, fontweight="bold")
    ax.set_xlim(trend_proportion.index.min(), trend_proportion.index.max())
    ax.set_ylim(0, 1.0)
    plt.legend(title="Research Fields", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> [Saved] Stacked area chart saved to: {output_path}")


def create_single_coder_validation_sample(clean_df, output_dir, n_random=50,n_borderline=20, random_state=20260824):
    """生成50篇分层随机样本和20篇边界样本；边界样本不用于总体准确率。"""
    blind_path = os.path.join(output_dir, "Top5_Validation_Sample_Blind.csv")
    key_path = os.path.join(output_dir, "Top5_Validation_Sample_Key.csv")
    if os.path.exists(blind_path) or os.path.exists(key_path):
        print(">> Validation sample already exists; existing files were not overwritten.")
        return

    n_random = min(n_random, len(clean_df))
    try:
        _, random_sample = train_test_split(
            clean_df, test_size=n_random,
            stratify=clean_df["Raw_Predicted_Field"], random_state=random_state
        )
    except ValueError:
        random_sample = clean_df.sample(n=n_random, random_state=random_state)

    remaining = clean_df.drop(index=random_sample.index).copy()
    remaining["_Risk_Index"] = (
        remaining["Classification_Score"].rank(method="average", pct=True)
        + remaining["Classification_Margin"].rank(method="average", pct=True)
    )
    borderline_sample = remaining.nsmallest(min(n_borderline, len(remaining)), "_Risk_Index")
    random_sample = random_sample.copy()
    borderline_sample = borderline_sample.copy()
    random_sample["Sample_Type"] = "Random"
    borderline_sample["Sample_Type"] = "Borderline_diagnostic"
    sample = pd.concat([random_sample, borderline_sample], ignore_index=True)
    sample = sample.sample(frac=1, random_state=random_state).reset_index(drop=True)
    sample["Validation_ID"] = [f"EN-AUDIT-{i:03d}" for i in range(1, len(sample) + 1)]

    for col in ["Journal", "Keywords"]:
        if col not in sample:
            sample[col] = ""
    blind = sample[[
        "Validation_ID", "Record_ID", "Sample_Type", "Year", "Journal",
        "Title", "Keywords", "Abstract"
    ]].copy()
    blind["Human_Field"] = ""
    blind["Human_Confidence"] = ""
    blind["Notes"] = ""
    key = sample[[
        "Validation_ID", "Record_ID", "Sample_Type", "Raw_Predicted_Field",
        "Predicted_Field", "Classification_Score", "Classification_Margin", "Evidence_Band"
    ]]
    blind.to_csv(blind_path, index=False, encoding="utf-8-sig")
    key.to_csv(key_path, index=False, encoding="utf-8-sig")
    print(f">> [Saved] Blind validation sample: {blind_path}")
    print("   Use only Sample_Type=Random to estimate overall accuracy.")


def evaluate_validation_if_complete(output_dir):
    """填写完盲审表后再次运行脚本，自动计算随机样本的审计指标。"""
    blind_path = os.path.join(output_dir, "Top5_Validation_Sample_Blind.csv")
    key_path = os.path.join(output_dir, "Top5_Validation_Sample_Key.csv")
    if not os.path.exists(blind_path) or not os.path.exists(key_path):
        return
    blind = pd.read_csv(blind_path, encoding="utf-8-sig")
    key = pd.read_csv(key_path, encoding="utf-8-sig")
    audit = blind.merge(key, on=["Validation_ID", "Record_ID", "Sample_Type"], validate="one_to_one")
    audit["Human_Field"] = audit["Human_Field"].fillna("").astype(str).str.strip()
    random_audit = audit[audit["Sample_Type"].eq("Random")].copy()
    if random_audit["Human_Field"].eq("").any():
        print(">> Validation labels are incomplete; metrics were not calculated yet.")
        return

    y_true, y_pred = random_audit["Human_Field"], random_audit["Raw_Predicted_Field"]
    n = len(random_audit)
    accuracy = accuracy_score(y_true, y_pred)
    z = 1.959963984540054
    denominator = 1 + z ** 2 / n
    center = (accuracy + z ** 2 / (2 * n)) / denominator
    half_width = z * np.sqrt(accuracy * (1 - accuracy) / n + z ** 2 / (4 * n ** 2)) / denominator
    metrics = pd.DataFrame([{
        "Sample": "Random stratified audit", "N": n,
        "Exact_Accuracy": accuracy,
        "Wilson_95_Lower": max(0, center - half_width),
        "Wilson_95_Upper": min(1, center + half_width),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }])
    metrics.to_csv(os.path.join(output_dir, "Top5_Validation_Metrics.csv"),
                   index=False, encoding="utf-8-sig")
    pd.crosstab(y_true, y_pred, rownames=["Human_Field"], colnames=["Machine_Field"]).to_csv(
        os.path.join(output_dir, "Top5_Validation_Confusion_Matrix.csv"), encoding="utf-8-sig"
    )
    print(">> [Saved] Validation accuracy, Wilson interval, Macro-F1 and confusion matrix.")

# ------------------ 主程序 ------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "Cleaned_Top5_Journals_Dataset.csv")
    df = load_ready_csv(csv_file)
    if not df.empty:
        df = classify_economics_fields(df)
        df.to_csv(os.path.join(base_dir, "Classified_Top5_Journals_Result.csv"), index=False, encoding="utf-8-sig")
        # 主图使用最终分类；原始最优匹配仅作敏感性对照。
        generate_field_trend_stacked_chart(
            df, os.path.join(base_dir, "Field_Trends_Stacked_Area.png"),
            field_col="Predicted_Field", title_suffix="Final classification"
        )
        generate_field_trend_stacked_chart(
            df, os.path.join(base_dir, "Field_Trends_Raw_Classification.png"),
            field_col="Raw_Predicted_Field", title_suffix="Raw classification"
        )
        create_single_coder_validation_sample(df, base_dir)
        evaluate_validation_if_complete(base_dir)
        print("\n>> All tasks completed successfully!")
    else:
        print("[Error] CSV file is empty or missing necessary data.")
