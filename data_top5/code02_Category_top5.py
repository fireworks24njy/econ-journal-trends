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
from sklearn.metrics.pairwise import cosine_similarity

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
    for col in ["Title", "Abstract", "Keywords", "Year", "Citations"]:
        if col not in df.columns:
            raise ValueError(f"Missing essential column in CSV: {col}")
    df["Title"] = df["Title"].fillna("").astype(str)
    df["Abstract"] = df["Abstract"].fillna("").astype(str)
    df["Keywords"] = df["Keywords"].fillna("").astype(str)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    df["Citations"] = pd.to_numeric(df["Citations"], errors="coerce").fillna(0).astype(int)
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
    # Separate mandatory review from a lower-risk sampling pool. This reduces
    # manual workload without silently treating every borderline case as certain.
    margins = best_scores - second_scores
    high_risk = (best_scores < 0.015) | (margins < 0.002)
    medium_risk = (~high_risk) & ((best_scores < 0.02) | (margins < 0.003))
    # Assign genuinely methodological papers and uncertain field matches to the
    # miscellaneous class instead of forcing a weak substantive-field label.
    ambiguous = high_risk & (raw_fields != "Miscellaneous & Methods")
    clean_df["Raw_Predicted_Field"] = raw_fields
    clean_df["Predicted_Field"] = np.where(
        ambiguous, "Miscellaneous & Methods", raw_fields
    )
    clean_df["Assignment_Basis"] = np.select(
        [ambiguous, raw_fields == "Miscellaneous & Methods"],
        ["Insufficient field evidence; assigned to miscellaneous", "Methods content"],
        default="Clear field evidence",
    )
    clean_df["Is_Ambiguous"] = ambiguous
    clean_df["Needs_Review"] = False
    clean_df["Review_Priority"] = np.select(
        [ambiguous, medium_risk], ["Medium", "Medium"], default="Low"
    )

    print("\n[Diagnostic] Field Classification Distribution:")
    print(clean_df["Predicted_Field"].value_counts().to_string())
    print("-" * 65)
    return clean_df

# ------------------ 趋势图（不变） ------------------
def generate_field_trend_stacked_chart(clean_df, output_path="Field_Trends_Stacked_Area.png"):
    print(">> Generating field trend stacked area chart...")
    min_year = int(clean_df["Year"].min())
    max_year = int(clean_df["Year"].max())
    all_years = range(min_year, max_year + 1)
    trend_pivot = clean_df.pivot_table(index="Year", columns="Predicted_Field", values="Title", aggfunc="count", fill_value=0)
    trend_pivot = trend_pivot.reindex(all_years, fill_value=0)
    trend_proportion = trend_pivot.div(trend_pivot.sum(axis=1), axis=0)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    trend_proportion.plot.area(ax=ax, cmap="tab10", alpha=0.85, linewidth=0.5)
    ax.set_title("Evolution of Research Field Proportions in Top-5 Economics Journals", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Publication Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("Proportion of Publications", fontsize=12, fontweight="bold")
    ax.set_xlim(trend_proportion.index.min(), trend_proportion.index.max())
    ax.set_ylim(0, 1.0)
    plt.legend(title="Research Fields", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> [Saved] Stacked area chart saved to: {output_path}")

# ------------------ 主程序 ------------------
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "Cleaned_Top5_Journals_Dataset.csv")
    df = load_ready_csv(csv_file)
    if not df.empty:
        df = classify_economics_fields(df)
        df.to_csv(os.path.join(base_dir, "Classified_Top5_Journals_Result.csv"), index=False, encoding="utf-8-sig")
        generate_field_trend_stacked_chart(df, os.path.join(base_dir, "Field_Trends_Stacked_Area.png"))
        print("\n>> All tasks completed successfully!")
    else:
        print("[Error] CSV file is empty or missing necessary data.")