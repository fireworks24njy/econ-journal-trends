import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'

from collections import Counter
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import STOPWORDS, WordCloud
from sentence_transformers import SentenceTransformer



# ==========================================
# 1. 直接加载已规范化的 CSV 数据
# ==========================================


def load_ready_csv(file_path="Cleaned_Top5_Journals_Dataset.csv"):
    print(f">> Loading pre-formatted CSV data from {file_path}...")
    df = pd.read_csv(file_path, encoding="utf-8-sig")

    # 确保必要列存在并处理空值
    for col in ["Title", "Abstract", "Keywords", "Year", "Citations"]:
        if col not in df.columns:
            raise ValueError(f"Missing essential column in CSV: {col}")

    df["Title"] = df["Title"].fillna("").astype(str)
    df["Abstract"] = df["Abstract"].fillna("").astype(str)
    df["Keywords"] = df["Keywords"].fillna("").astype(str)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    df["Citations"] = (
        pd.to_numeric(df["Citations"], errors="coerce").fillna(0).astype(int)
    )

    # 构筑加权全文本（标题x2 + 关键词x2 + 摘要）
    df["Full_Text"] = (
        (df["Title"] + ". ") * 2 + (df["Keywords"] + ". ") * 2 + df["Abstract"]
    )
    print(f">> Successfully loaded {len(df)} valid records.")
    return df


# ==========================================
# 2. 零样本语义相似度领域分类 (Zero-Shot Classification)
# ==========================================


# def classify_economics_fields(clean_df):
#     """基于 TF-IDF 余弦相似度，将文献自动归入论文中定义的 10 大经济学标准领域。"""
#     print(
#         ">> Performing Zero-Shot economic field classification (10 Standard Fields)..."
#     )
#     fields_dict = {
#         "Development Economics": "poverty aid agricultural health randomized trial field experiment developing country growth education sanitation",
#         "Economic History": "historical data economic history cliometrics long term historical development past century nineteenth century",
#         "Finance": "asset pricing portfolio stock stock market bond liquidity risk dividend equity option corporate finance",
#         "Industrial Organization": "firm market structure antitrust competition entry monopoly price oligopoly productivity markup",
#         "International Economics": "trade tariff export import international trade exchange rate globalization multinational gravity model",
#         "Labor Economics": "labor wage employment worker job earnings inequality skill education human capital migration",
#         "Macroeconomics": "macroeconomic monetary inflation fiscal dsge business cycle interest rate output central bank consumption",
#         "Microeconomics": "theory equilibrium game choice preference utility mechanism auction information contract bargaining",
#         "Public Finance": "tax taxation public expenditure government spending subsidy welfare revenue fiscal policy externality",
#         "Miscellaneous & Methods": "estimation identification instrumental variable asymptotic panel data method theory econometric miscellaneous",
#     }

#     field_names = list(fields_dict.keys())
#     field_texts = list(fields_dict.values())
#     corpus = clean_df["Full_Text"].tolist()

#     vectorizer = TfidfVectorizer(
#         stop_words="english", max_features=8000, ngram_range=(1, 2)
#     )
#     all_texts = corpus + field_texts
#     tfidf_matrix = vectorizer.fit_transform(all_texts)

#     doc_vectors = tfidf_matrix[: len(corpus)]
#     field_vectors = tfidf_matrix[len(corpus) :]

#     similarity_matrix = cosine_similarity(doc_vectors, field_vectors)
#     best_field_indices = similarity_matrix.argmax(axis=1)

#     clean_df["Predicted_Field"] = [field_names[idx] for idx in best_field_indices]

#     print("\n[Diagnostic] Field Classification Distribution:")
#     print(clean_df["Predicted_Field"].value_counts().to_string())
#     print("-" * 65)
#     return clean_df



def classify_economics_fields(clean_df):

    fields = {
        "Development Economics": "This field explores why persistent poverty and inequality remain entrenched in low-income regions, relying heavily on randomized controlled trials and quasi-experimental methods to measure the causal impacts of microfinance, health interventions, school construction, and agricultural technology adoption on household welfare and long-term growth trajectories.",
        "Economic History": "This field reconstructs the long-run evolution of economies by quantifying how past institutions, colonial legacies, warfare, and pandemic shocks have produced divergent development paths, using cliometric techniques and historical microdata to trace the persistent effects of events like the Industrial Revolution on contemporary demographic and income patterns.",
        "Finance": "This field investigates how capital is priced and allocated under uncertainty, centering on the joint dynamics of risk premia, liquidity, volatility, and arbitrage in equity, bond, and derivative markets, while also scrutinizing corporate financing decisions, banking fragility, and the destabilizing feedback loops that precipitate systemic crises.",
        "Industrial Organization": "This field dissects strategic interactions among profit-maximizing firms, focusing on how market power, entry deterrence, price discrimination, and vertical restraints shape observed outcomes, and employs structural models to evaluate the welfare consequences of mergers, antitrust interventions, and the regulatory challenges posed by multi-sided digital platforms.",
        "International Economics": "This field analyzes the causes and consequences of cross-border flows of goods, services, and capital, building upon Ricardian and Heckscher-Ohlin frameworks while incorporating firm-heterogeneity and global value chains to precisely quantify the distributional effects of tariffs, exchange rate movements, and trade liberalization agreements on domestic employment and productivity.",
        "Labor Economics": "This field studies the determination of wages, employment, and human capital returns, emphasizing how supply-and-demand forces interact with institutional frictions such as minimum wage laws, union coverage, and immigration policies to generate observed inequality, and increasingly deploys natural experiments to identify the labor-market impacts of automation and childcare access.",
        "Macroeconomics": "This field constructs general-equilibrium models of the entire economy, incorporating micro-founded expectations, price stickiness, and financial frictions to simulate how monetary policy rules, fiscal stimulus packages, and productivity disturbances propagate through business cycles, ultimately affecting inflation, aggregate output, and the natural rate of unemployment.",
        "Microeconomics": "This field provides the foundational theory of individual choice and strategic interaction, developing rigorous game-theoretic and mechanism-design paradigms to solve adverse selection, moral hazard, and coordination failures, which are then applied to redesign auctions, matching markets, and contracts in both private and public sectors.",
        "Public Finance": "This field evaluates the efficiency and equity implications of government intervention, deriving optimal tax schedules and public-good provision rules in the presence of externalities and behavioral biases, while empirically measuring the incidence of social insurance programs, environmental taxes, and education subsidies across heterogeneous households.",
        "Miscellaneous & Methods": "This field advances the quantitative toolkit of the discipline, specializing in the development and refinement of econometric identification strategies—including instrumental variables, regression discontinuity, difference-in-differences, and high-dimensional machine learning—to extract credible causal relationships from observational data and complex panel structures."
    }

    model = SentenceTransformer("BAAI/bge-base-en-v1.5")

    docs = clean_df["Full_Text"].fillna("").tolist()
    labels = list(fields.keys())

    doc_emb = model.encode(docs,
                           normalize_embeddings=True,
                           batch_size=32,
                           show_progress_bar=False)

    field_emb = model.encode(list(fields.values()),
                             normalize_embeddings=True)

    sim = cosine_similarity(doc_emb, field_emb)

    clean_df["Similarity"] = sim.max(axis=1)
    clean_df["Predicted_Field"] = [
        labels[i] if s >= 0.5 else "Miscellaneous"  # 相似度低于0.3则归入杂项
        for i, s in zip(sim.argmax(axis=1), sim.max(axis=1))
    ]

    print(clean_df["Predicted_Field"].value_counts())

    return clean_df

# ==========================================
# 3. 任务一：ESI 加权热点词云渲染
# ==========================================


def generate_topic_wordcloud(
    clean_df, output_image_path="Top5_Journals_ESI_Weighted_Topics.png"
):
    """计算 ESI 阶梯权重并生成高影响力热点领域词云。"""
    print(">> Calculating ESI percentile weights for word cloud...")
    clean_df["Percentile"] = clean_df.groupby("Year")["Citations"].rank(
        pct=True, method="max"
    )
    clean_df["ESI_Weight"] = np.select(
        [clean_df["Percentile"] >= 0.99, clean_df["Percentile"] >= 0.90],
        [10.0, 5.0],
        default=1.0,
    )

    academic_words = {
        "article",
        "paper",
        "study",
        "model",
        "results",
        "data",
        "using",
        "used",
        "use",
        "find",
        "found",
        "show",
        "shown",
        "provide",
        "effect",
        "effects",
        "evidence",
        "analysis",
        "based",
        "two",
        "new",
        "one",
        "within",
        "also",
        "may",
        "via",
        "table",
        "figure",
        "however",
        "often",
        "across",
        "set",
        "suggest",
        "propose",
        "well",
        "rather",
        "whether",
        "including",
        "overall",
        "important",
        "simple",
        "different",
        "without",
        "among",
        "author",
        "year",
        "sample",
        "firm",
        "firms",
        "market",
        "markets",
        "time",
        "high",
        "low",
        "large",
        "small",
        "years",
        "recent",
        "journal",
        "quarterly",
        "review",
        "economic",
        "economics",
        "business",
    }
    stopwords = set(STOPWORDS) | academic_words

    weighted_freq = Counter()
    for _, row in clean_df.iterrows():
        if pd.isna(row["Full_Text"]):
            continue
        words = re.findall(r"\b[a-z]{3,30}\b", row["Full_Text"].lower())
        for w in words:
            if w not in stopwords:
                weighted_freq[w] += row["ESI_Weight"]

    print(">>> Rendering ESI-weighted word cloud...")
    wc = WordCloud(
        width=1600,
        height=1000,
        background_color="white",
        colormap="inferno",
        max_words=150,
        min_font_size=12,
        max_font_size=160,
        random_state=42,
        prefer_horizontal=0.85,
    )
    wc.generate_from_frequencies(weighted_freq)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.suptitle(
        "Hot Research Topics in Top-5 Economics Journals\n(ESI Time-Normalized Weighted)",
        fontsize=20,
        fontweight="bold",
        color="#2c3e50",
        y=0.94,
        ha="center",
    )
    plt.subplots_adjust(top=0.86, bottom=0.04, left=0.04, right=0.96)
    plt.savefig(
        output_image_path,
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.3,
    )
    plt.close()
    print(f">> [Task 1 Saved] Word cloud saved to: {output_image_path}")


# ==========================================
# 4. 任务二：随年份变化趋势堆叠图 (Stacked Area Chart)
# ==========================================


def generate_field_trend_stacked_chart(
    clean_df, output_image_path="Field_Trends_Stacked_Area.png"
):
    print(">> Generating field trend stacked area chart...")

    trend_pivot = clean_df.pivot_table(
        index="Year",
        columns="Predicted_Field",
        values="Title",
        aggfunc="count",
        fill_value=0,
    )
    trend_proportion = trend_pivot.div(trend_pivot.sum(axis=1), axis=0)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    trend_proportion.plot.area(
        ax=ax, cmap="tab10", alpha=0.85, linewidth=0.5
    )

    ax.set_title(
        "Evolution of Research Field Proportions in Top-5 Economics Journals",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Publication Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("Proportion of Publications", fontsize=12, fontweight="bold")
    ax.set_xlim(trend_proportion.index.min(), trend_proportion.index.max())
    ax.set_ylim(0, 1.0)

    plt.legend(
        title="Research Fields",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        fontsize=10,
    )
    plt.tight_layout()

    plt.savefig(output_image_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> [Task 2 Saved] Stacked area trend chart saved to: {output_image_path}")


# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    csv_filename = "Cleaned_Top5_Journals_Dataset.csv"

    # Step 1: 加载已规范化的 CSV
    df_clean = load_ready_csv(csv_filename)

    if not df_clean.empty:
        # Step 2: 领域分类 (更新为 10 大标准领域)
        df_clean = classify_economics_fields(df_clean)

        # 导出带有分类标签的新表以便复查
        df_clean.to_csv(
            "Classified_Top5_Journals_Result.csv", index=False, encoding="utf-8-sig"
        )

        # Step 3: 绘制 ESI 加权热点词云
        generate_topic_wordcloud(df_clean)

        # Step 4: 绘制年度领域演变趋势堆叠图
        generate_field_trend_stacked_chart(df_clean)

        print("\n>> All tasks completed successfully!")
    else:
        print("[Error] CSV file is empty or missing necessary data.")