# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download

_MODEL = None

def get_model():
    """Load all-MiniLM-L6-v2 from ModelScope for English semantic matching."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    print(">> Downloading/Loading all-MiniLM-L6-v2 from ModelScope...")
    try:
        # 使用 ModelScope 下载模型到本地缓存目录
        model_dir = snapshot_download('AI-ModelScope/all-MiniLM-L6-v2')
        _MODEL = SentenceTransformer(model_dir)
        print(">> [OK] Model loaded successfully from ModelScope")
        return _MODEL
    except Exception as e:
        print(f"   [FAIL] Failed to load model from ModelScope: {e}")
        sys.exit(1)

def load_ready_csv(file_path="Cleaned_Top5_Journals_Dataset.csv"):
    """Load and preprocess the CSV dataset."""
    print(f">> Loading dataset from {file_path}...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    for col in ["Title", "Abstract", "Keywords", "Year", "Citations"]:
        if col not in df.columns:
            raise ValueError(f"Missing essential column: {col}")
            
    df["Title"] = df["Title"].fillna("").astype(str)
    df["Abstract"] = df["Abstract"].fillna("").astype(str)
    df["Keywords"] = df["Keywords"].fillna("").astype(str)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype(int)
    df["Citations"] = pd.to_numeric(df["Citations"], errors="coerce").fillna(0).astype(int)
    print(f">> Successfully loaded {len(df)} records.")
    return df

def encode_adaptive_texts(texts, model):
    """Encode long texts adaptively with chunking and max pooling."""
    max_len = model.max_seq_length
    tokenizer = model.tokenizer
    embeddings = []
    
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) <= max_len:
            emb = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
            embeddings.append(emb[0])
        else:
            chunk_embs = []
            for i in range(0, len(tokens), max_len):
                chunk_tokens = tokens[i:i+max_len]
                chunk_text = tokenizer.decode(chunk_tokens)
                chunk_emb = model.encode([chunk_text], normalize_embeddings=False, show_progress_bar=False)
                chunk_embs.append(chunk_emb[0])
            # 使用分段 max-pooling 聚合长文本向量
            max_emb = np.max(chunk_embs, axis=0)
            norm_emb = max_emb / np.linalg.norm(max_emb)
            embeddings.append(norm_emb)
            
    return np.array(embeddings)

def classify_economics_fields(clean_df, similarity_threshold=0.35):

    print(">> Performing zero-shot field classification (ModelScope Model + Keywords)...")
    fields = {
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

    model = get_model()
    
    docs = (clean_df["Title"].fillna("") + ". ") * 2 + (clean_df["Keywords"].fillna("") + ". ") * 2 + clean_df["Abstract"].fillna("")
    docs = docs.tolist()

    labels = list(fields.keys())

    # 编码论文和类别描
    doc_emb = encode_adaptive_texts(docs, model)
    field_emb = model.encode(list(fields.values()), normalize_embeddings=True, show_progress_bar=False)

    # 计算相似度
    sim = cosine_similarity(doc_emb, field_emb)
    best_indices = sim.argmax(axis=1)
    max_sims = sim.max(axis=1)

    clean_df["Similarity"] = max_sims
    clean_df["Predicted_Field"] = [
        labels[i] if s >= similarity_threshold else "Miscellaneous & Methods"
        for i, s in zip(best_indices, max_sims)
    ]

    print("\n【Field Distribution】")
    print(clean_df["Predicted_Field"].value_counts())
    print(f"\nAverage Max Similarity: {max_sims.mean():.4f} (Threshold used: {similarity_threshold})")
    return clean_df

def generate_field_trend_stacked_chart(clean_df, output_path="Field_Trends_Stacked_Area_ModelScope.png"):
    """Generate and save the stacked area chart for field trends."""
    print(">> Generating field trend stacked chart...")
    min_year = int(clean_df["Year"].min())
    max_year = int(clean_df["Year"].max())
    all_years = range(min_year, max_year + 1)
    
    trend_pivot = clean_df.pivot_table(index="Year", columns="Predicted_Field", values="Title", aggfunc="count", fill_value=0)
    trend_pivot = trend_pivot.reindex(all_years, fill_value=0)
    trend_proportion = trend_pivot.div(trend_pivot.sum(axis=1), axis=0)
    
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    trend_proportion.plot.area(ax=ax, cmap="tab10", alpha=0.85, linewidth=0.5)
    
    ax.set_title("Evolution of Research Field Proportions (ModelScope + Keywords)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Publication Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("Proportion of Publications", fontsize=12, fontweight="bold")
    ax.set_xlim(trend_proportion.index.min(), trend_proportion.index.max())
    ax.set_ylim(0, 1.0)
    plt.legend(title="Research Fields", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> [Saved] Chart saved to: {output_path}")

if __name__ == "__main__":
    csv_file = "Cleaned_Top5_Journals_Dataset.csv"
    df = load_ready_csv(csv_file)
    if not df.empty:
        df = classify_economics_fields(df, similarity_threshold=0.35)
        df.to_csv("Classified_Top5_Journals_ModelScope.csv", index=False, encoding="utf-8-sig")
        generate_field_trend_stacked_chart(df)
        print("\n>> All tasks completed successfully!")
    else:
        print("[Error] CSV file is empty.")