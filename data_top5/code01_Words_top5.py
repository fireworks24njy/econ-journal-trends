from collections import Counter
import glob
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from wordcloud import STOPWORDS, WordCloud

import nltk
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.stem import WordNetLemmatizer

# ==========================================
# 0. 相对路径注册与 WordNet 离线加载
# ==========================================
def init_wordnet():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    local_nltk_dir = os.path.join(base_dir, "nltk_data")
    
    if local_nltk_dir not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_dir)
    
    expected_wordnet_path = os.path.join(local_nltk_dir, "corpora", "wordnet")
    if not os.path.exists(expected_wordnet_path):
        print("\n" + "!" * 60)
        print("[WARNING] WordNet folder not found!")
        print(f"Expected path: {expected_wordnet_path}")
        print("Please check directory structure: nltk_data/corpora/wordnet")
        print("!" * 60 + "\n")

    try:
        lem = WordNetLemmatizer()
        lem.lemmatize("policies")
        print(">> [System] WordNet loaded successfully!")
        return lem
    except Exception as e:
        print(f"\n[Error] WordNet initialization failed: {e}")
        raise e

lemmatizer = init_wordnet()

# 全局超参数设定
MIN_FREQ, TITLE_W, KEY_W = 10, 2, 2

# ==========================================
# 定制停用词库（保持不变）
# ==========================================
GENERIC_STOP = {
    # 1. 深度拦截的抽象词 / 泛用趋势词 / 数据后缀词
    "growth", "grow", "grows", "growing", "grown",
    "dynamic", "dynamics", "rate", "rates", "change", "changes", "changed", "changing",
    "level", "levels", "trend", "trends", "long", "short", "high", "higher", "highest", "low", "lower", "lowest",
    "large", "larger", "largest", "small", "smaller", "smallest",
    "strong", "stronger", "strongest", "weak", "weaker", "weakest",
    "better", "best", "worse", "worst", "good", "bad", "great", "greater", "greatest",
    "main", "major", "minor", "key", "important", "importance", "overall", "relative", "relatively",
    "absolute", "recent", "recently", "significant", "significantly", "insignificant",
    "positive", "positively", "negative", "negatively", "new", "various", "several",
    "same", "different", "differences", "similar", "general", "specific", "broad", "narrow",
    "likely", "unlikely", "potential", "primary", "secondary", "substantial", "substantially",

    # 2. 泛用动作 / 趋势 / 结果动词
    "reduce", "reduces", "reduced", "reducing", "reduction", "reductions",
    "increase", "increases", "increased", "increasing",
    "decline", "declines", "declining", "declined",
    "improve", "improves", "improved", "improving", "improvement", "improvements",
    "enhance", "enhances", "enhanced", "enhancing",
    "affect", "affects", "affected", "affecting",
    "lead", "leads", "led", "leading",
    "drive", "drives", "driven", "driving",
    "use", "using", "used", "uses",
    "find", "finds", "found", "finding", "findings",
    "show", "shows", "shown", "showing",
    "provide", "provides", "provided", "providing",
    "examine", "examines", "examined", "examining",
    "analyze", "analyzes", "analyzed", "analyzing",
    "investigate", "investigates", "investigated", "investigating",
    "explore", "explores", "explored", "exploring",
    "discuss", "discusses", "discussed", "discussing",
    "consider", "considers", "considered", "considering",
    "suggest", "suggests", "suggested", "suggesting",
    "reveal", "reveals", "revealed", "revealing",

    # 3. 经济学泛用词 (非特定领域)
    "market", "markets", "firm", "firms", "company", "companies", "stock", "stocks", "equity", "equities",
    "policy", "policies", "economic", "economics", "economy", "economies", "business", "political", "politics",
    "agent", "agents", "consumer", "consumers", "household", "households", "worker", "workers", "employee", "employees",
    "employer", "employers", "investor", "investors", "trader", "traders", "bank", "banks", "banking", "banking_sector",
    "price", "prices", "pricing", "cost", "costs", "return", "returns", "profit", "profits", "profitability",
    "shock", "shocks", "spillover", "spillovers", "trade", "trading", "traded", "financial", "finance",
    "industry", "industries", "industrial", "sector", "sectors", "asset", "assets", "capital", "fund", "funds", "funding",

    # 4. 计量方法与实证套话
    "difference", "differences", "different", "difference_in_differences", "did", "regression", "panel",
    "estimate", "estimates", "estimated", "estimating", "estimation", "estimations", "estimator", "estimators",
    "measure", "measures", "measured", "measuring", "measurement", "measurements", "metric", "metrics",
    "variable", "variables", "factor", "factors", "model", "models", "modeling", "modeled", "specification", "specifications",
    "sample", "samples", "sampling", "data", "dataset", "datasets", "empirical", "evidence",
    "result", "results", "outcome", "outcomes", "effect", "effects", "impact", "impacts",
    "relationship", "relationships", "relation", "relations", "association", "associations", "associated",
    "link", "links", "linked", "channel", "channels", "mechanism", "mechanisms", "role", "roles", "driver", "drivers",
    "variance", "variation", "variations", "bias", "robust", "robustness", "significance", "heterogeneity", "heterogeneous", "causal", "causality",

    # 5. 论文结构与泛用表达套话
    "abstract", "paper", "article", "study", "studies", "author", "authors", "journal", "review", "quarterly",
    "university", "press", "table", "figure", "appendix", "section", "implication", "implications", "literature",
    "contribution", "contributions", "aim", "aims", "objective", "objectives", "conclusion", "conclusions",
    "introduction", "framework", "perspective", "approach", "method", "methods", "methodology", "performance", "performances",
    "behavior", "behaviors", "behavioral", "choice", "choices", "decision", "decisions",

    # 6. 时间 / 地理 / 其它泛词
    "time", "period", "periods", "year", "years", "annual", "century", "date",
    "country", "countries", "nation", "nations", "national", "international", "global", "local", "regional", "state", "states",
    "group", "groups", "individual", "individuals", "also", "however", "thus", "therefore", "furthermore", "moreover", "although",
    "within", "among", "across", "between", "via", "well", "may", "can", "could", "would", "should", "might", "one", "two",
    "three", "first", "second", "third", "both", "either", "neither", "american",
    "information", "social", "optimal", "network", "networks", "design", "set", "sets", "preference", "preferences",
    "type", "types", "case", "cases", "term", "terms", "form", "forms", "part", "parts", "point", "points", "way", "ways", "degree", "value", "values"
}

# 全局整合停用词表
STOP = set(STOPWORDS) | GENERIC_STOP


# ==========================================
# 1. 数据加载与论文分流预处理
# ==========================================
def _export_if_exists(df_subset, filename):
    col_map = {"TI": "Title", "DT": "Document_Type", "SO": "Journal", "PY": "Year", "AU": "Author", "DI": "DOI", "AB": "Abstract"}
    if not df_subset.empty:
        df_subset.rename(columns=col_map).to_csv(filename, index=False, encoding="utf-8-sig")
        print(f"   [Filter] Exported {len(df_subset)} records to: {filename}")


def load_and_preprocess_wos_data(file_pattern="*-*.txt"):
    all_files = glob.glob(file_pattern)
    if not all_files:
        print(f"[Error] No files matching '{file_pattern}' found.")
        return pd.DataFrame()

    print(f">> Loading {len(all_files)} data file(s)...")
    dfs = []
    for f in all_files:
        try:
            dfs.append(pd.read_csv(f, sep="\t", encoding="utf-8-sig", dtype=str, on_bad_lines="skip"))
        except UnicodeDecodeError:
            dfs.append(pd.read_csv(f, sep="\t", encoding="utf-8", dtype=str, errors="ignore"))

    df = pd.concat(dfs, ignore_index=True)
    
    for col in ["AB", "DT", "TI", "PY", "AU", "AF", "DI", "SO"]:
        if col not in df.columns:
            df[col] = np.nan

    missing_abs = df["AB"].isna() | (df["AB"].str.strip() == "")
    missing_meta = (
        (df["TI"].isna() | (df["TI"].str.strip() == "")) |
        (df["PY"].isna() | (df["PY"].str.strip() == "")) |
        ((df["AU"].isna() | (df["AU"].str.strip() == "")) & (df["AF"].isna() | (df["AF"].str.strip() == "")))
    )

    _export_if_exists(df[missing_abs], "Missing_Abstracts_Records.csv")
    _export_if_exists(df[~missing_abs & missing_meta], "Incomplete_Metadata_Records.csv")

    df_valid = df[~missing_abs & ~missing_meta].copy()
    if df_valid.empty:
        return pd.DataFrame()

    empty_str, empty_int = pd.Series("", index=df_valid.index), pd.Series(0, index=df_valid.index)
    de, id_col = df_valid.get("DE", empty_str).fillna(""), df_valid.get("ID", empty_str).fillna("")
    wc, sc = df_valid.get("WC", empty_str).fillna(""), df_valid.get("SC", empty_str).fillna("")
    cit = df_valid.get("TC", df_valid.get("Z9", empty_int))

    clean_df = pd.DataFrame({
        "Title": df_valid["TI"].str.strip(),
        "Abstract": df_valid["AB"].str.strip(),
        "Year": pd.to_numeric(df_valid["PY"], errors="coerce"),
        "Author": df_valid["AU"].fillna(df_valid["AF"]).str.strip(),
        "Keywords": (de + "; " + id_col).str.strip("; "),
        "JEL_Category": (wc + "; " + sc).str.replace(r"(;\s*)+", "; ", regex=True).str.strip("; "),
        "Citations": pd.to_numeric(cit, errors="coerce").fillna(0).astype(int),
    })

    valid_year = clean_df["Year"].between(2020, 2025, inclusive="both")
    if (~valid_year).any():
        print(f"   [Constraint] Dropped {(~valid_year).sum()} records outside target years.")

    clean_df = clean_df[valid_year].copy()
    clean_df["Year"] = clean_df["Year"].astype(int)
    print(f">> Retained {len(clean_df)} valid records for analysis.")
    return clean_df


# ==========================================
# 2. NLP 工具与自动短语挖掘
# ==========================================
def lemmatize_word(w):
    return lemmatizer.lemmatize(lemmatizer.lemmatize(w, pos="n"), pos="v")


def build_auto_phrases(df, top_n_bigrams=300, top_n_trigrams=100, min_colloc_freq=5):
    print(">> Mining domain phrases via NLTK Collocations & Strict Stopword Filter...")
    corpus_series = df["Title"].fillna("") + ". " + df["Keywords"].fillna("") + ". " + df["Abstract"].fillna("")
    corpus_tokens = [w for text in corpus_series for w in re.findall(r"\b[a-z]{2,40}\b", str(text).lower())]

    bcf = BigramCollocationFinder.from_words(corpus_tokens)
    bcf.apply_freq_filter(min_colloc_freq)
    raw_bigrams = bcf.nbest(BigramAssocMeasures.likelihood_ratio, top_n_bigrams)

    tcf = TrigramCollocationFinder.from_words(corpus_tokens)
    tcf.apply_freq_filter(min_colloc_freq)
    raw_trigrams = tcf.nbest(TrigramAssocMeasures.likelihood_ratio, top_n_trigrams)

    valid_phrases = []

    for tuple_phrase in raw_trigrams + raw_bigrams:
        lemmatized_tuple = [lemmatize_word(w) for w in tuple_phrase]
        if all(w not in STOP and len(w) >= 3 for w in lemmatized_tuple):
            phrase_str = " ".join(lemmatized_tuple)
            valid_phrases.append(phrase_str)

    print(f">> Mined {len(valid_phrases)} clean domain phrases.")
    return valid_phrases


# ==========================================
# 3. 文本清洗与分词统计（修改：单字词与短语独立）
# ==========================================
def norm_text(txt, discovered_phrases):
    """将文本中发现的短语替换为下划线形式，返回所有 token（含下划线短语和普通单词）"""
    txt = txt.lower()
    for p in sorted(discovered_phrases, key=len, reverse=True):
        if p in txt:
            txt = txt.replace(p, p.replace(" ", "_"))

    cleaned_words = []
    for w in re.findall(r"\b[a-z_]{2,40}\b", txt):
        if "_" in w:
            cleaned_words.append(w)
        elif len(w) >= 3:
            lemma = lemmatize_word(w)
            if lemma not in STOP:
                cleaned_words.append(lemma)
    return cleaned_words


def tokenize(df, discovered_phrases):
    """
    独立统计单字词和短语：
      - 单字词：直接对原始文本（不替换短语）进行词元化、词形还原、停用词过滤。
      - 短语：替换短语后，仅提取带下划线的 token。
    """
    print(">> Tokenizing and counting term frequencies...")
    single_counter = Counter()   # 单字词计数器
    phrase_counter = Counter()   # 短语计数器

    # ---- 1. 单字词统计：使用原始文本，不替换短语 ----
    # 拼接标题、关键词、摘要（应用权重）
    raw_text = (df["Title"].fillna("") + ". ") * TITLE_W + (df["Keywords"].fillna("") + ". ") * KEY_W + df["Abstract"].fillna("")
    for txt in raw_text:
        # 直接提取单词，不做任何替换
        for w in re.findall(r"\b[a-z]{2,40}\b", txt.lower()):
            lemma = lemmatize_word(w)
            if lemma not in STOP and len(lemma) >= 3:
                single_counter[lemma] += 1

    # ---- 2. 短语统计：替换短语后，只提取带下划线的 token ----
    weighted_text = (df["Title"].fillna("") + ". ") * TITLE_W + (df["Keywords"].fillna("") + ". ") * KEY_W + df["Abstract"].fillna("")
    for txt in weighted_text:
        # 使用 norm_text 进行短语替换并返回所有 token
        for token in norm_text(txt, discovered_phrases):
            if "_" in token:  # 只关心短语
                phrase_counter[token] += 1

    # 过滤低频词
    filtered_single = Counter({k: v for k, v in single_counter.items() if v >= MIN_FREQ})
    filtered_phrases = Counter({k: v for k, v in phrase_counter.items() if v >= MIN_FREQ})

    return filtered_single, filtered_phrases


# ==========================================
# 4. 控制台打印 (短语Top5 + 单词Top10) 与全量词云展示
# ==========================================
def generate_output(freq_all, freq_phrases, clean_df):
    """
    freq_all: 单字词计数器（不包含短语成分）
    freq_phrases: 短语计数器（仅含带下划线的短语）
    """
    # 导出清洗数据集
    clean_df[["Year", "Title", "Author", "Keywords", "JEL_Category", "Citations", "Abstract"]].to_csv(
        "Cleaned_Top5_Journals_Dataset.csv", index=False, encoding="utf-8-sig"
    )

    # 1. 提取 Top 5 纯短语 (必须带有下划线)
    top5_phrases = []
    for p, count in freq_phrases.most_common(50):
        if "_" in p:
            top5_phrases.append((p.replace("_", " "), count))
            if len(top5_phrases) == 5:
                break

    # 2. 提取 Top 10 纯单词 (不含下划线)
    top10_single_words = []
    for w, count in freq_all.most_common(200):
        if "_" not in w and w not in STOP:
            top10_single_words.append((w, count))
            if len(top10_single_words) == 10:
                break

    # 控制台格式化双榜单输出
    print("\n" + "=" * 55)
    print("      Top 5 Domain-Specific Key Phrases")
    print("=" * 55)
    for rank, (term, score) in enumerate(top5_phrases, 1):
        print(f" #{rank:<4} | {term:<30} | {score:<10}")

    print("\n" + "=" * 55)
    print("      Top 10 Domain-Specific Single Words")
    print("=" * 55)
    for rank, (term, score) in enumerate(top10_single_words, 1):
        print(f" #{rank:<4} | {term:<30} | {score:<10}")
    print("=" * 55 + "\n")

    # 3. 渲染词云图 (使用单字词计数器，短语显示时替换下划线为空格)
    wc_frequencies = {k.replace("_", " "): v for k, v in freq_all.items() if k not in STOP}
    
    wc = WordCloud(
        width=1600, height=1000, background_color="white", colormap="inferno", 
        max_words=120, min_font_size=12, max_font_size=160, relative_scaling=0.5, 
        random_state=42, prefer_horizontal=0.85
    ).generate_from_frequencies(wc_frequencies)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.suptitle("Top Research Topics in Top-5 Economics Journals\n(Automated N-gram Phrase Analysis)", 
                 fontsize=20, fontweight="bold", color="#2c3e50", y=0.94, ha="center")
    plt.subplots_adjust(top=0.86, bottom=0.04, left=0.04, right=0.96)
    
    plt.savefig("Top5_Economics_WordCloud.png", format="png", dpi=300, bbox_inches="tight", pad_inches=0.3)
    print(">> [System] Full WordCloud saved to 'Top5_Economics_WordCloud.png'")
    plt.show()


# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    file_pattern = "*-*.txt"
    if not glob.glob(file_pattern):
        print(f"[Error] No files matching '{file_pattern}' found in current directory.")
    else:
        df_clean = load_and_preprocess_wos_data(file_pattern)
        if not df_clean.empty:
            phrases = build_auto_phrases(df_clean, top_n_bigrams=300, top_n_trigrams=100, min_colloc_freq=5)
            word_counts, phrase_counts = tokenize(df_clean, phrases)
            generate_output(word_counts, phrase_counts, df_clean)