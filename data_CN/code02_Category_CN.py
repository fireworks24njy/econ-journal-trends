import io
import os
import re
import sys

import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# 设置中文字体与标准输出编码
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
if sys.version_info >= (3, 7):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
os.environ["PYTHONIOENCODING"] = "utf-8"


# 与英文分类代码完全一致的阈值标准
MIN_ASSIGNMENT_SCORE = 0.015
MIN_ASSIGNMENT_MARGIN = 0.002
MIN_HIGH_CONFIDENCE_SCORE = 0.020
MIN_HIGH_CONFIDENCE_MARGIN = 0.003
MISCELLANEOUS_FIELD = "方法与杂项"


FIELDS_DICT = {
    "发展经济学": (
        "贫困 援助 农业 农村 健康 教育 卫生 营养 福利 "
        "基建 扶贫 发展 转型 低收入 随机实验 劳动力 "
        "农户 乡村 产业结构 共同富裕"
    ),
    "经济史": (
        "历史 制度 殖民 工业化 人口 土地 战争 技术 贸易 "
        "城市化 路径 遗产 变迁 革命 持续性 工业 "
        "技术进步 市场化 传统 城市"
    ),
    "金融学": (
        "股票 债券 期权 股息 投资 融资 银行 信贷 债务 "
        "流动性 波动 风险 对冲 资产定价 公司金融 金融 "
        "资本 资产 杠杆 基金 上市公司 股东 信用 金融风险"
    ),
    "产业组织": (
        "垄断 合谋 寡头 竞争 定价 平台 企业 市场 进入 "
        "壁垒 集中度 加价率 生产率 反垄断 网络效应 "
        "产业 行业 产品 价格 制造业 供应链 企业创新 市场化"
    ),
    "国际经济学": (
        "贸易 关税 汇率 出口 进口 外资 跨国 全球化 协定 "
        "收支 账户 价值链 贸易成本 引力模型 跨国公司 "
        "国际 区域 跨境 开放 产业链 供应链"
    ),
    "劳动经济学": (
        "劳动 工资 就业 失业 移民 教育 培训 收入 性别 "
        "职业 人力资本 最低工资 劳动供给 劳动需求 工资差距 "
        "劳动力 流动 技能 分配 差距"
    ),
    "宏观经济学": (
        "货币 通胀 财政 利率 产出 消费 投资 增长 衰退 "
        "周期 债务 生产率 货币政策 中央银行 总需求 "
        "宏观 预期 稳定 供给 需求 要素生产率"
    ),
    "微观经济学": (
        "效用 均衡 博弈 拍卖 契约 信号 偏好 外部性 "
        "公共品 激励 信息 选择 纳什均衡 机制设计 道德风险 "
        "行为 价格 需求 约束 资源配置 决策"
    ),
    "公共财政": (
        "税收 财政 补贴 保险 福利 支出 征管 再分配 "
        "所得税 增值税 转移支付 公共品 最优税收 超额负担 "
        "失业保险 政府 地方政府 监管 分配 公共 共同富裕 环境"
    ),
    "方法与杂项": (
        "计量 回归 面板 工具变量 固定效应 因果 贝叶斯 "
        "机器学习 双重差分 断点回归 合成控制 广义矩 "
        "渐近分布 稳健性 匹配 实证 证据 指标 动态 "
        "案例研究 人工智能"
    ),
}


# ==========================================
# 1. 中文阈值型零样本领域分类
# ==========================================
def classify_chinese_economics_fields(clean_df):
    """使用TF-IDF和余弦相似度将中文论文归入十大经济学领域。

    文本权重为标题×2、关键词×2、摘要×1。分类规则与英文代码一致：
    1. 先保留余弦相似度最高的原始分类；
    2. 若最高相似度低于0.015或前两名差值低于0.002，则最终归入“方法与杂项”；
    3. 最高相似度不低于0.020且差值不低于0.003的记录标记为高置信分类。
    """
    print(">> Performing threshold-based zero-shot field classification for Chinese texts...")
    clean_df = clean_df.copy()

    field_names = list(FIELDS_DICT.keys())
    field_texts = list(FIELDS_DICT.values())

    # 结构加权：标题×2、关键词×2、摘要×1
    clean_df["Full_Text"] = (
        (clean_df["Title"] + "。") * 2
        + (clean_df["Keywords"] + "。") * 2
        + clean_df["Abstract"]
    )

    token_pattern = re.compile(r"^[\u4e00-\u9fa5a-zA-Z0-9]+$")

    def chinese_tokenizer(text):
        tokens = []
        for word in jieba.cut(str(text)):
            word = word.strip()
            if len(word) < 2:
                continue
            if not token_pattern.match(word):
                continue
            if re.fullmatch(r"[0-9]+", word):
                continue
            tokens.append(word)
        return tokens

    corpus = clean_df["Full_Text"].tolist()
    all_texts = corpus + field_texts

    # 参数尽可能与英文版本保持一致；中文使用结巴分词，不使用英文停用词表。
    vectorizer = TfidfVectorizer(
        tokenizer=chinese_tokenizer,
        token_pattern=None,
        lowercase=False,
        max_features=30000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1,
        norm="l2",
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    doc_vectors = tfidf_matrix[: len(corpus)]
    field_vectors = tfidf_matrix[len(corpus) :]
    similarity_matrix = cosine_similarity(doc_vectors, field_vectors)

    best_field_indices = similarity_matrix.argmax(axis=1)
    sorted_scores = np.sort(similarity_matrix, axis=1)
    best_scores = sorted_scores[:, -1]
    second_scores = sorted_scores[:, -2]
    margins = best_scores - second_scores
    raw_fields = np.array(
        [field_names[index] for index in best_field_indices], dtype=object
    )

    # 两级阈值与英文参考代码完全一致
    high_risk = (
        (best_scores < MIN_ASSIGNMENT_SCORE)
        | (margins < MIN_ASSIGNMENT_MARGIN)
    )
    medium_risk = (~high_risk) & (
        (best_scores < MIN_HIGH_CONFIDENCE_SCORE)
        | (margins < MIN_HIGH_CONFIDENCE_MARGIN)
    )
    ambiguous = high_risk & (raw_fields != MISCELLANEOUS_FIELD)
    high_confidence = (
        (best_scores >= MIN_HIGH_CONFIDENCE_SCORE)
        & (margins >= MIN_HIGH_CONFIDENCE_MARGIN)
    )

    # 保存原始分类、最终分类及置信度信息，供三种分类口径敏感性分析使用
    clean_df["Raw_Predicted_Field"] = raw_fields
    clean_df["Classification_Score"] = best_scores
    clean_df["Classification_Margin"] = margins
    clean_df["Predicted_Field"] = np.where(
        ambiguous, MISCELLANEOUS_FIELD, raw_fields
    )
    clean_df["Assignment_Basis"] = np.select(
        [ambiguous, raw_fields == MISCELLANEOUS_FIELD],
        ["领域证据不足，归入方法与杂项", "方法类内容"],
        default="领域证据较明确",
    )
    clean_df["Is_Ambiguous"] = ambiguous
    clean_df["Is_High_Confidence"] = high_confidence
    clean_df["Needs_Review"] = False
    clean_df["Review_Priority"] = np.select(
        [ambiguous, medium_risk], ["Medium", "Medium"], default="Low"
    )

    print("\n[Diagnostic] Raw classification distribution:")
    print(clean_df["Raw_Predicted_Field"].value_counts().to_string())
    print("\n[Diagnostic] Final classification distribution:")
    print(clean_df["Predicted_Field"].value_counts().to_string())
    print("\n[Diagnostic] Threshold summary:")
    print(f"   Total records: {len(clean_df)}")
    print(f"   Reassigned to miscellaneous: {int(ambiguous.sum())}")
    print(f"   Medium-risk records retained in raw field: {int(medium_risk.sum())}")
    print(f"   High-confidence records: {int(high_confidence.sum())}")
    print(f"   High-confidence share: {high_confidence.mean():.2%}")
    print("-" * 65)
    return clean_df


# ==========================================
# 2. 长期演变趋势堆叠图
# ==========================================
def generate_field_trend_stacked_chart(
    clean_df,
    output_image_path="Chinese_Field_Trends_Stacked_Area.png",
):
    print(">> Generating Chinese field trend stacked area chart...")

    trend_pivot = clean_df.pivot_table(
        index="Year",
        columns="Predicted_Field",
        values="Title",
        aggfunc="count",
        fill_value=0,
    )
    all_years = range(int(clean_df["Year"].min()), int(clean_df["Year"].max()) + 1)
    trend_pivot = trend_pivot.reindex(index=all_years, fill_value=0)
    trend_pivot = trend_pivot.reindex(columns=list(FIELDS_DICT.keys()), fill_value=0)
    trend_proportion = trend_pivot.div(trend_pivot.sum(axis=1), axis=0)

    sns.set_theme(
        style="whitegrid",
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial Unicode MS"],
            "axes.unicode_minus": False,
        },
    )

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    trend_proportion.plot.area(
        ax=ax, cmap="tab10", alpha=0.85, linewidth=0.5
    )
    ax.set_title(
        "中文期刊十大经济学领域研究占比演变趋势",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("出版年份", fontsize=12, fontweight="bold")
    ax.set_ylabel("发文占比", fontsize=12, fontweight="bold")
    ax.set_xlim(trend_proportion.index.min(), trend_proportion.index.max())
    ax.set_ylim(0, 1.0)
    ax.legend(
        title="研究领域",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(output_image_path, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f">> [Saved] Stacked area trend chart saved to: {output_image_path}")


# ==========================================
# 3. 主程序入口
# ==========================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filename = os.path.join(base_dir, "Cleaned_Custom_Dataset.csv")

    try:
        print(f">> Loading pre-formatted CSV data from {csv_filename}...")
        df_clean = pd.read_csv(csv_filename, encoding="utf-8-sig")

        required_columns = ["Title", "Abstract", "Keywords", "Year"]
        for column in required_columns:
            if column not in df_clean.columns:
                raise ValueError(f"Missing essential column in CSV: {column}")

        for column in ["Title", "Abstract", "Keywords"]:
            df_clean[column] = df_clean[column].fillna("").astype(str)

        df_clean["Year"] = pd.to_numeric(df_clean["Year"], errors="coerce")
        df_clean = df_clean[df_clean["Year"].between(2020, 2025)].copy()
        df_clean["Year"] = df_clean["Year"].astype(int)

        if df_clean.empty:
            print("[Error] CSV file is empty or contains no records from 2020 to 2025.")
            sys.exit(1)

        df_clean = classify_chinese_economics_fields(df_clean)

        export_csv_path = os.path.join(
            base_dir, "Cleaned_Custom_Dataset_Classified.csv"
        )
        df_clean.to_csv(export_csv_path, index=False, encoding="utf-8-sig")
        print(f">> Exported classified dataset to: {export_csv_path}")

        chart_path = os.path.join(
            base_dir, "Chinese_Field_Trends_Stacked_Area.png"
        )
        generate_field_trend_stacked_chart(df_clean, chart_path)
        print("\n>> All classification and trend tasks completed successfully!")

    except FileNotFoundError:
        print(
            f"[Error] File '{csv_filename}' not found. "
            "Please run the data preprocessing step first."
        )
