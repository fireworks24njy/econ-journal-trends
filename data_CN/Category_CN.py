import io
import os
import re
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 设置中文字体与标准输出编码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
if sys.version_info >= (3, 7):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

# ==========================================
# 1. 中文零样本语义相似度领域分类 (10大标准领域)
# ==========================================
def classify_chinese_economics_fields(clean_df):
    """基于 TF-IDF 与余弦相似度，将中文文献自动归入 10 大经济学标准领域。
    采用文本结构加权：标题 * 2、关键词 * 2、摘要 * 1。
    """
    print(">> Performing Zero-Shot economic field classification for Chinese texts...")
    fields_dict = {
    "发展经济学": "贫困 援助 农业 健康 教育 卫生设施 农村 乡村 基础设施 随机实验 田野调查 结构转型 低收入 家庭福利 发展中国家",
    "经济史": "经济史 历史数据 档案 清代 民国 古代 近代 十九世纪 工业革命 殖民 路径依赖 历史变迁 历史冲击",
    "金融学": "金融 金融市场 资本市场 资产定价 投资组合 股票 债券 银行 信贷 融资 流动性 风险 波动 股息 期权 公司金融 债务 对冲 系统性风险 交易",
    "产业组织": "企业 市场结构 反垄断 市场竞争 市场进入 垄断 价格 合谋 寡头 生产率 加价率 规模经济 网络效应 平台 数字平台 供应链",
    "国际经济学": "国际贸易 出口 进口 关税 汇率 跨国投资 对外投资 外商投资 价值链 引力模型 区域协定 国际收支 全球化 跨国公司 一带一路",
    "劳动经济学": "工资 就业 失业 劳动力 劳动者 收入不平等 教育回报 人力资本 移民 培训 劳动供给 劳动需求 性别差距 收入分配",
    "宏观经济学": "宏观经济 货币政策 通货膨胀 财政政策 经济周期 利率 产出 央行 消费 投资 总需求 经济增长 房地产",
    "微观经济学": "微观经济 效用 博弈论 均衡 拍卖 机制设计 契约 逆向选择 道德风险 信号 外部性 公共品 福利 实验经济学",
    "公共财政": "税收 财政支出 政府支出 补贴 社会福利 转移支付 再分配 社会保障 公共品 最优税收 超额负担 地方财政 政府债务",
    "方法与杂项": "计量经济学 估计方法 识别策略 工具变量 面板数据 双重差分 断点回归 固定效应 广义矩估计 贝叶斯 因果推断 机器学习"
    }

    field_names = list(fields_dict.keys())
    field_texts = list(fields_dict.values())

    # 结构加权：标题 * 2、关键词 * 2、摘要 * 1
    clean_df["Full_Text"] = (
        (clean_df["Title"] + " ") * 2 +
        (clean_df["Keywords"] + " ") * 2 +
        clean_df["Abstract"]
    )

    corpus = clean_df["Full_Text"].tolist()
    all_texts = corpus + field_texts

    # 中文术语经常不会被分词器切成与领域词典完全相同的形式（例如“金融市场”
    # 与“金融”）。字符 n-gram 能保留这些重叠，同时移除 jieba 这一非必要运行依赖。
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 4),
        min_df=1,
        max_features=30000,
        sublinear_tf=True,
        norm="l2",
    )
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    doc_vectors = tfidf_matrix[:len(corpus)]
    field_vectors = tfidf_matrix[len(corpus):]

    similarity_matrix = cosine_similarity(doc_vectors, field_vectors)
    best_field_indices = similarity_matrix.argmax(axis=1)
    sorted_scores = np.sort(similarity_matrix, axis=1)
    best_scores = sorted_scores[:, -1]
    second_scores = sorted_scores[:, -2]

    clean_df["Predicted_Field"] = [field_names[idx] for idx in best_field_indices]
    clean_df["Classification_Score"] = best_scores
    clean_df["Classification_Margin"] = best_scores - second_scores
    clean_df["Needs_Review"] = (best_scores < 0.03) | ((best_scores - second_scores) < 0.005)

    print("\n[Diagnostic] Chinese Field Classification Distribution:")
    print(clean_df["Predicted_Field"].value_counts().to_string())
    print("-" * 65)
    return clean_df

# ==========================================
# 2. 长期演变趋势堆叠图 (Stacked Area Chart)
# ==========================================
def generate_field_trend_stacked_chart(clean_df, output_image_path="Chinese_Field_Trends_Stacked_Area.png"):
    print(">> Generating Chinese field trend stacked area chart...")
    
    trend_pivot = clean_df.pivot_table(
        index='Year', 
        columns='Predicted_Field', 
        values='Title', 
        aggfunc='count', 
        fill_value=0
    )
    trend_proportion = trend_pivot.div(trend_pivot.sum(axis=1), axis=0)

    # 显式传入 rc 字体配置，防止 Seaborn 覆盖后导致中文乱码或字形缺失警告
    sns.set_theme(
        style="whitegrid",
        rc={
            "font.family": "sans-serif",
            "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial Unicode MS"],
            "axes.unicode_minus": False
        }
    )

    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

    # 绘制堆叠面积图
    trend_proportion.plot.area(ax=ax, cmap='tab10', alpha=0.85, linewidth=0.5)

    ax.set_title("中文期刊10大经济学领域研究占比演变趋势", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("出版年份 (Publication Year)", fontsize=12, fontweight='bold')
    ax.set_ylabel("发文占比 (Proportion of Publications)", fontsize=12, fontweight='bold')
    ax.set_xlim(trend_proportion.index.min(), trend_proportion.index.max())
    ax.set_ylim(0, 1.0)

    plt.legend(title="研究领域", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, fontsize=10)
    plt.tight_layout()

    plt.savefig(output_image_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f">> [Task Saved] Stacked area trend chart saved to: {output_image_path}")

# ==========================================
# 3. 主程序入口 (Main Execution Block)
# ==========================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filename = os.path.join(base_dir, "Cleaned_Custom_Dataset.csv")

    try:
        print(f">> Loading pre-formatted CSV data from {csv_filename}...")
        df_clean = pd.read_csv(csv_filename, encoding='utf-8-sig')

        # 确保关键列存在并处理空值
        for col in ['Title', 'Abstract', 'Keywords', 'Year']:
            if col not in df_clean.columns:
                raise ValueError(f"Missing essential column in CSV: {col}")

        df_clean['Title'] = df_clean['Title'].fillna('').astype(str)
        df_clean['Abstract'] = df_clean['Abstract'].fillna('').astype(str)
        df_clean['Keywords'] = df_clean['Keywords'].fillna('').astype(str)
        df_clean['Year'] = pd.to_numeric(df_clean['Year'], errors='coerce').fillna(0).astype(int)

        if not df_clean.empty:
            # Step 1: 执行中文零样本语义相似度领域分类（内部已应用标题*2、关键词*2、摘要*1）
            df_clean = classify_chinese_economics_fields(df_clean)

            # 导出带有分类标签的新表
            export_csv_path = os.path.join(base_dir, "Cleaned_Custom_Dataset_Classified.csv")
            df_clean.to_csv(export_csv_path, index=False, encoding='utf-8-sig')
            print(f">> Exported classified dataset to: {export_csv_path}")

            # Step 2: 绘制长期演变趋势堆叠图
            generate_field_trend_stacked_chart(
                df_clean, os.path.join(base_dir, "Chinese_Field_Trends_Stacked_Area.png")
            )

            print("\n>> All classification and trend tasks completed successfully!")
        else:
            print("[Error] CSV file is empty or missing necessary data.")

    except FileNotFoundError:
        print(f"[Error] File '{csv_filename}' not found. Please ensure the data preprocessing step has been run first.")
