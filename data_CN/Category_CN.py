import io
import os
import re
import sys
import jieba
import matplotlib.pyplot as plt
import pandas as pd
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
        "发展经济学": "贫困 援助 农业 健康 随机试验 田野实验 发展中国家 经济增长 教育 卫生",
        "经济史": "历史数据 经济史 计量历史学 长期历史发展 过去世纪 十九世纪",
        "金融学": "资产定价 投资组合 股票 股票市场 债券 流动性 风险 股利 期权 公司金融",
        "产业组织": "企业 市场结构 反垄断 竞争 进入 垄断 价格 寡头 生产率 加价",
        "国际经济学": "贸易 关税 出口 进口 国际贸易 汇率 全球化 跨国公司 引力模型",
        "劳动经济学": "劳动力 工资 就业 工人 岗位 收入不平等 技能 教育 人力资本 迁移",
        "宏观经济学": "宏观经济 货币 通货膨胀 财政 动态随机一般均衡 商业周期 利率 产出 中央银行 消费",
        "微观经济学": "理论 均衡 博弈 选择 偏好 效用 机制设计 拍卖 信息合约 讨价还价",
        "公共财政": "税收 财政支出 政府支出 补贴 福利 财政政策 外部性 收入",
        "方法与杂项": "估计 识别 工具变量 渐近理论 面板数据 方法 计量经济学 杂项"
    }

    field_names = list(fields_dict.keys())
    field_texts = list(fields_dict.values())

    # 结构加权：标题 * 2、关键词 * 2、摘要 * 1
    clean_df["Full_Text"] = (
        (clean_df["Title"] + " ") * 2 +
        (clean_df["Keywords"] + " ") * 2 +
        clean_df["Abstract"]
    )

    def chinese_tokenizer(text):
        return [
            w.strip() for w in jieba.cut(text) 
            if len(w.strip()) >= 2 and not re.match(r'^[0-9]+$', w.strip())
        ]

    corpus = clean_df["Full_Text"].tolist()
    all_texts = corpus + field_texts

    vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer, max_features=8000)
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    doc_vectors = tfidf_matrix[:len(corpus)]
    field_vectors = tfidf_matrix[len(corpus):]

    similarity_matrix = cosine_similarity(doc_vectors, field_vectors)
    best_field_indices = similarity_matrix.argmax(axis=1)

    clean_df["Predicted_Field"] = [field_names[idx] for idx in best_field_indices]

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
    csv_filename = "Cleaned_Custom_Dataset.csv"

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
            export_csv_path = "Cleaned_Custom_Dataset_Classified.csv"
            df_clean.to_csv(export_csv_path, index=False, encoding='utf-8-sig')
            print(f">> Exported classified dataset to: {export_csv_path}")

            # Step 2: 绘制长期演变趋势堆叠图
            generate_field_trend_stacked_chart(df_clean)

            print("\n>> All classification and trend tasks completed successfully!")
        else:
            print("[Error] CSV file is empty or missing necessary data.")

    except FileNotFoundError:
        print(f"[Error] File '{csv_filename}' not found. Please ensure the data preprocessing step has been run first.")