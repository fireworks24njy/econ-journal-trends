import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
from wordcloud import WordCloud
from collections import Counter
import os,sys,io
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS'] 
plt.rcParams['axes.unicode_minus'] = False
if sys.version_info >= (3, 7):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"
# ==========================================
# 1. 数据加载与中文文本解析（适配无引用字段）
# ==========================================
def load_and_preprocess_custom_data(file_pattern="CN*-*.txt"):
    """加载自定义中文期刊文本数据，自动识别可能存在的引用字段或直接跳过。"""
    all_files = glob.glob(file_pattern)
    if not all_files:
        print(f"[Error] No files matching '{file_pattern}' found in current directory.")
        return pd.DataFrame()
        
    print(f">> Loading {len(all_files)} data file(s)...")
    records = []
    for f in all_files:
        try:
            with open(f, 'r', encoding='utf-8-sig') as file:
                content = file.read()
        except UnicodeDecodeError:
            with open(f, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                
        # 按空行分割多条记录
        raw_records = content.strip().split('\n\n')
        for raw_rec in raw_records:
            rec_dict = {}
            lines = raw_rec.strip().split('\n')
            for line in lines:
                if ':' in line or '：' in line:
                    parts = re.split(r'[:：]', line, maxsplit=1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        val = parts[1].strip()
                        rec_dict[key] = val
            if rec_dict:
                records.append(rec_dict)
                
    if not records:
        print("[Error] No valid records parsed from files.")
        return pd.DataFrame()
        
    df = pd.DataFrame(records)
    
    # 字段映射匹配当前数据集的实际中文标签
    col_mapping = {
        'Title-题名': 'Title',
        'Summary-摘要': 'Abstract',
        'Year-年': 'Year',
        'Author-作者': 'Author',
        'Keyword-关键词': 'Keywords',
        'Source-文献来源': 'Journal'
    }
    
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})
    
    for col in ['Title', 'Abstract', 'Year', 'Author', 'Keywords', 'Journal']:
        if col not in df.columns:
            df[col] = ''
            
    # 如果完全没有引用字段，则初始化为 0
    if 'Citations' not in df.columns:
        df['Citations'] = 0
    else:
        df['Citations'] = pd.to_numeric(df['Citations'], errors='fillna').fillna(0)

    # 完整性过滤（检查核心字段是否缺失）
    missing_abs = df['Abstract'].isna() | (df['Abstract'].str.strip() == '')
    missing_meta = (df['Title'].isna() | (df['Title'].str.strip() == '')) | \
                   (df['Year'].isna() | (df['Year'].str.strip() == ''))

    df_valid = df[~missing_abs & ~missing_meta].copy()
    if df_valid.empty: 
        return pd.DataFrame()

    # 构建清洗后的标准数据集
    clean_df = pd.DataFrame({
        'Title': df_valid['Title'].str.strip(),
        'Abstract': df_valid['Abstract'].str.strip(),
        'Year': pd.to_numeric(df_valid['Year'], errors='coerce'),
        'Author': df_valid['Author'].str.strip(),
        'Keywords': df_valid['Keywords'].str.strip(),
        'Journal': df_valid['Journal'].str.strip(),
        'Citations': df_valid['Citations']
    })

    # 年份合理性过滤
    valid_year = (clean_df['Year'] >= 1900) & (clean_df['Year'] <= 2026) & clean_df['Year'].notna()
    clean_df = clean_df[valid_year].copy()
    clean_df['Year'] = clean_df['Year'].astype(int)
    
    print(f">> Retained {len(clean_df)} valid records for text analysis.")
    return clean_df


# ==========================================
# 2. 文本结构加权与中文分词处理
# ==========================================
def tokenize_with_structural_weight(clean_df, extra_stopwords=None):
    """
    在无引用数据的情况下，采用文本结构加权法：
    - 标题（Title）词语权重：2.0
    - 关键词（Keywords）词语权重：2.0
    - 摘要（Abstract）词语权重：1.0
    """
    print(">> Processing Chinese text with structural weighting (Title/Keywords x3, Abstract x1)...")
    
    academic_stopwords = {
        '研究', '分析', '本文', '结果', '表明', '发现', '通过', '基于', '进行', '影响', 
        '作用', '机制', '发展', '建设', '水平', '提升', '推进', '促进', '有效', '显著', 
        '问题', '我国', '企业', '产业', '经济', '模式', '路径', '政策', '关系', '创新', 
        '制度', '管理', '应用', '评估', '特征', '框架', '视角', '实践', '过程', '多元', 
        '协同', '系统', '空间', '结构', '要素', '配置', '效应', '增长', '一个', '这种', 
        '以及', '可以', '需要', '对于', '具有', '同时', 'et', 'al'
    }
    stopwords = academic_stopwords | set(extra_stopwords or [])
        
    weighted_freq = Counter()
    
    for _, row in clean_df.iterrows():
        # 对标题分词（赋予权值 2.0）
        if pd.notna(row['Title']) and row['Title'].strip():
            for w in jieba.cut(row['Title']):
                w = w.strip()
                if len(w) >= 2 and w not in stopwords and not re.match(r'^[0-9]+$', w):
                    weighted_freq[w] += 2.0
                    
        # 对关键词分词（赋予权值 3.0）
        if pd.notna(row['Keywords']) and row['Keywords'].strip():
            # 关键词有时用分号或空格分隔，也可以整体切词
            for w in re.split(r'[;；,\s]+', row['Keywords']):
                w = w.strip()
                if len(w) >= 2 and w not in stopwords and not re.match(r'^[0-9]+$', w):
                    weighted_freq[w] += 2.0
                    
        # 对摘要分词（赋予权值 1.0）
        if pd.notna(row['Abstract']) and row['Abstract'].strip():
            for w in jieba.cut(row['Abstract']):
                w = w.strip()
                if len(w) >= 2 and w not in stopwords and not re.match(r'^[0-9]+$', w):
                    weighted_freq[w] += 1.0
                
    return weighted_freq


# ==========================================
# 3. 统计输出与词云渲染
# ==========================================
def generate_topic_wordcloud(weighted_freq, output_image_path="Structural_Weighted_Topics.png"):
    """打印 Top 20 高权重研究词表格并渲染生成高清中文学术词云。"""
    print("\n" + "="*65)
    print(" Top 20 Research Topics (Structural Weighted Score)")
    print("="*65)
    print(f" {'Rank':<5} | {'Term':<22} | {'Weighted Score':<15}")
    print("-" * 65)
    for rank, (word, score) in enumerate(weighted_freq.most_common(20), 1):
        print(f" #{rank:<4} | {word:<22} | {score:.2f}")
    print("="*65 + "\n")

    print(">>> Rendering word cloud...")
    wc = WordCloud(
        font_path='simhei.ttf',  
        width=1600,
        height=1000,
        background_color='white',
        colormap='inferno',  
        max_words=150,
        min_font_size=12,
        max_font_size=160,
        random_state=42,
        prefer_horizontal=0.85
    )
    wc.generate_from_frequencies(weighted_freq)
    
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')  
    
    fig.suptitle("中文期刊高频研究主题与结构加权分析\n(Structural Weighting: Title/Keywords x2, Abstract x1)", 
                 fontsize=20, fontweight='bold', color='#2c3e50', y=0.94, ha='center')
    
    plt.subplots_adjust(top=0.86, bottom=0.04, left=0.04, right=0.96)
    plt.savefig(output_image_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.show()
    print(f">>> Word cloud saved to: {output_image_path}")


# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    file_pattern = "*-*.txt"
    if not glob.glob(file_pattern):
        print(f"[Error] No files matching '{file_pattern}' found. Please check data files.")
        exit(1)
        
    # Step 1: 加载并执行文本解析
    df_clean = load_and_preprocess_custom_data(file_pattern)
    
    if not df_clean.empty:
        export_csv_path = "Cleaned_Custom_Dataset.csv"
        df_clean[['Year', 'Title', 'Author', 'Keywords', 'Journal', 'Citations', 'Abstract']].to_csv(
            export_csv_path, index=False, encoding='utf-8-sig'
        )
        print(f">> Exported clean dataset to: {export_csv_path}")
        
        # Step 2: 文本结构加权分词与词频聚合
        custom_stops = ['期刊', '管理世界', '中国社会科学', '经济研究'] 
        word_frequencies = tokenize_with_structural_weight(df_clean, extra_stopwords=custom_stops)
        
        # Step 3: 数据输出与词云渲染
        generate_topic_wordcloud(word_frequencies)
    else:
        print("[Error] Processing aborted: No valid records remaining after filtration.")