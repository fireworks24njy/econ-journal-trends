import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

# ==========================================
# 1. 数据加载与精细化预处理
# ==========================================
def load_and_preprocess_wos_data(file_pattern="*-*.txt"):
    """加载 WoS 文本数据，执行三路分流过滤与年份字段合理性约束。"""
    all_files = glob.glob(file_pattern)
    if not all_files:
        print(f"[Error] No files matching '{file_pattern}' found in current directory.")
        return pd.DataFrame()
        
    print(f">> Loading {len(all_files)} data file(s)...")
    df_list = []
    for f in all_files:
        try:
            df_list.append(pd.read_csv(f, sep='\t', encoding='utf-8-sig', dtype=str))
        except UnicodeDecodeError:
            df_list.append(pd.read_csv(f, sep='\t', encoding='utf-8', dtype=str, errors='ignore'))
            
    df = pd.concat(df_list, ignore_index=True)
    for col in ['AB', 'DT', 'TI', 'PY', 'AU', 'AF', 'DI', 'SO']:
        if col not in df.columns: df[col] = np.nan

    # 定义数据分流条件
    missing_abs = df['AB'].isna() | (df['AB'].str.strip() == '')
    missing_meta = (df['TI'].isna() | (df['TI'].str.strip() == '')) | \
                   (df['PY'].isna() | (df['PY'].str.strip() == '')) | \
                   ((df['AU'].isna() | (df['AU'].str.strip() == '')) & (df['AF'].isna() | (df['AF'].str.strip() == '')))

    col_map = {'TI': 'Title', 'DT': 'Document_Type', 'SO': 'Journal', 'PY': 'Year', 'AU': 'Author', 'DI': 'DOI', 'AB': 'Abstract'}

    # 分流导出：无摘要记录
    if missing_abs.any():
        df[missing_abs].rename(columns=col_map).to_csv("Missing_Abstracts_Records.csv", index=False, encoding='utf-8-sig')
        print(f"   [Filter] Exported {missing_abs.sum()} records without abstract to Missing_Abstracts_Records.csv")
    
    # 分流导出：核心元数据缺失记录
    incomplete = ~missing_abs & missing_meta
    if incomplete.any():
        df[incomplete].rename(columns=col_map).to_csv("Incomplete_Metadata_Records.csv", index=False, encoding='utf-8-sig')
        print(f"   [Filter] Exported {incomplete.sum()} records with incomplete metadata to Incomplete_Metadata_Records.csv")

    # 留存参与计算的完整记录
    df_valid = df[~missing_abs & ~missing_meta].copy()
    if df_valid.empty: 
        return pd.DataFrame()

    # 安全提取可选字段（防止缺失列报错，保证返回 Series 并正确填充空字符串/数字）
    empty_str = pd.Series('', index=df_valid.index)
    empty_int = pd.Series(0, index=df_valid.index)
    
    de = df_valid['DE'].fillna('') if 'DE' in df_valid else empty_str
    id_col = df_valid['ID'].fillna('') if 'ID' in df_valid else empty_str
    wc = df_valid['WC'].fillna('') if 'WC' in df_valid else empty_str
    sc = df_valid['SC'].fillna('') if 'SC' in df_valid else empty_str
    cit = df_valid['TC'] if 'TC' in df_valid else (df_valid['Z9'] if 'Z9' in df_valid else empty_int)

    # 构建清洗后的标准数据集
    clean_df = pd.DataFrame({
        'Title': df_valid['TI'].str.strip(),
        'Abstract': df_valid['AB'].str.strip(),
        'Year': pd.to_numeric(df_valid['PY'], errors='coerce'),
        'Author': df_valid['AU'].fillna(df_valid['AF']).str.strip(),
        'Keywords': (de + '; ' + id_col).str.strip('; '),
        'JEL_Category': (wc + '; ' + sc).str.replace(r'(;\s*)+', '; ', regex=True).str.strip('; '),
        'Citations': pd.to_numeric(cit, errors='coerce').fillna(0).astype(int)
    })

    # 年份合理性硬性过滤 (彻底拦截如 113 等列错位/乱码脏数据)
    valid_year = (clean_df['Year'] >= 1900) & (clean_df['Year'] <= 2026) & clean_df['Year'].notna()
    if (~valid_year).any():
        dropped_years = clean_df.loc[~valid_year, 'Year'].dropna().unique().tolist()
        print(f"   [Constraint] Dropped {(~valid_year).sum()} invalid year records with values: {dropped_years}")
        
    clean_df = clean_df[valid_year].copy()
    clean_df['Year'] = clean_df['Year'].astype(int)
    
    # 构建词频加权全文本 (标题权重x2 + 关键词权重x2 + 摘要)
    clean_df['Full_Text'] = (clean_df['Title'] + ". ") * 2 + (clean_df['Keywords'] + ". ") * 2 + clean_df['Abstract']
    print(f">> Retained {len(clean_df)} perfect records for final ESI analysis.")
    return clean_df


# ==========================================
# 2. ESI 阶梯分位数计算与深度词频赋权
# ==========================================
def tokenize_and_esi_weight(clean_df, extra_stopwords=None):
    """按年份独立计算引用量分位数，应用 ESI 阶梯权重 (Top1%=10, Top10%=5, 其它=1)。"""
    print(">> Calculating time-normalized ESI percentile weights...")
    
    clean_df['Percentile'] = clean_df.groupby('Year')['Citations'].rank(pct=True, method='max')
    clean_df['ESI_Weight'] = np.select(
        [clean_df['Percentile'] >= 0.99, clean_df['Percentile'] >= 0.90],
        [10.0, 5.0], 
        default=1.0
    )
    
    # 诊断采样：用 sort_values + head 替代 apply，彻底规避 Pandas DeprecationWarning
    sample_show = (clean_df.sort_values(['Year', 'Citations'], ascending=[True, False])
                           .groupby('Year').head(2)[['Year', 'Citations', 'Percentile', 'ESI_Weight']])
    print("\n[Diagnostic] Weight assignment sample (Top 2 citations per year):")
    print(sample_show.to_string(index=False))
    print("-" * 65)

    # 停用词设置（剔除一般学术套话与泛化通用词）
    academic_words = {
        'article', 'paper', 'study', 'model', 'results', 'data', 'using', 'used', 'use',
        'find', 'found', 'show', 'shown', 'provide', 'effect', 'effects', 'evidence',
        'analysis', 'based', 'two', 'new', 'one', 'within', 'also', 'may', 'via',
        'table', 'figure', 'however', 'often', 'across', 'set', 'suggest', 'propose',
        'well', 'rather', 'whether', 'including', 'overall', 'important', 'simple',
        'different', 'without', 'among', 'author', 'year', 'sample', 'firm', 'firms',
        'market', 'markets', 'time', 'high', 'low', 'large', 'small', 'years', 'recent'
    }
    stopwords = set(STOPWORDS) | academic_words | set(extra_stopwords or [])
        
    # 纯正则表达式分词与阶梯词频累计
    weighted_freq = Counter()
    for _, row in clean_df.iterrows():
        if pd.isna(row['Full_Text']): continue
        words = re.findall(r'\b[a-z]{2,30}\b', row['Full_Text'].lower())
        for w in words:
            if w not in stopwords:
                weighted_freq[w] += row['ESI_Weight']
                
    return weighted_freq


# ==========================================
# 3. 结果统计打印与词云渲染
# ==========================================
def generate_topic_wordcloud(weighted_freq, output_image_path="Top5_Journals_ESI_Weighted_Topics.png"):
    """打印 Top 20 高权重研究词表格并渲染生成高清学术词云。"""
    print("\n" + "="*65)
    print(" Top 20 Research Topics (ESI Normalized Weight: Top1%=10, Top10%=5)")
    print("="*65)
    print(f" {'Rank':<5} | {'Term':<22} | {'Weighted Score':<15}")
    print("-" * 65)
    for rank, (word, score) in enumerate(weighted_freq.most_common(20), 1):
        print(f" #{rank:<4} | {word:<22} | {score:.2f}")
    print("="*65 + "\n")

    print(">>> Rendering word cloud...")
    wc = WordCloud(
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
    
    # 使用画布级总标题 suptitle 锚定在顶端，彻底避免被词云遮挡或裁切
    fig.suptitle("What Do English Top-5 Economics Journals Care About?\n(Time-Normalized ESI Percentile Weighting)", 
                 fontsize=20, fontweight='bold', color='#2c3e50', y=0.94, ha='center')
    
    # 调整子图下移，给上方的 suptitle 留出安全空间
    plt.subplots_adjust(top=0.86, bottom=0.04, left=0.04, right=0.96)
    
    # 强制将标题和图完整打包保存
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
        
    # Step 1: 加载并执行数据分流清洗与合法性校验
    df_clean = load_and_preprocess_wos_data(file_pattern)
    
    if not df_clean.empty:
        # 导出完全纯净的结构化 CSV 主表备用
        export_csv_path = "Cleaned_Top5_Journals_Dataset.csv"
        df_clean[['Year', 'Title', 'Author', 'Keywords', 'JEL_Category', 'Citations', 'Abstract']].to_csv(
            export_csv_path, index=False, encoding='utf-8-sig'
        )
        print(f">> Exported clean dataset to: {export_csv_path}")
        
        # Step 2: 分位数赋权与加权词频聚合
        custom_stops = ['journal', 'quarterly', 'review', 'economic', 'economics', 'business'] 
        word_frequencies = tokenize_and_esi_weight(df_clean, extra_stopwords=custom_stops)
        
        # Step 3: 数据输出与词云渲染
        generate_topic_wordcloud(word_frequencies)
    else:
        print("[Error] Processing aborted: No valid records remaining after filtration.")