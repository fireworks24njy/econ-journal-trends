import re, glob, sys, io, os, math
from collections import Counter
import pandas as pd, numpy as np, matplotlib.pyplot as plt, jieba
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
if sys.version_info >= (3, 7):
    sys.stdout.reconfigure(encoding='utf-8')
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

# ======================== 停用词表 ========================
STOPWORDS_CN = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上',
    '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这',
    '那', '它', '他', '她', '们', '与', '或', '但', '而', '且', '并', '等', '于', '之', '其',
    '所', '以', '因', '为', '对', '向', '从', '把', '被', '让', '给', '跟', '与', '及', '亦',
    '很', '太', '非常', '十分', '极其', '格外', '分外', '相当', '比较', '较为', '更', '更加',
    '最', '极为', '特别', '尤其', '实在', '确实', '确', '真', '真正', '简直', '几乎', '大约',
    '大概', '左右', '上下', '前后', '始终', '一直', '一向', '向来', '从来', '总是', '经常',
    '常常', '时常', '往往', '不断', '反复', '再三', '已经', '曾经', '刚刚', '才',
    '就', '马上', '立刻', '顿时', '渐渐', '逐渐', '逐步', '慢慢', '忽然', '偶尔', '有时',
    '任何', '一切', '所有', '全部', '全', '都', '凡是', '大多', '多数', '少数', '若干',
    '许多', '很多', '大量', '少量', '一点', '一些', '有些', "家庭", "地方",
    '因此', '因而', '从而', '于是', '所以', '故', '故此', '由此', '为此', '基于此',
    '然而', '不过', '只是', '但是', '可是', '却', '虽然', '尽管', '即使', '即便',
    '如果', '假如', '倘若', '若', '要是', '只要', '除非', '无论', '不论', '不管',
    '而且', '并且', '此外', '另外', '再者', '加之', '以及', '及其', '连同', '总之',
    '总而言之', '综上所述', '如上所述', '如前所述', "明显", "显著", "效果", "方面",
    '研究', '分析', '本文', '我们', '作者', '结果', '表明', '发现', '通过', '基于',
    '进行', '影响', '作用', '机制', '过程', '问题', '关系', '视角', '框架', '实践',
    '评估', '特征', '应用', '管理', '模式', '路径', '政策', '制度', '创新', '结构',
    '要素', '配置', '效应', '提升', '推进', '促进', '发展', '建设', '水平', '有效',
    '我国', '模型', '数据', '样本', '变量', "构建", "国家", "研究",  # 已添加
    '回归', '估计', '检验', '假设', '理论', '文献', '综述', '结论', '讨论', '建议',
    '贡献', '局限', '不足', '展望', '前言', '引言', '背景', '目的', '意义', '方法',
    '策略', '方案', '体系', '系统', '空间', '时间', '阶段', '时期', '年代', '年份',
    '提供', '降低', '基础', '异质性', '提高', '增加', '减少', '变化', '差异',
    '显著地', '显著性', '显著影响', '正向', '负向', '正相关', '负相关',
    '稳健', '稳健性', '稳健性检验', '控制', '控制变量', '控制组',
    '固定', '固定效应', '随机', '随机效应', '工具', '工具变量', '内生', '内生性',
    '滞后', '滞后项', '差分', '一阶差分', '平方', '交叉项', '交互项',
    '回归模型', '线性回归', '非线性', '多元回归', '最小二乘法', '极大似然', '广义矩估计',
    '面板数据', '截面数据', '时间序列', '混合数据', '描述性统计', '相关性分析',
    '主成分', '因子分析', '聚类', '分类', '预测', '拟合', '残差', '方差',
    '标准差', '均值', '中位数', '众数', '百分比', "增长", "衰退",
    '期刊', '管理世界', '中国社会科学', '经济研究', "经济", "经济学", "时代", "全球",
    '世界', '地区', '进一步', '形成', '具有', '产生', '主要', '利用',
    '发挥', '体现', '表现', '构成', '实现', '推动', '支撑', '带动',
    '当前', '目前', '未来', '长期', '短期', '整体', '局部', '内部',
    '外部', '直接', '间接', '初步', '充分', '必要', '根本',
    '对于', '来自', '如何', '能够', '可以', '之间', '按照', '关于', '根据',
    '这样', '这种', '那些', '这些', '什么', '为什么', '等等', '成为', '作为', "重要", "同时",
    '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
    '百', '千', '万', '亿', '第一', '第二', '第三',
    '最后', '首先', '其次', '再次', '此外', '以及', '及其', '等等', '因为', '所以',
    '不仅', '而且', '并且', '或者', '要么', '还是', '以便', '从而', '于是',
}

# ======================== 1. 数据加载 ========================
def load_and_preprocess_custom_data(file_pattern="CN*-*.txt"):
    all_files = glob.glob(file_pattern)
    if not all_files:
        print(f"[Error] No files matching '{file_pattern}' found.")
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
        for raw_rec in content.strip().split('\n\n'):
            rec_dict = {}
            for line in raw_rec.strip().split('\n'):
                if ':' in line or '：' in line:
                    parts = re.split(r'[:：]', line, maxsplit=1)
                    if len(parts) == 2:
                        rec_dict[parts[0].strip()] = parts[1].strip()
            if rec_dict:
                records.append(rec_dict)
    if not records:
        print("[Error] No valid records parsed.")
        return pd.DataFrame()
    df = pd.DataFrame(records)
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
    if 'Citations' not in df.columns:
        df['Citations'] = 0
    else:
        df['Citations'] = pd.to_numeric(df['Citations'], errors='coerce').fillna(0)
    # 先统一缺失值和字段类型，避免数值型年份或空值导致字符串处理报错
    text_columns = ['Title', 'Abstract', 'Year', 'Author', 'Keywords', 'Journal']
    for col in text_columns:
        df[col] = df[col].fillna('').astype(str)

    df = df[
        (df['Abstract'].str.strip() != '')
        & (df['Title'].str.strip() != '')
        & (df['Year'].str.strip() != '')
    ]
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df[df['Year'].between(2020, 2025)].copy()
    df['Year'] = df['Year'].astype(int)

    # 统一首尾空格和连续空白，防止仅因排版空格不同而无法识别重复记录
    def normalize_whitespace(value):
        return re.sub(r'\s+', ' ', str(value)).strip()

    clean_df = pd.DataFrame({
        'Title': df['Title'].map(normalize_whitespace),
        'Abstract': df['Abstract'].map(normalize_whitespace),
        'Year': df['Year'],
        'Author': df['Author'].map(normalize_whitespace),
        'Keywords': df['Keywords'].map(normalize_whitespace),
        'Journal': df['Journal'].map(normalize_whitespace),
        'Citations': pd.to_numeric(df['Citations'], errors='coerce').fillna(0)
    })

    # 仅删除核心字段完全一致的重复记录。
    # 不按“年份+标题”直接去重，以免误删题名相同但作者、期刊或摘要不同的文章。
    duplicate_columns = [
        'Year', 'Title', 'Author', 'Keywords',
        'Journal', 'Citations', 'Abstract'
    ]
    before_dedup = len(clean_df)
    exact_duplicate_mask = clean_df.duplicated(
        subset=duplicate_columns, keep='first'
    )
    exact_duplicate_count = int(exact_duplicate_mask.sum())
    clean_df = clean_df.loc[~exact_duplicate_mask].reset_index(drop=True)

    # 仅报告可能需要人工核查的“同年同标题”记录，不自动删除。
    title_year_duplicate_mask = clean_df.duplicated(
        subset=['Year', 'Title'], keep=False
    )
    title_year_candidate_count = int(title_year_duplicate_mask.sum())

    print(f">> Retained {before_dedup} valid records for 2020-2025 before deduplication.")
    print(f">> Removed {exact_duplicate_count} fully duplicated record(s).")
    print(
        f">> {len(clean_df)} record(s) remain after field cleaning and exact "
        "deduplication (before subject filtering)."
    )
    if title_year_candidate_count:
        print(
            f">> Review note: {title_year_candidate_count} record(s) share the same "
            "year and title but differ in other fields; they were retained."
        )
    return clean_df

# ======================== 2. 过滤《中国社会科学》中的非经济学文章 ========================
def filter_chinese_social_science(df):
    before_filter = len(df)
    non_eco_markers = [
        '中国式现代化', '马克思主义', '中国特色社会主义', '共产党', '党的领导',
        '毛泽东思想', '邓小平理论', '三个代表', '科学发展观', '习近平',
        '中华民族伟大复兴', '中国梦', '历史唯物主义', '辩证唯物主义',
        '哲学', '政治学', '法学', '文学', '历史学', '社会学', '人类学',
        '意识形态', '精神文明', '道德', '伦理', '宗教', '艺术',
        '霸权', '制衡', '地缘', '地缘政治', '军事', '国际关系', '外交', '多极', 
        '冷战', '安全', '战略', '大国关系', '国际政治', '威慑', '武装', '海军', '海上格局'
    ]
    
    strong_eco_keywords = [
        '金融', '财政', 'GDP', '通胀', '通货膨胀', '汇率', '利率', '货币', '微观经济',
        '工资', '税收', '社保', '宏观经济', '资本市场', '数字经济',
        '计量', '实证', '面板', '双重差分', '内生性', '博弈'
    ]
    
    weak_eco_keywords = [
        '经济', '贸易', '市场', '企业', '产业', '投资', '消费', '价格',
        '就业', '债务', '资本', '收入', '分配', '贫富', '福利',
        '农村', '农业', '工业', '服务业', '改革', '开放', '全球化', '跨国',
        '生产率', '效率', '增长', '发展', '结构', '转型', '升级', '创新',
        '模型', '数据', '回归', '因果', '政策', '治理', '制度', '产权', '契约', '合同', '拍卖', '均衡'
    ]

    def is_economics(row):
        title = str(row['Title'])
        abstract = str(row['Abstract'])
        text = title + " " + abstract
        
        for marker in non_eco_markers:
            if marker in text:
                if not any(kw in text for kw in strong_eco_keywords):
                    return False
                    
        strong_hits = sum(1 for kw in strong_eco_keywords if kw in text)
        weak_hits = sum(1 for kw in weak_eco_keywords if kw in text)
        
        return (strong_hits >= 1) or (weak_hits >= 3)
    
    css = df[df['Journal'] == '中国社会科学'].copy()
    others = df[df['Journal'] != '中国社会科学'].copy()
    if not css.empty:
        css_filtered = css[css.apply(is_economics, axis=1)]
        print(f">> Filtered out {len(css) - len(css_filtered)} non-economics articles from '中国社会科学'.")
    else:
        css_filtered = css
    df_final = pd.concat([others, css_filtered], ignore_index=True)
    print(
        f">> Subject filtering: {before_filter} -> {len(df_final)} record(s); "
        f"removed {before_filter - len(df_final)} record(s)."
    )
    return df_final

# ======================== 3. 短语挖掘（增强过滤） ========================
def build_chinese_phrases(df, stopwords, min_freq=2, llr_threshold=0.1):
    print(f">> Mining phrases via LLR (min_freq={min_freq}, llr_threshold={llr_threshold})...")
    corpus = []
    word_freq = Counter()
    token_pattern = re.compile(r'^[\u4e00-\u9fa5a-zA-Z0-9]+$')
    
    for _, row in df.iterrows():
        text = (row['Title'] if pd.notna(row['Title']) else '') + ' ' + \
               (row['Keywords'] if pd.notna(row['Keywords']) else '') + ' ' + \
               (row['Abstract'] if pd.notna(row['Abstract']) else '')
        tokens = jieba.cut(text)
        filtered = []
        for w in tokens:
            w = w.strip()
            if len(w) < 2 or not token_pattern.match(w):
                continue
            filtered.append(w)
            word_freq[w] += 1
        if filtered:
            corpus.append(' '.join(filtered))
    
    total_words = sum(word_freq.values())
    if total_words == 0:
        print("   [Warning] No valid words found in corpus.")
        return []
    
    vectorizer = CountVectorizer(ngram_range=(2, 3), min_df=min_freq, tokenizer=lambda x: x.split())
    try:
        X = vectorizer.fit_transform(corpus)
    except Exception as e:
        print(f"   [Error] CountVectorizer failed: {e}")
        return []
    
    ngram_freq = np.array(X.sum(axis=0)).flatten()
    ngram_names = vectorizer.get_feature_names_out()
    print(f"   [Debug] Found {len(ngram_names)} candidate n-grams (with min_freq={min_freq}).")
    if len(ngram_names) == 0:
        return []
    
    phrases_with_llr = []
    N = total_words
    for idx, phrase in enumerate(ngram_names):
        freq = ngram_freq[idx]
        if freq < min_freq:
            continue
        parts = phrase.split()
        
        if len(parts) == 2:
            a, b = parts
            n1 = word_freq.get(a, 0)
            n2 = word_freq.get(b, 0)
            if n1 == 0 or n2 == 0:
                continue
            E = (n1 * n2) / N
        elif len(parts) == 3:
            a, b, c = parts
            n1 = word_freq.get(a, 0)
            n2 = word_freq.get(b, 0)
            n3 = word_freq.get(c, 0)
            if n1 == 0 or n2 == 0 or n3 == 0:
                continue
            E = (n1 * n2 * n3) / (N * N)
        else:
            continue
        
        k = freq
        try:
            llr = 2 * (k * math.log(k / E) + (N - k) * math.log((N - k) / (N - E)))
        except ValueError:
            llr = 0
        
        if llr >= llr_threshold:
            phrases_with_llr.append(('_'.join(parts), llr, freq))
    
    phrases_with_llr.sort(key=lambda x: x[1], reverse=True)
    print("\n   [Debug] Top 20 phrases by LLR (threshold=0.1):")
    for i, (p, llr, f) in enumerate(phrases_with_llr[:20], 1):
        print(f"      {i:2d}. {p}  (LLR={llr:.2f}, freq={f})")
    
    all_phrases = [p for p, _, _ in phrases_with_llr]
    
    # 过滤资助计划词组和学术套话词组
    exact_exclude = {
        '资助_计划', '青年_学者', '乌家培_资助', '信息_乌家培', '乌家培_资助_计划', '信息管理_领域', '领域_青年_学者', '信息_信息管理',
        '中国_信息_乌家培', '信息_乌家培_资助'
    }
    forbidden_words = {'表明', '发现', '结果'}
    filtered_phrases = []
    for p in all_phrases:
        if p in exact_exclude:
            continue
        parts = p.split('_')
        if all(part in stopwords for part in parts):
            continue
        # 使用包含匹配，排除“结果表明”“研究发现”等学术套话变体
        if any(
            forbidden in part
            for part in parts
            for forbidden in forbidden_words
        ):
            continue
        filtered_phrases.append(p)
    
    filtered_phrases = sorted(set(filtered_phrases), key=lambda x: -len(x))
    print(f"\n>> After filtering, kept {len(filtered_phrases)} economic-related phrases.")
    return filtered_phrases

# ======================== 4. 文本归一化（仅用于短语统计） ========================
def norm_text_cn(text, phrase_list, stopwords):
    if not isinstance(text, str) or not text.strip():
        return []
    for phrase in sorted(phrase_list, key=len, reverse=True):
        orig = phrase.replace('_', '')
        if orig in text:
            text = text.replace(orig, phrase)
    pattern = re.compile(r'[\u4e00-\u9fa5a-zA-Z0-9]+(?:_[\u4e00-\u9fa5a-zA-Z0-9]+)+')
    found = pattern.findall(text)
    for fp in found:
        text = text.replace(fp, ' ' + fp + ' ')
    parts = text.split()
    tokens = []
    for p in parts:
        if '_' in p:
            tokens.append(p)
        else:
            for w in jieba.lcut(p):
                w = w.strip()
                if len(w) < 2 or w in stopwords or not re.match(r'^[\u4e00-\u9fa5a-zA-Z0-9]+$', w):
                    continue
                tokens.append(w)
    return tokens

# ======================== 5. 结构加权统计（单字词与短语独立） ========================
def tokenize_with_structural_weight(df, phrase_list, stopwords, title_weight=2, keyword_weight=2):
    print(">> Tokenizing with weighting: Title×{}, Keywords×{}, Abstract×1".format(title_weight, keyword_weight))
    print(">> Phrase counting mode: inclusive (nested phrases are counted independently).")
    word_counter = Counter()
    phrase_counter = Counter()
    
    sorted_phrases = sorted(phrase_list, key=len, reverse=True)
    token_pattern = re.compile(r'^[\u4e00-\u9fa5a-zA-Z0-9]+$')

    # 使用独立分词器统计普通词，避免动态加入候选短语后改变基础分词结果
    word_tokenizer = jieba.Tokenizer()
    
    for _, row in df.iterrows():
        title = row['Title'] if pd.notna(row['Title']) else ''
        keywords = row['Keywords'] if pd.notna(row['Keywords']) else ''
        abstract = row['Abstract'] if pd.notna(row['Abstract']) else ''
        
        # --- 1) 单字词统计（不替换短语） ---
        def count_words(text, weight):
            if not text:
                return
            for w in word_tokenizer.lcut(text):
                w = w.strip()
                if len(w) < 2 or w in stopwords or not token_pattern.match(w):
                    continue
                word_counter[w] += weight
        
        count_words(title, title_weight)
        count_words(keywords, keyword_weight)
        count_words(abstract, 1)
        
        # --- 2) 短语统计（包含式、相互独立） ---
        # 不再先替换较长短语。每个候选短语都直接在原始文本中统计，
        # 因此“企业全要素生产率”和“全要素生产率”可以同时被计数。
        def count_phrases(text, weight):
            if not text:
                return
            for phrase in sorted_phrases:
                orig = phrase.replace('_', '')
                if not orig:
                    continue
                occurrence_count = len(re.findall(re.escape(orig), text))
                if occurrence_count:
                    phrase_counter[phrase] += occurrence_count * weight
        
        count_phrases(title, title_weight)
        count_phrases(keywords, keyword_weight)
        count_phrases(abstract, 1)
    
    return word_counter, phrase_counter

# ======================== 6. 输出双榜单与词云 ========================
def generate_output(word_counter, phrase_counter, df_clean,
                    output_csv="Cleaned_Custom_Dataset.csv",
                    hot_terms_csv="Chinese_Hot_Terms.csv",
                    wordcloud_path="Chinese_Phrase_WordCloud.png"):
    df_clean[['Year', 'Title', 'Author', 'Keywords', 'Journal', 'Citations', 'Abstract']].to_csv(
        output_csv, index=False, encoding='utf-8-sig'
    )
    print(f">> Exported clean dataset to {output_csv}")
    
    print("\n[DEBUG] phrase_counter items with '_' (top 10):")
    phr_items = [(w, c) for w, c in phrase_counter.most_common(50) if '_' in w]
    if phr_items:
        for w, c in phr_items[:10]:
            print(f"   {w}: {c}")
    else:
        print("   (No phrases with '_' found in phrase_counter)")
    
    top5_phrases = []
    for w, cnt in phrase_counter.most_common(100):
        if '_' in w:
            clean_w = w.replace('_', ' ')
            if clean_w not in STOPWORDS_CN:
                top5_phrases.append((clean_w, cnt))
                if len(top5_phrases) == 5:
                    break
                    
    top10_single = []
    for w, cnt in word_counter.most_common(200):
        clean_w = w.replace('_', ' ')
        if '_' not in w and clean_w not in STOPWORDS_CN and len(clean_w) > 1:
            top10_single.append((clean_w, cnt))
            if len(top10_single) == 10:
                break
                
    print("\n" + "=" * 55)
    print("      Top 5 领域复合短语")
    print("=" * 55)
    if top5_phrases:
        for rank, (term, score) in enumerate(top5_phrases, 1):
            print(f" #{rank:<4} | {term:<30} | {score:<10}")
    else:
        print(" [提示] 未能提取到复合短语，请检查 phrase_counter 是否有 '_' 项。")
        
    print("\n" + "=" * 55)
    print("      Top 10 领域核心单字/单词")
    print("=" * 55)
    for rank, (term, score) in enumerate(top10_single, 1):
        print(f" #{rank:<4} | {term:<30} | {score:<10}")
    print("=" * 55 + "\n")
    
    # 合并普通词与复合短语，使调试榜单和词云采用一致的统计口径。
    # 若同一表面词同时出现在两个计数器中，取较大的独立频次，避免重复相加。
    wc_freq = {}
    for term, freq in word_counter.items():
        display_term = term.replace('_', '')
        if display_term not in STOPWORDS_CN and freq >= 3:
            wc_freq[display_term] = max(wc_freq.get(display_term, 0), freq)
    for phrase, freq in phrase_counter.items():
        display_term = phrase.replace('_', '')
        if display_term not in STOPWORDS_CN and freq >= 3:
            wc_freq[display_term] = max(wc_freq.get(display_term, 0), freq)

    # 导出与词云完全同源的频率表，便于核对每个词的字体大小。
    normalized_words = {
        term.replace('_', ''): freq
        for term, freq in word_counter.items()
        if term.replace('_', '') not in STOPWORDS_CN
    }
    normalized_phrases = {
        term.replace('_', ''): freq
        for term, freq in phrase_counter.items()
        if term.replace('_', '') not in STOPWORDS_CN
    }
    hot_term_rows = []
    for term, freq in sorted(wc_freq.items(), key=lambda item: item[1], reverse=True):
        in_words = term in normalized_words
        in_phrases = term in normalized_phrases
        source = '普通词与复合短语' if in_words and in_phrases else (
            '复合短语' if in_phrases else '普通词'
        )
        hot_term_rows.append({
            'Term': term,
            'Weighted_Frequency': freq,
            'Source': source
        })
    pd.DataFrame(hot_term_rows).to_csv(
        hot_terms_csv, index=False, encoding='utf-8-sig'
    )
    print(f">> Exported word-cloud frequencies to {hot_terms_csv}")

    print("\n[DEBUG] Combined word-cloud frequencies (top 20):")
    for rank, row in enumerate(hot_term_rows[:20], 1):
        print(
            f" #{rank:<3} | {row['Term']:<24} | "
            f"{row['Weighted_Frequency']:<8} | {row['Source']}"
        )
    if not wc_freq:
        print("[Error] No words for word cloud.")
        return
        
    font_path = 'simhei.ttf'
    if not os.path.exists(font_path):
        for f in ['C:/Windows/Fonts/simhei.ttf', '/System/Library/Fonts/PingFang.ttc']:
            if os.path.exists(f):
                font_path = f
                break
        else:
            print("[Warning] No Chinese font found, using default.")
            font_path = None
            
    wc = WordCloud(
        font_path=font_path,
        width=1600, height=1000,
        background_color='white',
        colormap='inferno',
        max_words=120,
        min_font_size=12,
        max_font_size=160,
        relative_scaling=0.5,
        random_state=42,
        prefer_horizontal=0.85
    ).generate_from_frequencies(wc_freq)
    
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    fig.suptitle("中文经济管理学研究热点词云\n(核心词与复合短语；标题×2，关键词×2，摘要×1)",
                 fontsize=20, fontweight='bold', color='#2c3e50', y=0.94, ha='center')
    plt.subplots_adjust(top=0.86, bottom=0.04, left=0.04, right=0.96)
    plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f">> Word cloud saved to {wordcloud_path}")
    plt.show()

# ======================== 7. 主程序 ========================
if __name__ == "__main__":
    file_pattern = "CN*-*.txt"
    if not glob.glob(file_pattern):
        print(f"[Error] No files matching '{file_pattern}' found.")
        exit(1)
    df_clean = load_and_preprocess_custom_data(file_pattern)
    if df_clean.empty:
        print("[Error] No valid data for 2020-2025.")
        exit(1)
    df_clean = filter_chinese_social_science(df_clean)
    if df_clean.empty:
        print("[Error] No articles remaining after filtering.")
        exit(1)
    print(f">> Final dataset for subsequent analysis: {len(df_clean)} record(s).")
    phrase_list = build_chinese_phrases(df_clean, STOPWORDS_CN, min_freq=2, llr_threshold=0.1)
    word_counter, phrase_counter = tokenize_with_structural_weight(
        df_clean, phrase_list, STOPWORDS_CN, title_weight=2, keyword_weight=2
    )
    generate_output(word_counter, phrase_counter, df_clean)
    print("\n>> All tasks completed successfully!")
