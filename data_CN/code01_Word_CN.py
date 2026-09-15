"""中文期刊数据清洗、人工复核回填与热词统计。

处理顺序：解析原始记录 → 限定年份和期刊 → 删除非论文 →
回填《中国社会科学》人工判断 → 去重 → 导出数据、热词和词云。
"""

from __future__ import annotations

import hashlib
import html
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# 中文路径与绘图缓存。
SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

import jieba
import matplotlib
import pandas as pd
from wordcloud import WordCloud

matplotlib.use("Agg")  # 无图形界面的服务器也能保存图片。
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def configure_chinese_runtime() -> None:
    """配置UTF-8输出和中文字体。"""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    plt.rcParams["font.sans-serif"] = [
        "SimHei", "Microsoft YaHei", "PingFang SC",
        "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    jieba.setLogLevel(20)


configure_chinese_runtime()


# ============================== 配置 ==============================

BASE_DIR = SCRIPT_DIR
INPUT_GLOB = "CN*-*.txt"
TARGET_YEARS = range(2020, 2026)
TARGET_JOURNALS = {"经济研究", "管理世界", "中国社会科学"}

MANUAL_REVIEW_FILE = "CN_Social_Sciences_Manual_Review.csv"
MANUAL_GUIDE_FILE = "CN_Social_Sciences_Manual_Guide.txt"

CLEAN_DATA_FILE = "Cleaned_Custom_Dataset.csv"
HOT_TERMS_FILE = "Chinese_Hot_Terms.csv"
WORDCLOUD_FILE = "Chinese_Phrase_WordCloud.png"

# 文本权重：标题×2，关键词×2，摘要×1。
TEXT_WEIGHTS = {"Title": 2, "Keywords": 2, "Abstract": 1}

# 保留至少2字符的中文、英文或数字词。
TOKEN_PATTERN = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9]+$")

# 停用词：虚词、论文套话、方法词和宽泛词。
STOPWORDS_CN = set(
    """
    的 了 在 是 有 和 与 或 及 而 并 且 对 从 于 为 之 其 所 以 因 被 把
    这 那 这些 那些 一个 一些 有些 所有 全部 任何 多数 少数 若干 许多 很多
    不 都 也 就 已经 曾经 仍然 可能 可以 能够 需要 应当 以及 同时 进一步
    因此 因而 从而 所以 然而 不过 但是 虽然 尽管 如果 无论 不仅 而且 此外
    首先 其次 再次 最后 总之 当前 目前 未来 长期 短期 整体 内部 外部
    主要 重要 基本 根本 显著 明显 有效 相关 直接 间接 不同 其中
    研究 分析 本文 作者 结果 表明 发现 通过 基于 进行 影响 作用 机制
    问题 关系 视角 框架 实践 评估 特征 应用 管理 模式 路径 政策 制度
    创新 结构 要素 配置 效应 提升 推进 促进 发展 建设 水平 实现 推动
    形成 构建 提供 发挥 体现 表现 变化 差异 增加 减少 提高 降低
    方法 理论 文献 综述 结论 讨论 建议 贡献 局限 不足 展望 前言 引言 背景
    回归 估计 检验 假设 模型 数据 样本 变量 稳健 稳健性 控制变量
    固定效应 随机效应 工具变量 内生性 滞后项 差分 交互项
    线性回归 非线性 多元回归 最小二乘法 极大似然 广义矩估计
    面板数据 截面数据 时间序列 描述性统计 相关性分析 主成分 因子分析
    聚类 分类 预测 拟合 残差 方差 标准差 均值 中位数 百分比
    中国 我国 国家 地方 家庭 经济 经济学 增长 衰退 世界 全球 地区 时代
    期刊 管理世界 中国社会科学 经济研究
    一 二 三 四 五 六 七 八 九 十 第一 第二 第三
    """.split()
)


# ============================== 文本与解析 ==============================

def normalize_text(value: object) -> str:
    """统一实体、Unicode和空白。"""
    text = "" if pd.isna(value) else str(value)
    text = html.unescape(text).replace("<正>", " ")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b-\u200f\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def normalized_key(value: object) -> str:
    """生成标准化判重键。"""
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", normalize_text(value).lower())


def parse_cnki_records(content: str) -> list[dict[str, str]]:
    """按题名字段切分CNKI记录。"""
    blocks = re.split(r"(?m)(?=^Title-题名\s*[:：])", content)
    field_start = re.compile(r"^([^\r\n:：]+-[^\r\n:：]+)\s*[:：]\s*(.*)$")
    records: list[dict[str, str]] = []

    for block in blocks:
        if not re.match(r"^Title-题名\s*[:：]", block.lstrip()):
            continue
        record: dict[str, str] = {}
        current_field: str | None = None
        for line in block.splitlines():
            match = field_start.match(line.strip())
            if match:
                current_field = match.group(1).strip()
                record[current_field] = match.group(2).strip()
            elif current_field and line.strip():
                record[current_field] += " " + line.strip()
        if record:
            records.append(record)
    return records


def read_text(path: Path) -> str:
    """读取UTF-8文本。"""
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def export_audit(df: pd.DataFrame, filename: str, base_dir: Path) -> None:
    """导出非空审计表。"""
    if df.empty:
        return
    df.to_csv(base_dir / filename, index=False, encoding="utf-8-sig")
    print(f"   [审计] {filename}: {len(df)} 条")


def load_source_data(base_dir: Path = BASE_DIR) -> pd.DataFrame:
    """载入并限定年份、期刊。"""
    files = sorted(base_dir.glob(INPUT_GLOB))
    if not files:
        raise FileNotFoundError(f"未找到 {INPUT_GLOB}；请把原始txt与代码放在同一文件夹。")

    records: list[dict[str, str]] = []
    for path in files:
        records.extend(parse_cnki_records(read_text(path)))
    if not records:
        raise ValueError("原始文件中未解析到有效记录。")

    rename_map = {
        "Title-题名": "Title",
        "Summary-摘要": "Abstract",
        "Year-年": "Year",
        "Author-作者": "Author",
        "Keyword-关键词": "Keywords",
        "Source-文献来源": "Journal",
        "CLC-中图分类号": "CLC",
    }
    df = pd.DataFrame(records).rename(columns=rename_map)
    text_columns = ["Title", "Abstract", "Year", "Author", "Keywords", "Journal", "CLC"]
    for column in text_columns:
        if column not in df:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str)

    # 缺少题名、摘要或年份则排除。
    complete = (
        df["Title"].str.strip().ne("")
        & df["Abstract"].str.strip().ne("")
        & df["Year"].str.strip().ne("")
    )
    df = df.loc[complete].copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df.loc[df["Year"].isin(TARGET_YEARS)].copy()
    df["Year"] = df["Year"].astype(int)

    clean = pd.DataFrame({
        "Title": df["Title"].map(normalize_text),
        "Abstract": df["Abstract"].map(normalize_text),
        "Year": df["Year"],
        "Author": df["Author"].map(normalize_text),
        "Keywords": df["Keywords"].map(normalize_text),
        "Journal": df["Journal"].map(normalize_text),
        "CLC": df["CLC"].map(normalize_text),
    })

    # 仅保留三本目标期刊。
    in_scope = clean["Journal"].isin(TARGET_JOURNALS)
    export_audit(clean.loc[~in_scope], "Excluded_CN_Out_of_Scope.csv", base_dir)
    result = clean.loc[in_scope].reset_index(drop=True)
    print(f">> 载入 {len(files)} 个文件；年份与期刊筛选后保留 {len(result)} 条。")
    return result


# ============================== 论文体裁筛选 ==============================

# 公告、招聘和编辑性材料。
ADMIN_TITLE_PATTERN = re.compile(
    r"(?:评选|会议|论坛|颁奖典礼|资助计划).*公告$|征文启事$|"
    r"诚聘|招聘|招募|欢迎报名|研修班|大讲堂|"
    r"(?:学院|学科|中心).*简介$|院庆|主编寄语$|关于稿件写作|"
    r"微信公众号|荣获|热烈庆祝|隆重出版$|会议召开$|论坛召开$"
)

# 明确的非论文体裁；理论论文不因无计量方法被删除。
NON_PAPER_PATTERN = re.compile(
    r"笔谈|书评|(?:^|——)评[《〈]|(?:序言|导言|译者序)$|"
    r"访谈|专访|对话|(?:论坛|会议|研讨会).*综述|综述$|"
    r"重要讲话精神|(?:讲话|致辞|贺词|获奖感言)$|"
    r"编者按|卷首语|书讯|会议纪要|学术自传|年谱|"
    r"——纪念|纪念.*(?:周年|诞辰|逝世)"
)


def filter_research_articles(df: pd.DataFrame, base_dir: Path = BASE_DIR) -> pd.DataFrame:
    """删除非论文并记录原因。"""
    missing_author = df["Author"].str.strip().eq("")
    administrative = df["Title"].str.contains(ADMIN_TITLE_PATTERN, na=False)
    non_paper = df["Title"].str.contains(NON_PAPER_PATTERN, na=False)
    remove = missing_author | administrative | non_paper

    excluded = df.loc[remove].copy()
    reasons = pd.Series("", index=df.index, dtype="object")
    reasons.loc[missing_author] = "缺少作者，通常为公告、宣传或编辑性记录"
    reasons.loc[administrative & ~missing_author] = "公告、招聘、宣传或编辑性材料"
    reasons.loc[non_paper & ~missing_author & ~administrative] = "笔谈、书评、序言、访谈或综述等非论文体裁"
    excluded["Removal_Reason"] = reasons.loc[remove]
    export_audit(excluded, "Excluded_CN_NonArticles.csv", base_dir)

    result = df.loc[~remove].reset_index(drop=True)
    print(f">> 非论文筛选删除 {remove.sum()} 条，保留 {len(result)} 篇研究论文候选记录。")
    return result


# ============================== 人工复核 ==============================

MANUAL_TEMPLATE_COLUMNS = [
    "Record_ID", "Year", "Title", "Author", "Keywords",
    "Journal", "CLC", "Abstract", "Manual_Decision",
]
MANUAL_REQUIRED_COLUMNS = ["Record_ID", "Manual_Decision"]
DECISION_ALIASES = {
    "1": "KEEP", "KEEP": "KEEP", "保留": "KEEP",
    "0": "EXCLUDE", "EXCLUDE": "EXCLUDE", "删除": "EXCLUDE",
}


def make_record_id(row: pd.Series) -> str:
    """生成稳定ID，防止人工决定错配。"""
    identity = "|".join([
        str(row["Year"]),
        normalized_key(row["Title"]),
        normalized_key(row["Author"]),
        normalized_key(row["Journal"]),
    ])
    return "CSS_" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]


def build_manual_template(css: pd.DataFrame) -> pd.DataFrame:
    review = css.copy()
    review.insert(0, "Record_ID", review.apply(make_record_id, axis=1))
    review["Manual_Decision"] = ""
    return review[MANUAL_TEMPLATE_COLUMNS]


def write_manual_guide(path: Path) -> None:
    """保存人工复核说明。"""
    path.write_text(
        """《中国社会科学》经济学论文人工复核说明

1. 填1（保留）：主要研究对象或主要贡献在解释经济行为、经济机制或经济结果。
   理论经济学、政治经济学、经济史均可保留，不要求使用回归分析。

2. 填0（排除）：主要贡献属于纯法学、哲学、文学、一般历史、社会学或人类学，
   经济内容仅是背景、案例或政策语境。

3. 跨学科论文：看核心问题与核心贡献，不凭“经济”“实证”等单个词判断。

4. 程序已先删除笔谈、书评、序言、访谈、讲话和会议综述等非论文体裁。

填写要求：只填写Manual_Decision；可用1/0、KEEP/EXCLUDE或保留/删除。
""",
        encoding="utf-8",
    )


def read_csv_robust(path: Path) -> pd.DataFrame:
    """兼容UTF-8和GB18030。"""
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str, keep_default_na=False)
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error or UnicodeDecodeError("utf-8", b"", 0, 1, "无法识别编码")


def normalize_decisions(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper().map(DECISION_ALIASES)


def apply_manual_review(df: pd.DataFrame, base_dir: Path = BASE_DIR) -> pd.DataFrame | None:
    """校验并回填人工决定。"""
    css = df.loc[df["Journal"].eq("中国社会科学")].copy()
    others = df.loc[~df["Journal"].eq("中国社会科学")].copy()
    if css.empty:
        return df.reset_index(drop=True)

    template = build_manual_template(css)
    if template["Record_ID"].duplicated().any():
        duplicated = template.loc[template["Record_ID"].duplicated(keep=False)]
        export_audit(duplicated, "Manual_Review_Duplicate_IDs.csv", base_dir)
        print("[错误] 源数据产生重复Record_ID，请先核对重复记录。")
        return None

    review_path = base_dir / MANUAL_REVIEW_FILE
    if not review_path.exists():
        template.to_csv(review_path, index=False, encoding="utf-8-sig")
        write_manual_guide(base_dir / MANUAL_GUIDE_FILE)
        print(f">> 已生成 {MANUAL_REVIEW_FILE}，共 {len(template)} 条；请填写后再次运行。")
        return None

    review = read_csv_robust(review_path)
    missing = [column for column in MANUAL_REQUIRED_COLUMNS if column not in review]
    if missing:
        print(f"[错误] 人工表缺少必要列：{missing}")
        return None

    review["Record_ID"] = review["Record_ID"].str.strip()
    if review["Record_ID"].duplicated().any():
        print("[错误] 人工表存在重复Record_ID。")
        return None

    current_ids = set(template["Record_ID"])
    reviewed_ids = set(review["Record_ID"])
    if current_ids != reviewed_ids:
        missing_rows = template.loc[~template["Record_ID"].isin(reviewed_ids)].assign(
            Mismatch_Type="Missing_from_manual_file"
        )
        extra_rows = review.loc[~review["Record_ID"].isin(current_ids)].assign(
            Mismatch_Type="Not_in_current_source"
        )
        export_audit(
            pd.concat([missing_rows, extra_rows], ignore_index=True, sort=False),
            "Manual_Review_ID_Mismatch.csv",
            base_dir,
        )
        print("[错误] 人工表与当前源数据的Record_ID不一致，主流程已停止。")
        return None

    decisions = normalize_decisions(review["Manual_Decision"])
    invalid = decisions.isna()
    if invalid.any():
        audit_columns = [
            column for column in ["Record_ID", "Year", "Title", "Manual_Decision"]
            if column in review
        ]
        export_audit(
            review.loc[invalid, audit_columns],
            "Manual_Review_Incomplete_or_Invalid.csv",
            base_dir,
        )
        print(f"[错误] {invalid.sum()} 条决定为空或无效；请使用1/0。")
        return None

    decision_map = pd.Series(decisions.values, index=review["Record_ID"])
    css["Record_ID"] = css.apply(make_record_id, axis=1)
    css["_Decision"] = css["Record_ID"].map(decision_map)

    excluded = css.loc[css["_Decision"].eq("EXCLUDE")].drop(columns="_Decision")
    export_audit(excluded, "Excluded_CN_Manual_Classification.csv", base_dir)

    kept = css.loc[css["_Decision"].eq("KEEP")].drop(columns=["Record_ID", "_Decision"])
    result = pd.concat([others, kept], ignore_index=True)
    print(f">> 人工复核：保留 {len(kept)} 条，排除 {len(excluded)} 条。")
    return result


# ============================== 去重 ==============================

def deduplicate_records(df: pd.DataFrame, base_dir: Path = BASE_DIR) -> pd.DataFrame:
    """按标题+作者或标题+摘要判重。"""
    work = df.copy()
    work["_TitleKey"] = work["Title"].map(normalized_key)
    work["_AuthorKey"] = work["Author"].map(normalized_key)
    work["_AbstractKey"] = work["Abstract"].map(normalized_key)

    # 先删完全重复，再查跨批次重复。
    exact = work.duplicated(
        ["Year", "_TitleKey", "_AuthorKey", "_AbstractKey", "Journal"], keep="first"
    )
    remaining = work.loc[~exact].copy()
    repeated = (
        remaining.duplicated(["_TitleKey", "_AuthorKey"], keep="first")
        | remaining.duplicated(["_TitleKey", "_AbstractKey"], keep="first")
    )
    duplicates = pd.concat([work.loc[exact], remaining.loc[repeated]])
    key_columns = ["_TitleKey", "_AuthorKey", "_AbstractKey"]
    export_audit(duplicates.drop(columns=key_columns), "Duplicate_CN_Records.csv", base_dir)

    result = remaining.loc[~repeated].drop(columns=key_columns).reset_index(drop=True)
    print(f">> 去重删除 {len(df) - len(result)} 条。")
    return result


# ============================== 热词与短语 ==============================

def tokenize(text: str, stopwords: set[str] | None = None) -> list[str]:
    """中文分词并过滤无效字符。"""
    words = [word.strip() for word in jieba.lcut(text or "")]
    return [
        word for word in words
        if len(word) >= 2
        and TOKEN_PATTERN.fullmatch(word)
        and (stopwords is None or word not in stopwords)
    ]


def approximate_llr(observed: int, expected: float, total: int) -> float:
    """计算短语对数似然比。"""
    if observed <= 0 or expected <= 0 or observed >= total or expected >= total:
        return 0.0
    return 2 * (
        observed * math.log(observed / expected)
        + (total - observed) * math.log((total - observed) / (total - expected))
    )


def discover_phrases(
    df: pd.DataFrame,
    min_doc_freq: int = 2,
    llr_threshold: float = 0.1,
) -> list[str]:
    """发现至少出现在两篇论文中的二元、三元短语。"""
    word_freq: Counter[str] = Counter()
    ngram_freq: Counter[tuple[str, ...]] = Counter()
    document_freq: Counter[tuple[str, ...]] = Counter()
    total_words = 0

    for row in df.itertuples(index=False):
        seen_in_document: set[tuple[str, ...]] = set()
        # 分字段处理，避免跨字段拼词。
        for field in TEXT_WEIGHTS:
            words = tokenize(str(getattr(row, field)))
            word_freq.update(words)
            total_words += len(words)
            for size in (2, 3):
                ngrams = [tuple(words[i:i + size]) for i in range(len(words) - size + 1)]
                ngram_freq.update(ngrams)
                seen_in_document.update(ngrams)
        document_freq.update(seen_in_document)

    exact_exclude = {
        "资助_计划", "青年_学者", "乌家培_资助", "乌家培_资助_计划",
        "信息管理_领域", "领域_青年_学者",
    }
    formula_words = {"表明", "发现", "结果"}
    scored: list[tuple[str, float]] = []

    for parts, observed in ngram_freq.items():
        if document_freq[parts] < min_doc_freq:
            continue
        expected = math.prod(word_freq[word] for word in parts) / (total_words ** (len(parts) - 1))
        phrase = "_".join(parts)
        score = approximate_llr(observed, expected, total_words)
        # 删除资助名称和论文套话。
        if (
            phrase not in exact_exclude
            and not all(word in STOPWORDS_CN for word in parts)
            and not any(marker in word for word in parts for marker in formula_words)
            and score >= llr_threshold
        ):
            scored.append((phrase, score))

    scored.sort(key=lambda item: (-item[1], -len(item[0]), item[0]))
    phrases = [phrase for phrase, _ in scored]
    print(f">> 短语发现保留 {len(phrases)} 个候选短语。")
    return phrases


def count_hot_terms(
    df: pd.DataFrame,
    phrases: list[str],
) -> tuple[Counter[str], Counter[str]]:
    """普通词与短语独立计数；二元、三元短语可同时命中。"""
    word_counter: Counter[str] = Counter()
    phrase_counter: Counter[str] = Counter()
    phrase_map = {tuple(phrase.split("_")): phrase for phrase in phrases}

    for row in df.itertuples(index=False):
        for field, weight in TEXT_WEIGHTS.items():
            text = str(getattr(row, field))
            words = tokenize(text)

            # 普通词去停用词；短语按完整序列匹配。
            kept_words = [word for word in words if word not in STOPWORDS_CN]
            word_counter.update(
                {word: count * weight for word, count in Counter(kept_words).items()}
            )
            for size in (2, 3):
                for index in range(len(words) - size + 1):
                    phrase = phrase_map.get(tuple(words[index:index + size]))
                    if phrase:
                        phrase_counter[phrase] += weight
    return word_counter, phrase_counter


def find_chinese_font(base_dir: Path) -> str | None:
    """查找可用中文字体。"""
    candidates = [
        base_dir / "simhei.ttf",
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    return next((str(path) for path in candidates if path.exists()), None)


def export_results(
    df: pd.DataFrame,
    word_counter: Counter[str],
    phrase_counter: Counter[str],
    base_dir: Path = BASE_DIR,
) -> None:
    """导出数据、热词和词云。"""
    output_columns = ["Year", "Title", "Author", "Keywords", "Journal", "CLC", "Abstract"]
    df[output_columns].to_csv(base_dir / CLEAN_DATA_FILE, index=False, encoding="utf-8-sig")

    # 普通词与短语重合时取较大频次。
    combined: dict[str, int] = {}
    sources: dict[str, set[str]] = {}
    for source_name, counter in (("普通词", word_counter), ("复合短语", phrase_counter)):
        for term, frequency in counter.items():
            surface = term.replace("_", "")
            if surface in STOPWORDS_CN or frequency < 3:
                continue
            combined[surface] = max(combined.get(surface, 0), frequency)
            sources.setdefault(surface, set()).add(source_name)

    rows = [
        {
            "Term": term,
            "Weighted_Frequency": frequency,
            "Source": "与".join(sorted(sources[term])),
        }
        for term, frequency in sorted(combined.items(), key=lambda item: (-item[1], item[0]))
    ]
    pd.DataFrame(rows).to_csv(base_dir / HOT_TERMS_FILE, index=False, encoding="utf-8-sig")

    if not combined:
        print("[警告] 没有可用于词云的词语；已跳过图片生成。")
        return

    font_path = find_chinese_font(base_dir)
    if font_path is None:
        print("[警告] 未找到中文字体；已生成CSV，但跳过词云以避免乱码。")
        return

    wordcloud = WordCloud(
        font_path=font_path,
        width=1600,
        height=1000,
        background_color="white",
        colormap="inferno",
        max_words=120,
        min_font_size=12,
        max_font_size=160,
        relative_scaling=0.5,
        random_state=42,
        prefer_horizontal=0.85,
    ).generate_from_frequencies(combined)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    title_font = FontProperties(fname=font_path, size=20, weight="bold")
    fig.suptitle(
        "中文经济管理学研究热点词云\n"
        "（核心词与复合短语；标题×2，关键词×2，摘要×1）",
        fontproperties=title_font,
        color="#2C3E50",
        y=0.96,
    )
    fig.subplots_adjust(top=0.86, bottom=0.02, left=0.02, right=0.98)
    fig.savefig(base_dir / WORDCLOUD_FILE, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    print(f">> 已导出：{CLEAN_DATA_FILE}、{HOT_TERMS_FILE}、{WORDCLOUD_FILE}")


# ============================== 主程序 ==============================

def run(base_dir: Path = BASE_DIR) -> pd.DataFrame | None:
    df = load_source_data(base_dir)
    df = filter_research_articles(df, base_dir)
    df = apply_manual_review(df, base_dir)
    if df is None:
        print("[暂停] 请完成或修正人工复核表后重新运行。")
        return None

    df = deduplicate_records(df, base_dir)
    if df.empty:
        raise ValueError("全部记录均被过滤，未生成最终数据。")

    phrases = discover_phrases(df)
    word_counter, phrase_counter = count_hot_terms(df, phrases)
    export_results(df, word_counter, phrase_counter, base_dir)
    print(f">> 全部完成，最终保留 {len(df)} 篇论文。")
    return df


if __name__ == "__main__":
    try:
        run()
    except Exception as error:
        print(f"[错误] {error}")
        raise SystemExit(1) from error
