"""Clean English Top-5 journal records and create a topic word cloud.

Rules:
1. Keep English research articles published in 2020--2025.
2. Exclude comments, replies, corrections, notices, and administrative items.
3. Deduplicate by DOI; use title-year-journal only when DOI is missing.
4. Do not read, weight, or export citation counts.
5. Save every exclusion and manual metadata correction for audit.
"""

from __future__ import annotations

import io
import os
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
    from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
except ImportError as exc:
    raise ImportError("NLTK is required. Install it with: pip install nltk") from exc

try:
    from wordcloud import STOPWORDS as WORDCLOUD_STOPWORDS
    from wordcloud import WordCloud
except ImportError:  # Cleaning remains available without the plotting package.
    WORDCLOUD_STOPWORDS = set()
    WordCloud = None

# ============================== Configuration ==============================

BASE_DIR = Path(__file__).resolve().parent
RAW_FILES = (
    "0-500.txt",
    "501-1000.txt",
    "1001-1500.txt",
    "1501-2000.txt",
    "2001-2500.txt",
    "2501-2658.txt",
)

YEAR_START, YEAR_END = 2020, 2025
ALLOWED_TYPES = {"Article", "Article; Early Access"}
TOP5_JOURNALS = {
    "AMERICAN ECONOMIC REVIEW",
    "ECONOMETRICA",
    "JOURNAL OF POLITICAL ECONOMY",
    "QUARTERLY JOURNAL OF ECONOMICS",
    "REVIEW OF ECONOMIC STUDIES",
}

TITLE_WEIGHT = 2
KEYWORD_WEIGHT = 2
ABSTRACT_WEIGHT = 1
MIN_TERM_FREQUENCY = 10
MIN_PHRASE_FREQUENCY = 5
LLR_THRESHOLD = 10.0
MAX_BIGRAMS = 300
MAX_TRIGRAMS = 100

# Verified against the official AEA records.
TITLE_CORRECTIONS = {
    "happy times measuring happiness using response times": {
        "PY": "2023",
        "DI": "10.1257/aer.20211051",
        "Reason": "WoS year and DOI metadata error",
    }
}
DOI_YEAR_CORRECTIONS = {
    "10.1257/aer.20211145": 2025,
}

# Patterns are anchored to structural title forms to avoid false exclusions.
NON_RESEARCH_TITLE_RULES = {
    "Comment or discussion": (
        r"(?i)^\s*(?:a\s+)?comments?\s+(?:on|to)\b"
        r"|^\s*(?:a\s+)?discussion\s+(?:of|on)\b"
        r"|:\s*(?:a\s+)?comments?\s+(?:on|to)\b.*$"
        r"|:\s*(?:a\s+)?comments?t?\s*$"
    ),
    "Reply, response or rejoinder": (
        r"(?i)^\s*(?:a\s+)?(?:reply|response|rejoinder)\s+(?:to|on)\b"
        r"|:\s*(?:a\s+)?(?:reply|response|rejoinder)(?:\s+to\b.*)?\s*$"
    ),
    "Correction or retraction": (
        r"(?i)^\s*(?:erratum|corrigendum|correction\s+(?:to|of)|"
        r"retraction|withdrawal|addendum)\b"
        r"|:\s*(?:erratum|corrigendum|correction|retraction)\s*$"
    ),
    "Editorial or administrative item": (
        r"(?i)^\s*(?:editorial|editor['’]?s note|publisher['’]?s note|"
        r"notice|announcement|call for papers|forthcoming papers|recent referees|"
        r"jpe turnaround times|report of independent auditor|"
        r"the econometric society .*annual reports?|in memoriam|obituary)\b"
    ),
}

# Remove academic boilerplate, but retain substantive economic concepts.
ACADEMIC_STOPWORDS = {
    "abstract", "analysis", "analyze", "approach", "article", "author",
    "conclusion", "consider", "data", "dataset", "difference", "effect",
    "empirical", "estimate", "evidence", "examine", "find", "finding",
    "framework", "implication", "increase", "investigate", "literature",
    "measure", "method", "model", "objective", "outcome", "paper",
    "provide", "regression", "result", "robust", "sample", "section",
    "show", "significant", "specification", "study", "table", "use",
    "using", "variable", "year",
}
BASIC_ENGLISH_STOPWORDS = set("""
    a about above after again against all am an and any are as at be because been
    before being below between both but by can could did do does doing down during
    each few for from further had has have having he her here hers herself him
    himself his how i if in into is it its itself just me more most my myself no
    nor not now of off on once only or other our ours ourselves out over own same
    she should so some such than that the their theirs them themselves then there
    these they this those through to too under until up very was we were what when
    where which while who whom why will with would you your yours yourself
""".split())
STOPWORDS = (
    {word.casefold() for word in WORDCLOUD_STOPWORDS}
    | BASIC_ENGLISH_STOPWORDS
    | ACADEMIC_STOPWORDS
)


# ============================== Text utilities =============================

def configure_utf8() -> None:
    """Use UTF-8 for Chinese paths and console messages on Windows."""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
        elif hasattr(stream, "buffer"):
            setattr(sys, name, io.TextIOWrapper(stream.buffer, encoding="utf-8"))


def normalize_doi(value: object) -> str:
    doi = str(value or "").strip().casefold()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    return re.sub(r"^doi:\s*", "", doi).strip()


def normalize_title(value: object) -> str:
    """Create a punctuation-insensitive title key."""
    return re.sub(r"[^\w]+", " ", str(value or "").casefold()).strip()


def lemmatize(token: str) -> str:
    """Apply conservative plural normalization without external corpora."""
    irregular = {
        "analyses": "analysis", "children": "child", "countries": "country",
        "economies": "economy", "firms": "firm", "indices": "index",
        "men": "man", "policies": "policy", "studies": "study", "women": "woman",
    }
    protected = {"analysis", "basis", "bias", "business", "crisis", "economics", "news", "series"}
    if token in irregular:
        return irregular[token]
    if token in protected:
        return token
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def text_tokens(text: object) -> list[str]:
    words = re.findall(r"\b[a-z]{2,40}\b", str(text).casefold())
    return [lemmatize(word) for word in words]


# ============================== Data cleaning ==============================

def read_wos_files(base_dir: Path = BASE_DIR) -> pd.DataFrame:
    """Read the six named WoS exports and reject accidental file mixing."""
    paths = [base_dir / name for name in RAW_FILES]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {', '.join(missing)}")

    frames = []
    reference_columns = None
    for path in paths:
        frame = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig")
        if reference_columns is None:
            reference_columns = frame.columns.tolist()
        elif frame.columns.tolist() != reference_columns:
            raise ValueError(f"Column mismatch in {path.name}")
        frame["_Source_File"] = path.name
        frame["_Source_Row"] = range(2, len(frame) + 2)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    print(f"[Load] {len(data):,} records from {len(paths)} files.")
    return data


def apply_manual_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Correct only verified metadata errors and export the changes."""
    records = []
    title_key = df["TI"].map(normalize_title)

    for key, correction in TITLE_CORRECTIONS.items():
        mask = title_key.eq(key)
        for index in df.index[mask]:
            old_year, old_doi = df.at[index, "PY"], df.at[index, "DI"]
            df.at[index, "PY"] = correction["PY"]
            df.at[index, "DI"] = correction["DI"]
            records.append({
                "Source_File": df.at[index, "_Source_File"],
                "Source_Row": df.at[index, "_Source_Row"],
                "Title": df.at[index, "TI"],
                "Old_Year": old_year,
                "Corrected_Year": correction["PY"],
                "Old_DOI": old_doi,
                "Corrected_DOI": correction["DI"],
                "Reason": correction["Reason"],
            })

    doi_key = df["DI"].map(normalize_doi)
    for doi, year in DOI_YEAR_CORRECTIONS.items():
        mask = doi_key.eq(doi) & df["PY"].ne(str(year))
        for index in df.index[mask]:
            old_year = df.at[index, "PY"]
            df.at[index, "PY"] = str(year)
            records.append({
                "Source_File": df.at[index, "_Source_File"],
                "Source_Row": df.at[index, "_Source_Row"],
                "Title": df.at[index, "TI"],
                "Old_Year": old_year,
                "Corrected_Year": year,
                "Old_DOI": df.at[index, "DI"],
                "Corrected_DOI": df.at[index, "DI"],
                "Reason": "Verified publication-year conflict",
            })

    columns = [
        "Source_File", "Source_Row", "Title", "Old_Year", "Corrected_Year",
        "Old_DOI", "Corrected_DOI", "Reason",
    ]
    pd.DataFrame(records, columns=columns).to_csv(
        BASE_DIR / "Top5_Manual_Corrections.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return df


def best_duplicate_mask(candidate: pd.DataFrame, keys: list[str]) -> pd.Series:
    """Mark duplicate rows while retaining the most complete final record."""
    ranked = candidate.assign(
        _FinalArticle=candidate["DT"].eq("Article").astype(int),
        _AbstractLength=candidate["AB"].str.len(),
        _HasRecordID=candidate["UT"].ne("").astype(int),
    ).sort_values(
        keys + ["_FinalArticle", "_AbstractLength", "_HasRecordID", "_Input_Order"],
        ascending=[True] * len(keys) + [False, False, False, True],
        kind="stable",
    )
    return ranked.duplicated(keys, keep="first").reindex(candidate.index, fill_value=False)


def clean_records(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply transparent, sequential rules and write a complete audit trail."""
    df = raw.copy()
    required = {"TI", "AB", "PY", "DT", "SO", "LA", "AU", "AF", "DI", "UT"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing WoS column(s): {sorted(missing)}")

    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].fillna("").astype(str).str.strip()

    df["_Input_Order"] = range(len(df))
    df = apply_manual_corrections(df)
    df["_Year"] = pd.to_numeric(df["PY"], errors="coerce")
    df["_DOI_Key"] = df["DI"].map(normalize_doi)
    df["_Title_Key"] = df["TI"].map(normalize_title)
    df["_Journal_Key"] = df["SO"].str.upper().str.strip()

    active = pd.Series(True, index=df.index)
    excluded, flow = [], []

    def exclude(mask: pd.Series, reason: str) -> None:
        nonlocal active
        hit = active & mask.fillna(False)
        before = int(active.sum())
        if hit.any():
            rows = df.loc[hit].copy()
            rows["Exclusion_Reason"] = reason
            excluded.append(rows)
        active &= ~hit
        flow.append({
            "Step": reason,
            "N_Before": before,
            "N_Excluded": int(hit.sum()),
            "N_After": int(active.sum()),
        })

    # Eligibility rules precede completeness checks so audit reasons stay meaningful.
    exclude(df["TI"].eq(""), "Missing title")
    exclude(df["_Year"].isna(), "Missing or invalid year")
    exclude(~df["_Year"].between(YEAR_START, YEAR_END), "Outside 2020-2025")
    exclude(~df["_Journal_Key"].isin(TOP5_JOURNALS), "Not a designated Top-5 journal")
    exclude(~df["LA"].str.casefold().eq("english"), "Non-English record")
    exclude(~df["DT"].isin(ALLOWED_TYPES), "Non-article document type")

    for label, pattern in NON_RESEARCH_TITLE_RULES.items():
        exclude(df["TI"].str.contains(pattern, regex=True, na=False), f"Non-research item: {label}")

    # Abstracts are required because topic frequencies use comparable text fields.
    exclude(df["AB"].eq(""), "Missing abstract")
    exclude(df["AU"].eq("") & df["AF"].eq(""), "Missing author")

    candidate = df.loc[active].copy()
    with_doi = candidate[candidate["_DOI_Key"].ne("")]
    duplicate_doi = best_duplicate_mask(with_doi, ["_DOI_Key"])
    exclude(df.index.to_series().isin(duplicate_doi.index[duplicate_doi]), "Duplicate DOI")

    # The fallback key applies only when at least one duplicate record lacks a DOI.
    candidate = df.loc[active].copy()
    fallback_keys = ["_Title_Key", "_Year", "_Journal_Key"]
    duplicated_group = candidate.duplicated(fallback_keys, keep=False)
    group_has_doi = candidate.groupby(fallback_keys, dropna=False)["_DOI_Key"].transform(
        lambda values: values.ne("").any()
    )
    missing_doi_duplicate = duplicated_group & candidate["_DOI_Key"].eq("") & group_has_doi
    no_doi_pool = candidate[duplicated_group & ~group_has_doi]
    duplicate_without_doi = best_duplicate_mask(no_doi_pool, fallback_keys)
    fallback_indexes = set(candidate.index[missing_doi_duplicate]) | set(
        duplicate_without_doi.index[duplicate_without_doi]
    )
    exclude(
        df.index.to_series().isin(fallback_indexes),
        "Duplicate title-year-journal with missing DOI",
    )

    audit_columns = [
        "_Source_File", "_Source_Row", "UT", "TI", "SO", "DT", "PY", "DI",
        "Exclusion_Reason",
    ]
    exclusion_log = (
        pd.concat(excluded, ignore_index=True)[audit_columns]
        if excluded else pd.DataFrame(columns=audit_columns)
    )
    exclusion_log.rename(columns={
        "_Source_File": "Source_File", "_Source_Row": "Source_Row",
        "UT": "Record_ID", "TI": "Title", "SO": "Journal",
        "DT": "Document_Type", "PY": "Source_Year", "DI": "DOI",
    }).to_csv(BASE_DIR / "Top5_Cleaning_Exclusion_Log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(flow).to_csv(
        BASE_DIR / "Top5_Cleaning_Flow.csv", index=False, encoding="utf-8-sig"
    )

    valid = df.loc[active].copy()
    blank = pd.Series("", index=valid.index)
    author = valid["AU"].where(valid["AU"].ne(""), valid["AF"])
    keywords = (
        valid.get("DE", blank).fillna("") + "; " + valid.get("ID", blank).fillna("")
    ).str.replace(r"(?:;\s*)+", "; ", regex=True).str.strip("; ")
    categories = (
        valid.get("WC", blank).fillna("") + "; " + valid.get("SC", blank).fillna("")
    ).str.replace(r"(?:;\s*)+", "; ", regex=True).str.strip("; ")

    clean = pd.DataFrame({
        "Record_ID": valid["UT"].where(
            valid["UT"].ne(""),
            valid["_Source_File"] + ":" + valid["_Source_Row"].astype(str),
        ),
        "Year": valid["_Year"].astype(int),
        "Title": valid["TI"],
        "Author": author,
        "Journal": valid["SO"],
        "Document_Type": valid["DT"],
        "DOI": valid["_DOI_Key"],
        "Keywords": keywords,
        "WoS_Category": categories,
        "Abstract": valid["AB"],
    }).sort_values(["Year", "Journal", "Title"], kind="stable").reset_index(drop=True)

    print(f"[Clean] Retained {len(clean):,} research articles.")
    return clean


# ============================== Topic extraction ============================

def mine_phrases(df: pd.DataFrame) -> list[str]:
    """Mine document-bounded bigrams and trigrams using NLTK LLR."""
    documents = (
        df["Title"].fillna("") + ". "
        + df["Keywords"].fillna("") + ". "
        + df["Abstract"].fillna("")
    )
    tokenized = [text_tokens(text) for text in documents]

    def valid(*words: str) -> bool:
        return all(len(word) >= 3 and word not in STOPWORDS for word in words)

    def select(finder, measure, limit: int) -> list[str]:
        finder.apply_freq_filter(MIN_PHRASE_FREQUENCY)
        finder.apply_ngram_filter(lambda *words: not valid(*words))
        scored = finder.score_ngrams(measure)
        return [
            " ".join(term)
            for term, score in scored
            if score >= LLR_THRESHOLD
        ][:limit]

    bigrams = select(
        BigramCollocationFinder.from_documents(tokenized),
        BigramAssocMeasures.likelihood_ratio,
        MAX_BIGRAMS,
    )
    trigrams = select(
        TrigramCollocationFinder.from_documents(tokenized),
        TrigramAssocMeasures.likelihood_ratio,
        MAX_TRIGRAMS,
    )
    return trigrams + bigrams


def weighted_documents(df: pd.DataFrame) -> pd.Series:
    return (
        (df["Title"].fillna("") + ". ") * TITLE_WEIGHT
        + (df["Keywords"].fillna("") + ". ") * KEYWORD_WEIGHT
        + (df["Abstract"].fillna("") + ". ") * ABSTRACT_WEIGHT
    )


def count_topics(df: pd.DataFrame, phrases: list[str]) -> tuple[Counter, Counter]:
    """Count single terms and phrases separately on the same weighted text."""
    single_counts, phrase_counts = Counter(), Counter()
    phrase_patterns = [
        (phrase, re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)"))
        for phrase in sorted(phrases, key=len, reverse=True)
    ]

    for text in weighted_documents(df):
        normalized = " ".join(text_tokens(text))
        single_counts.update(
            token for token in normalized.split()
            if len(token) >= 3 and token not in STOPWORDS
        )
        for phrase, pattern in phrase_patterns:
            count = len(pattern.findall(normalized))
            if count:
                phrase_counts[phrase] += count

    singles = Counter({k: v for k, v in single_counts.items() if v >= MIN_TERM_FREQUENCY})
    multiwords = Counter({k: v for k, v in phrase_counts.items() if v >= MIN_TERM_FREQUENCY})
    return singles, multiwords


def save_frequency_table(counter: Counter, path: Path, term_type: str) -> None:
    rows = [
        {"Rank": rank, "Term": term, "Frequency": frequency, "Type": term_type}
        for rank, (term, frequency) in enumerate(counter.most_common(), 1)
    ]
    pd.DataFrame(rows, columns=["Rank", "Term", "Frequency", "Type"]).to_csv(
        path, index=False, encoding="utf-8-sig"
    )


def create_wordcloud(single_counts: Counter, phrase_counts: Counter) -> None:
    """Plot core terms and multiword phrases with their observed frequencies."""
    if WordCloud is None:
        print("[Warning] Word cloud skipped. Install it with: pip install wordcloud")
        return

    frequencies = dict(single_counts)
    frequencies.update(phrase_counts)
    if not frequencies:
        raise ValueError("No terms meet the minimum-frequency threshold.")

    cloud = WordCloud(
        width=1800,
        height=1100,
        background_color="white",
        colormap="plasma",
        max_words=140,
        min_font_size=12,
        max_font_size=170,
        relative_scaling=0.5,
        prefer_horizontal=0.85,
        random_state=42,
    ).generate_from_frequencies(frequencies)

    fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        "Research Topics in Top-5 Economics Journals\n"
        "(Core Terms and Multiword Phrases; Title ×2, Keywords ×2, Abstract ×1)",
        fontsize=20,
        color="#2F4354",
        pad=20,
    )
    fig.tight_layout()
    fig.savefig(BASE_DIR / "Top5_Economics_WordCloud.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_top_terms(single_counts: Counter, phrase_counts: Counter) -> None:
    print("\nTop 10 terms")
    for rank, (term, count) in enumerate(single_counts.most_common(10), 1):
        print(f"{rank:>2}. {term:<28} {count:,}")

    print("\nTop 5 phrases")
    for rank, (term, count) in enumerate(phrase_counts.most_common(5), 1):
        print(f"{rank:>2}. {term:<28} {count:,}")


def main() -> None:
    configure_utf8()
    raw = read_wos_files()
    clean = clean_records(raw)
    clean.to_csv(
        BASE_DIR / "Cleaned_Top5_Journals_Dataset.csv",
        index=False,
        encoding="utf-8-sig",
    )

    phrases = mine_phrases(clean)
    single_counts, phrase_counts = count_topics(clean, phrases)
    save_frequency_table(single_counts, BASE_DIR / "Top5_Word_Frequencies.csv", "word")
    save_frequency_table(phrase_counts, BASE_DIR / "Top5_Phrase_Frequencies.csv", "phrase")
    create_wordcloud(single_counts, phrase_counts)
    print_top_terms(single_counts, phrase_counts)
    print("\n[Done] Cleaning data, audit files, frequencies, and figure were exported.")


if __name__ == "__main__":
    main()
