<div align="center">


# econ-journal-trends

### Research Topics and Field Trends in Chinese and English Economics Journals, 2020–2025

Cross-lingual text mining · Zero-shot field classification · Trend analysis · Textual concentration

[Latest Report](Phase%205%20-%20September%2016.pdf) · [English Analysis](data_top5/) · [Chinese Analysis](data_CN/) · [Cross-language Comparison](data_Comparison/) · [Classification Validation](data_classification/)

</div>

---

## About

This project examines research topics and field composition in leading Chinese and English economics journals from 2020 to 2025.

Using bibliographic records from Web of Science and CNKI, it combines natural language processing with statistical analysis to:

* extract high-frequency words and multiword expressions;
* classify papers into ten economics fields;
* validate classification accuracy through stratified sampling and independent blind review;
* estimate annual changes in field composition;
* compare Chinese and English journal samples;
* measure textual concentration within the English economics Top 5.

> **Language note:** This README is written in English for broader accessibility. The accompanying phase reports are written in Chinese.

The complete research design, statistical results, robustness checks, and discussion are available in the [latest report](Phase%205%20-%20September%2016.pdf).

## Data

| Corpus  | Source         | Journal coverage                                           |  Period   | Final sample |
| :------ | :------------- | :--------------------------------------------------------- | :-------: | -----------: |
| English | Web of Science | *AER*, *Econometrica*, *JPE*, *QJE*, and *ReStud*          | 2020–2025 |    **2,467** |
| Chinese | CNKI           | 《经济研究》《管理世界》《中国社会科学》中的经济学相关文章 | 2020–2025 |    **1,592** |

For the English corpus, only research articles are retained; comments, book reviews, corrections, and duplicate records are excluded.

For the Chinese corpus, interviews, book reviews, discussion columns, and other non-research materials are excluded. Candidate papers from 《中国社会科学》 are additionally reviewed to identify economics-related research.

## Research Workflow

```text
Raw records
    │
    ├── Data cleaning and standardization
    │
    ├── Weighted text construction
    │       └── Title × 2 + Keywords × 2 + Abstract
    │
    ├── Word and phrase extraction
    │       └── Unigrams + LLR-filtered bigrams and trigrams
    │
    ├── Research-field classification
    │       └── TF-IDF + Vector Space Model
    │
    ├── Classification validation
    │       └── Stratified sampling + independent blind review
    │
    └── Statistical analysis
            ├── Annual field trends
            ├── Chinese–English comparison
            └── Within-field textual concentration
```

## Methods at a Glance

| Task                    | Main approach                                                |
| :---------------------- | :----------------------------------------------------------- |
| Text construction       | Titles and keywords receive weight 2; abstracts receive weight 1 |
| Phrase extraction       | Log-likelihood ratio for bigrams and trigrams, with $G^2 \ge 10$ |
| Field classification    | TF-IDF weighting, vector space representation, and cosine similarity |
| Semantic benchmark      | Sentence-BERT embeddings for the English corpus              |
| Validation              | Stratified random sampling and independent blind human review |
| Trend estimation        | Multinomial and binary Logit models                          |
| Distributional tests    | Pearson chi-square tests and Cramér’s $V$                    |
| Local pattern detection | Breakpoint scans and permutation tests                       |
| Multiple testing        | Benjamini–Hochberg correction                                |
| Sensitivity analysis    | Alternative classification rules and leave-one-year-out estimation |
| Textual concentration   | Top-20 term share under multiple weighting and sampling specifications |

## Research-Field Classification

Each paper is assigned to one of the ten economics fields defined in the research report:

| No.  | Field                   |
| :--: | :---------------------- |
|  1   | Development Economics   |
|  2   | Economic History        |
|  3   | Finance                 |
|  4   | Industrial Organization |
|  5   | International Economics |
|  6   | Labor Economics         |
|  7   | Macroeconomics          |
|  8   | Microeconomics          |
|  9   | Public Finance          |
|  10  | Miscellaneous& Methods  |

Three classification specifications are retained:

* **Final classification:** the primary classification after applying confidence and adjustment rules;
* **Raw classification:** the field with the highest original similarity score;
* **High-confidence classification:** a restricted specification excluding low-confidence observations.

The main results are compared across these specifications to assess their sensitivity to classification choices.

## Classification Validation

From each language corpus, 100 papers are selected through stratified random sampling. The validation samples cover all candidate fields and deliberately allocate 20% of observations to low-confidence or boundary cases.

Sampling weights are used to recover accuracy estimates for the full sample. Each selected paper is independently blind-reviewed using its title, keywords, and abstract.

* **Top-1 accuracy** measures whether the model’s first-ranked field agrees with the independently blind-reviewed field.
* **Top-2 accuracy** measures whether the blind-reviewed field appears among the model’s two highest-ranked fields.

### Validation Results

| Corpus           | Weighted Top-1 accuracy | Weighted Top-2 accuracy |
| :--------------- | ----------------------: | ----------------------: |
| English Top 5    |               **79.5%** |               **91.0%** |
| Chinese journals |               **76.5%** |                       — |

The validation analysis further shows that:

* fields with relatively distinctive terminology, including International Economics, Finance, and Labor Economics, achieve comparatively strong classification performance;
* most disagreements occur near substantive boundaries, particularly between Microeconomics, Macroeconomics, and Miscellaneous& Methods;
* classification errors are not concentrated in particular years;
* among reviewed low-confidence observations, approximately **68.4%** are interdisciplinary, broadly framed, or primarily methodological.

### Comparison with Sentence-BERT

| Method           | English weighted agreement |
| :--------------- | -------------------------: |
| **TF-IDF + VSM** |                  **79.5%** |
| Sentence-BERT    |                      61.6% |

TF-IDF + VSM is retained as the primary classification method because it performs better in the validation sample while preserving interpretability and a consistent framework across Chinese and English texts.

## Main Findings

### 1. High-Frequency Terms

| Corpus           | Representative phrases                                       | Representative words                               |
| :--------------- | :----------------------------------------------------------- | :------------------------------------------------- |
| English Top 5    | `monetary policy`, `labor market`, `long run`, `interest rate`, `business cycle` | `market`, `policy`, `information`, `firm`, `price` |
| Chinese journals | 高质量发展、数字经济、地方政府、要素生产率、实体经济         | 企业、市场、数字、风险、政府                       |

### 2. English Top 5 Trends

* The overall field composition remains highly stable from 2020 to 2025.
* Neither the multinomial Logit model nor the Pearson chi-square test identifies a significant systematic change in the full field distribution.
* Labor Economics shows a moderate upward tendency, with an annual odds ratio of **1.082**, but the result does not remain significant after correction across ten fields ($q_{\mathrm{BH}} = 0.2026$).
* Public Finance reaches a temporary low in 2022, while Industrial Organization declines during 2023–2024. Neither pattern remains significant at the 5% level after multiple-testing correction, so both are treated as exploratory findings.

### 3. Chinese-Journal Trends

* Under the final classification, annual field composition differs statistically across years, but the effect size is small (Cramér’s $V = 0.0886$).
* The significance of the overall difference is sensitive to the classification specification.
* International Economics shows a relatively clear upward pattern, although the corrected stage-comparison result is slightly above the conventional 5% threshold ($q_{\mathrm{BH}} = 0.0594$).
* Leave-one-year-out analysis indicates that this result is substantially influenced by the high share observed in 2025. It is therefore interpreted as a stage-specific upward signal around 2025 rather than an established long-run trend.

### 4. Chinese–English Comparison

* The average field composition differs significantly between the two journal groups.
* The English Top 5 sample places relatively greater emphasis on Microeconomics.
* Finance, Industrial Organization, and Public Finance account for larger shares of the Chinese journal sample.
* Whether year is treated as a continuous or categorical variable, the analysis does not identify a significant difference in the overall temporal trajectories of the two groups from 2020 to 2025.

### 5. Textual Concentration

| Comparison                                          | Main result                                                  | Interpretation                                               |
| :-------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Labor Economics vs. Development Economics           | Labor Economics is higher by **3.96 percentage points**; $q_{\mathrm{BH}} = 0.0010$ | The most stable concentration difference across specifications |
| Microeconomics vs. Macroeconomics                   | **13.94% vs. 13.90%**                                        | No substantive difference in overall concentration           |
| Industrial Organization vs. International Economics | Industrial Organization is more concentrated                 | Sensitive to term granularity                                |
| Finance vs. Public Finance                          | Finance is more concentrated; $q_{\mathrm{BH}} = 0.0461$     | Also sensitive to term granularity                           |

The greater visibility of microeconomic terms in the full-sample frequency analysis does not imply that Microeconomics has a higher within-field concentration than Macroeconomics. It mainly reflects its larger publication base and the widespread use of terms such as information, equilibrium, and mechanism design across applied fields.

## Robustness Checks

The main analyses are evaluated under several alternative specifications.

### Classification robustness

* final classification;
* raw highest-similarity classification;
* high-confidence classification.

### Trend robustness

* year as a continuous variable;
* year as a categorical variable;
* breakpoint scans;
* permutation tests;
* leave-one-year-out estimation;
* Benjamini–Hochberg correction.

### Concentration robustness

* pooled term-frequency weighting;
* equal weighting across papers;
* equal weighting across years;
* document-occurrence counts;
* bigrams only;
* equal-sample-size bootstrap resampling.

These checks are used to distinguish stable results from findings that depend on a particular classification rule, time specification, or term-counting method.

## Repository Structure

```text
econ-journal-trends/
├── data_CN/                         # Chinese data and field-trend analysis
│   ├── chinese_field_analysis_output/
│   ├── code01_Word_CN.py
│   ├── code02_Category_CN.py
│   └── code03_chinese_journal_field_analysis.py
│
├── data_top5/                       # English Top 5 data and analysis
│   ├── code01_Words_top5.py
│   ├── code02_Category_top5.py
│   ├── code03_Category_Sentence-BERT.py
│   ├── code04_trend_analysis.py
│   ├── code05_english_all_field_concentration_analysis.py
│   ├── trend_analysis_outputs.zip
│   └── english_all_field_concentration_output.zip
│
├── data_Comparison/                # Chinese–English comparison
│   ├── cn_en_group_comparison_analysis.py
│   └── cn_en_group_comparison_output/
│
├── data_classification/            # Classification validation
│   ├── bilingual_classification_validation.py
│   ├── EN_Validation.csv
│   └── CN_Validation.csv
│
├── figure/                         # Figures used in the reports
├── (Galofré-Vilà, 2026).pdf         # Reference paper
├── Phase 1 - July 22.pdf
├── Phase 2 - August 04.pdf
├── Phase 3 - August 15.pdf
├── Phase 4 - August 31.pdf
├── Phase 5 - September 16.pdf       # Latest full report
└── README.md
```

## Reproduction Guide

The scripts follow the order of the research workflow:

1. Run the word and phrase extraction scripts:

   * `data_CN/code01_Word_CN.py`
   * `data_top5/code01_Words_top5.py`

2. Generate the field classifications:

   * `data_CN/code02_Category_CN.py`
   * `data_top5/code02_Category_top5.py`

3. Run the Sentence-BERT benchmark:

   * `data_top5/code03_Category_Sentence-BERT.py`

4. Validate the Chinese and English classifications:

   * `data_classification/bilingual_classification_validation.py`

5. Estimate field trends:

   * `data_CN/code03_chinese_journal_field_analysis.py`
   * `data_top5/code04_trend_analysis.py`

6. Compare the Chinese and English journal samples:

   * `data_Comparison/cn_en_group_comparison_analysis.py`

7. Analyze within-field textual concentration:

   * `data_top5/code05_english_all_field_concentration_analysis.py`

Some scripts depend on intermediate datasets generated in earlier stages. Local file paths may need to be adjusted before execution.

## Scope and Interpretation

This project should be interpreted within three boundaries:

1. **Classification uncertainty.**
   The ten-field structure provides a consistent basis for comparison, but some papers genuinely span multiple fields. Top-1 assignments necessarily simplify these interdisciplinary cases. The Top-2 results, low-confidence labels, and alternative classification specifications are therefore reported alongside the main classification.

2. **Sample scope.**
   The results describe the selected English Top 5 and Chinese journal samples. They should not be treated as estimates of the field composition of all economics research published in English or Chinese.

3. **Meaning of textual concentration.**
   The concentration indicator is constructed from words and multiword expressions. It measures how strongly a field’s textual usage is concentrated among its most frequent terms, not whether the field’s underlying research questions are inherently narrow or homogeneous. Synonyms, general-purpose terms, and alternative phrasing may also affect the measured concentration.

## Reports

| Stage | Report                                                       | Main focus                                       |
| :---: | :----------------------------------------------------------- | :----------------------------------------------- |
|   1   | [Phase 1 – July 22](Phase%201%20-%20July%2022.pdf)           | Initial data processing and exploratory analysis |
|   2   | [Phase 2 – August 04](Phase%202%20-%20August%2004.pdf)       | Text extraction and classification development   |
|   3   | [Phase 3 – August 15](Phase%203%20-%20August%2015.pdf)       | Field classification and trend analysis          |
|   4   | [Phase 4 – August 31](Phase%204%20-%20August%2031.pdf)       | Extended statistical tests and comparisons       |
|   5   | [Phase 5 – September 16](Phase%205%20-%20September%2016.pdf) | Current complete report                          |

> The phase reports are written in Chinese. Phase 5 is the latest and most complete version of the study.
