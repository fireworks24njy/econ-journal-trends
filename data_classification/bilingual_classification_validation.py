"""Bilingual blind validation for the CN/EN economics-field classifiers.

Usage
-----
1. Generate two sealed 100-record audits:
   python bilingual_classification_validation.py generate

2. Fill only EN_Validation_Blind.csv and CN_Validation_Blind.csv, then evaluate:
   python bilingual_classification_validation.py evaluate

3. About seven days later, create and fill the delayed retest:
   python bilingual_classification_validation.py create-retest
   python bilingual_classification_validation.py evaluate-retest
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


FIELD_ORDER = [
    "Development Economics",
    "Economic History",
    "Finance",
    "Industrial Organization",
    "International Economics",
    "Labor Economics",
    "Macroeconomics",
    "Microeconomics",
    "Public Finance",
    "Miscellaneous & Methods",
]

CN_TO_EN = {
    "发展经济学": "Development Economics",
    "经济史": "Economic History",
    "金融学": "Finance",
    "产业组织": "Industrial Organization",
    "国际经济学": "International Economics",
    "劳动经济学": "Labor Economics",
    "宏观经济学": "Macroeconomics",
    "微观经济学": "Microeconomics",
    "公共财政": "Public Finance",
    "方法与杂项": "Miscellaneous & Methods",
    "未分类": "Unclassified",
}
EN_TO_CN = {v: k for k, v in CN_TO_EN.items()}
CONFIDENCE_MAP = {
    "high": "High", "medium": "Medium", "low": "Low",
    "高": "High", "中": "Medium", "低": "Low",
}
BASE_DIR = Path(__file__).resolve().parent

# Default classified datasets on Windows.  Raw strings are required here so
# backslashes such as ``\D`` are treated as path separators, not escapes.
EN_CLASSIFIED_FILE = Path(
    r"D:\Personal\Desktop\data\data_classification\Classified_Top5_Journals_Result.csv"
)
CN_CLASSIFIED_FILE = Path(
    r"D:\Personal\Desktop\data\data_classification\Cleaned_Custom_Dataset_Classified.csv"
)


def _bool_series(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _normalise_field(value: object) -> str:
    text = str(value).strip()
    return CN_TO_EN.get(text, text)


def _normalise_confidence(value: object) -> str:
    text = str(value).strip()
    return CONFIDENCE_MAP.get(text.lower(), CONFIDENCE_MAP.get(text, text))


def _read_classified(path: Path, language: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "Title", "Abstract", "Raw_Predicted_Field", "Classification_Score",
        "Classification_Margin", "Is_Ambiguous",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if df.empty:
        raise ValueError(f"{path.name} is empty")

    df = df.reset_index(drop=True).copy()
    df["Source_Row"] = np.arange(1, len(df) + 1)
    df["Record_ID"] = [f"{language}{i:06d}" for i in df["Source_Row"]]
    df["Keywords"] = df.get("Keywords", "").fillna("").astype(str)
    df["Title"] = df["Title"].fillna("").astype(str).str.strip()
    df["Abstract"] = df["Abstract"].fillna("").astype(str).str.strip()
    if df["Title"].eq("").any() or df["Abstract"].eq("").any():
        raise ValueError(f"{path.name} contains empty titles or abstracts")

    df["Machine_Field"] = df["Raw_Predicted_Field"].map(_normalise_field)
    unknown = sorted(set(df["Machine_Field"]) - set(FIELD_ORDER))
    if unknown:
        raise ValueError(f"{path.name} contains unknown machine fields: {unknown}")
    df["Is_Ambiguous"] = _bool_series(df["Is_Ambiguous"])
    # Abstention is not a substantive methods category.
    df["Final_Field"] = np.where(df["Is_Ambiguous"], "Unclassified", df["Machine_Field"])
    return df


def _write_csv(df: pd.DataFrame, path: Path, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; use --overwrite only if replacement is intended")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _draw_field_sample(
    group: pd.DataFrame,
    n_per_field: int,
    n_ambiguous: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    total_n = min(n_per_field, len(group))
    ambiguous = group[group["Is_Ambiguous"]]
    clear = group[~group["Is_Ambiguous"]]

    take_ambiguous = min(n_ambiguous, len(ambiguous), total_n)
    take_clear = min(total_n - take_ambiguous, len(clear))
    take_ambiguous = min(total_n - take_clear, len(ambiguous))
    if take_clear + take_ambiguous != total_n:
        raise ValueError(f"Cannot draw {total_n} records from field {group['Machine_Field'].iloc[0]}")

    parts = []
    for subset, n in ((clear, take_clear), (ambiguous, take_ambiguous)):
        if n:
            parts.append(subset.sample(n=n, random_state=int(rng.integers(0, 2**32 - 1))))
    return pd.concat(parts, ignore_index=False)


def _generate_one(
    source_path: Path,
    language: str,
    output_dir: Path,
    n_per_field: int,
    n_ambiguous: int,
    seed: int,
    overwrite: bool,
) -> None:
    df = _read_classified(source_path, language)
    counts = df.groupby(["Machine_Field", "Is_Ambiguous"]).size().rename("Population_N")
    rng = np.random.default_rng(seed + (0 if language == "EN" else 1))
    selected = []

    for field in FIELD_ORDER:
        group = df[df["Machine_Field"].eq(field)]
        if len(group) < n_per_field:
            raise ValueError(f"{language} field {field} has only {len(group)} records")
        desired_ambiguous = 0 if field == "Miscellaneous & Methods" else n_ambiguous
        selected.append(_draw_field_sample(group, n_per_field, desired_ambiguous, rng))

    sample = pd.concat(selected, ignore_index=True)
    sample = sample.sample(frac=1, random_state=seed).reset_index(drop=True)
    sample["Validation_ID"] = [f"{language}-AUDIT-{i:03d}" for i in range(1, len(sample) + 1)]

    sample_n = sample.groupby(["Machine_Field", "Is_Ambiguous"]).size().rename("Sample_n")
    sample = sample.join(counts, on=["Machine_Field", "Is_Ambiguous"])
    sample = sample.join(sample_n, on=["Machine_Field", "Is_Ambiguous"])
    sample["Sampling_Weight"] = sample["Population_N"] / sample["Sample_n"]

    blind = sample[["Validation_ID", "Record_ID", "Title", "Keywords", "Abstract"]].copy()
    blind["Human_Field"] = ""
    blind["Human_Confidence"] = ""
    blind["Notes"] = ""

    key_cols = [
        "Validation_ID", "Record_ID", "Source_Row", "Machine_Field", "Final_Field",
        "Is_Ambiguous", "Classification_Score", "Classification_Margin",
        "Population_N", "Sample_n", "Sampling_Weight",
    ]
    key = sample[key_cols].copy()
    key.insert(0, "Language", language)

    _write_csv(blind, output_dir / f"{language}_Validation_Blind.csv", overwrite)
    _write_csv(key, output_dir / f"{language}_Validation_Key.csv", overwrite)


def generate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    _generate_one(Path(args.en), "EN", output_dir, args.n_per_field, args.n_ambiguous, args.seed, args.overwrite)
    _generate_one(Path(args.cn), "CN", output_dir, args.n_per_field, args.n_ambiguous, args.seed, args.overwrite)
    print(f"Generated two blinded audits in {output_dir.resolve()}")
    print("Fill only the two *_Validation_Blind.csv files; keep both key files sealed.")


def _merge_completed(blind_path: Path, key_path: Path, language: str) -> pd.DataFrame:
    blind = pd.read_csv(blind_path, encoding="utf-8-sig")
    key = pd.read_csv(key_path, encoding="utf-8-sig")
    audit = blind.merge(
        key,
        on=["Validation_ID", "Record_ID"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    bad = audit["_merge"].ne("both")
    if bad.any():
        ids = audit.loc[bad, "Validation_ID"].astype(str).tolist()[:10]
        raise ValueError(f"{language}: blind/key IDs do not match: {ids}")
    audit = audit.drop(columns="_merge")

    audit["Human_Field"] = audit["Human_Field"].fillna("").map(_normalise_field)
    if audit["Human_Field"].eq("").any():
        n = int(audit["Human_Field"].eq("").sum())
        raise ValueError(f"{language}: {n} Human_Field labels are blank")
    invalid = sorted(set(audit["Human_Field"]) - set(FIELD_ORDER))
    if invalid:
        raise ValueError(f"{language}: invalid Human_Field labels: {invalid}")

    audit["Human_Confidence"] = audit["Human_Confidence"].fillna("").map(_normalise_confidence)
    invalid_conf = sorted(set(audit["Human_Confidence"]) - {"High", "Medium", "Low"})
    if invalid_conf:
        raise ValueError(f"{language}: invalid or blank Human_Confidence values: {invalid_conf}")
    audit["Is_Ambiguous"] = _bool_series(audit["Is_Ambiguous"])
    return audit


def _weighted_accuracy(y_true: pd.Series, y_pred: pd.Series, weight: pd.Series) -> float:
    return float(np.average(y_true.to_numpy() == y_pred.to_numpy(), weights=weight.to_numpy()))


def _metric_bundle(audit: pd.DataFrame) -> dict[str, float]:
    weight = audit["Sampling_Weight"].astype(float)
    clear = ~audit["Is_Ambiguous"]
    ambiguous = audit["Is_Ambiguous"]
    result = {
        "Raw_Weighted_Accuracy": _weighted_accuracy(audit["Human_Field"], audit["Machine_Field"], weight),
        "Raw_Weighted_Macro_F1": float(f1_score(
            audit["Human_Field"], audit["Machine_Field"], labels=FIELD_ORDER,
            average="macro", sample_weight=weight, zero_division=0,
        )),
        "Coverage": float(np.average(clear, weights=weight)),
        "Final_Conservative_Accuracy": _weighted_accuracy(audit["Human_Field"], audit["Final_Field"], weight),
        "Clear_Conditional_Accuracy": np.nan,
        "Ambiguous_Diagnostic_Accuracy": np.nan,
    }
    if clear.any():
        result["Clear_Conditional_Accuracy"] = _weighted_accuracy(
            audit.loc[clear, "Human_Field"], audit.loc[clear, "Machine_Field"], weight.loc[clear]
        )
    if ambiguous.any():
        result["Ambiguous_Diagnostic_Accuracy"] = _weighted_accuracy(
            audit.loc[ambiguous, "Human_Field"], audit.loc[ambiguous, "Machine_Field"], weight.loc[ambiguous]
        )
    return result


def _stratified_bootstrap(audit: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strata = [g.reset_index(drop=True) for _, g in audit.groupby(["Machine_Field", "Is_Ambiguous"], sort=False)]
    records = []
    for _ in range(repetitions):
        draw = pd.concat(
            [g.iloc[rng.integers(0, len(g), len(g))] for g in strata],
            ignore_index=True,
        )
        records.append(_metric_bundle(draw))
    return pd.DataFrame(records)


def _save_confusions(audit: pd.DataFrame, language: str, output_dir: Path) -> None:
    weight = audit["Sampling_Weight"].astype(float)
    raw = confusion_matrix(
        audit["Human_Field"], audit["Machine_Field"], labels=FIELD_ORDER, sample_weight=weight
    )
    raw_df = pd.DataFrame(raw, index=FIELD_ORDER, columns=FIELD_ORDER)
    raw_df.index.name = "Human_Field"
    raw_df.to_csv(output_dir / f"{language}_Weighted_Confusion_Matrix.csv", encoding="utf-8-sig")
    row_sum = raw_df.sum(axis=1).replace(0, np.nan)
    raw_df.div(row_sum, axis=0).to_csv(
        output_dir / f"{language}_Row_Normalized_Confusion_Matrix.csv", encoding="utf-8-sig"
    )

    final_labels = FIELD_ORDER + ["Unclassified"]
    final = confusion_matrix(
        audit["Human_Field"], audit["Final_Field"], labels=final_labels, sample_weight=weight
    )
    final_df = pd.DataFrame(final, index=final_labels, columns=final_labels)
    final_df.index.name = "Human_Field"
    final_df.to_csv(output_dir / f"{language}_Final_Weighted_Confusion_Matrix.csv", encoding="utf-8-sig")


def _evaluate_one(language: str, output_dir: Path, bootstrap: int, seed: int) -> pd.DataFrame:
    audit = _merge_completed(
        output_dir / f"{language}_Validation_Blind.csv",
        output_dir / f"{language}_Validation_Key.csv",
        language,
    )
    point = _metric_bundle(audit)
    boot = _stratified_bootstrap(audit, bootstrap, seed + (0 if language == "EN" else 1))
    rows = []
    for metric, estimate in point.items():
        values = boot[metric].dropna()
        rows.append({
            "Language": language,
            "Metric": metric,
            "Estimate": estimate,
            "CI_95_Lower": values.quantile(0.025) if len(values) else np.nan,
            "CI_95_Upper": values.quantile(0.975) if len(values) else np.nan,
            "Bootstrap_Repetitions": bootstrap,
            "Audit_N": len(audit),
        })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / f"{language}_Validation_Metrics.csv", index=False, encoding="utf-8-sig")

    precision, recall, f1, support = precision_recall_fscore_support(
        audit["Human_Field"], audit["Machine_Field"], labels=FIELD_ORDER,
        sample_weight=audit["Sampling_Weight"], zero_division=0,
    )
    pd.DataFrame({
        "Language": language,
        "Field": FIELD_ORDER,
        "Weighted_Precision": precision,
        "Weighted_Recall": recall,
        "Weighted_F1": f1,
        "Weighted_Support": support,
    }).to_csv(output_dir / f"{language}_Field_Metrics.csv", index=False, encoding="utf-8-sig")

    _save_confusions(audit, language, output_dir)
    errors = audit[audit["Human_Field"].ne(audit["Machine_Field"])].copy()
    error_cols = [
        "Validation_ID", "Record_ID", "Title", "Human_Field", "Machine_Field",
        "Final_Field", "Human_Confidence", "Is_Ambiguous", "Classification_Score",
        "Classification_Margin", "Sampling_Weight", "Notes",
    ]
    errors[error_cols].to_csv(
        output_dir / f"{language}_Misclassified_Cases.csv", index=False, encoding="utf-8-sig"
    )
    return metrics


def evaluate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    results = [
        _evaluate_one("EN", output_dir, args.bootstrap, args.seed),
        _evaluate_one("CN", output_dir, args.bootstrap, args.seed),
    ]
    pd.concat(results, ignore_index=True).to_csv(
        output_dir / "Bilingual_Validation_Summary.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Validation outputs saved in {output_dir.resolve()}")


def _create_retest_one(language: str, output_dir: Path, n_retest: int, seed: int, overwrite: bool) -> None:
    audit = _merge_completed(
        output_dir / f"{language}_Validation_Blind.csv",
        output_dir / f"{language}_Validation_Key.csv",
        language,
    )
    rng = np.random.default_rng(seed + (0 if language == "EN" else 1))
    selected_indices: list[int] = []
    per_field = max(1, n_retest // len(FIELD_ORDER))
    for field in FIELD_ORDER:
        candidates = audit.index[audit["Human_Field"].eq(field)].difference(selected_indices)
        n = min(per_field, len(candidates))
        if n:
            selected_indices.extend(rng.choice(candidates.to_numpy(), size=n, replace=False).tolist())
    if len(selected_indices) < n_retest:
        remaining = audit.index.difference(selected_indices)
        extra = min(n_retest - len(selected_indices), len(remaining))
        selected_indices.extend(rng.choice(remaining.to_numpy(), size=extra, replace=False).tolist())
    retest = audit.loc[selected_indices].copy()
    retest = retest.sample(frac=1, random_state=seed).reset_index(drop=True)
    retest["Retest_ID"] = [f"{language}-RETEST-{i:03d}" for i in range(1, len(retest) + 1)]

    blind = retest[["Retest_ID", "Record_ID", "Title", "Keywords", "Abstract"]].copy()
    blind["Retest_Field"] = ""
    blind["Retest_Confidence"] = ""
    blind["Notes"] = ""
    key = retest[[
        "Retest_ID", "Validation_ID", "Record_ID", "Human_Field", "Human_Confidence"
    ]].rename(columns={
        "Human_Field": "Original_Human_Field",
        "Human_Confidence": "Original_Human_Confidence",
    })
    key.insert(0, "Language", language)

    _write_csv(blind, output_dir / f"{language}_Retest_Blind.csv", overwrite)
    _write_csv(key, output_dir / f"{language}_Retest_Key.csv", overwrite)


def create_retest(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    _create_retest_one("EN", output_dir, args.n_retest, args.seed, args.overwrite)
    _create_retest_one("CN", output_dir, args.n_retest, args.seed, args.overwrite)
    print("Delayed blinded retest files created. Do not open the two retest key files.")


def _evaluate_retest_one(language: str, output_dir: Path) -> dict[str, object]:
    blind = pd.read_csv(output_dir / f"{language}_Retest_Blind.csv", encoding="utf-8-sig")
    key = pd.read_csv(output_dir / f"{language}_Retest_Key.csv", encoding="utf-8-sig")
    data = blind.merge(key, on=["Retest_ID", "Record_ID"], how="outer", validate="one_to_one", indicator=True)
    if data["_merge"].ne("both").any():
        raise ValueError(f"{language}: retest blind/key IDs do not match")
    data["Retest_Field"] = data["Retest_Field"].fillna("").map(_normalise_field)
    if data["Retest_Field"].eq("").any():
        raise ValueError(f"{language}: retest labels are incomplete")
    invalid = sorted(set(data["Retest_Field"]) - set(FIELD_ORDER))
    if invalid:
        raise ValueError(f"{language}: invalid retest labels: {invalid}")

    original = data["Original_Human_Field"].map(_normalise_field)
    repeated = data["Retest_Field"]
    matrix = confusion_matrix(original, repeated, labels=FIELD_ORDER)
    matrix_df = pd.DataFrame(matrix, index=FIELD_ORDER, columns=FIELD_ORDER)
    matrix_df.index.name = "Original_Human_Field"
    matrix_df.to_csv(output_dir / f"{language}_Retest_Confusion_Matrix.csv", encoding="utf-8-sig")
    return {
        "Language": language,
        "Retest_N": len(data),
        "Raw_Agreement": float((original == repeated).mean()),
        "Cohens_Kappa": float(cohen_kappa_score(original, repeated, labels=FIELD_ORDER)),
    }


def evaluate_retest(args: argparse.Namespace) -> None:
    output_dir = Path(args.output)
    metrics = pd.DataFrame([
        _evaluate_retest_one("EN", output_dir),
        _evaluate_retest_one("CN", output_dir),
    ])
    metrics.to_csv(output_dir / "Bilingual_Retest_Metrics.csv", index=False, encoding="utf-8-sig")
    print(f"Retest metrics saved in {output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blind validation of CN/EN economics-field classification")
    sub = parser.add_subparsers(dest="command", required=True)

    generate_parser = sub.add_parser("generate", help="generate blinded audit samples and sealed keys")
    generate_parser.add_argument("--en", default=str(EN_CLASSIFIED_FILE))
    generate_parser.add_argument("--cn", default=str(CN_CLASSIFIED_FILE))
    generate_parser.add_argument("--output", default=str(BASE_DIR / "classification_validation"))
    generate_parser.add_argument("--n-per-field", type=int, default=10)
    generate_parser.add_argument("--n-ambiguous", type=int, default=2)
    generate_parser.add_argument("--seed", type=int, default=20260831)
    generate_parser.add_argument("--overwrite", action="store_true")
    generate_parser.set_defaults(func=generate)

    evaluate_parser = sub.add_parser("evaluate", help="evaluate completed initial blind labels")
    evaluate_parser.add_argument("--output", default=str(BASE_DIR / "classification_validation"))
    evaluate_parser.add_argument("--bootstrap", type=int, default=1000)
    evaluate_parser.add_argument("--seed", type=int, default=20260831)
    evaluate_parser.set_defaults(func=evaluate)

    retest_parser = sub.add_parser("create-retest", help="create delayed blinded retest files")
    retest_parser.add_argument("--output", default=str(BASE_DIR / "classification_validation"))
    retest_parser.add_argument("--n-retest", type=int, default=20)
    retest_parser.add_argument("--seed", type=int, default=20260907)
    retest_parser.add_argument("--overwrite", action="store_true")
    retest_parser.set_defaults(func=create_retest)

    retest_eval_parser = sub.add_parser("evaluate-retest", help="evaluate completed delayed retest")
    retest_eval_parser.add_argument("--output", default=str(BASE_DIR / "classification_validation"))
    retest_eval_parser.set_defaults(func=evaluate_retest)
    return parser


def _columns_complete(path: Path, columns: list[str]) -> bool:
    data = pd.read_csv(path, encoding="utf-8-sig")
    if any(column not in data.columns for column in columns):
        return False
    return all(data[column].fillna("").astype(str).str.strip().ne("").all() for column in columns)


def _auto_run(parser: argparse.ArgumentParser) -> None:
    output_dir = BASE_DIR / "classification_validation"
    audit_blinds = [output_dir / f"{lang}_Validation_Blind.csv" for lang in ("EN", "CN")]
    audit_keys = [output_dir / f"{lang}_Validation_Key.csv" for lang in ("EN", "CN")]
    audit_files = audit_blinds + audit_keys

    if not any(path.exists() for path in audit_files):
        arguments = parser.parse_args(["generate"])
        arguments.func(arguments)
        return
    if not all(path.exists() for path in audit_files):
        raise FileNotFoundError("The initial audit is incomplete. Restore the missing blind/key file.")
    if not all(_columns_complete(path, ["Human_Field", "Human_Confidence"]) for path in audit_blinds):
        print("Fill EN_Validation_Blind.csv and CN_Validation_Blind.csv, then run again.")
        return

    summary = output_dir / "Bilingual_Validation_Summary.csv"
    if not summary.exists() or max(path.stat().st_mtime for path in audit_blinds) > summary.stat().st_mtime:
        arguments = parser.parse_args(["evaluate"])
        arguments.func(arguments)
        return

    retest_blinds = [output_dir / f"{lang}_Retest_Blind.csv" for lang in ("EN", "CN")]
    retest_keys = [output_dir / f"{lang}_Retest_Key.csv" for lang in ("EN", "CN")]
    retest_files = retest_blinds + retest_keys
    if not any(path.exists() for path in retest_files):
        due = datetime.fromtimestamp(summary.stat().st_mtime) + timedelta(days=7)
        if datetime.now() < due:
            print(f"Initial evaluation complete. Run again on or after {due:%Y-%m-%d}.")
            return
        arguments = parser.parse_args(["create-retest"])
        arguments.func(arguments)
        return
    if not all(path.exists() for path in retest_files):
        raise FileNotFoundError("The retest is incomplete. Restore the missing blind/key file.")
    if not all(_columns_complete(path, ["Retest_Field", "Retest_Confidence"]) for path in retest_blinds):
        print("Fill EN_Retest_Blind.csv and CN_Retest_Blind.csv, then run again.")
        return

    retest_metrics = output_dir / "Bilingual_Retest_Metrics.csv"
    if not retest_metrics.exists() or max(path.stat().st_mtime for path in retest_blinds) > retest_metrics.stat().st_mtime:
        arguments = parser.parse_args(["evaluate-retest"])
        arguments.func(arguments)
        return
    print("All outputs are current.")


if __name__ == "__main__":
    command_parser = build_parser()
    if len(sys.argv) == 1:
        _auto_run(command_parser)
    else:
        arguments = command_parser.parse_args()
        arguments.func(arguments)
