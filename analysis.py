"""
End-to-end analysis and ML pipeline for Shaastra Institute Open House data.

Run:
    python analysis.py

The script loads ./IOH.csv, creates plots in ./plots, trains multiple models,
and prints a non-technical publicity insights report.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CSV_PATH = Path("IOH.csv")
PLOTS_DIR = Path("plots")
RANDOM_STATE = 42
SIGNIFICANT_F1_DIFF = 0.02


warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="darkgrid", palette="Set2")


def clean_empty_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convert blank-looking strings into real missing values."""
    return df.replace(r"^\s*$", np.nan, regex=True)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def detect_target_column(df: pd.DataFrame) -> tuple[str | None, pd.Series | None]:
    """Find the attendance/conversion column and return a clean binary target."""
    preferred_names = [
        "ispresent",
        "present",
        "attended",
        "attendance",
        "isattended",
        "converted",
        "conversion",
        "showedup",
        "checkin",
        "checkedin",
        "status",
    ]
    positive_words = {"1", "yes", "y", "true", "present", "attended", "converted", "showed", "checkedin", "visited"}
    negative_words = {"0", "no", "n", "false", "absent", "notattended", "notpresent", "registered", "missed"}

    candidates: list[tuple[int, str, pd.Series]] = []
    for col in df.columns:
        raw = df[col]
        non_null = raw.dropna()
        if non_null.empty:
            continue

        norm_name = normalize_column_name(col)
        name_score = 0
        for idx, preferred in enumerate(preferred_names):
            if preferred in norm_name:
                name_score = 100 - idx
                break

        mapped = non_null.astype(str).str.strip().str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
        unique_values = set(mapped.unique())
        is_binary_numeric = pd.to_numeric(non_null, errors="coerce").dropna().isin([0, 1]).all()
        is_binary_text = unique_values.issubset(positive_words | negative_words) and len(unique_values) <= 2

        if is_binary_numeric or is_binary_text or name_score > 0:
            target = raw.copy()
            if is_binary_numeric:
                target = pd.to_numeric(target, errors="coerce")
            else:
                cleaned = raw.astype(str).str.strip().str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)
                target = cleaned.map({word: 1 for word in positive_words})
                target = target.fillna(cleaned.map({word: 0 for word in negative_words}))

            target = pd.to_numeric(target, errors="coerce")
            if target.dropna().nunique() == 2:
                binary_score = 25
            elif target.dropna().nunique() == 1 and name_score > 0:
                binary_score = 5
            else:
                continue

            candidates.append((name_score + binary_score, col, target))

    if not candidates:
        return None, None

    candidates.sort(reverse=True, key=lambda item: item[0])
    _, target_col, target = candidates[0]
    target = target.fillna(target.mode().iloc[0] if not target.mode().empty else 0).astype(int)
    return target_col, target


def find_first_column(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Find a column whose name contains any of the requested keywords."""
    for col in df.columns:
        norm = normalize_column_name(col)
        if any(keyword in norm for keyword in keywords):
            return col
    return None


def detect_datetime_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """Try to identify useful datetime columns without forcing IDs into dates."""
    datetime_cols: list[str] = []
    name_hints = ["time", "date", "created", "registered", "timestamp", "submitted"]
    for col in df.columns:
        if col in exclude:
            continue
        norm = normalize_column_name(col)
        if not any(hint in norm for hint in name_hints):
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        if parsed.notna().mean() >= 0.5:
            datetime_cols.append(col)
    return datetime_cols


def add_datetime_features(df: pd.DataFrame, datetime_cols: list[str]) -> tuple[pd.DataFrame, str | None]:
    """Add hour/day/month features and keep the best datetime column for insights."""
    df = df.copy()
    timeline_col = None
    for col in datetime_cols:
        parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        if parsed.notna().sum() == 0:
            continue
        safe = normalize_column_name(col)
        df[f"{safe}_hour"] = parsed.dt.hour
        df[f"{safe}_dayofweek"] = parsed.dt.dayofweek
        df[f"{safe}_month"] = parsed.dt.month
        df[f"{safe}_day"] = parsed.dt.day
        if timeline_col is None:
            df[f"{safe}_parsed_datetime"] = parsed
            timeline_col = f"{safe}_parsed_datetime"
    return df, timeline_col


def apply_ioh_cleaning_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Apply data-specific cleaning rules for the IOH registration export."""
    df = df.copy()

    if "userId" in df.columns:
        df = df.drop(columns=["userId"])

    if "companyName" in df.columns:
        df["hasCompany"] = df["companyName"].notna().astype(int)
        df = df.drop(columns=["companyName"])

    if {"collegeName", "category"}.issubset(df.columns):
        non_student_mask = df["category"].ne("COLLEGE_STUDENT")
        df.loc[non_student_mask & df["collegeName"].isna(), "collegeName"] = "Not Applicable"
        df.loc[df["category"].eq("COLLEGE_STUDENT") & df["collegeName"].isna(), "collegeName"] = "Missing College"
        df.loc[df["collegeName"].notna(), "collegeName"] = (
            df.loc[df["collegeName"].notna(), "collegeName"].astype(str).str.strip().str.title()
        )

    if "groupSize" in df.columns:
        group_size = pd.to_numeric(df["groupSize"], errors="coerce")
        df["groupSize"] = group_size
        df["groupSize_capped"] = group_size.clip(upper=6)
        df["isGroup"] = (group_size > 1).astype(int)

    return df


def make_placeholder_plot(filename: str, title: str, message: str) -> None:
    """Save a readable placeholder when a requested plot cannot be computed."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=13, wrap=True)
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_registration_vs_attendance(df: pd.DataFrame, target_col: str) -> None:
    counts = df[target_col].value_counts().reindex([0, 1], fill_value=0)
    labels = ["Registered only", "Attended"]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=labels, y=counts.values, ax=ax)
    ax.set_title("Registration vs Attendance", fontsize=15, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Number of people")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value, str(int(value)), ha="center", va="bottom", weight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "registration_vs_attendance.png", dpi=160)
    plt.close(fig)


def save_conversion_by_group(df: pd.DataFrame, target_col: str, group_col: str | None, filename: str, title: str) -> None:
    if not group_col or group_col not in df.columns:
        make_placeholder_plot(filename, title, "No matching college/department column was found.")
        return

    grouped = (
        df.dropna(subset=[group_col])
        .groupby(group_col)
        .agg(conversion_rate=(target_col, "mean"), registrations=(target_col, "size"))
        .sort_values(["conversion_rate", "registrations"], ascending=[False, False])
        .head(15)
        .sort_values("conversion_rate")
    )
    if grouped.empty:
        make_placeholder_plot(filename, title, "Not enough non-empty group data to compute conversion rates.")
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    sns.barplot(x=grouped["conversion_rate"] * 100, y=grouped.index, ax=ax, orient="h")
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel("Conversion rate (%)")
    ax.set_ylabel(group_col)
    for value, label in zip(grouped["conversion_rate"] * 100, ax.patches):
        ax.text(value + 0.5, label.get_y() + label.get_height() / 2, f"{value:.1f}%", va="center")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_conversion_rate_chart(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    filename: str,
    title: str,
    horizontal: bool = False,
) -> None:
    if group_col not in df.columns:
        make_placeholder_plot(filename, title, f"No {group_col} column was found.")
        return

    grouped = (
        df.dropna(subset=[group_col])
        .groupby(group_col)
        .agg(conversion_rate=(target_col, "mean"), registrations=(target_col, "size"))
        .sort_values("conversion_rate", ascending=False)
    )
    if grouped.empty:
        make_placeholder_plot(filename, title, f"No usable {group_col} values were found.")
        return

    plot_df = grouped.reset_index()
    plot_df["conversion_pct"] = plot_df["conversion_rate"] * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    if horizontal:
        sns.barplot(data=plot_df, x="conversion_pct", y=group_col, ax=ax, orient="h")
        ax.set_xlabel("Conversion rate (%)")
        ax.set_ylabel(group_col)
        for value, patch in zip(plot_df["conversion_pct"], ax.patches):
            ax.text(value + 0.5, patch.get_y() + patch.get_height() / 2, f"{value:.1f}%", va="center")
    else:
        sns.barplot(data=plot_df, x=group_col, y="conversion_pct", ax=ax)
        ax.set_xlabel(group_col)
        ax.set_ylabel("Conversion rate (%)")
        ax.tick_params(axis="x", rotation=25)
        for idx, value in enumerate(plot_df["conversion_pct"]):
            ax.text(idx, value, f"{value:.1f}%", ha="center", va="bottom", weight="bold")
    ax.set_title(title, fontsize=15, weight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(feature_names: list[str], importances: np.ndarray, filename: str = "feature_importance.png") -> None:
    order = np.argsort(importances)[-20:]
    if len(order) == 0 or np.nan_to_num(importances).sum() == 0:
        make_placeholder_plot(filename, "Feature Importance", "Feature importance was not available for the best model.")
        return

    plot_df = pd.DataFrame({"feature": np.array(feature_names)[order], "importance": importances[order]})
    plot_df = plot_df.sort_values("importance")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.barplot(data=plot_df, x="importance", y="feature", ax=ax, orient="h")
    ax.set_title("Top Model Feature Importances", fontsize=15, weight="bold")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_plot(cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix", fontsize=15, weight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(["Not attended", "Attended"])
    ax.set_yticklabels(["Not attended", "Attended"], rotation=0)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def save_roc_curve(y_test: pd.Series, y_score: np.ndarray, auc: float) -> None:
    if len(np.unique(y_test)) < 2:
        make_placeholder_plot("roc_curve.png", "ROC Curve", "ROC curve needs both attendance classes in the test data.")
        return
    fpr, tpr, _ = roc_curve(y_test, y_score)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
    ax.set_title("ROC Curve", fontsize=15, weight="bold")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "roc_curve.png", dpi=160)
    plt.close(fig)


def save_top_colleges(df: pd.DataFrame, college_col: str | None) -> None:
    if not college_col or college_col not in df.columns:
        make_placeholder_plot("top_10_colleges_registration_count.png", "Top 10 Colleges by Registration Count", "No college column was found.")
        return

    counts = df[college_col].dropna().value_counts().head(10).sort_values()
    if counts.empty:
        make_placeholder_plot("top_10_colleges_registration_count.png", "Top 10 Colleges by Registration Count", "College values were empty.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(x=counts.values, y=counts.index, ax=ax, orient="h")
    ax.set_title("Top 10 Colleges by Registration Count", fontsize=15, weight="bold")
    ax.set_xlabel("Registrations")
    ax.set_ylabel(college_col)
    for value, patch in zip(counts.values, ax.patches):
        ax.text(value + 0.2, patch.get_y() + patch.get_height() / 2, str(int(value)), va="center")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "top_10_colleges_registration_count.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_registration_vs_attendance_by_category(df: pd.DataFrame, target_col: str) -> None:
    if "category" not in df.columns:
        make_placeholder_plot(
            "registration_vs_attendance_by_category.png",
            "Registrations vs Actual Attendance by Category",
            "No category column was found.",
        )
        return

    category_order = ["COLLEGE_STUDENT", "FAMILY", "OTHERS", "INDUSTRY_PROFESSIONAL", "INSTITUTE"]
    summary = (
        df.groupby("category")
        .agg(registered=(target_col, "size"), attended=(target_col, "sum"))
        .reindex(category_order)
        .dropna(how="all")
        .reset_index()
    )
    if summary.empty:
        make_placeholder_plot(
            "registration_vs_attendance_by_category.png",
            "Registrations vs Actual Attendance by Category",
            "No category values were found.",
        )
        return

    summary[["registered", "attended"]] = summary[["registered", "attended"]].astype(int)
    plot_df = summary.melt(
        id_vars="category",
        value_vars=["registered", "attended"],
        var_name="Metric",
        value_name="Count",
    )
    plot_df["Metric"] = plot_df["Metric"].map({"registered": "Total registered", "attended": "Actual attended"})

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_df, x="category", y="Count", hue="Metric", ax=ax)
    ax.set_title("Registrations vs Actual Attendance by Category", fontsize=15, weight="bold")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=9)
    ax.legend(title="")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "registration_vs_attendance_by_category.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def smooth_target_encode(train_df, test_df, col, target, smoothing=10):
    global_mean = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(['mean', 'count'])
    stats['smoothed'] = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
    train_encoded = train_df[col].map(stats['smoothed']).fillna(global_mean)
    test_encoded = test_df[col].map(stats['smoothed']).fillna(global_mean)
    return train_encoded, test_encoded


def apply_target_encoding_from_train(train_df: pd.DataFrame, df_to_encode: pd.DataFrame, col: str, target: str, smoothing: int = 10) -> pd.Series:
    global_mean = train_df[target].mean()
    stats = train_df.groupby(col)[target].agg(["mean", "count"])
    stats["smoothed"] = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
    return df_to_encode[col].map(stats["smoothed"]).fillna(global_mean)


def add_smoothed_target_features(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()
    for col in ["collegeName", "city", "state"]:
        train_encoded[f"{col}_target"], test_encoded[f"{col}_target"] = smooth_target_encode(
            train_df, test_df, col, target_col
        )
    return train_encoded, test_encoded


def build_model_matrix(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Keep exactly the raw columns needed to create the final feature set."""
    required_cols = [
        "groupSize_capped",
        "isGroup",
        "hasCompany",
        "registerType",
        "category",
        "collegeName",
        "city",
        "state",
        target_col,
    ]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required model columns after cleaning: {missing}")
    return df[required_cols].copy()


def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    final_cols = [
        "groupSize_capped",
        "isGroup",
        "hasCompany",
        "registerType",
        "category",
        "collegeName_target",
        "city_target",
        "state_target",
    ]
    return df[final_cols].copy()


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = ["groupSize_capped", "isGroup", "hasCompany", "collegeName_target", "city_target", "state_target"]
    categorical_cols = ["registerType", "category"]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", encoder),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


def get_feature_names(preprocessor: ColumnTransformer, numeric_cols: list[str], categorical_cols: list[str]) -> list[str]:
    feature_names = list(numeric_cols)
    if categorical_cols:
        encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        feature_names.extend(encoder.get_feature_names_out(categorical_cols).tolist())
    return feature_names


def model_importance(model, feature_names: list[str]) -> np.ndarray:
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_)
    if hasattr(model, "coef_"):
        return np.abs(np.asarray(model.coef_).ravel())
    return np.zeros(len(feature_names))


def choose_cv_splits(y: pd.Series) -> int:
    smallest_class = int(y.value_counts().min()) if y.nunique() > 1 else 0
    if smallest_class < 2:
        return 0
    return min(5, smallest_class)


def cross_val_target_encoded_model(model, model_df: pd.DataFrame, target_col: str, splits: int) -> np.ndarray:
    scores = []
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE)
    y_base = model_df[target_col]
    for train_idx, valid_idx in cv.split(model_df.drop(columns=[target_col]), y_base):
        fold_train = model_df.iloc[train_idx].copy()
        fold_valid = model_df.iloc[valid_idx].copy()
        fold_train, fold_valid = add_smoothed_target_features(fold_train, fold_valid, target_col)
        X_fold_train = select_final_features(fold_train)
        y_fold_train = fold_train[target_col]
        X_fold_valid = select_final_features(fold_valid)
        y_fold_valid = fold_valid[target_col]
        preprocessor, _, _ = build_preprocessor(X_fold_train)
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", clone(model))])
        pipeline.fit(X_fold_train, y_fold_train)
        y_score = pipeline.predict_proba(X_fold_valid)[:, 1]
        scores.append(roc_auc_score(y_fold_valid, y_score))
    return np.asarray(scores)


def print_publicity_metrics(
    df: pd.DataFrame,
    target_col: str,
    college_col: str | None,
    timeline_col: str | None,
    predicted_probabilities: pd.Series,
) -> None:
    print_section("PUBLICITY METRICS AND SIMPLE INSIGHTS")
    print("Data source: all counts and rates below are computed from IOH.csv.")
    print("Model-estimated values are explicitly marked as model estimates.")
    overall_rate = df[target_col].mean() * 100
    print(f"Overall conversion rate: {overall_rate:.2f}%")

    if "category" in df.columns:
        category_rates = df.groupby("category")[target_col].mean().sort_values(ascending=False)
        institute_rate = category_rates.get("INSTITUTE")
        others_rate = category_rates.get("OTHERS")
        if pd.notna(institute_rate):
            top_note = "highest of all segments, prioritize them" if category_rates.index[0] == "INSTITUTE" else "prioritize based on this strong conversion"
            print(f"\nINSTITUTE category has {institute_rate * 100:.1f}% conversion - {top_note}.")
        if pd.notna(others_rate):
            low_note = "lowest, reconsider outreach effort here" if category_rates.index[-1] == "OTHERS" else "review outreach effort here"
            print(f"OTHERS category has only {others_rate * 100:.1f}% conversion - {low_note}.")

    if "registerType" in df.columns:
        register_rates = df.groupby("registerType")[target_col].mean()
        group_rate = register_rates.get("GROUP")
        individual_rate = register_rates.get("INDIVIDUAL")
        if pd.notna(group_rate) and pd.notna(individual_rate):
            lift = (group_rate - individual_rate) * 100
            print(
                f"GROUP registrations convert {lift:.1f} percentage points better than INDIVIDUAL "
                f"({group_rate * 100:.1f}% vs {individual_rate * 100:.1f}%) - encourage group sign-ups."
            )

    if "state" in df.columns:
        state_rates = df.groupby("state")[target_col].mean().sort_values()
        if not state_rates.empty:
            lowest_state = state_rates.index[0]
            lowest_rate = state_rates.iloc[0] * 100
            print(f"{lowest_state} has the lowest state conversion ({lowest_rate:.1f}%) - needs targeted follow-up.")
        karnataka_rate = state_rates.get("Karnataka")
        if pd.notna(karnataka_rate):
            print(f"Karnataka conversion: {karnataka_rate * 100:.1f}% - targeted follow-up recommended.")

    if "hasCompany" in df.columns:
        has_company_rates = df.groupby("hasCompany")[target_col].mean()
        known_company_rate = has_company_rates.get(1)
        if pd.notna(known_company_rate):
            count = int((df["hasCompany"] == 1).sum())
            print(f"hasCompany = 1 conversion: {known_company_rate * 100:.1f}% from {count} industry/company-known registrations.")

    if college_col and college_col in df.columns:
        college_stats = (
            df.dropna(subset=[college_col])
            .groupby(college_col)
            .agg(conversion_rate=(target_col, "mean"), registrations=(target_col, "size"))
            .sort_values(["conversion_rate", "registrations"], ascending=[False, False])
        )
        print("\nTop 5 colleges by conversion:")
        if college_stats.empty:
            print("Not enough college data.")
        else:
            for college, row in college_stats.head(5).iterrows():
                print(f"- {college}: {row['conversion_rate'] * 100:.1f}% conversion from {int(row['registrations'])} registrations")

        print("\nHigh-registration but low-conversion colleges needing attention:")
        if college_stats.empty:
            print("Not enough college data.")
        else:
            attention = college_stats[college_stats["conversion_rate"] < df[target_col].mean()].sort_values(
                ["registrations", "conversion_rate"], ascending=[False, True]
            )
            for college, row in attention.head(5).iterrows():
                print(f"- {college}: {int(row['registrations'])} registrations, {row['conversion_rate'] * 100:.1f}% conversion")
            if attention.empty:
                print("No college with below-overall conversion was found.")

        print("\nNeeds Attention colleges (50+ registrations and below 35% conversion):")
        if college_stats.empty:
            print("Not enough college data.")
        else:
            needs_attention = college_stats[
                (college_stats["registrations"] >= 50) & (college_stats["conversion_rate"] < 0.35)
            ].sort_values(["registrations", "conversion_rate"], ascending=[False, True])
            if needs_attention.empty:
                print("No college matched the 50+ registrations and below 35% conversion rule.")
            else:
                for college, row in needs_attention.iterrows():
                    print(f"- {college}: {int(row['registrations'])} registrations, {row['conversion_rate'] * 100:.1f}% conversion")
    else:
        print("\nCollege-level insights unavailable because no college column was found.")

    if timeline_col and timeline_col in df.columns and df[timeline_col].notna().any():
        time_df = df.dropna(subset=[timeline_col]).copy()
        time_df["registration_day"] = time_df[timeline_col].dt.day_name()
        time_df["registration_hour"] = time_df[timeline_col].dt.hour
        best_day = time_df["registration_day"].value_counts().idxmax()
        best_hour = int(time_df["registration_hour"].value_counts().idxmax())
        print(f"\nBest day and time for registrations: {best_day} around {best_hour:02d}:00")
    else:
        print("\nBest day and time for registrations: unavailable because no datetime column was found.")

    segment_candidates = [
        find_first_column(df, ["gender", "sex"]),
        find_first_column(df, ["year", "batch"]),
        find_first_column(df, ["branch", "department", "dept", "category"]),
        college_col,
        find_first_column(df, ["city"]),
        find_first_column(df, ["state"]),
    ]
    best_segment = None
    risk_segment = None
    for col in [c for c in segment_candidates if c and c in df.columns]:
        stats = (
            df.dropna(subset=[col])
            .groupby(col)
            .agg(conversion_rate=(target_col, "mean"), registrations=(target_col, "size"))
        )
        if stats.empty:
            continue
        top = stats.sort_values(["conversion_rate", "registrations"], ascending=[False, False]).head(1)
        low = stats.sort_values(["conversion_rate", "registrations"], ascending=[True, False]).head(1)
        if best_segment is None or top.iloc[0]["conversion_rate"] > best_segment[2]:
            best_segment = (col, top.index[0], top.iloc[0]["conversion_rate"], top.iloc[0]["registrations"])
        if risk_segment is None or low.iloc[0]["conversion_rate"] < risk_segment[2]:
            risk_segment = (col, low.index[0], low.iloc[0]["conversion_rate"], low.iloc[0]["registrations"])

    if best_segment:
        col, value, rate, count = best_segment
        print(f"\nBest-converting demographic segment: {col} = {value}, {rate * 100:.1f}% from {int(count)} registrations")
    else:
        print("\nBest-converting demographic segment: not enough segment data.")

    df_with_risk = df.copy()
    df_with_risk["predicted_attendance_probability"] = predicted_probabilities
    at_risk = df_with_risk[df_with_risk[target_col] == 0].sort_values("predicted_attendance_probability").head(10)
    print("\nAt-risk segment based on model-estimated attendance probability:")
    if risk_segment:
        col, value, rate, count = risk_segment
        print(f"- Lowest observed conversion segment: {col} = {value}, {rate * 100:.1f}% from {int(count)} registrations")
    if not at_risk.empty:
        avg_prob = at_risk["predicted_attendance_probability"].mean() * 100
        print(f"- Lowest-probability registered group average model-estimated attendance probability: {avg_prob:.1f}%")


def executive_summary(df: pd.DataFrame, target_col: str, college_col: str | None, best_model_name: str, best_auc: float) -> None:
    print_section("EXECUTIVE SUMMARY")
    total = len(df)
    attended = int(df[target_col].sum())
    conversion = df[target_col].mean() * 100
    print(f"{total:,} registrations were analyzed, with {attended:,} attendees and an overall conversion rate of {conversion:.2f}%.")
    print(f"The best predictive model was {best_model_name} with ROC-AUC {best_auc:.3f}.")

    recommendations = []
    if college_col and college_col in df.columns:
        college_stats = (
            df.dropna(subset=[college_col])
            .groupby(college_col)
            .agg(conversion_rate=(target_col, "mean"), registrations=(target_col, "size"))
        )
        if not college_stats.empty:
            below_overall = college_stats[college_stats["conversion_rate"] < df[target_col].mean()].sort_values(
                ["registrations", "conversion_rate"], ascending=[False, True]
            )
            top_conversion = college_stats.sort_values(
                ["conversion_rate", "registrations"], ascending=[False, False]
            ).head(1)
            if not below_overall.empty:
                college = below_overall.index[0]
                row = below_overall.iloc[0]
                recommendations.append(
                    f"Follow up first with {college}, because IOH.csv shows {int(row['registrations'])} registrations and {row['conversion_rate'] * 100:.1f}% conversion."
                )
            if not top_conversion.empty:
                college = top_conversion.index[0]
                row = top_conversion.iloc[0]
                recommendations.append(
                    f"Study outreach used for {college}, because IOH.csv shows {row['conversion_rate'] * 100:.1f}% conversion from {int(row['registrations'])} registrations."
                )
    recommendations.append("Use the model-estimated low-probability attendee list only as a follow-up prioritization aid, not as a confirmed attendance fact.")
    print("\nRecommended actions:")
    for item in recommendations:
        print(f"- {item}")

    print(
        "\nIMPORTANT: The best achievable AUC with this dataset is limited by the\n"
        "features available. Target encoding improved signal extraction but the\n"
        "fundamental ceiling is the data itself. Collecting registration timestamp,\n"
        "distance from venue, and past IOH attendance history in future years will\n"
        "significantly improve prediction power."
    )


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)

    print_section("LOAD AND INSPECT")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH.resolve()}")

    df = pd.read_csv(CSV_PATH)
    df = clean_empty_values(df)
    print(f"Shape: {df.shape}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nNull values:")
    print(df.isna().sum())
    print("\nFirst 5 rows:")
    print(df.head())

    if "isPresent" in df.columns:
        target_col = "isPresent"
        y = pd.to_numeric(df[target_col], errors="coerce")
        y = y.fillna(y.mode().iloc[0] if not y.mode().empty else 0).astype(int)
    else:
        target_col, y = detect_target_column(df)
    if target_col is None or y is None:
        raise ValueError("Could not auto-detect a binary attendance/conversion column.")

    df[target_col] = y
    print(f"\nDetected attendance/conversion column: {target_col}")
    if target_col == "isPresent":
        print("Target meaning: 0 = did not attend, 1 = attended")
    print(f"Target distribution:\n{df[target_col].value_counts().sort_index()}")

    if df[target_col].nunique() < 2:
        raise ValueError("The attendance/conversion column has only one class, so classification cannot be trained.")

    print_section("DATA CLEANING")
    df = apply_ioh_cleaning_rules(df)
    print("Applied IOH-specific cleaning:")
    print("- Dropped userId because it is an ID, not an analysis/model feature.")
    print("- Converted companyName into hasCompany and dropped raw companyName.")
    print("- Filled non-student collegeName nulls with Not Applicable and student nulls with Missing College.")
    print("- Normalized collegeName with strip/title case for case-insensitive grouping.")
    print("- Added groupSize_capped and isGroup while keeping raw groupSize.")

    print_section("PREPROCESS")
    datetime_cols = detect_datetime_columns(df, exclude={target_col})
    df, timeline_col = add_datetime_features(df, datetime_cols)
    if datetime_cols:
        print(f"Datetime columns detected: {datetime_cols}")
    else:
        print("No datetime columns detected. Date/time insights will be unavailable.")

    model_df = build_model_matrix(df, target_col)
    y = model_df[target_col]
    stratify = y if y.value_counts().min() >= 2 else None
    train_df, test_df = train_test_split(
        model_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    train_df, test_df = add_smoothed_target_features(train_df, test_df, target_col)
    X_train = select_final_features(train_df)
    y_train = train_df[target_col]
    X_test = select_final_features(test_df)
    y_test = test_df[target_col]
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)
    print(f"Final numerical features: {numeric_cols}")
    print("Final categorical features one-hot encoded: registerType, category")
    print("Target-encoded features: collegeName_target, city_target, state_target")
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    print_section("MODEL TRAINING")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = {}
    fitted_pipelines = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[("preprocessor", clone(preprocessor)), ("model", clone(model))])
        pipeline.fit(X_train, y_train)
        y_pred_model = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_score) if y_test.nunique() > 1 else np.nan
        f1 = f1_score(y_test, y_pred_model, zero_division=0)
        results[name] = {"roc_auc": auc, "f1": f1}
        fitted_pipelines[name] = pipeline
        print(f"{name} ROC-AUC: {auc:.3f} | F1-score: {f1:.3f}")

    baseline_pred = test_df["category"].eq("INSTITUTE").astype(int)
    baseline_score = baseline_pred.astype(float)
    baseline_auc = roc_auc_score(y_test, baseline_score) if y_test.nunique() > 1 else np.nan
    baseline_f1 = f1_score(y_test, baseline_pred, zero_division=0)
    print(f"Rule Baseline ROC-AUC: {baseline_auc:.3f} | F1-score: {baseline_f1:.3f}")

    best_auc_model = max(results, key=lambda key: -1 if np.isnan(results[key]["roc_auc"]) else results[key]["roc_auc"])
    best_f1_model = max(results, key=lambda key: results[key]["f1"])
    f1_gap = results[best_f1_model]["f1"] - results[best_auc_model]["f1"]
    if best_f1_model != best_auc_model and f1_gap >= SIGNIFICANT_F1_DIFF:
        best_model_name = best_f1_model
        selection_reason = (
            f"selected by F1-score because it beats the ROC-AUC leader by {f1_gap:.3f}, "
            f"which is at least the {SIGNIFICANT_F1_DIFF:.2f} significance threshold"
        )
    else:
        best_model_name = best_auc_model
        selection_reason = "selected by ROC-AUC because F1-score did not differ significantly"
    best_pipeline = fitted_pipelines[best_model_name]
    best_auc = results[best_model_name]["roc_auc"]
    best_f1 = results[best_model_name]["f1"]
    y_pred = best_pipeline.predict(X_test)
    y_score = best_pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nBest model: {best_model_name}")
    print(f"Selection reason: {selection_reason}")
    print(f"Best model ROC-AUC: {best_auc:.3f} | F1-score: {best_f1:.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(cm)

    cv_splits = choose_cv_splits(y)
    if cv_splits:
        cv_scores = cross_val_target_encoded_model(
            clone(best_pipeline.named_steps["model"]), model_df, target_col, cv_splits
        )
        print(f"\n{cv_splits}-fold cross-val ROC-AUC scores: {np.round(cv_scores, 3)}")
        print(f"Mean CV ROC-AUC: {cv_scores.mean():.3f}")
    else:
        print("\nCross-validation skipped because one class has fewer than 2 rows.")

    feature_names = get_feature_names(best_pipeline.named_steps["preprocessor"], numeric_cols, categorical_cols)
    importances = model_importance(best_pipeline.named_steps["model"], feature_names)
    feature_importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    print("\nTop feature importances:")
    print(feature_importance_df.to_string(index=False))

    print_section("SAVING VISUALIZATIONS")
    college_col = find_first_column(df, ["college", "institution", "university", "department", "dept"])
    save_registration_vs_attendance(df, target_col)
    save_conversion_by_group(
        df,
        target_col,
        college_col,
        "conversion_rate_by_college_department.png",
        "Conversion Rate by College/Department - Top 15",
    )
    save_conversion_rate_chart(df, target_col, "category", "conversion_rate_by_category.png", "Conversion Rate by Category")
    save_conversion_rate_chart(df, target_col, "registerType", "conversion_rate_by_register_type.png", "Conversion Rate by Register Type")
    save_conversion_rate_chart(df, target_col, "state", "conversion_rate_by_state.png", "Conversion Rate by State", horizontal=True)
    save_feature_importance(feature_names, importances)
    save_confusion_matrix_plot(cm)
    save_roc_curve(y_test, y_score, best_auc)
    save_registration_vs_attendance_by_category(df, target_col)
    save_top_colleges(df, college_col)
    print(f"Plots saved to: {PLOTS_DIR.resolve()}")

    full_encoded_df = model_df.copy()
    for encoded_col in ["collegeName", "city", "state"]:
        full_encoded_df[f"{encoded_col}_target"] = apply_target_encoding_from_train(
            train_df, full_encoded_df, encoded_col, target_col
        )
    X_all = select_final_features(full_encoded_df)
    full_probabilities = pd.Series(best_pipeline.predict_proba(X_all)[:, 1], index=df.index)
    print_publicity_metrics(df, target_col, college_col, timeline_col, full_probabilities)
    executive_summary(df, target_col, college_col, best_model_name, best_auc)


if __name__ == "__main__":
    main()
