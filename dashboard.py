from pathlib import Path
from html import escape

import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from analysis import (
    RANDOM_STATE,
    SIGNIFICANT_F1_DIFF,
    add_smoothed_target_features,
    apply_ioh_cleaning_rules,
    build_model_matrix,
    build_preprocessor,
    select_final_features,
)


CSV_PATH = Path("IOH.csv")
PLOTS_DIR = Path("plots")
SUMMARY_PATH = Path("executive_summary.txt")
TARGET_COL = "isPresent"


st.set_page_config(
    page_title="IOH Attendance Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1320px;
    }
    h1 {
        font-size: 2.1rem !important;
        letter-spacing: 0;
        color: #172033;
    }
    h2, h3 {
        letter-spacing: 0;
        color: #172033;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #132238 0%, #1d3557 100%);
        border-right: 0;
    }
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 8px;
        padding: 8px 10px;
        margin-bottom: 6px;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.15);
    }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e4e8ef;
        border-radius: 8px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
    }
    [data-testid="stMetricLabel"] {
        color: #5f6b7a;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #101828;
        font-weight: 750;
    }
    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #eef5ff 100%);
        border: 1px solid #dbe6f5;
        border-radius: 8px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 760;
        color: #14213d;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #526071;
        max-width: 900px;
    }
    .section-note {
        background: #ffffff;
        border-left: 4px solid #2f6fed;
        border-radius: 6px;
        padding: 12px 14px;
        margin: 8px 0 18px 0;
        color: #364152;
    }
    .insight-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 12px;
        margin: 8px 0 18px 0;
    }
    .insight-card {
        background: #ffffff;
        border: 1px solid #e4e8ef;
        border-radius: 8px;
        padding: 15px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        min-height: 104px;
    }
    .insight-kicker {
        color: #667085;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 750;
        margin-bottom: 6px;
    }
    .insight-value {
        color: #172033;
        font-size: 1.35rem;
        font-weight: 800;
        margin-bottom: 4px;
    }
    .insight-text {
        color: #526071;
        font-size: 0.94rem;
        line-height: 1.35;
    }
    .model-card {
        background: #ffffff;
        border: 1px solid #e4e8ef;
        border-radius: 8px;
        padding: 15px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        min-height: 92px;
    }
    .model-label {
        color: #667085;
        font-size: 0.82rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 8px;
    }
    .model-value {
        color: #172033;
        font-size: 1.28rem;
        font-weight: 800;
        line-height: 1.25;
        overflow-wrap: anywhere;
    }
    .recommendation {
        background: #ffffff;
        border: 1px solid #e4e8ef;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .recommendation-title {
        font-weight: 750;
        color: #172033;
        margin-bottom: 4px;
    }
    .recommendation-text {
        color: #4b5563;
    }
    .summary-panel {
        background: #ffffff;
        border: 1px solid #e4e8ef;
        border-radius: 8px;
        padding: 18px 20px;
        margin-top: 10px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    .summary-section {
        font-size: 1.08rem;
        color: #172033;
        font-weight: 800;
        margin: 18px 0 8px 0;
    }
    .summary-row {
        display: grid;
        grid-template-columns: minmax(190px, 260px) 1fr;
        gap: 14px;
        padding: 8px 0;
        border-bottom: 1px solid #f0f3f7;
    }
    .summary-label {
        color: #526071;
        font-weight: 700;
    }
    .summary-value {
        color: #172033;
    }
    .small-label {
        color: #667085;
        font-size: 0.86rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 700;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_cleaned_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    return apply_ioh_cleaning_rules(df)


@st.cache_data(show_spinner=False)
def train_cached_model(cleaned_df: pd.DataFrame) -> dict:
    model_df = build_model_matrix(cleaned_df, TARGET_COL)
    y = model_df[TARGET_COL]
    train_df, test_df = train_test_split(
        model_df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    train_df, test_df = add_smoothed_target_features(train_df, test_df, TARGET_COL)
    X_train = select_final_features(train_df)
    y_train = train_df[TARGET_COL]
    X_test = select_final_features(test_df)
    y_test = test_df[TARGET_COL]
    preprocessor, _, _ = build_preprocessor(X_train)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        pipeline = Pipeline(
            steps=[("preprocessor", clone(preprocessor)), ("model", clone(model))]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_score = pipeline.predict_proba(X_test)[:, 1]
        results[name] = {
            "roc_auc": roc_auc_score(y_test, y_score),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

    baseline_pred = test_df["category"].eq("INSTITUTE").astype(int)
    results["Rule Baseline"] = {
        "roc_auc": roc_auc_score(y_test, baseline_pred.astype(float)),
        "f1": f1_score(y_test, baseline_pred, zero_division=0),
    }

    ml_results = {k: v for k, v in results.items() if k != "Rule Baseline"}
    best_auc_model = max(ml_results, key=lambda key: ml_results[key]["roc_auc"])
    best_f1_model = max(ml_results, key=lambda key: ml_results[key]["f1"])
    f1_gap = ml_results[best_f1_model]["f1"] - ml_results[best_auc_model]["f1"]
    if best_f1_model != best_auc_model and f1_gap >= SIGNIFICANT_F1_DIFF:
        selected_model = best_f1_model
        selection_basis = "F1-score"
    else:
        selected_model = best_auc_model
        selection_basis = "ROC-AUC"

    return {
        "results": results,
        "selected_model": selected_model,
        "selection_basis": selection_basis,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
    }


def show_plot(filename: str, title: str) -> None:
    path = PLOTS_DIR / filename
    st.markdown(f"### {title}")
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"{filename} not found. Run analysis.py to generate this plot.")


def conversion_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.dropna(subset=[group_col])
        .groupby(group_col)
        .agg(registrations=(TARGET_COL, "size"), attendees=(TARGET_COL, "sum"))
        .assign(conversion_rate=lambda x: x["attendees"] / x["registrations"] * 100)
        .sort_values("conversion_rate", ascending=False)
    )


def prepare_display_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    table = df[["registrations", "conversion_rate"]].reset_index()
    table = table.rename(
        columns={
            table.columns[0]: label,
            "registrations": "Registrations",
            "conversion_rate": "Conversion Rate",
        }
    )
    table["Conversion Rate"] = table["Conversion Rate"].map(lambda value: f"{value:.2f}%")
    return table.fillna("")


def table_height(row_count: int, max_height: int) -> int:
    header_height = 38
    row_height = 35
    padding = 8
    return min(max_height, header_height + row_height * max(row_count, 1) + padding)


def show_clean_table(
    df: pd.DataFrame,
    label: str,
    max_height: int,
    label_width: str = "large",
    registrations_label: str = "Registrations",
    conversion_label: str = "Conversion Rate",
) -> None:
    table = prepare_display_table(df, label)
    table = table.rename(
        columns={
            "Registrations": registrations_label,
            "Conversion Rate": conversion_label,
        }
    )
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=table_height(len(table), max_height),
        column_config={
            label: st.column_config.TextColumn(label, width=label_width),
            registrations_label: st.column_config.NumberColumn(registrations_label, width="small"),
            conversion_label: st.column_config.TextColumn(conversion_label, width="small"),
        },
    )


def page_intro(title: str, description: str) -> None:
    st.markdown(f"## {title}")
    st.markdown(f'<div class="section-note">{description}</div>', unsafe_allow_html=True)


def recommendation_card(title: str, text: str) -> None:
    st.markdown(
        (
            '<div class="recommendation">'
            f'<div class="recommendation-title">{title}</div>'
            f'<div class="recommendation-text">{text}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def insight_card(kicker: str, value: str, text: str) -> str:
    return (
        '<div class="insight-card">'
        f'<div class="insight-kicker">{kicker}</div>'
        f'<div class="insight-value">{value}</div>'
        f'<div class="insight-text">{text}</div>'
        "</div>"
    )


def insight_grid(cards: list[tuple[str, str, str]]) -> None:
    html = '<div class="insight-grid">'
    for kicker, value, text in cards:
        html += insight_card(kicker, value, text)
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def model_card(label: str, value: str) -> None:
    st.markdown(
        (
            '<div class="model-card">'
            f'<div class="model-label">{label}</div>'
            f'<div class="model-value">{value}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_summary_text(text: str) -> None:
    rows: list[tuple[str, str | None]] = []
    current_label = None
    current_value = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or set(line) in ({"="}, {"-"}):
            continue
        if raw_line.startswith("  ") and current_label:
            current_value = f"{current_value} {line}"
            rows[-1] = (current_label, current_value)
        elif line.isupper() and ":" not in line:
            current_label = None
            current_value = ""
            rows.append((line.title(), None))
        elif ":" in line:
            label, value = line.split(":", 1)
            current_label = label.strip()
            current_value = value.strip()
            rows.append((current_label, current_value))
        else:
            current_label = None
            current_value = ""
            rows.append((line, ""))

    html = '<div class="summary-panel">'
    for label, value in rows:
        if value is None:
            html += f'<div class="summary-section">{escape(label)}</div>'
        elif value == "":
            html += f'<p class="summary-value">{escape(label)}</p>'
        else:
            html += (
                '<div class="summary-row">'
                f'<div class="summary-label">{escape(label)}</div>'
                f'<div class="summary-value">{escape(value)}</div>'
                '</div>'
            )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


if not CSV_PATH.exists():
    st.error("IOH.csv was not found. Add it to the project root.")
    st.stop()

df = load_cleaned_data()

st.sidebar.markdown("## IOH Dashboard")
st.sidebar.markdown("Attendance intelligence for the publicity team.")
page = st.sidebar.radio(
    "Go to",
    [
        "Overview",
        "Category Analysis",
        "College Performance",
        "Geographic Breakdown",
        "Prediction Model",
        "Recommendations",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("IOH.csv | Previous IOH event")

st.markdown(
    """
    <div class="hero">
        <div class="small-label">Registration Analytics</div>
        <div class="hero-title">Registration & Attendance Dashboard</div>
        <div class="hero-subtitle">
            Interactive view of IOH registration conversion, category behaviour,
            college performance, geographic gaps, and model outputs.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

total_registrations = len(df)
total_attendees = int(df[TARGET_COL].sum())
total_no_shows = total_registrations - total_attendees
conversion_rate = df[TARGET_COL].mean() * 100

if page == "Overview":
    page_intro(
        "Overview",
        "High-level attendance health for the previous IOH registration cycle.",
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Registrations", f"{total_registrations:,}")
    c2.metric("Total Attendees", f"{total_attendees:,}")
    c3.metric("Total No-Shows", f"{total_no_shows:,}")
    c4.metric("Conversion Rate", f"{conversion_rate:.2f}%")

    insight_grid(
        [
            (
                "Attendance Reality",
                "39.17%",
                "Only about 4 in 10 registrations converted into actual attendance.",
            ),
            (
                "Largest Gap",
                f"{total_no_shows:,}",
                "Registered participants did not attend, so follow-up quality matters.",
            ),
        ]
    )

    left, right = st.columns(2)
    with left:
        show_plot("registration_vs_attendance.png", "Registration vs Attendance")
    with right:
        show_plot(
            "registration_vs_attendance_by_category.png",
            "Registrations vs Attendance by Category",
        )

elif page == "Category Analysis":
    page_intro(
        "Category Analysis",
        "Compare conversion across participant categories and registration types.",
    )
    insight_grid(
        [
            (
                "Best Segment",
                "INSTITUTE",
                "72.6% conversion makes this the strongest category for turnout.",
            ),
            (
                "Registration Type Lift",
                "+8.2 pts",
                "GROUP registrations convert better than INDIVIDUAL registrations.",
            ),
        ]
    )
    left, right = st.columns(2)
    with left:
        show_plot("conversion_rate_by_category.png", "Conversion Rate by Category")
    with right:
        show_plot(
            "conversion_rate_by_register_type.png",
            "Conversion Rate by Register Type",
        )

    st.subheader("Category Metrics")
    category_table = conversion_summary(df, "category")
    show_clean_table(category_table, "Category", 250)

    st.subheader("Registration Type Metrics")
    register_table = conversion_summary(df, "registerType")
    show_clean_table(register_table, "Registration Type", 170)

elif page == "College Performance":
    page_intro(
        "College Performance",
        "Identify colleges that convert well and colleges that need direct follow-up.",
    )
    insight_grid(
        [
            (
                "Needs Attention",
                "CIT",
                "105 registrations with only 29.5% conversion.",
            ),
            (
                "Strong Volume",
                "Velammal",
                "235 registrations with above-average conversion.",
            ),
        ]
    )
    left, right = st.columns(2)
    with left:
        show_plot(
            "top_10_colleges_registration_count.png",
            "Top 10 Colleges by Registration Count",
        )
    with right:
        show_plot(
            "conversion_rate_by_college_department.png",
            "Conversion Rate by College",
        )

    college_table = conversion_summary(df, "collegeName")
    attention = college_table[
        (college_table["registrations"] >= 50)
        & (college_table["conversion_rate"] < 35)
    ]
    st.subheader("Needs Attention: 50+ Registrations and Below 35% Conversion")
    show_clean_table(attention, "College", 150)

    st.subheader("Top Colleges by Registration Volume")
    top_volume = college_table.sort_values(
        ["registrations", "conversion_rate"], ascending=[False, False]
    ).head(25)
    show_clean_table(top_volume, "College", 360)

elif page == "Geographic Breakdown":
    page_intro(
        "Geographic Breakdown",
        "Review conversion by state and city to understand where outreach needs tuning.",
    )
    insight_grid(
        [
            (
                "Weakest State",
                "Karnataka",
                "29.6% conversion means the outreach approach needs redesign.",
            ),
            (
                "Strong Base",
                "Tamil Nadu",
                "Largest registration base with roughly 42% conversion.",
            ),
        ]
    )
    show_plot("conversion_rate_by_state.png", "Conversion Rate by State")

    left, right = st.columns(2)
    with left:
        st.subheader("State Metrics")
        state_table = conversion_summary(df, "state")
        show_clean_table(
            state_table,
            "State",
            360,
            label_width="medium",
            registrations_label="Regs",
            conversion_label="Conv %",
        )
    with right:
        st.subheader("City Metrics")
        city_table = conversion_summary(df, "city").sort_values(
            ["registrations", "conversion_rate"], ascending=[False, False]
        )
        show_clean_table(
            city_table.head(12),
            "City",
            500,
            label_width="medium",
            registrations_label="Regs",
            conversion_label="Conv %",
        )

elif page == "Prediction Model":
    model_summary = train_cached_model(df)
    page_intro(
        "Prediction Model",
        "Cached model comparison using the same cleaned features and held-out test split.",
    )
    selected_model = model_summary["selected_model"]
    selected_metrics = model_summary["results"][selected_model]

    c1, c2 = st.columns([2, 1])
    with c1:
        model_card("Selected Model", selected_model)
    with c2:
        model_card("Selection Basis", model_summary["selection_basis"])

    c3, c4 = st.columns(2)
    c3.metric("ROC-AUC", f"{selected_metrics['roc_auc']:.3f}")
    c4.metric("F1-Score", f"{selected_metrics['f1']:.3f}")

    insight_grid(
        [
            (
                "Model Signal",
                f"{selected_metrics['roc_auc']:.3f}",
                "Better than random guessing, but limited by available registration data.",
            ),
            (
                "Baseline Check",
                "INSTITUTE Rule",
                "Simple category rule is shown beside ML models for context.",
            ),
        ]
    )

    st.subheader("Model Comparison")
    results_df = (
        pd.DataFrame(model_summary["results"])
        .T.reset_index()
        .rename(columns={"index": "Model", "roc_auc": "ROC-AUC", "f1": "F1-Score"})
    )
    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True,
        height=table_height(len(results_df), 210),
        column_config={
            "Model": st.column_config.TextColumn("Model", width="large"),
            "ROC-AUC": st.column_config.NumberColumn("ROC-AUC", format="%.3f", width="small"),
            "F1-Score": st.column_config.NumberColumn("F1-Score", format="%.3f", width="small"),
        },
    )

    st.caption(
        f"Train/Test Split: {model_summary['train_rows']:,} training rows and "
        f"{model_summary['test_rows']:,} held-out test rows."
    )

    left, right = st.columns(2)
    with left:
        show_plot("roc_curve.png", "ROC Curve")
    with right:
        show_plot("confusion_matrix.png", "Confusion Matrix")
    show_plot("feature_importance.png", "Feature Importance")

elif page == "Recommendations":
    page_intro(
        "Recommendations",
        "Action items for the publicity team based on the conversion analysis.",
    )
    recommendation_card(
        "Follow up with high-volume low-conversion colleges",
        "Prioritize Chennai Institute of Technology and Sathyabama Institute of "
        "Science and Technology because both have strong registration volume but "
        "weak attendance conversion.",
    )
    recommendation_card(
        "Prioritize INSTITUTE registrations",
        "The INSTITUTE category has the strongest conversion rate, so dedicated "
        "outreach here is likely to produce actual attendance.",
    )
    recommendation_card(
        "Encourage group sign-ups",
        "GROUP registrations convert better than INDIVIDUAL registrations, so "
        "campaigns should make group registration feel natural and easy.",
    )
    recommendation_card(
        "Redesign Karnataka outreach",
        "Karnataka registrations convert poorly, so the issue is not just volume. "
        "Messaging, channels, and follow-up need a different approach.",
    )

    if SUMMARY_PATH.exists():
        st.subheader("Executive Summary")
        render_summary_text(SUMMARY_PATH.read_text(encoding="utf-8"))
    else:
        st.info("executive_summary.txt was not found.")
