"""
MentorMatch Dashboard — shared utilities
Data loading, cleaning, theming, and reusable model helpers.
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.io as pio
import plotly.graph_objects as go

# ---------------------------------------------------------------- THEME ----
PALETTE = {
    "primary":  "#4C6FFF",   # indigo blue
    "secondary":"#00C2A8",   # teal
    "accent":   "#FFB547",   # amber
    "purple":   "#9B6DFF",
    "pink":     "#FF6B9D",
    "slate":    "#64748B",
    "positive": "#00C2A8",
    "limited":  "#FF8A5C",   # orange (deliberately NOT red/green pairing)
    "bg":       "#FFFFFF",
}
CAT_SEQ = ["#4C6FFF", "#00C2A8", "#FFB547", "#9B6DFF", "#FF6B9D",
           "#38BDF8", "#A3E635", "#FB923C", "#F472B6", "#94A3B8"]
OUTCOME_COLORS = {"Positive": PALETTE["positive"], "Limited": PALETTE["limited"]}
TIER_COLORS = {"Free": "#94A3B8", "Basic": "#4C6FFF", "Premium": "#FFB547"}

def apply_theme():
    """Register a cohesive LIGHT plotly template + page CSS. Call once per page."""
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#1E293B"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=CAT_SEQ,
        xaxis=dict(gridcolor="rgba(100,116,139,.18)", zerolinecolor="rgba(100,116,139,.35)"),
        yaxis=dict(gridcolor="rgba(100,116,139,.18)", zerolinecolor="rgba(100,116,139,.35)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=60, r=20, b=40, l=50),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#1E293B",
                        bordercolor="#CBD5E1"),
    )
    pio.templates["mentormatch"] = tpl
    pio.templates.default = "plotly_white+mentormatch"
    st.markdown("""
    <style>
      .block-container {padding-top: 2rem;}
      div[data-testid="stMetric"] {
          background: linear-gradient(135deg, rgba(76,111,255,.08), rgba(0,194,168,.07));
          border: 1px solid rgba(76,111,255,.30);
          border-radius: 12px; padding: 14px 18px;
      }
      div[data-testid="stMetric"] label {color:#475569;}
      .mm-callout {
          background: linear-gradient(135deg, rgba(255,181,71,.15), rgba(255,107,157,.08));
          border-left: 4px solid #F59E0B; border-radius: 8px;
          padding: 16px 20px; margin: 8px 0 16px 0; color:#1E293B;
      }
      .mm-ok {
          background: rgba(0,194,168,.10); border-left: 4px solid #00A98F;
          border-radius: 8px; padding: 14px 18px; margin: 8px 0 16px 0; color:#1E293B;
      }
      h1, h2, h3 {letter-spacing: -.02em;}
    </style>""", unsafe_allow_html=True)

def callout(html, kind="warn"):
    cls = "mm-callout" if kind == "warn" else "mm-ok"
    st.markdown(f'<div class="{cls}">{html}</div>', unsafe_allow_html=True)

# ------------------------------------------------------- PRETTY LABELS ----
_SPECIAL = {"aed": "(AED)", "pct": "%", "id": "ID", "avg": "Avg.",
            "num": "No. of", "pc1": "PC1", "pc2": "PC2", "pc3": "PC3"}

def pretty(name) -> str:
    """'mentor_match_score' -> 'Mentor Match Score'; 'income_growth_pct' ->
    'Income Growth %'. Used everywhere a column name is shown to the user."""
    words = str(name).replace("_", " ").split()
    return " ".join(_SPECIAL.get(w.lower(), w[:1].upper() + w[1:]) for w in words)

def pretty_list(names) -> str:
    return ", ".join(pretty(n) for n in names)

def px_labels(df) -> dict:
    """labels= dict for plotly express so axes/legends show pretty names."""
    return {c: pretty(c) for c in df.columns}

# ----------------------------------------------------------------- DATA ----
NUMERIC_COLS = ["age", "years_experience", "monthly_income_aed", "months_active",
                "sessions_attended", "messages_sent", "num_mentors",
                "mentor_match_score", "avg_session_rating", "response_time_hours",
                "goals_completed", "income_growth_pct", "skills_gained",
                "confidence_score"]

# Feature sets defined once, imported by the modeling tabs (spec sections 7–10)
OUTCOME_FEATURES = ["age", "years_experience", "education_level", "career_stage",
                    "employment_status", "goal_type", "primary_challenge",
                    "sessions_attended", "messages_sent", "num_mentors",
                    "mentor_match_score", "avg_session_rating",
                    "response_time_hours", "skills_gained",
                    "mentor_industry_match", "months_active"]
LEAKAGE_COLS = ["income_growth_pct", "goals_completed"]      # excluded from Tab 4
BORDERLINE_COL = "confidence_score"                          # sensitivity toggle

CONVERSION_FEATURES = ["signup_channel", "referred_friend", "mentor_match_score",
                       "sessions_attended", "messages_sent", "avg_session_rating",
                       "primary_challenge", "goal_type", "target_industry",
                       "career_stage", "response_time_hours"]

CLUSTER_FEATURES = ["months_active", "sessions_attended", "messages_sent",
                    "num_mentors", "mentor_match_score", "avg_session_rating",
                    "goals_completed", "skills_gained", "confidence_score",
                    "response_time_hours"]

from pathlib import Path

def _find_csv() -> str | None:
    """Locate the dataset wherever it was uploaded: data/, repo root, cwd,
    or anywhere in the repo tree. Returns None if genuinely absent."""
    base = Path(__file__).resolve().parent
    fname = "MentorMatch_UAE_Clean.csv"
    for c in [base / "data" / fname, base / fname,
              Path.cwd() / "data" / fname, Path.cwd() / fname]:
        if c.exists():
            return str(c)
    hits = sorted(base.glob(f"**/{fname}")) or sorted(base.glob("**/MentorMatch*.csv"))
    return str(hits[0]) if hits else None

DATA_PATH = _find_csv()

def _require_data() -> str:
    if DATA_PATH is None:
        st.error(
            "**Dataset not found.** The app looked for "
            "`MentorMatch_UAE_Clean.csv` in the `data/` folder, the repo root, "
            "and everywhere in the repository — it isn't there.\n\n"
            "**Fix:** on GitHub, click **Add file → Upload files**, drag in "
            "`MentorMatch_UAE_Clean.csv` (repo root is fine — the app will find "
            "it), commit, then reboot the app from **Manage app → ⋮ → Reboot**.")
        st.stop()
    return DATA_PATH

@st.cache_data(show_spinner=False)
def load_raw(path: str | None = None) -> pd.DataFrame:
    return pd.read_csv(path or _require_data())

@st.cache_data(show_spinner=False)
def clean_data(path: str | None = None):
    """Return (clean_df, cleaning_log). Log records before/after for each fix."""
    df = pd.read_csv(path or _require_data())
    log = []

    # 1. gender casing (Male/MALE, Female/FEMALE)
    before = df["gender"].value_counts().to_dict()
    df["gender"] = df["gender"].str.strip().str.title()
    log.append({"step": "Standardize gender casing",
                "before": before, "after": df["gender"].value_counts().to_dict(),
                "why": "Same category split across casings would double-count "
                       "levels in cross-tabs and one-hot encoding."})

    # 2. emirate leading whitespace (' Dubai' vs 'Dubai')
    before = df["emirate"].value_counts().to_dict()
    df["emirate"] = df["emirate"].str.strip()
    log.append({"step": "Strip emirate whitespace",
                "before": before, "after": df["emirate"].value_counts().to_dict(),
                "why": "' Dubai' and 'Dubai' are the same emirate; whitespace "
                       "duplicates fragment group-bys and filters."})

    # 3. median-impute the two small-gap numerics (<1% missing each)
    for col in ["monthly_income_aed", "avg_session_rating"]:
        n_missing = int(df[col].isna().sum())
        med = float(df[col].median())
        df[col] = df[col].fillna(med)
        log.append({"step": f"Median-impute {col}",
                    "before": {"missing": n_missing},
                    "after": {"missing": 0, "imputed_value": round(med, 2)},
                    "why": f"Only {n_missing}/{len(df)} rows (<1%) missing — too few to "
                           "bias the median, and dropping them would discard otherwise "
                           "complete records. Imputation is stated explicitly, not silent."})

    # 4. structural missingness — encode, do NOT impute
    n_free = int((df["subscription_tier"] == "Free").sum())
    df["churned"] = df["churned"].fillna("Never Subscribed")
    # days_to_subscribe left as NaN on purpose; add explicit converted flag
    df["converted"] = (df["subscription_tier"] != "Free").astype(int)
    log.append({"step": "Encode structural missingness (churned / days_to_subscribe)",
                "before": {"NaN churned rows": n_free},
                "after": {"churned = 'Never Subscribed'": n_free,
                          "days_to_subscribe": "left NaN by design"},
                "why": "These fields are undefined for Free users — the value does not "
                       "exist, it is not unobserved. Median/mode imputation would invent "
                       "churn behavior for users who never subscribed. This is a finding "
                       "about the funnel, not a data bug."})

    # 5. note the emirate coverage (all 7 emirates present, incl. synthetic
    #    augmentation for Fujairah / Umm Al Quwain — see README)
    n_fuj = int((df["emirate"] == "Fujairah").sum())
    n_uaq = int((df["emirate"] == "Umm Al Quwain").sum())
    log.append({"step": "Emirate coverage — all 7 emirates",
                "before": {"emirates in original survey": 5},
                "after": {"emirates": 7, "Fujairah rows": n_fuj,
                          "Umm Al Quwain rows": n_uaq},
                "why": "The original survey had no respondents from Fujairah or "
                       "Umm Al Quwain. Representative rows for both were generated "
                       "by bootstrap-sampling profiles from the smaller existing "
                       "emirates (with light jitter and new IDs), so every UAE "
                       "emirate is analysable. Flagged transparently: these rows "
                       "are synthetic, not survey responses."})

    # 6. age / experience bands for association mining
    df["age_band"] = pd.cut(df["age"], bins=[17, 24, 30, 40, 100],
                            labels=["18-24", "25-30", "31-40", "40+"])
    df["experience_band"] = pd.cut(df["years_experience"], bins=[-0.1, 2, 5, 10, 100],
                                   labels=["0-2 yrs", "3-5 yrs", "6-10 yrs", "10+ yrs"])
    log.append({"step": "Derive age_band / experience_band",
                "before": {"columns": 33}, "after": {"columns": int(df.shape[1])},
                "why": "Association rule mining (Tab 8) needs categorical items; "
                       "binning continuous variables makes them minable."})

    return df, log

# ---------------------------------------------------------- MODEL HELPERS ----
def encode_features(df, features, scale=True):
    """One-hot encode categoricals + optionally scale numerics. Returns (X, names)."""
    from sklearn.preprocessing import StandardScaler
    X = pd.get_dummies(df[features], drop_first=True)
    names = X.columns.tolist()
    if scale:
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=names, index=df.index)
    return X, names

def classification_suite():
    """The four spec-mandated classifiers with sensible small-data settings."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    return {
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8,
                                                random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200,
                                                        max_depth=3, random_state=42),
    }

@st.cache_data(show_spinner="Training & cross-validating models…")
def run_classification(df_key: str, X: pd.DataFrame, y: pd.Series):
    """5-fold stratified CV + a held-out split for confusion matrices/ROC.
    df_key busts the cache when the feature set changes."""
    from sklearn.model_selection import (StratifiedKFold, cross_validate,
                                         train_test_split)
    from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score,
                                 accuracy_score, precision_score, recall_score,
                                 f1_score)
    models = classification_suite()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          stratify=y, random_state=42)
    results, details = {}, {}
    for name, model in models.items():
        cv = cross_validate(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)
        nice = {"accuracy": "Accuracy", "precision": "Precision",
                "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"}
        results[name] = {f"CV {nice[m]}":
                         (cv[f"test_{m}"].mean(), cv[f"test_{m}"].std())
                         for m in scoring}
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        proba = model.predict_proba(Xte)[:, 1]
        fpr, tpr, _ = roc_curve(yte, proba)
        details[name] = {
            "cm": confusion_matrix(yte, pred),
            "roc": (fpr, tpr, roc_auc_score(yte, proba)),
            "holdout": {"Accuracy": accuracy_score(yte, pred),
                        "Precision": precision_score(yte, pred),
                        "Recall": recall_score(yte, pred),
                        "F1": f1_score(yte, pred),
                        "ROC-AUC": roc_auc_score(yte, proba)},
            "importance": (dict(zip(X.columns, model.feature_importances_))
                           if hasattr(model, "feature_importances_") else None),
        }
    return results, details

def metric_table(results: dict) -> pd.DataFrame:
    rows = []
    for model, metrics in results.items():
        row = {"Model": model}
        for m, (mean, std) in metrics.items():
            row[m] = f"{mean:.3f} ± {std:.3f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model")

def cm_figure(cm, labels, title):
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"Pred {l}" for l in labels], y=[f"True {l}" for l in labels],
        text=cm, texttemplate="%{text}", colorscale=[[0, "#EEF2FF"], [1, "#4C6FFF"]],
        showscale=False))
    fig.update_layout(title=title, height=360)
    return fig
