"""Tab 4 — Classification of career_outcome with leakage-free features."""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from helpers import (apply_theme, clean_data, callout, encode_features,
                           run_classification, metric_table, cm_figure,
                           OUTCOME_FEATURES, BORDERLINE_COL, PALETTE, CAT_SEQ,
                           pretty, pretty_list)

apply_theme()
st.title("🎯 Tab 4 — Classification: Career Outcome")

df, _ = clean_data()

callout("🧪 <b>Leakage-free by construction.</b> <code>income_growth_pct</code> and "
        "<code>goals_completed</code> are excluded — the label was derived from them "
        "(see Tabs 1–2), so including them would let the model 'predict' the rule it "
        "was built on. <code>confidence_score</code> is <b>borderline</b> (r ≈ 0.79 "
        "with income growth): plausibly a <i>co-outcome</i> self-reported after "
        "progress rather than a leading predictor. Rather than silently including or "
        "excluding it, the toggle below runs the sensitivity both ways.")

include_conf = st.toggle("Include confidence_score (sensitivity analysis)",
                         value=False)
features = OUTCOME_FEATURES + ([BORDERLINE_COL] if include_conf else [])
st.caption(f"Feature set ({len(features)}): " + pretty_list(features))

y = (df["career_outcome"] == "Positive").astype(int)
X, names = encode_features(df, features, scale=True)  # scaling: required for KNN,
# harmless for trees — one pipeline keeps the comparison apples-to-apples.
st.caption(f"After one-hot encoding: **{X.shape[1]} features**. Stratified 5-fold CV "
           f"+ 25% stratified hold-out (classes {int((y==0).sum())}/{int((y==1).sum())} "
           "— mildly imbalanced, stratification preserves the ratio in every fold).")

with st.expander("Why compare all four models instead of picking one?"):
    st.markdown("""
- **KNN** — distance baseline. *Expected to underperform*: one-hot encoding many
  categoricals creates a high-dimensional sparse space where Euclidean distance loses
  meaning (curse of dimensionality). Included to demonstrate that failure mode, not to win.
- **Decision Tree** — interpretable, handles mixed data natively, but a single tree
  overfits; serves as the interpretability baseline the ensembles must beat.
- **Random Forest** — bagging averages away the single tree's variance; expected to
  outperform it and to supply feature importances.
- **Gradient Boosting** — sequential error-correction, often the best raw accuracy on
  tabular data this size — but with ~1,286 rows it risks overfitting, which is exactly
  why the comparison is judged on **5-fold CV**, not a single lucky split.
""")

results, details = run_classification(f"outcome|conf={include_conf}", X, y)

st.subheader("Cross-validated comparison (mean ± std across 5 folds)")
st.dataframe(metric_table(results), use_container_width=True)
callout("📏 Accuracy alone is not enough: predicting 'Positive' for everyone scores "
        f"~{y.mean():.0%} trivially. Precision, recall, F1 and ROC-AUC are reported "
        "for every model, and the CV std shows whether differences are real or fold "
        "noise.", kind="ok")

# ------------------------------------------------------- per-model deep dive ----
st.subheader("Model deep-dive")
model_name = st.selectbox("Select model", list(details.keys()), index=2)
d = details[model_name]

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(cm_figure(d["cm"], ["Limited", "Positive"],
                              f"{model_name} — hold-out confusion matrix"),
                    use_container_width=True)
with c2:
    fig = go.Figure()
    for name, dd in details.items():
        fpr, tpr, auc = dd["roc"]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"{name} (AUC {auc:.3f})",
                                 line=dict(width=4 if name == model_name else 1.5)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance",
                             line=dict(dash="dot", color="#64748B")))
    fig.update_layout(title="ROC curves — hold-out set", height=380,
                      xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("**Hold-out metrics** — " + " · ".join(
    f"{k}: `{v:.3f}`" for k, v in d["holdout"].items()))

if d["importance"]:
    imp = (pd.Series(d["importance"]).sort_values(ascending=True).tail(15))
    imp.index = [pretty(i) for i in imp.index]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                           marker=dict(color=imp.values, colorscale=[
                               [0, "#C7D2FE"], [1, PALETTE["primary"]]])))
    fig.update_layout(title=f"{model_name} — top 15 feature importances",
                      height=520, xaxis_title="importance")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("This connects back to Tab 3's narrative: check whether "
                "`mentor_match_score` and `sessions_attended` survive as top "
                "predictors once every other feature competes with them — or "
                "whether something else (e.g. `skills_gained`, `months_active`) "
                "carries the signal.")

callout("⚖️ <b>Reading the result honestly.</b> Without the leakage columns, "
        "expect honest, moderate performance (ROC-AUC well below the ~1.0 a leaky "
        "model would fake). If the confidence_score toggle materially lifts AUC, "
        "that is itself evidence it behaves like a co-outcome — a discussion point, "
        "not a free accuracy boost.")
