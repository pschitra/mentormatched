"""Tab 6 — Classification of subscription conversion (Free vs paid) + churn subset."""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from helpers import (apply_theme, clean_data, callout, encode_features,
                           run_classification, metric_table, cm_figure,
                           CONVERSION_FEATURES, PALETTE, pretty, pretty_list)

apply_theme()
st.title("💳 Tab 6 — Classification: Subscription Conversion")

df, _ = clean_data()

callout("🎯 <b>Different target, different feature rationale — not Tab 4 reused.</b> "
        "Target is a derived binary <code>converted</code> "
        f"(Free = 0: {int((df.converted==0).sum())} · Basic/Premium = 1: "
        f"{int((df.converted==1).sum())}). "
        "<code>referred_friend</code> and <code>signup_channel</code> enter here "
        "because they shape the decision to pay — but they were irrelevant to career "
        "outcome. <code>churned</code> and <code>days_to_subscribe</code> are "
        "excluded: they only exist <i>after</i> conversion, so using them to predict "
        "conversion would be circular. <code>mentor_match_score</code> appears in "
        "<b>both</b> this model and Tab 4's — that overlap is itself a finding.")

y = df["converted"]
X, names = encode_features(df, CONVERSION_FEATURES, scale=True)
st.caption(f"Features ({len(CONVERSION_FEATURES)} raw → {X.shape[1]} encoded): "
           + pretty_list(CONVERSION_FEATURES))

results, details = run_classification("conversion", X, y)

st.subheader("Cross-validated comparison (same KNN/DT/RF/GB structure as Tab 4)")
st.dataframe(metric_table(results), use_container_width=True)

model_name = st.selectbox("Model deep-dive", list(details.keys()), index=2)
d = details[model_name]
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(cm_figure(d["cm"], ["Free", "Converted"],
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
    fig.update_layout(title="ROC — conversion models", height=380,
                      xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("**Hold-out metrics** — " + " · ".join(
    f"{k}: `{v:.3f}`" for k, v in d["holdout"].items()))

if d["importance"]:
    imp = pd.Series(d["importance"]).sort_values(ascending=True).tail(15)
    imp.index = [pretty(i) for i in imp.index]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                           marker=dict(color=imp.values, colorscale=[
                               [0, "#FDE68A"], [1, PALETTE["accent"]]])))
    fig.update_layout(title=f"{model_name} — top conversion drivers", height=520)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.info("💎 Retention (churn), time-to-subscribe, and career outcomes for the "
        "paid base now have their own dedicated tab: **Tab 7 — Paid Users**.")
