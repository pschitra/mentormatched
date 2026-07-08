"""Tab 1 — Data Cleaning: before/after audit of every fix."""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from helpers import apply_theme, load_raw, clean_data, callout, PALETTE

apply_theme()
st.title("🧹 Tab 1 — Data Cleaning")

raw = load_raw()
clean, log = clean_data()

c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(raw):,}", "unchanged — no rows dropped")
c2.metric("Columns (raw → clean)", f"{raw.shape[1]} → {clean.shape[1]}",
          "+ converted, age_band, experience_band")
n_imp = int(raw["monthly_income_aed"].isna().sum() + raw["avg_session_rating"].isna().sum())
c3.metric("Cells imputed", f"{n_imp}",
          f"{n_imp/(raw.shape[0]*raw.shape[1]):.2%} of all cells — median, stated explicitly")

st.subheader("Structural missingness — a finding, not a bug")
n_free = int(raw["churned"].isna().sum())
callout(f"🔍 <b>{n_free:,} of {len(raw):,} rows have NaN for <code>churned</code> and "
        "<code>days_to_subscribe</code> — every one of them a Free-tier user.</b> "
        "These users never subscribed, so there is nothing to measure: the values are "
        "<i>undefined</i>, not <i>unobserved</i>. Median/mode imputation would invent "
        "churn behavior for users who never had a subscription to churn from. "
        "Instead, <code>churned</code> is encoded as an explicit "
        "<b>'Never Subscribed'</b> category, <code>days_to_subscribe</code> is left "
        f"NaN by design, and churn analysis is scoped to the <b>{int(len(raw)-n_free):,} "
        "paid users only</b> — see Tab 7.")

miss = raw.isna().groupby(raw["subscription_tier"]).sum()[
    ["churned", "days_to_subscribe", "monthly_income_aed", "avg_session_rating"]]
fig = go.Figure()
for col, color in zip(miss.columns, ["#FF8A5C", "#FFB547", "#4C6FFF", "#00C2A8"]):
    fig.add_bar(name=col, x=miss.index, y=miss[col], marker_color=color)
fig.update_layout(barmode="group", title="Missing values by subscription tier — "
                  "missingness is perfectly aligned with the Free tier",
                  yaxis_title="missing cells", height=400)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Cleaning log — expand each step for before/after evidence")
for i, entry in enumerate(log, 1):
    with st.expander(f"Step {i}: {entry['step']}", expanded=(i <= 2)):
        b, a = st.columns(2)
        with b:
            st.markdown("**Before**")
            st.json(entry["before"])
        with a:
            st.markdown("**After**")
            st.json(entry["after"])
        st.markdown(f"**Why:** {entry['why']}")

st.subheader("Gender & emirate — before vs after")
b, a = st.columns(2)
with b:
    fig = go.Figure(go.Bar(x=raw["gender"].value_counts().index,
                           y=raw["gender"].value_counts().values,
                           marker_color="#FF8A5C"))
    fig.update_layout(title="Gender (raw) — 4 apparent levels", height=330)
    st.plotly_chart(fig, use_container_width=True)
with a:
    fig = go.Figure(go.Bar(x=clean["gender"].value_counts().index,
                           y=clean["gender"].value_counts().values,
                           marker_color="#00C2A8"))
    fig.update_layout(title="Gender (clean) — 2 true levels", height=330)
    st.plotly_chart(fig, use_container_width=True)

b, a = st.columns(2)
with b:
    vc = raw["emirate"].value_counts()
    fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color="#FF8A5C"))
    fig.update_layout(title="Emirate (raw) — whitespace duplicates", height=330)
    st.plotly_chart(fig, use_container_width=True)
with a:
    vc = clean["emirate"].value_counts()
    fig = go.Figure(go.Bar(x=vc.index, y=vc.values, marker_color="#00C2A8"))
    fig.update_layout(title="Emirate (clean) — consolidated", height=330)
    st.plotly_chart(fig, use_container_width=True)

callout("✅ <b>What was deliberately NOT done:</b> no rows dropped, no outliers "
        "removed, <code>days_to_subscribe</code> NaNs preserved, and the leakage "
        "columns (<code>income_growth_pct</code>, <code>goals_completed</code>) kept "
        "in the data — they are excluded at <i>feature-selection</i> time in the "
        "classification tabs, where the exclusion can be justified in context.",
        kind="ok")
