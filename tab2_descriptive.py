"""Tab 2 — Descriptive Analytics: cross-tabs, correlation heatmap, leakage callout."""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from helpers import (apply_theme, clean_data, callout, NUMERIC_COLS,
                           OUTCOME_COLORS, pretty, px_labels)

apply_theme()
st.title("📊 Tab 2 — Descriptive Analytics")

df, _ = clean_data()

# ---------------------------------------------------------------- slicers ----
st.sidebar.header("Slicers")
emirates = st.sidebar.multiselect("Emirate", sorted(df["emirate"].unique()))
regions = st.sidebar.multiselect("Nationality region",
                                 sorted(df["nationality_region"].unique()))
stages = st.sidebar.multiselect("Career stage", sorted(df["career_stage"].unique()))

view = df.copy()
if emirates: view = view[view["emirate"].isin(emirates)]
if regions:  view = view[view["nationality_region"].isin(regions)]
if stages:   view = view[view["career_stage"].isin(stages)]
st.caption(f"Filtered view: **{len(view):,}** of {len(df):,} mentees")

# --------------------------------------------------------- leakage callout ----
callout("🚨 <b>Descriptive finding with modeling consequences — target leakage.</b> "
        f"<code>career_outcome</code> is almost perfectly separated by "
        f"<code>income_growth_pct</code> (Limited avg "
        f"{df.loc[df.career_outcome=='Limited','income_growth_pct'].mean():.1f}% vs "
        f"Positive avg {df.loc[df.career_outcome=='Positive','income_growth_pct'].mean():.1f}%) "
        f"and by <code>goals_completed</code> "
        f"({df.loc[df.career_outcome=='Limited','goals_completed'].mean():.1f} vs "
        f"{df.loc[df.career_outcome=='Positive','goals_completed'].mean():.1f}). "
        "The label was evidently <i>derived from</i> these fields. Any classifier "
        "fed these columns would score near-100% by rediscovering the labeling rule "
        "— impressive-looking, meaningless. They are therefore <b>excluded from "
        "Tab 4's feature set</b>, and <code>income_growth_pct</code> becomes the "
        "<b>regression target in Tab 5</b> instead, where its granularity is an "
        "asset rather than a leak.")

l, r = st.columns(2)
with l:
    fig = px.violin(view, x="career_outcome", y="income_growth_pct",
                    color="career_outcome", color_discrete_map=OUTCOME_COLORS,
                    box=True, points=False, labels=px_labels(view),
                    title="Income Growth % separates the label almost perfectly")
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with r:
    fig = px.histogram(view, x="goals_completed", color="career_outcome",
                       color_discrete_map=OUTCOME_COLORS, barmode="group",
                       labels=px_labels(view),
                       title="Goals Completed — same story in discrete form")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------- cross-tabs ----
st.subheader("Career outcome by segment (row-normalised %)")
CROSS_VARS = ["career_stage", "goal_type", "primary_challenge", "education_level",
              "target_industry", "mentor_industry_match", "subscription_tier",
              "employment_status"]
var = st.selectbox("Cross-tabulate Career Outcome against:", CROSS_VARS,
                   format_func=pretty)
ct = pd.crosstab(view[var], view["career_outcome"], normalize="index") * 100
ct = ct.sort_values("Positive")
fig = go.Figure()
for outcome in ["Positive", "Limited"]:
    if outcome in ct.columns:
        fig.add_bar(name=outcome, y=ct.index.astype(str), x=ct[outcome],
                    orientation="h", marker_color=OUTCOME_COLORS[outcome],
                    hovertemplate="%{y}: %{x:.1f}% " + outcome + "<extra></extra>")
fig.update_layout(barmode="stack", title=f"Career Outcome share by {pretty(var)}",
                  xaxis_title="% of segment", height=90 + 42 * len(ct))
st.plotly_chart(fig, use_container_width=True)

counts = pd.crosstab(view[var], view["career_outcome"])
with st.expander("Raw counts"):
    st.dataframe(counts, use_container_width=True)

# --------------------------------------------------------- correlation heat ----
st.subheader("Correlation heatmap — numeric features")
corr = view[NUMERIC_COLS].corr()
labels_c = [pretty(c) for c in corr.columns]
fig = go.Figure(go.Heatmap(
    z=corr.values.round(2), x=labels_c, y=labels_c,
    text=corr.values.round(2), texttemplate="%{text}",
    colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
    hovertemplate="%{y} × %{x}: r = %{z}<extra></extra>"))
fig.update_layout(height=650, title="Pearson Correlations")
st.plotly_chart(fig, use_container_width=True)

top = (corr.where(~np.eye(len(corr), dtype=bool)).abs().stack()
       .sort_values(ascending=False).drop_duplicates().head(6))
callout("🔎 <b>What the heatmap flags for later tabs:</b> "
        f"<code>confidence_score</code> × <code>income_growth_pct</code> r = "
        f"{corr.loc['confidence_score','income_growth_pct']:.2f} — borderline "
        "leakage, handled as a sensitivity analysis in Tab 4. The engagement-volume "
        "trio (<code>sessions_attended</code>, <code>messages_sent</code>, "
        "<code>months_active</code>) inter-correlate, which is the multicollinearity "
        "motivation for regularization in Tab 5 (confirmed there with VIF).")

# ------------------------------------------------------------- distributions ----
st.subheader("Population profile")
c1, c2, c3 = st.columns(3)
with c1:
    fig = px.histogram(view, x="age", nbins=25, title="Age", labels=px_labels(view))
    fig.update_layout(height=320); st.plotly_chart(fig, use_container_width=True)
with c2:
    fig = px.pie(view, names="nationality_region", hole=.45,
                 title="Nationality Region")
    fig.update_layout(height=320); st.plotly_chart(fig, use_container_width=True)
with c3:
    fig = px.pie(view, names="subscription_tier", hole=.45,
                 title="Subscription Tier")
    fig.update_layout(height=320); st.plotly_chart(fig, use_container_width=True)
