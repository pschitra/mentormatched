"""Home — overview and navigation guide."""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import streamlit as st
import plotly.express as px
from helpers import (apply_theme, clean_data, callout, PALETTE,
                           OUTCOME_COLORS, TIER_COLORS, px_labels)

apply_theme()

df, _ = clean_data()

st.title("🧭 MentorMatch Analytics Dashboard")
st.caption(f"Career mentorship platform · UAE · {len(df):,} mentees "
           f"· {df.shape[1] - 4} survey variables · all 7 emirates")

st.markdown("""
This dashboard answers **two distinct business questions** — deliberately kept separate
throughout the analysis:

1. **What drives subscription conversion** (Free → Basic/Premium)? → *Tabs 3 & 6*
2. **What drives positive career outcomes** for mentees? → *Tabs 3, 4, 5*

Use the sidebar to navigate the nine analysis tabs.
""")

callout("⚠️ <b>Read the Data Cleaning tab first.</b> The dataset contains a "
        "<b>target-leakage trap</b> (<code>income_growth_pct</code> and "
        "<code>goals_completed</code> almost perfectly determine "
        "<code>career_outcome</code>) and <b>structural missingness</b> "
        "(<code>churned</code>/<code>days_to_subscribe</code> are undefined for the "
        f"{int((df.converted==0).sum())} Free users). Both are handled explicitly — never silently imputed.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Mentees", f"{len(df):,}")
c2.metric("Converted to paid", f"{df['converted'].sum():,}",
          f"{df['converted'].mean():.0%} of base")
c3.metric("Positive career outcomes", f"{(df['career_outcome']=='Positive').sum():,}",
          f"{(df['career_outcome']=='Positive').mean():.0%} of base")
c4.metric("Avg income growth", f"{df['income_growth_pct'].mean():.1f}%")

st.divider()
left, right = st.columns(2)

with left:
    fig = px.sunburst(df, path=["subscription_tier", "career_outcome"],
                      color="subscription_tier", color_discrete_map=TIER_COLORS,
                      labels=px_labels(df),
                      title="Subscription Tier → Career Outcome")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

with right:
    counts = (df.groupby(["emirate", "career_outcome"]).size()
                .reset_index(name="mentees"))
    fig = px.bar(counts, x="emirate", y="mentees", color="career_outcome",
                 labels={"emirate": "Emirate", "mentees": "Mentees",
                         "career_outcome": "Career Outcome"},
                 color_discrete_map=OUTCOME_COLORS, barmode="group",
                 title="Mentees by Emirate & Career Outcome")
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.markdown("""
| Tab | Question it answers | Method |
|---|---|---|
| 1 · Data Cleaning | What was fixed, and what is *deliberately* left alone? | Auditable log |
| 2 · Descriptive | What does the population look like? | Cross-tabs, correlations |
| 3 · Diagnostic | *Why* do outcomes and conversion differ? | Segment drill-downs |
| 4 · Classification — Outcome | Can we predict career outcome (leakage-free)? | KNN / DT / RF / GB |
| 5 · Regression — Income | What drives income growth magnitude? | OLS / Ridge / Lasso / ElasticNet |
| 6 · Classification — Conversion | Who converts to paid? | KNN / DT / RF / GB |
| 7 · Paid Users | Who churns, and how do paying mentees fare? | Subset analytics |
| 8 · Clustering | What engagement personas exist? | K-Means + hierarchical |
| 9 · Association Rules | Which attribute combos co-occur? | Apriori |
| 10 · Findings | So what — and what should MentorMatch *do*? | Synthesis |
""")
