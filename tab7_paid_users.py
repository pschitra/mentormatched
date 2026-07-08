"""Tab 7 — Paid Users: churn, time-to-subscribe, and career outcomes (n = paid only)."""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from helpers import (apply_theme, clean_data, callout, PALETTE, TIER_COLORS,
                     OUTCOME_COLORS, pretty, px_labels)

apply_theme()
st.title("💎 Tab 7 — Paid Users: Retention & Career Outcomes")

df, _ = clean_data()
paid = df[df["converted"] == 1].copy()
free = df[df["converted"] == 0]

callout("📌 <b>Why this tab only looks at paid users.</b> <code>churned</code> and "
        "<code>days_to_subscribe</code> are <i>undefined</i> for Free users — they "
        f"never subscribed, so there is nothing to measure. Every figure below is "
        f"scoped to the <b>{len(paid):,} paying mentees</b> where these fields "
        "actually exist. Comparisons against Free users are made only on fields "
        "defined for both groups.")

churn_rate = (paid["churned"] == "Yes").mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Paid users", f"{len(paid):,}",
          f"{len(paid)/len(df):.0%} of all mentees")
c2.metric("Churn rate", f"{churn_rate:.1%}")
c3.metric("Median days to subscribe", f"{paid['days_to_subscribe'].median():.0f}")
c4.metric("Positive career outcomes (paid)",
          f"{(paid['career_outcome']=='Positive').mean():.0%}",
          f"vs {(free['career_outcome']=='Positive').mean():.0%} among Free")

# ------------------------------------------------------------ conversion lag ----
st.subheader("How quickly do paying users convert?")
l, r = st.columns(2)
with l:
    fig = px.histogram(paid, x="days_to_subscribe", color="subscription_tier",
                       color_discrete_map=TIER_COLORS, nbins=30,
                       labels=px_labels(paid),
                       title="Days to Subscribe, by tier")
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)
with r:
    lag = paid.copy()
    lag["lag_band"] = pd.cut(lag["days_to_subscribe"], [-1, 7, 30, 90, 10_000],
                             labels=["≤ 1 week", "8–30 days", "31–90 days", "90+ days"])
    lb = (lag.groupby("lag_band", observed=True)
             .agg(churn=("churned", lambda s: (s == "Yes").mean() * 100),
                  positive=("career_outcome",
                            lambda s: (s == "Positive").mean() * 100))
             .reset_index())
    fig = go.Figure()
    fig.add_bar(name="Churn %", x=lb["lag_band"], y=lb["churn"],
                marker_color=PALETTE["limited"])
    fig.add_bar(name="Positive outcome %", x=lb["lag_band"], y=lb["positive"],
                marker_color=PALETTE["positive"])
    fig.update_layout(barmode="group", height=380,
                      title="Churn and outcomes by conversion speed",
                      yaxis_title="% of band")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------------- churn ----
st.subheader("Who churns?")
l, r = st.columns(2)
with l:
    cr = (paid.groupby("subscription_tier", observed=True)["churned"]
          .apply(lambda s: (s == "Yes").mean() * 100).reset_index(name="Churn %"))
    fig = px.bar(cr, x="subscription_tier", y="Churn %",
                 color="subscription_tier",
                 labels={"subscription_tier": "Subscription Tier"},
                 color_discrete_map=TIER_COLORS, title="Churn rate by paid tier")
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with r:
    eng = (paid.assign(ch=paid["churned"].eq("Yes"))
           .groupby("ch")[["sessions_attended", "messages_sent",
                           "mentor_match_score", "avg_session_rating"]].mean().T)
    eng.columns = ["Retained", "Churned"]
    eng.index = [pretty(i) for i in eng.index]
    fig = go.Figure()
    fig.add_bar(name="Retained", x=eng.index, y=eng["Retained"],
                marker_color=PALETTE["secondary"])
    fig.add_bar(name="Churned", x=eng.index, y=eng["Churned"],
                marker_color=PALETTE["limited"])
    fig.update_layout(barmode="group", height=380,
                      title="Engagement profile — churned vs. retained")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------- paid-user career outcomes ----
st.subheader("Career outcomes within the paid base")
l, r = st.columns(2)
with l:
    tier_oc = (paid.groupby("subscription_tier", observed=True)["career_outcome"]
               .apply(lambda s: (s == "Positive").mean() * 100)
               .reset_index(name="Positive %"))
    fig = px.bar(tier_oc, x="subscription_tier", y="Positive %",
                 color="subscription_tier", color_discrete_map=TIER_COLORS,
                 labels={"subscription_tier": "Subscription Tier"},
                 title="Positive career outcomes: Basic vs. Premium")
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
with r:
    fig = px.violin(paid, x="subscription_tier", y="income_growth_pct",
                    color="subscription_tier", color_discrete_map=TIER_COLORS,
                    box=True, labels=px_labels(paid),
                    title="Income Growth % distribution by paid tier")
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

retained = paid[paid["churned"] != "Yes"]
churned_u = paid[paid["churned"] == "Yes"]
callout(f"🔗 <b>Retention and outcomes travel together.</b> Retained paid users show "
        f"{(retained['career_outcome']=='Positive').mean():.0%} positive outcomes and "
        f"{retained['income_growth_pct'].mean():.1f}% average income growth, vs. "
        f"{(churned_u['career_outcome']=='Positive').mean():.0%} and "
        f"{churned_u['income_growth_pct'].mean():.1f}% among churned users. This is "
        "correlational — users who progress have a reason to stay — but it means "
        "retention efforts and outcome efforts point at the same users: the "
        "low-engagement, low-match segment identified in the clustering tab.",
        kind="ok")

st.subheader("Paid vs. Free — on fields defined for both")
comp = (df.groupby("subscription_tier", observed=True)
        .agg(**{"Mentees": ("respondent_id", "count"),
                "Positive Outcome %": ("career_outcome",
                                       lambda s: round((s == "Positive").mean()*100, 1)),
                "Avg. Income Growth %": ("income_growth_pct",
                                         lambda s: round(s.mean(), 1)),
                "Avg. Sessions": ("sessions_attended", lambda s: round(s.mean(), 1)),
                "Avg. Mentor Match Score": ("mentor_match_score",
                                            lambda s: round(s.mean(), 1))}))
st.dataframe(comp, use_container_width=True)
st.caption("Premium and Basic users out-perform Free users on outcomes — but "
           "remember the selection caveat: motivated mentees may both pay and "
           "progress. The conversion model (Tab 6) and the limitations in Tab 10 "
           "address this directly.")
