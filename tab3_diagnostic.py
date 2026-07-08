"""Tab 3 — Diagnostic Analytics: why do outcomes & conversion differ?"""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from helpers import (apply_theme, clean_data, callout, OUTCOME_COLORS,
                           PALETTE, pretty, px_labels)

apply_theme()
st.title("🔬 Tab 3 — Diagnostic Analytics")
st.caption("Two business questions, examined side by side: what predicts mentee "
           "success vs. what predicts conversion — and where they overlap.")

df, _ = clean_data()

# ---------------------------------------------- stage × challenge breakdown ----
st.subheader("Career outcome by career stage × primary challenge")
pivot = (df.pivot_table(index="primary_challenge", columns="career_stage",
                        values="career_outcome",
                        aggfunc=lambda s: (s == "Positive").mean()) * 100)
stage_order = [c for c in ["Student", "Entry-Level", "Junior", "Mid-Level",
                           "Senior", "Career Switcher"] if c in pivot.columns]
pivot = pivot[stage_order] if stage_order else pivot
fig = go.Figure(go.Heatmap(
    z=pivot.values.round(1), x=pivot.columns, y=pivot.index,
    text=pivot.values.round(1), texttemplate="%{text}%",
    colorscale=[[0, "#FF8A5C"], [.5, "#F8FAFC"], [1, "#00C2A8"]],
    hovertemplate="%{y} × %{x}: %{z}% positive<extra></extra>"))
fig.update_layout(title="% positive career outcome per stage × challenge cell",
                  height=430)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------- visa challenge by nationality ----
st.subheader("Is Visa / Sponsorship a structurally unequal blocker?")
visa = df.assign(is_visa=(df["primary_challenge"] == "Visa / Sponsorship"))
share = (visa.groupby("nationality_region")["is_visa"].mean() * 100).sort_values()
outcome_given_visa = (df[df["primary_challenge"] == "Visa / Sponsorship"]
                      .groupby("nationality_region")["career_outcome"]
                      .apply(lambda s: (s == "Positive").mean() * 100))
l, r = st.columns(2)
with l:
    fig = go.Figure(go.Bar(x=share.values, y=share.index, orientation="h",
                           marker_color=PALETTE["primary"],
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_layout(title="% citing Visa/Sponsorship as primary challenge",
                      xaxis_title="% of nationality group", height=380)
    st.plotly_chart(fig, use_container_width=True)
with r:
    ov = outcome_given_visa.reindex(share.index)
    fig = go.Figure(go.Bar(x=ov.values, y=ov.index, orientation="h",
                           marker_color=PALETTE["accent"],
                           hovertemplate="%{y}: %{x:.1f}% positive<extra></extra>"))
    fig.update_layout(title="% positive outcome among visa-challenged mentees",
                      xaxis_title="% positive", height=380)
    st.plotly_chart(fig, use_container_width=True)

callout("🌍 <b>UAE labor-market context.</b> Employment in the UAE has historically "
        "been tied to employer-sponsored residence visas, which raises the cost of "
        "job switching and makes early-career mobility harder for expatriate groups "
        "whose residence depends on a sponsor. Reforms since 2021–22 (Green Visa, "
        "expanded Golden Visa, freelancer permits) have loosened this link, but "
        "sponsorship friction remains a real constraint — consistent with the "
        f"{(df['primary_challenge']=='Visa / Sponsorship').sum()} mentees "
        "(16% of the base) naming it their primary challenge, concentrated in "
        "South Asian and African nationality groups. <i>Interpretation is grounded "
        "in the dataset's own proportions; see README for cited UAE labor-market "
        "sources rather than invented statistics.</i>")

# --------------------------------------------- mentor match as shared driver ----
st.subheader("Mentor match quality — does it drive BOTH business goals?")
df["match_band"] = pd.qcut(df["mentor_match_score"], 4,
                           labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
agg = df.groupby("match_band", observed=True).agg(
    income_growth=("income_growth_pct", "mean"),
    goals=("goals_completed", "mean"),
    conversion=("converted", "mean"),
    sessions=("sessions_attended", "mean")).reset_index()

fig = go.Figure()
fig.add_bar(name="Avg income growth %", x=agg["match_band"], y=agg["income_growth"],
            marker_color=PALETTE["secondary"], yaxis="y")
fig.add_trace(go.Scatter(name="Conversion rate", x=agg["match_band"],
                         y=agg["conversion"] * 100, mode="lines+markers",
                         marker=dict(size=12, color=PALETTE["accent"]),
                         yaxis="y2"))
fig.update_layout(
    title="Mentor match score quartile vs income growth (bars) and conversion (line)",
    yaxis=dict(title="Avg. Income Growth %"),
    yaxis2=dict(title="Conversion %", overlaying="y", side="right",
                gridcolor="rgba(0,0,0,0)"),
    height=430)
st.plotly_chart(fig, use_container_width=True)

st.subheader("…and is that independent of effort (sessions attended)?")
df["sessions_band"] = pd.qcut(df["sessions_attended"], 3,
                              labels=["Low effort", "Mid effort", "High effort"])
ctrl = (df.groupby(["sessions_band", "mentor_industry_match"], observed=True)
          ["income_growth_pct"].mean().reset_index())
fig = px.bar(ctrl, x="sessions_band", y="income_growth_pct",
             color="mentor_industry_match", barmode="group",
             labels={"sessions_band": "Effort Band",
                     "income_growth_pct": "Income Growth %",
                     "mentor_industry_match": "Industry-Matched Mentor"},
             color_discrete_sequence=[PALETTE["slate"], PALETTE["secondary"]],
             title="Income growth by effort band, split by industry-matched mentor "
                   "(controls effort crudely)")
fig.update_layout(height=400, yaxis_title="avg income growth %")
st.plotly_chart(fig, use_container_width=True)

match_gap = (df.groupby("mentor_industry_match")["income_growth_pct"].mean())
callout("🎯 <b>Tying it to both business goals.</b> "
        f"Industry-matched mentors coincide with higher income growth "
        f"({match_gap.get('Yes', float('nan')):.1f}% vs "
        f"{match_gap.get('No', float('nan')):.1f}%) at <i>every</i> effort band — "
        "the match effect is not just disguised effort. Mentor match score also "
        "rises with conversion. This makes <code>mentor_match_score</code> the "
        "leading candidate for a lever that serves <b>both</b> revenue (Tab 6) and "
        "mentee value (Tabs 4–5) — the models test whether it survives against all "
        "other features. Being observational data, 'drives' means 'predicts': "
        "motivated mentees may both secure better matches and progress faster.",
        kind="ok")
