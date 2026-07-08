"""Tab 9 — Findings & Prescriptive Recommendations (cross-tab synthesis)."""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from helpers import apply_theme, clean_data, callout, PALETTE

apply_theme()
st.title("🧭 Tab 9 — Findings & Prescriptive Recommendations")

df, _ = clean_data()

st.subheader("1 · What genuinely predicts what (post-leakage-correction)")
match_gap = df.groupby("mentor_industry_match")["income_growth_pct"].mean()
q4 = df[df["mentor_match_score"] >= df["mentor_match_score"].quantile(.75)]
q1 = df[df["mentor_match_score"] <= df["mentor_match_score"].quantile(.25)]

c1, c2, c3 = st.columns(3)
c1.metric("Income growth: industry-matched vs not",
          f"{match_gap.get('Yes', 0):.1f}% vs {match_gap.get('No', 0):.1f}%")
c2.metric("Conversion: top vs bottom match-score quartile",
          f"{q4['converted'].mean():.0%} vs {q1['converted'].mean():.0%}")
c3.metric("Positive outcome: top vs bottom match quartile",
          f"{(q4['career_outcome']=='Positive').mean():.0%} vs "
          f"{(q1['career_outcome']=='Positive').mean():.0%}")

st.markdown("""
| Question | Signal that survives scrutiny | Where shown |
|---|---|---|
| **Mentee success** | Engagement volume (sessions/months), `skills_gained`, mentor match quality (`mentor_match_score`, `mentor_industry_match`) — with honest, moderate model performance once leakage is removed | Tabs 4–5 |
| **Conversion** | `mentor_match_score`, early engagement (sessions/messages), `referred_friend`, `signup_channel` | Tab 6 |
| **The overlap** | `mentor_match_score` predicts **both** — the one lever serving revenue *and* mentee value simultaneously | Tabs 3, 4, 6 |
| **Structural friction** | Visa/Sponsorship challenge concentrates in South Asian / African groups and co-occurs with Limited outcomes | Tabs 3, 8 |
""")

callout("🔑 <b>The headline finding is the overlap:</b> mentor match quality is the "
        "only feature that ranks highly in both the outcome model and the conversion "
        "model. Improving matching is therefore not a revenue-vs-mission trade-off — "
        "it is the same investment counted twice.", kind="ok")

st.subheader("2 · Prescriptive actions")
st.markdown("""
**Tied to conversion (revenue):**
1. **Proactive mentor re-matching** for users in the bottom match-score quartile —
   they convert at roughly half the rate of the top quartile; a re-match nudge within
   the first month targets the funnel where it leaks.
2. **Referral amplification** — `referred_friend` is a conversion-specific driver;
   formalize a referral incentive rather than leaving it organic.
3. **Channel re-weighting** — shift acquisition spend toward the signup channels the
   Tab 6 importances rank highest for converted users.

**Tied to mentee outcomes (mission):**
4. **Visa/Sponsorship playbook** — the 206 visa-challenged mentees (concentrated in
   sponsorship-dependent nationality groups) show weaker outcomes; build dedicated
   content and mentor pools with UAE visa-pathway experience (Green/Golden/freelance
   permits) instead of generic career advice.
5. **Industry-match guarantee** — matched mentors coincide with higher income growth
   at every effort level; make industry match a default constraint of the matching
   algorithm, not a premium perk.
6. **Early-engagement onboarding** — the clustering tab's low-engagement segment shows
   the weakest outcomes *and* conversion; trigger structured session-scheduling nudges
   in the first 60 days.

**Tied to retention (paid base):**
7. **Churn-risk watchlist** — churned paid users show lower sessions and match scores;
   flag paid accounts whose engagement drops below their segment median for
   mentor-led re-engagement.
""")

st.subheader("3 · Limitations — stated plainly")
callout("""
⚠️ <b>This is observational data, not an experiment.</b>
<ul>
<li><b>Correlation ≠ causation:</b> a mentor match "predicting" income growth could
reflect selection — more motivated mentees may both seek better matches and progress
faster. Forcing re-matches on other mentees is not guaranteed to reproduce the gap;
an A/B test of the re-matching policy is the honest next step.</li>
<li><b>Label provenance:</b> <code>career_outcome</code> appears to be derived from
<code>income_growth_pct</code>/<code>goals_completed</code>; all classification results
must be read as predicting that <i>constructed</i> label, not an external ground truth.</li>
<li><b>Self-report risk:</b> <code>confidence_score</code> (and possibly income
figures) are self-reported, plausibly <i>after</i> progress occurred — the Tab 4
sensitivity toggle quantifies how much this ambiguity moves the results.</li>
<li><b>Survivorship & scope:</b> churn is only observable for the 580 paid users;
Free-user "churn" (silent disengagement) is invisible in this schema.</li>
<li><b>Cross-sectional snapshot:</b> no time dimension beyond
<code>months_active</code>; trajectories are inferred, not observed.</li>
</ul>""")

st.subheader("4 · One picture of the whole funnel")
funnel = go.Figure(go.Funnel(
    y=["All mentees", "Converted to paid", "Paid & retained",
       "Paid, retained & positive outcome"],
    x=[len(df),
       int(df["converted"].sum()),
       int(((df["converted"] == 1) & (df["churned"] != "Yes")).sum()),
       int(((df["converted"] == 1) & (df["churned"] != "Yes")
            & (df["career_outcome"] == "Positive")).sum())],
    marker_color=[PALETTE["slate"], PALETTE["primary"],
                  PALETTE["secondary"], PALETTE["accent"]],
    textinfo="value+percent initial"))
funnel.update_layout(title="Acquire → convert → retain → succeed", height=430)
st.plotly_chart(funnel, use_container_width=True)
