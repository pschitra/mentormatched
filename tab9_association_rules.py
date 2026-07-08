"""Tab 8 — Association Rule Mining (Apriori) on categorical attributes."""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from helpers import apply_theme, clean_data, callout, PALETTE, pretty

apply_theme()
st.title("🕸️ Tab 8 — Association Rule Mining")

df, _ = clean_data()

ITEM_COLS = ["goal_type", "primary_challenge", "career_stage", "education_level",
             "target_industry", "mentor_industry_match", "subscription_tier",
             "career_outcome", "employment_status", "signup_channel",
             "referred_friend", "age_band", "experience_band"]

callout("⚙️ <b>Why Apriori rather than FP-Growth?</b> With only ~13 categorical "
        "attributes (a few dozen distinct items) and 1,286 transactions, Apriori's "
        "candidate-generation cost is trivial, and its levelwise output maps directly "
        "onto business-readable rules. FP-Growth's compressed-tree advantage only "
        "pays off at much larger item-set scale (thousands of items / millions of "
        "transactions). The choice is fitness-for-purpose, not arbitrary.")

c1, c2, c3 = st.columns(3)
with c1:
    min_sup = st.slider("Min support", 0.02, 0.30, 0.05, 0.01)
with c2:
    min_conf = st.slider("Min confidence", 0.30, 0.95, 0.60, 0.05)
with c3:
    min_lift = st.slider("Min lift", 1.0, 3.0, 1.1, 0.05)

@st.cache_data(show_spinner="Mining rules…")
def mine(min_sup: float):
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules
    tx = df[ITEM_COLS].astype(str).apply(
        lambda r: [f"{pretty(c)} = {r[c]}" for c in ITEM_COLS], axis=1).tolist()
    te = TransactionEncoder()
    arr = te.fit(tx).transform(tx)
    basket = pd.DataFrame(arr, columns=te.columns_)
    freq = apriori(basket, min_support=min_sup, use_colnames=True, max_len=3)
    if freq.empty:
        return pd.DataFrame()
    rules = association_rules(freq, metric="confidence", min_threshold=0.01)
    rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
    return rules

rules = mine(min_sup)
if rules.empty:
    st.warning("No frequent itemsets at this support — lower the min support slider.")
    st.stop()

view = rules[(rules["confidence"] >= min_conf) & (rules["lift"] >= min_lift)].copy()
st.caption(f"{len(view):,} rules pass the thresholds "
           f"(of {len(rules):,} mined at support ≥ {min_sup:.2f}).")

focus = st.selectbox(
    "Focus consequent (business lens)",
    ["All", "Career Outcome = Limited", "Career Outcome = Positive",
     "Subscription Tier = Free", "Subscription Tier = Premium"])
if focus != "All":
    view = view[view["consequents"] == focus]

show = (view[["antecedents", "consequents", "support", "confidence", "lift"]]
        .sort_values("lift", ascending=False).reset_index(drop=True))
st.dataframe(show.style.format({"support": "{:.3f}", "confidence": "{:.3f}",
                                "lift": "{:.2f}"}),
             use_container_width=True, height=380)

# ------------------------------------------------------------ bubble chart ----
st.subheader("Rule landscape — support × confidence, bubble = lift")
top = view.nlargest(60, "lift")
fig = go.Figure(go.Scatter(
    x=top["support"], y=top["confidence"], mode="markers",
    marker=dict(size=np.clip(top["lift"] * 14, 8, 46),
                color=top["lift"], colorscale="Viridis",
                colorbar=dict(title="lift"), line=dict(width=0.5, color="#FFFFFF")),
    text=("<b>IF</b> " + top["antecedents"] + "<br><b>THEN</b> " + top["consequents"]),
    hovertemplate="%{text}<br>support %{x:.3f} · confidence %{y:.3f}<extra></extra>"))
fig.update_layout(height=520, xaxis_title="support", yaxis_title="confidence",
                  title="Hover any bubble to read the rule")
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------ outcome rules ----
st.subheader("Rules that land on career_outcome — the business-critical subset")
oc = rules[rules["consequents"].str.startswith("Career Outcome =")].copy()
oc = oc[oc["lift"] > 1.05].nlargest(12, "lift")
if not oc.empty:
    oc["rule"] = oc["antecedents"] + " → " + oc["consequents"]
    colors = [PALETTE["limited"] if "Limited" in c else PALETTE["positive"]
              for c in oc["consequents"]]
    fig = go.Figure(go.Bar(x=oc["lift"], y=oc["rule"], orientation="h",
                           marker_color=colors,
                           hovertemplate="%{y}<br>lift %{x:.2f}<extra></extra>"))
    fig.update_layout(title="Top outcome rules by lift (orange → Limited, teal → Positive)",
                      height=90 + 42 * len(oc), xaxis_title="lift")
    st.plotly_chart(fig, use_container_width=True)
    callout("💡 Rules like <i>'Visa/Sponsorship challenge + Career Switch goal → "
            "Limited outcome'</i> become explorable here rather than a static list — "
            "raise the lift slider to keep only the strongest signals, and remember "
            "lift > 1 means <i>co-occurrence above chance</i>, not causation.",
            kind="ok")
else:
    st.info("No outcome-consequent rules above lift 1.05 at current support — "
            "lower min support to surface them.")
