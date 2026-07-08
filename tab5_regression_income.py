"""Tab 5 — Regression on income_growth_pct with VIF-motivated regularization."""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from helpers import (apply_theme, clean_data, callout, encode_features,
                           OUTCOME_FEATURES, PALETTE, pretty)

apply_theme()
st.title("📈 Tab 5 — Regression: Income Growth %")

df, _ = clean_data()

st.markdown("Target: **`income_growth_pct`** — the continuous, information-rich "
            "version of career outcome. Features: the same process/engagement set as "
            "Tab 4 (minus `career_outcome` and `goals_completed`, both co-outcomes of "
            "the same success — including either would reintroduce leakage in reverse).")

FEATURES = OUTCOME_FEATURES  # confidence_score excluded for the same co-outcome reason
y = df["income_growth_pct"]
X, names = encode_features(df, FEATURES, scale=True)

# --------------------------------------------------------------------- VIF ----
st.subheader("Step 1 — VIF: is regularization actually needed?")

@st.cache_data(show_spinner="Computing VIF…")
def compute_vif(X_df: pd.DataFrame) -> pd.DataFrame:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xv = X_df.astype(float).values
    return (pd.DataFrame({
        "feature": X_df.columns,
        "VIF": [variance_inflation_factor(Xv, i) for i in range(Xv.shape[1])]})
        .sort_values("VIF", ascending=False))

numeric_only = [c for c in ["age", "years_experience", "sessions_attended",
                            "messages_sent", "num_mentors", "mentor_match_score",
                            "avg_session_rating", "response_time_hours",
                            "skills_gained", "months_active"]]
vif = compute_vif(X[numeric_only])
fig = go.Figure(go.Bar(
    x=vif["VIF"], y=[pretty(f) for f in vif["feature"]], orientation="h",
    marker_color=[PALETTE["limited"] if v > 5 else PALETTE["primary"]
                  for v in vif["VIF"]]))
fig.add_vline(x=5, line_dash="dot", line_color="#FFB547",
              annotation_text="VIF = 5 (common concern threshold)")
fig.update_layout(title="Variance Inflation Factor — numeric engagement features",
                  height=420, xaxis_title="VIF")
st.plotly_chart(fig, use_container_width=True)

high_vif = vif[vif["VIF"] > 5]["feature"].tolist()
if high_vif:
    callout(f"⚠️ <b>VIF confirms multicollinearity</b> in: "
            f"{', '.join(f'<code>{c}</code>' for c in high_vif)}. "
            "The engagement-volume trio (sessions, messages, months active) proxy the "
            "same latent 'usage volume'. OLS coefficients on correlated features are "
            "unstable and inflated-variance — <b>this is the actual justification for "
            "regularization</b>, not a checklist of four models.")
else:
    callout("ℹ️ VIF values are moderate (all below 5). Multicollinearity exists but is "
            "mild — sessions_attended, messages_sent, and months_active still share "
            "variance as engagement-volume proxies, so regularization is shown as a "
            "robustness comparison: if Ridge/Lasso barely beat OLS, that is itself the "
            "honest finding (the win is small because the problem is small).", "ok")

# ------------------------------------------------------------------- models ----
st.subheader("Step 2 — OLS vs Ridge vs Lasso vs Elastic Net")
with st.expander("Model roles (why all four)"):
    st.markdown("""
- **OLS** — the baseline that *demonstrates the problem*: with correlated regressors its
  coefficients are individually unstable even when predictions are fine.
- **Ridge** — shrinks correlated coefficients *toward each other*; right tool if
  multicollinearity is the issue and no feature should be eliminated.
- **Lasso** — zeroes coefficients; right tool if some features are redundant
  (e.g. `messages_sent` given `sessions_attended`).
- **Elastic Net** — the blend; justified by showing where pure Ridge (too many small
  nonzero coefficients) or pure Lasso (too-aggressive zeroing among correlated twins)
  struggles.
""")

@st.cache_data(show_spinner="Cross-validating regressions…")
def regression_cv(Xv: pd.DataFrame, yv: pd.Series):
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.model_selection import KFold, cross_validate
    kf = KFold(5, shuffle=True, random_state=42)
    out = {}
    for name, model in {
            "Linear (OLS)": LinearRegression(),
            "Ridge (α=1)": Ridge(alpha=1.0),
            "Lasso (α=0.1)": Lasso(alpha=0.1, max_iter=20000),
            "Elastic Net (α=0.1, l1=0.5)": ElasticNet(alpha=0.1, l1_ratio=0.5,
                                                      max_iter=20000)}.items():
        cv = cross_validate(model, Xv, yv, cv=kf,
                            scoring=["r2", "neg_root_mean_squared_error"])
        out[name] = {"R² (CV)": cv["test_r2"].mean(),
                     "R² std": cv["test_r2"].std(),
                     "RMSE (CV)": -cv["test_neg_root_mean_squared_error"].mean()}
    return pd.DataFrame(out).T

st.dataframe(regression_cv(X, y).style.format("{:.3f}"), use_container_width=True)

# ------------------------------------------------------- interactive lambda ----
st.subheader("Step 3 — Live shrinkage: move λ, watch coefficients")
c1, c2 = st.columns([1, 1])
with c1:
    reg_type = st.radio("Regularizer", ["Ridge", "Lasso", "Elastic Net"],
                        horizontal=True)
with c2:
    log_alpha = st.slider("log₁₀(λ)", -3.0, 2.0, 0.0, 0.1)
alpha = 10 ** log_alpha
st.caption(f"λ = {alpha:.4g}")

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
model = {"Ridge": Ridge(alpha=alpha),
         "Lasso": Lasso(alpha=alpha, max_iter=20000),
         "Elastic Net": ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=20000)}[reg_type]
model.fit(X, y)
pred = model.predict(X)
r2 = r2_score(y, pred)
rmse = float(np.sqrt(mean_squared_error(y, pred)))
n_zero = int(np.sum(np.abs(model.coef_) < 1e-8))

m1, m2, m3 = st.columns(3)
m1.metric("R² (in-sample)", f"{r2:.3f}")
m2.metric("RMSE", f"{rmse:.2f} pct-pts")
m3.metric("Coefficients zeroed", f"{n_zero} / {len(model.coef_)}")

coef = pd.Series(model.coef_, index=X.columns)
top = coef.reindex(coef.abs().sort_values(ascending=True).tail(18).index)
top.index = [pretty(i) for i in top.index]
fig = go.Figure(go.Bar(
    x=top.values, y=top.index, orientation="h",
    marker_color=[PALETTE["secondary"] if v > 0 else PALETTE["limited"]
                  for v in top.values]))
fig.update_layout(title=f"{reg_type} coefficients at λ={alpha:.4g} "
                  "(standardized scale — magnitudes comparable)",
                  height=560, xaxis_title="coefficient")
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------- shrinkage paths ----
st.subheader("Step 4 — Full shrinkage path (the 'why regularization helps' story)")

@st.cache_data(show_spinner="Tracing coefficient paths…")
def shrinkage_path(reg_type: str, Xv: pd.DataFrame, yv: pd.Series):
    alphas = np.logspace(-3, 2, 30)
    paths, r2s = [], []
    for a in alphas:
        m = {"Ridge": Ridge(alpha=a),
             "Lasso": Lasso(alpha=a, max_iter=20000),
             "Elastic Net": ElasticNet(alpha=a, l1_ratio=0.5, max_iter=20000)}[reg_type]
        m.fit(Xv, yv)
        paths.append(m.coef_)
        r2s.append(r2_score(yv, m.predict(Xv)))
    return alphas, np.array(paths), r2s

alphas, paths, r2s = shrinkage_path(reg_type, X, y)
watch = [c for c in ["sessions_attended", "messages_sent", "months_active",
                     "mentor_match_score", "skills_gained",
                     "mentor_industry_match_Yes"] if c in X.columns]
fig = go.Figure()
for c in watch:
    i = list(X.columns).index(c)
    fig.add_trace(go.Scatter(x=alphas, y=paths[:, i], mode="lines",
                             name=pretty(c)))
fig.update_layout(title=f"{reg_type}: coefficient shrinkage as λ grows "
                  "(watch the correlated engagement trio converge/zero-out)",
                  xaxis=dict(type="log", title="λ (log scale)"),
                  yaxis_title="coefficient", height=460)
st.plotly_chart(fig, use_container_width=True)

callout("📖 <b>How to read the paths.</b> Under Ridge, the correlated trio "
        "(sessions / messages / months) are shrunk <i>together</i> — the model shares "
        "credit across them. Under Lasso, one of the trio typically survives while its "
        "redundant twins are zeroed — automatic feature selection. If OLS ≈ Ridge ≈ "
        "Lasso in CV R², the honest conclusion is that multicollinearity was mild and "
        "regularization is insurance, not a rescue. The R² level itself (moderate, "
        "not spectacular) is the leakage-free reality: engagement explains part of "
        "income growth, not all of it.", kind="ok")
