"""Tab 7 — K-Means engagement segments + hierarchical validation."""
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from helpers import (apply_theme, clean_data, callout, CLUSTER_FEATURES,
                           PALETTE, CAT_SEQ, pretty, pretty_list)

apply_theme()
st.title("🧩 Tab 7 — Clustering: Mentee Engagement Segments")

df, _ = clean_data()

callout("🧮 <b>Behavioral features only — demographics deliberately excluded.</b> "
        "K-Means needs continuous, scaled, Euclidean-meaningful inputs; mixing in "
        "one-hot nationality/industry dummies would distort distances (a 0→1 dummy "
        "jump is not commensurate with a scaled numeric difference). A demographic "
        "segmentation would need k-modes or Gower distance — out of scope here, "
        "named as a limitation. Scaling is mandatory: <code>months_active</code> "
        "spans 0–24 while <code>mentor_match_score</code> reaches 99; unscaled, the "
        "widest-range feature would dominate every distance.")
st.caption("Features: " + pretty_list(CLUSTER_FEATURES))

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

Xs = StandardScaler().fit_transform(df[CLUSTER_FEATURES])

# ------------------------------------------------------- elbow + silhouette ----
@st.cache_data(show_spinner="Scanning K = 2…15 (elbow + silhouette)…")
def scan_k(X_bytes: bytes, shape):
    X = np.frombuffer(X_bytes).reshape(shape)
    ks = range(2, 16)
    inertia, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        inertia.append(km.inertia_)
        sil.append(silhouette_score(X, km.labels_))
    return list(ks), inertia, sil

ks, inertia, sil = scan_k(Xs.tobytes(), Xs.shape)

l, r = st.columns(2)
with l:
    fig = go.Figure(go.Scatter(x=ks, y=inertia, mode="lines+markers",
                               marker_color=PALETTE["primary"]))
    fig.update_layout(title="Elbow method — inertia vs K", height=380,
                      xaxis_title="K", yaxis_title="inertia")
    st.plotly_chart(fig, use_container_width=True)
with r:
    fig = go.Figure(go.Scatter(x=ks, y=sil, mode="lines+markers",
                               marker_color=PALETTE["secondary"]))
    best_sil_k = ks[int(np.argmax(sil))]
    fig.add_vline(x=best_sil_k, line_dash="dot", line_color="#FFB547",
                  annotation_text=f"silhouette max: K={best_sil_k}")
    fig.update_layout(title="Silhouette score vs K", height=380,
                      xaxis_title="K", yaxis_title="mean silhouette")
    st.plotly_chart(fig, use_container_width=True)

callout(f"🔎 <b>Choosing K from both criteria, not the elbow's visual kink alone.</b> "
        f"The elbow flattens gradually (typical, and why elbow-alone is ambiguous); "
        f"the silhouette peaks at <b>K = {best_sil_k}</b>. Where they disagree, "
        "prefer the silhouette (it measures separation quality directly) unless the "
        "elbow strongly contradicts it. Use the selector below to override and "
        "inspect any K.")

k = st.slider("K (clusters)", 2, 15, int(best_sil_k))
km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
df["cluster"] = km.labels_.astype(str)
st.metric("Silhouette at chosen K", f"{silhouette_score(Xs, km.labels_):.3f}")

# ------------------------------------------------------------------ PCA 3D ----
st.subheader("3D view via PCA — with honesty about what you're looking at")
pca = PCA(n_components=3, random_state=42)
P = pca.fit_transform(Xs)
var = pca.explained_variance_ratio_
st.caption(f"Raw 10-dimensional data cannot be plotted directly in 3D, so the axes "
           f"are the first 3 principal components, explaining "
           f"**{var[0]:.0%} + {var[1]:.0%} + {var[2]:.0%} = {var.sum():.0%}** of "
           f"total variance. The remaining {1-var.sum():.0%} is invisible here — "
           "the plot is a faithful sketch, not the full geometry.")
p3 = pd.DataFrame(P, columns=["PC1", "PC2", "PC3"])
p3["cluster"] = df["cluster"].values
fig = px.scatter_3d(p3, x="PC1", y="PC2", z="PC3", color="cluster",
                    color_discrete_sequence=CAT_SEQ, opacity=0.75,
                    title=f"K-Means (K={k}) in PCA space")
fig.update_traces(marker_size=3.5)
fig.update_layout(height=620)
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------- profiles ----
st.subheader("Segment profiles")
prof = df.groupby("cluster")[CLUSTER_FEATURES].mean()
profz = (prof - prof.mean()) / prof.std()
fig = go.Figure(go.Heatmap(
    z=profz.values.round(2), x=[pretty(c) for c in profz.columns],
    y=["Segment " + c for c in profz.index],
    text=prof.values.round(1), texttemplate="%{text}",
    colorscale="RdBu", zmid=0,
    hovertemplate="%{y} · %{x}<br>mean = %{text} (z = %{z})<extra></extra>"))
fig.update_layout(title="Cluster means (cell text = raw mean, color = z-score vs "
                  "other clusters)", height=160 + 60 * k)
st.plotly_chart(fig, use_container_width=True)

biz = df.groupby("cluster").agg(
    n=("respondent_id", "count"),
    conversion=("converted", "mean"),
    positive_outcome=("career_outcome", lambda s: (s == "Positive").mean()),
    income_growth=("income_growth_pct", "mean")).round(3)
biz["conversion"] = (biz["conversion"] * 100).round(1)
biz["positive_outcome"] = (biz["positive_outcome"] * 100).round(1)
st.dataframe(biz.rename(columns={
    "n": "Mentees", "conversion": "Conversion %",
    "positive_outcome": "Positive outcome %",
    "income_growth": "Avg income growth %"}), use_container_width=True)

# ----------------------------------------------------- hierarchical check ----
st.subheader("Validation — hierarchical clustering on the same features")
callout("🧷 Second method, same data: if Ward-linkage hierarchical clustering "
        "suggests a similar K, the segments are less likely an artifact of K-Means' "
        "spherical-cluster assumption.")

@st.cache_data(show_spinner="Building dendrogram…")
def dendro(X_bytes: bytes, shape):
    from scipy.cluster.hierarchy import linkage
    X = np.frombuffer(X_bytes).reshape(shape)
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), size=min(400, len(X)), replace=False)  # readability
    return linkage(X[idx], method="ward")

Z = dendro(Xs.tobytes(), Xs.shape)
from scipy.cluster.hierarchy import dendrogram as scipy_dendro
dd = scipy_dendro(Z, no_plot=True, truncate_mode="lastp", p=30)
fig = go.Figure()
for xs, ys in zip(dd["icoord"], dd["dcoord"]):
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                             line=dict(color=PALETTE["primary"], width=1.5),
                             showlegend=False, hoverinfo="y"))
fig.update_layout(title="Ward dendrogram (400-mentee sample, truncated to 30 leaves) "
                  "— count the branches crossed by a horizontal cut",
                  xaxis=dict(showticklabels=False), yaxis_title="merge distance",
                  height=460)
st.plotly_chart(fig, use_container_width=True)

from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import adjusted_rand_score
rng = np.random.RandomState(42)
idx = rng.choice(len(Xs), size=min(400, len(Xs)), replace=False)
agree_rows = []
for kk in range(2, 7):
    labels_h = fcluster(Z, t=kk, criterion="maxclust")
    labels_k = KMeans(n_clusters=kk, n_init=10, random_state=42).fit(Xs[idx]).labels_
    sil_h = silhouette_score(Xs[idx], labels_h) if len(set(labels_h)) > 1 else np.nan
    agree_rows.append({"K": kk,
                       "Hierarchical silhouette": round(float(sil_h), 3),
                       "Agreement with K-Means (ARI)":
                           round(float(adjusted_rand_score(labels_h, labels_k)), 3)})
st.dataframe(pd.DataFrame(agree_rows).set_index("K"), use_container_width=True)
st.caption("Adjusted Rand Index (ARI) measures how similarly the two methods "
           "partition the same 400-mentee sample (1 = identical, 0 = chance). "
           "If the hierarchical silhouette also peaks near the K-Means silhouette's "
           "choice, the segments are unlikely to be an artifact of K-Means' "
           "spherical-cluster assumption.")
