# app_pitch1.py
# 🔬 Quantum Kernel DevKit v1.3.0 — Cancer Atlas (IC50 vs Quantum Minima)

import os, re, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------- Config ----------------
CACHE_ROOT = os.path.join("cache", "v1.3.0")
DEFAULT_TOPK = 25
MAX_SCATTER = 3000
UMAP_SAMPLE = 800
KNN_K = 8

st.set_page_config(
    page_title="Quantum Kernel DevKit — Cancer Atlas",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TITLE = "🔬 Quantum Kernel DevKit v1.3.0 — Cancer Atlas (IC50 vs Quantum Minima)"

def header_ratios_for(title: str):
    L, R = 1, 2
    n = len(title)
    if n > 80:   M = 16
    elif n > 60: M = 14
    elif n > 45: M = 12
    else:        M = 10
    return [L, M, R]

# Optional helpers
try:
    from helpers_demo import load_ccle_meta_expr_cohort, make_umap
    HAS_HELPERS = True
except Exception:
    HAS_HELPERS = False

# -------- Cohort aliases --------
COHORT_ALIASES = {
    "LUAD": "Lung Cancer", "LUSC": "Lung Cancer", "NSCLC": "Lung Cancer", "SCLC": "Lung Cancer",
    "LUNG": "Lung Cancer",
    "BRCA": "Breast Cancer", "TNBC": "Breast Cancer", "BREAST": "Breast Cancer",
    "GBM": "Brain Cancer", "GLIO": "Brain Cancer", "BRAIN": "Brain Cancer",
    "LEUK": "Leukemia", "AML": "Leukemia", "ALL": "Leukemia", "CML": "Leukemia",
    "SK": "Skin Cancer", "MEL": "Skin Cancer", "SKIN": "Skin Cancer",
    "PAAD": "Pancreatic Cancer", "PANC": "Pancreatic Cancer",
    "BONE": "Bone Cancer", "OS": "Bone Cancer", "OSTEO": "Bone Cancer",
}

# ---- Canonicalize cohort names & dedupe ----
ALIAS_MAP_LOWER = {k.lower(): v for k, v in COHORT_ALIASES.items()}
CANONS_LOWER    = {v.lower(): v for v in COHORT_ALIASES.values()}

def canonical_label(name: str) -> str:
    n = _pretty_name(name).strip()
    key = n.lower()
    if key in CANONS_LOWER:        # already a canonical label
        return CANONS_LOWER[key]
    if key in ALIAS_MAP_LOWER:     # alias like "lung", "LUAD", "NSCLC", etc.
        return ALIAS_MAP_LOWER[key]
    return n                       # leave unknowns as-is

# ---------------- Utilities ----------------
def _pretty_name(s): return re.sub(r"_+", " ", s)

@st.cache_data(show_spinner=False)
def _list_cohort_files(root):
    out = []
    if not os.path.isdir(root): return out
    for d in os.listdir(root):
        sub = os.path.join(root, d)
        if os.path.isdir(sub):
            pars = [p for p in os.listdir(sub) if p.lower().endswith(".parquet")]
            if pars: out.append((d, os.path.join(sub, pars[0])))
    return sorted(out, key=lambda x: x[0].lower())

@st.cache_data(show_spinner=False)
def _load_cache(path):
    df = pd.read_parquet(path)
    need = ["DepMap_ID","DRUG_NAME","quantum_minima","ic50","ic50_rank","Q_MEAN","n"]
    for c in need:
        if c not in df.columns: df[c] = np.nan
    for c in ["quantum_minima","ic50","ic50_rank","Q_MEAN","n"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _safe_dir_and_name(label):
    safe = re.sub(r"[^\w\-]+", "_", label)
    d = os.path.join(CACHE_ROOT, safe)
    return d, safe

def _umap_path_for(label):
    cohort_dir, safe = _safe_dir_and_name(label)
    return os.path.join(cohort_dir, f"{safe}_umap.parquet")

@st.cache_data(show_spinner=False)
def _load_or_build_umap_cached(label):
    umap_path = _umap_path_for(label)
    if os.path.exists(umap_path):
        try:
            emb = pd.read_parquet(umap_path)
            if {"x","y","z","label"}.issubset(set(emb.columns)) and len(emb) > 0:
                return emb
        except Exception:
            pass
    if not HAS_HELPERS:
        return pd.DataFrame(columns=["x","y","z","label"])

    expr, meta, _ = load_ccle_meta_expr_cohort(label)
    if expr.empty or meta.empty:
        return pd.DataFrame(columns=["x","y","z","label"])

    if expr.shape[0] > UMAP_SAMPLE:
        keep = np.random.RandomState(42).choice(expr.index, UMAP_SAMPLE, replace=False)
        expr = expr.loc[keep]
        meta = meta[meta["DepMap_ID"].isin(expr.index)]

    emb_np = make_umap(expr)
    if emb_np is None or len(emb_np) == 0:
        return pd.DataFrame(columns=["x","y","z","label"])

    lab_col = "primary_disease" if "primary_disease" in meta.columns else "lineage"
    emb = pd.DataFrame(emb_np, columns=["x","y","z"], index=expr.index)
    emb["label"] = meta.set_index("DepMap_ID").loc[emb.index, lab_col].astype(str).values

    try:
        os.makedirs(os.path.dirname(umap_path), exist_ok=True)
        emb.to_parquet(umap_path, index=True)
    except Exception:
        pass
    return emb

def _rank_ic50(df):
    sub = df[df["ic50"].notna()].copy()
    if sub.empty: return pd.DataFrame(columns=["DRUG_NAME","IC50_MEDIAN","n"])
    g = sub.groupby("DRUG_NAME")["ic50"]
    out = pd.DataFrame({"DRUG_NAME": g.median().index, "IC50_MEDIAN": g.median().values, "n": g.count().values})
    return out.sort_values(["IC50_MEDIAN","n"], ascending=[True, False]).reset_index(drop=True)

def _rank_quantum(df):
    sub = df[df["Q_MEAN"].notna()].copy()
    if sub.empty: return pd.DataFrame(columns=["DRUG_NAME","Q_MEAN","n"])
    g = sub.groupby("DRUG_NAME")["Q_MEAN"]
    out = pd.DataFrame({"DRUG_NAME": g.mean().index, "Q_MEAN": g.mean().values, "n": sub.groupby("DRUG_NAME")["Q_MEAN"].count().values})
    return out.sort_values(["Q_MEAN","n"], ascending=[True, False]).reset_index(drop=True)

def _drug_suggestions(query, all_drugs, limit=30):
    if not query: return sorted(all_drugs)[:limit]
    q = query.strip().lower()
    starts = [d for d in all_drugs if str(d).lower().startswith(q)]
    contains = [d for d in all_drugs if q in str(d).lower() and d not in starts]
    return (starts + contains)[:limit]

def _suggest_cohorts(query, available_labels):
    if not query: return available_labels
    q = query.strip().lower()
    hits = []
    for alias, canon in COHORT_ALIASES.items():
        if alias.lower().startswith(q) or q in alias.lower():
            if canon in available_labels: hits.append(canon)
    for name in available_labels:
        if name.lower().startswith(q) or q in name.lower(): hits.append(name)
    seen, out = set(), []
    for h in hits:
        if h not in seen:
            out.append(h); seen.add(h)
    return out if out else available_labels

def _top_drug_by_metric(df):
    best_ic50 = (df.dropna(subset=["ic50"])
                   .sort_values(["DepMap_ID","ic50"], ascending=[True,True])
                   .groupby("DepMap_ID").first())
    ic50_drug = best_ic50["DRUG_NAME"] if "DRUG_NAME" in best_ic50.columns else pd.Series(dtype=object)

    best_qm = (df.dropna(subset=["quantum_minima"])
                 .sort_values(["DepMap_ID","quantum_minima"], ascending=[True,True])
                 .groupby("DepMap_ID").first())
    qm_drug = best_qm["DRUG_NAME"] if "DRUG_NAME" in best_qm.columns else pd.Series(dtype=object)
    return ic50_drug, qm_drug

def _umap_winner_and_sensitivity(df):
    if df.empty or "DepMap_ID" not in df.columns:
        return pd.Series(dtype=object), pd.Series(dtype=float)

    ic_best = (df.dropna(subset=["ic50"])
                 .sort_values(["DepMap_ID","ic50"], ascending=[True, True])
                 .groupby("DepMap_ID")["ic50"].first())
    qm_best = (df.dropna(subset=["quantum_minima"])
                 .sort_values(["DepMap_ID","quantum_minima"], ascending=[True, True])
                 .groupby("DepMap_ID")["quantum_minima"].first())

    if len(ic_best) > 1:
        ic_rank = ic_best.rank(method="dense", ascending=True)
        ic_pct  = 1.0 - (ic_rank - 1) / (len(ic_rank) - 1)
    else:
        ic_pct = pd.Series(dtype=float)

    if len(qm_best) > 1:
        qm_rank = qm_best.rank(method="dense", ascending=True)
        qm_pct  = 1.0 - (qm_rank - 1) / (len(qm_rank) - 1)
    else:
        qm_pct = pd.Series(dtype=float)

    common = ic_best.index.intersection(qm_best.index)
    winners = pd.Series(np.where(ic_pct.reindex(common) > qm_pct.reindex(common),
                                 "IC50-better", "Quantum-better"), index=common)

    sens = pd.Series(index=common, dtype=float)
    sens[winners == "IC50-better"] = ic_pct.reindex(common)[winners == "IC50-better"]
    sens[winners == "Quantum-better"] = qm_pct.reindex(common)[winners == "Quantum-better"]

    only_ic = ic_best.index.difference(qm_best.index)
    only_qm = qm_best.index.difference(ic_best.index)
    if len(only_ic):
        winners = winners.reindex(winners.index.union(only_ic))
        winners.loc[only_ic] = "IC50-better"
        sens = sens.reindex(sens.index.union(only_ic))
        sens.loc[only_ic] = ic_pct.reindex(only_ic)
    if len(only_qm):
        winners = winners.reindex(winners.index.union(only_qm))
        winners.loc[only_qm] = "Quantum-better"
        sens = sens.reindex(sens.index.union(only_qm))
        sens.loc[only_qm] = qm_pct.reindex(only_qm)

    winners = winners.fillna("Unknown")
    sens = sens.fillna(0.0).clip(0.0, 1.0)
    return winners, sens

def _knn_density_xyz(df_xyz, k=8):
    if df_xyz.empty: return pd.Series(dtype=float, index=df_xyz.index)
    X = df_xyz[["x","y","z"]].to_numpy(float)
    n = len(X)
    D = np.sqrt(((X[:,None,:] - X[None,:,:])**2).sum(axis=2))
    D.sort(axis=1)
    k = max(1, min(k, n-1))
    mean_nn = D[:, 1:k+1].mean(axis=1)
    dens = 1.0 / (mean_nn + 1e-9)
    order = np.argsort(dens)
    pct = np.empty_like(dens)
    pct[order] = np.linspace(0, 1, len(dens))
    return pd.Series(pct, index=df_xyz.index)

# ---------- Leaderboard HTML ----------
def _leaderboard_html(df, value_cols):
    if df.empty:
        return "<div>No rows.</div>"
    dfv = df.head(DEFAULT_TOPK).reset_index(drop=True).copy()
    dfv.insert(0, "⭐", ["★"] + [""]*(len(dfv)-1))
    def _fmt(x, col):
        if col in ("IC50_MEDIAN","Q_MEAN"):
            try: return f"{float(x):.4f}"
            except: return str(x)
        if col == "n":
            try: return f"{int(x)}"
            except: return str(x)
        return str(x)
    cols = ["⭐","DRUG_NAME"] + value_cols + ["n"]
    rows_html = []
    for i, r in dfv.iterrows():
        cells = "".join(f"<td style='padding:8px 10px;'>{_fmt(r[c], c)}</td>" for c in cols)
        style = ' style="background-color:rgba(0,224,198,0.18);"' if i == 0 else ""
        rows_html.append(f"<tr{style}>{cells}</tr>")
    table = f"""
    <div style="overflow:auto;">
      <table class="tiny" style="width:100%; border-collapse:collapse;">
        <thead><tr>{"".join(f"<th style='text-align:left;padding:8px 10px;text-transform:uppercase;letter-spacing:.5px;'>{c}</th>" for c in cols)}</tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    </div>
    """
    return table

# ---------------- Theme (CSS) ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Sora:wght@600;700;800&display=swap');

.stApp {
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(255,146,209,0.25), transparent 60%),
    radial-gradient(1000px 800px at 110% 10%, rgba(0,232,255,0.20), transparent 60%),
    radial-gradient(700px 500px at 30% 100%, rgba(112,104,255,0.22), transparent 60%),
    linear-gradient(135deg, #e3a1ff 0%, #88d1ff 40%, #7ff0d2 70%, #c6a3ff 100%);
  background-attachment: fixed;
}
.block-container { padding-top: 0.6rem; }
[data-testid="stSidebar"] { display: none; }

.header-bar{
  position: sticky; top: 0; z-index: 50;
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  background: linear-gradient(135deg, rgba(227,161,255,.60), rgba(136,209,255,.60), rgba(127,240,210,.60), rgba(198,163,255,.60));
  padding-bottom: 6px;
}

.glass {
  background: rgba(255,255,255,0.30);
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255,255,255,0.25);
  border-radius: 18px;
  padding: 1.1rem 1.2rem;
  box-shadow: 0 8px 28px rgba(7,10,38,0.18);
  margin-bottom: 16px;
  overflow: hidden;
}
.chart-spacer { height: 10px; }

.title-wrap { margin: 2px 0 8px 0; }
.app-title {
  font-family: "Sora", system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans";
  font-weight: 800;
  font-size: clamp(40px, 6.2vw, 64px);
  line-height: 1.06;
  letter-spacing: 0.2px;
  color: #0b1220;
  text-shadow: 0 2px 16px rgba(255,255,255,0.35);
  word-break: break-word;
}

.stApp * { font-family: "Inter", system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans"; }
.app-subtitle { font-family: "Sora"; font-weight: 700; font-size: clamp(18px, 2vw, 26px); color: #0b1220; margin-top: 0.25rem; }
.loaded-badge { color: #059669; font-weight: 700; }

.menu-wrap { min-width: 260px; max-width: 360px; }

input, textarea, select, .stTextInput>div>div>input, .stSelectbox div[data-baseweb="select"]>div {
  border-radius: 12px !important; background: rgba(255,255,255,0.75) !important;
}
.stSlider>div>div>div>div { background: linear-gradient(90deg, #00e0c6, #7aa7ff) !important; }
.stSlider [data-baseweb="slider"]>div { background-color: rgba(255,255,255,0.55) !important; }

button, .stButton>button {
  border-radius: 12px !important; background: linear-gradient(135deg, #00e0c6, #7aa7ff);
  border: 0; color: #051018; font-weight: 700; box-shadow: 0 6px 18px rgba(15,110,180,0.25);
}
button:hover, .stButton>button:hover { filter: brightness(1.05); }
button[kind="secondary"] { white-space: nowrap; }

[data-testid="stDataFrame"]{
  background: rgba(255,255,255,0.65); border: 1px solid rgba(255,255,255,0.55);
  border-radius: 16px; box-shadow: 0 6px 18px rgba(7,10,38,0.12); overflow: hidden;
}
[data-testid="stDataFrame"] div[role="grid"]{ outline:none !important; border:none !important; }
[data-testid="stDataFrame"] *:focus{ outline:none !important; }
[data-testid="stDataFrame"] > div > div{ overflow:auto !important; }
[data-testid="stDataFrame"] thead th, [data-testid="stDataFrame"] tbody td { border-right:0 !important; }
[data-testid="stDataFrame"] > div{ scrollbar-gutter:stable both-edges; }

.footer { font-size: 12px; color: #0b1220; }
</style>
""", unsafe_allow_html=True)

# ---------------- Data discovery ----------------
# ---------------- Data discovery ----------------
files = _list_cohort_files(CACHE_ROOT)
if not files:
    st.error(f"No caches found in {CACHE_ROOT}. Build them first.")
    st.stop()

# Deduplicate by canonical label; prefer a folder that already matches the canonical name
path_by_label = {}
for raw_name, path in files:
    canon = canonical_label(raw_name)
    if (canon not in path_by_label) or (_pretty_name(raw_name).strip().lower() == canon.lower()):
        path_by_label[canon] = path

all_labels = sorted(path_by_label.keys())

# ---------------- Header + Cohort (STICKY) ----------------
st.markdown('<div class="header-bar">', unsafe_allow_html=True)

hdr_left, hdr_mid, hdr_right = st.columns(header_ratios_for(TITLE))

with hdr_left:
    try:
        menu = st.popover("☰", use_container_width=False)
    except Exception:
        menu = st.expander("☰ Filters", expanded=False)

with hdr_mid:
    st.markdown(f'<div class="title-wrap"><div class="app-title">{TITLE}</div></div>', unsafe_allow_html=True)

with hdr_right:
    st.button("Deploy", use_container_width=True)

row_a, row_b = st.columns([0.72, 0.28])
with row_a:
    cohort = st.selectbox("Pick cohort", all_labels, index=0, label_visibility="collapsed", key="cohortpick")
with row_b:
    df = _load_cache(path_by_label[cohort])
    st.markdown('<div style="padding-top:6px"><span class="loaded-badge">✅ Data loaded</span></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # end sticky header-bar

# ---------------- Filters inside the popover ----------------
with menu:
    st.markdown('<div class="menu-wrap">', unsafe_allow_html=True)

    st.markdown("### Filters")
    q_cohort = st.text_input("Search cancer type (e.g., SK, LUAD)", value="", key="q_cohort")
    _ = _suggest_cohorts(q_cohort, all_labels)
    st.divider()

    _max_n = int(df["n"].max()) if df["n"].notna().any() else 1
    _default_min_n = min(5, max(1, _max_n))

    min_n = st.slider("Min. samples per drug (n)", 1, max(1, _max_n), _default_min_n, key="min_n")

    all_drugs = sorted(set(df["DRUG_NAME"].dropna().astype(str)))
    drug_query = st.text_input("Search drug name", key="drug_query")
    drug_suggest = _drug_suggestions(drug_query, all_drugs, limit=30)
    drug_pick = st.selectbox("Pick a drug (optional)", ["(any)"] + drug_suggest, index=0, key="drug_pick")
    cell_q = st.text_input("Search DepMap ID", key="cell_q")
    st.caption("Selections apply instantly.")

    cA, cB = st.columns(2)
    with cA:
        if st.button("Reset filters"):
            st.session_state["q_cohort"] = ""
            st.session_state["min_n"] = _default_min_n
            st.session_state["drug_query"] = ""
            st.session_state["drug_pick"] = "(any)"
            st.session_state["cell_q"] = ""
            st.experimental_rerun()

    with cB:
        top_ic50 = _rank_ic50(df).head(DEFAULT_TOPK).rename(columns={"IC50_MEDIAN": "VALUE"})
        top_qm   = _rank_quantum(df).head(DEFAULT_TOPK).rename(columns={"Q_MEAN": "VALUE"})
        out = pd.concat(
            [
                top_ic50.assign(LIST="IC50"),
                top_qm.assign(LIST="Quantum"),
            ],
            ignore_index=True
        )
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download top lists CSV", data=csv_bytes, file_name="toplists.csv", mime="text/csv")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Apply filters ----------------
dfv = df.copy()
dfv = dfv[dfv["n"] >= st.session_state.get("min_n", _default_min_n)]
if st.session_state.get("drug_pick", "(any)") != "(any)":
    dfv = dfv[dfv["DRUG_NAME"].astype(str) == st.session_state["drug_pick"]]
elif st.session_state.get("drug_query", ""):
    dfv = dfv[dfv["DRUG_NAME"].astype(str).str.lower().str.contains(st.session_state["drug_query"].strip().lower())]
if st.session_state.get("cell_q", ""):
    dfv = dfv[dfv["DepMap_ID"].astype(str).str.upper().str.contains(st.session_state["cell_q"].strip().upper())]

# ---------------- Per-sample shortlists ----------------
st.markdown('<div class="glass"><div class="app-subtitle">Per-sample shortlists</div><div class="chart-spacer"></div>', unsafe_allow_html=True)

ic50_rows = (
    dfv.dropna(subset=["ic50"])
       .sort_values(["DepMap_ID","ic50"], ascending=[True,True])
       .groupby("DepMap_ID").head(1)
       .sort_values("ic50")
       .head(int(DEFAULT_TOPK))[["DepMap_ID","DRUG_NAME","ic50","ic50_rank","n"]]
)
qm_rows = (
    dfv.dropna(subset=["quantum_minima"])
       .sort_values(["DepMap_ID","quantum_minima"], ascending=[True,True])
       .groupby("DepMap_ID").head(1)
       .sort_values("quantum_minima")
       .head(int(DEFAULT_TOPK))[["DepMap_ID","DRUG_NAME","quantum_minima","Q_MEAN","n"]]
)

cL2, cR2 = st.columns(2, gap="large")
with cL2:
    st.caption("Best per sample by IC50 (lower is better)")
    st.dataframe(ic50_rows, use_container_width=True, hide_index=True, height=420)
with cR2:
    st.caption("Best per sample by Quantum minima (lower is better)")
    st.dataframe(qm_rows, use_container_width=True, hide_index=True, height=420)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Split leaderboards ----------------
colL, colR = st.columns(2, gap="large")
with colL:
    htmlL = _leaderboard_html(_rank_ic50(dfv), ["IC50_MEDIAN"])
    st.markdown(
        f'<div class="glass"><div class="app-subtitle">🏥 IC50 (wet-lab) — top drugs</div><div class="chart-spacer"></div>{htmlL}</div>',
        unsafe_allow_html=True
    )
with colR:
    htmlR = _leaderboard_html(_rank_quantum(dfv), ["Q_MEAN"])
    st.markdown(
        f'<div class="glass"><div class="app-subtitle">🧠 Quantum minima — top drugs</div><div class="chart-spacer"></div>{htmlR}</div>',
        unsafe_allow_html=True
    )

# ---------------- Scatter ----------------
st.markdown('<div class="glass"><div class="app-subtitle">Scatter — compare measures (subsampled)</div><div class="chart-spacer"></div>', unsafe_allow_html=True)
plot_df = dfv.dropna(subset=["quantum_minima","ic50"]).copy()
if len(plot_df) >= 10:
    plot_df = plot_df.sample(min(len(plot_df), MAX_SCATTER), random_state=42)
    plot_df["winner"] = np.where(
        plot_df["ic50"].rank(method="dense") < plot_df["quantum_minima"].rank(method="dense"),
        "IC50-better", "Quantum-better"
    )
    fig = px.scatter(
        plot_df, x="quantum_minima", y="ic50",
        color="winner",
        color_discrete_map={"IC50-better":"#e53935","Quantum-better":"#1e88e5"},
        hover_data=["DRUG_NAME","DepMap_ID","Q_MEAN","n"],
        title="Lower-left is best; color shows which metric wins per row",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.9, line=dict(width=0)))
    fig.update_layout(margin=dict(l=0,r=0,t=40,b=0), legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough rows with both minima and IC50 to plot.")
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Topographic UMAP ----------------
st.markdown('<div class="glass"><div class="app-subtitle">Topographic UMAP — high sensitivity</div><div class="chart-spacer"></div>', unsafe_allow_html=True)

emb = _load_or_build_umap_cached(cohort)
if emb is None or emb.empty:
    st.info("UMAP not available. (If helpers are present, it will auto-build once.)")
else:
    winners, sens = _umap_winner_and_sensitivity(dfv)
    ic50_drug, qm_drug = _top_drug_by_metric(dfv)

    emb2 = emb.copy()
    for c in ["x","y","z"]:
        emb2[c] = pd.to_numeric(emb2[c], errors="coerce")
    emb2 = emb2.dropna(subset=["x","y","z"])

    emb2["winner"] = winners.reindex(emb2.index).fillna("Unknown")
    emb2["sens"] = sens.reindex(emb2.index).fillna(0.0).clip(0.0, 1.0)

    best_ic = ic50_drug.reindex(emb2.index).astype(object)
    best_qm = qm_drug.reindex(emb2.index).astype(object)
    emb2["top_drug"] = np.where(
        emb2["winner"] == "IC50-better", best_ic.fillna("—"),
        np.where(emb2["winner"] == "Quantum-better", best_qm.fillna("—"), "—")
    )

    emb2["density_pct"] = _knn_density_xyz(emb2, k=KNN_K)
    emb2["size"] = (4 + (12 - 4) * emb2["sens"]).astype(float)

    color_map = {"IC50-better": "#e53935", "Quantum-better": "#1e88e5", "Unknown": "#9e9e9e"}

    fig3d = px.scatter_3d(
        emb2, x="x", y="y", z="z",
        color="winner",
        color_discrete_map=color_map,
        hover_name=emb2.index,
        hover_data={"winner": True, "top_drug": True, "sens": ':.2f', "density_pct": ':.2f', "x": False, "y": False, "z": False},
    )
    fig3d.update_traces(marker=dict(size=emb2["size"], opacity=0.85))

    q90 = np.quantile(emb2["sens"], 0.90) if len(emb2) else 1.0
    hot = emb2[emb2["sens"] >= q90]
    if len(hot):
        halo_size = float(emb2["size"].max() * 1.8)
        halo_text = (
            "DepMap_ID: %{customdata[0]}<br>"
            "Winner: %{customdata[1]}<br>"
            "Top drug: %{customdata[2]}<br>"
            "Confidence (sens): %{customdata[3]:.2f}<br>"
            "Local density: %{customdata[4]:.2f}"
        )
        fig3d.add_trace(go.Scatter3d(
            x=hot["x"], y=hot["y"], z=hot["z"],
            mode="markers",
            marker=dict(size=halo_size, opacity=0.20, color=hot["winner"].map(color_map)),
            customdata=np.stack([
                hot.index.to_numpy(),
                hot["winner"].to_numpy(),
                hot["top_drug"].to_numpy(),
                hot["sens"].to_numpy(),
                hot["density_pct"].to_numpy()
            ], axis=1),
            hovertemplate=halo_text,
            name="High-sensitivity hotspot",
            showlegend=False,
        ))

    fig3d.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title_text="",
        title=f"{cohort} — Topographic UMAP (two-tone, high-sensitivity regions)"
    )
    st.plotly_chart(fig3d, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Footer ----------------
st.markdown("""
<div class="glass footer">
  <strong>Gfam Quantum Kernel DevKit v1.3.0 — Cancer Atlas (IC50 vs Quantum Minima) — 2025</strong><br/>
  This interactive is provided for pitch/demo purposes only. It summarizes public cell-line data and experimental metrics (IC50) alongside simulated quantum minima scores <em>(equation-backed & reproducible)</em>. Values are not medical advice and should not be used for patient treatment decisions. Any interpretations require independent validation in appropriate wet-lab and clinical settings. © 2025 Gfam Quantum.
</div>
""", unsafe_allow_html=True)
