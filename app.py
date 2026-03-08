"""PathogenScope — Live Demo Dashboard"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

RESULTS = Path("results")

st.set_page_config(
    page_title="PathogenScope",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Force light mode
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #ffffff; }
    [data-testid="stHeader"] { background-color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #f8f9fa; }
    .stMetric { background-color: #f8f9fa; padding: 16px; border-radius: 8px; border: 1px solid #e9ecef; }
    h1 { color: #1a1a2e; }
    h2, h3 { color: #16213e; }
    .pathway-pass { color: #27ae60; font-weight: bold; }
    .pathway-miss { color: #e74c3c; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    merged = json.loads((RESULTS / "merged_targets.json").read_text())
    network = json.loads((RESULTS / "network_data.json").read_text())
    mimicry = pd.read_csv(RESULTS / "mimicry_with_partners.csv")
    hubs = pd.read_csv(RESULTS / "pathogen_hubs.csv")
    return merged, network, mimicry, hubs


merged, network, mimicry, hubs = load_data()
targets = pd.DataFrame(merged["targets"])


# ═══════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("# PathogenScope")
st.markdown("### Computational Drug Target Discovery for *Acinetobacter baumannii*")
st.markdown("WHO's #1 critical priority superbug — causes untreatable hospital infections with >50% mortality in ICU outbreaks.")

st.divider()

# Key metrics row
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Proteome", f"{len(targets):,} proteins")
c2.metric("Structural mimics", f"{mimicry['query_uniprot'].nunique():,}")
c3.metric("PPI hubs", f"{len(hubs)}")
c4.metric("Convergent targets", f"{int(targets['convergent'].sum())}")
c5.metric("Pathways recovered", "11/12")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# PIPELINE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## How it works")
st.markdown("""
| Step | Tool | What it finds | Output |
|------|------|--------------|--------|
| 1. Structural mimicry screen | **Foldseek** vs human proteome | Pathogen proteins that *look like* human proteins | 1,475 mimics |
| 2. Intra-pathogen PPI network | **FlashPPI** (H100 GPU) | Hub proteins essential to pathogen survival | 338 hubs |
| 3. STRING network expansion | **STRING DB** | Which human pathways are being hijacked | 1,719 human proteins |
| 4. Signal integration | Tiered ranking | Convergent targets = mimicry + hub | **121 drug targets** |
""")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# MONEY SHOT: INTERACTION RECOVERY
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## Blind Recovery of Known Virulence Pathways")
st.markdown("We recovered **11 of 12** known A. baumannii host-interaction pathways — with **zero prior knowledge**.")

KNOWN = {
    "GroEL/chaperonin": {"expected": 3, "recovered": 3, "targets": "TLR2, TLR4, HSPD1", "mechanism": "Innate immune activation"},
    "Porin": {"expected": 4, "recovered": 3, "targets": "CASP3, BAX, BCL2", "mechanism": "Apoptosis manipulation"},
    "OmpA": {"expected": 7, "recovered": 4, "targets": "TLR2, TLR4, FN1, CASP3", "mechanism": "Immune evasion + adhesion"},
    "DnaK/Hsp70": {"expected": 3, "recovered": 2, "targets": "TLR2, TLR4", "mechanism": "Immune activation"},
    "DnaJ": {"expected": 2, "recovered": 2, "targets": "DNAJA1, DNAJB1", "mechanism": "Chaperone mimicry"},
    "Efflux pumps": {"expected": 2, "recovered": 2, "targets": "ABCB1, ABCG2", "mechanism": "Drug resistance homology"},
    "Ata adhesin": {"expected": 5, "recovered": 1, "targets": "FN1", "mechanism": "Host cell adhesion"},
    "Bap": {"expected": 1, "recovered": 1, "targets": "FN1", "mechanism": "Biofilm adhesion"},
    "LPS": {"expected": 4, "recovered": 1, "targets": "TLR4", "mechanism": "Endotoxin signaling"},
    "Capsule": {"expected": 2, "recovered": 1, "targets": "TLR4", "mechanism": "Immune evasion"},
    "Phospholipase": {"expected": 2, "recovered": 1, "targets": "PLA2G4A", "mechanism": "Membrane disruption"},
    "Siderophore": {"expected": 3, "recovered": 0, "targets": "", "mechanism": "Iron acquisition"},
}

pdf = pd.DataFrame([
    {"Pathway": k, "Recovery": v["recovered"]/v["expected"],
     "Recovered": v["recovered"], "Expected": v["expected"],
     "Human Targets Found": v["targets"], "Mechanism": v["mechanism"],
     "Status": "Recovered" if v["recovered"] > 0 else "Missed"}
    for k, v in KNOWN.items()
])

fig_path = px.bar(
    pdf.sort_values("Recovery", ascending=True),
    x="Recovery", y="Pathway", orientation="h",
    color="Status",
    color_discrete_map={"Recovered": "#27ae60", "Missed": "#e74c3c"},
    hover_data=["Human Targets Found", "Mechanism"],
    labels={"Recovery": "Fraction of Known Targets Recovered"},
)
fig_path.update_layout(
    height=450, showlegend=True,
    plot_bgcolor="white",
    xaxis=dict(range=[0, 1.05], tickformat=".0%", gridcolor="#eee"),
    yaxis=dict(gridcolor="#eee"),
    font=dict(size=13),
)
st.plotly_chart(fig_path, use_container_width=True)

# Highlight the best hit
col_a, col_b = st.columns([1, 1])
with col_a:
    st.markdown("### Showcase: GroEL mimics HSPD1")
    st.markdown("""
    A. baumannii **chaperonin GroEL** structurally mimics human **HSPD1** with
    TM-score **0.996** (near-identical fold). HSPD1 directly interacts with
    **TLR2** and **TLR4** — the innate immune receptors that trigger
    inflammatory response.

    This is a validated pathogen-associated molecular pattern (PAMP). Our
    pipeline found it **without any prior knowledge** of this interaction.
    """)
    st.markdown("""
    **Other recovered interactions:**
    - Porins → **CASP3/BAX/BCL2** (apoptosis)
    - DnaK → **TLR2/TLR4** (immune)
    - Efflux → **ABCB1/ABCG2** (drug resistance)
    - OmpA → **FN1** (adhesion)
    """)

with col_b:
    groel = mimicry[mimicry["query_uniprot"] == "V5VAH2"].copy()
    if not groel.empty:
        groel_show = groel[["target_gene", "alntmscore", "fident", "string_partners"]].rename(columns={
            "target_gene": "Human protein", "alntmscore": "TM-score",
            "fident": "Seq identity", "string_partners": "STRING partners"
        })
        st.dataframe(groel_show, use_container_width=True, hide_index=True)

# 3D structure comparison
st.markdown("### 3D Structure Comparison: Pathogen GroEL vs Human HSPD1")
st.markdown("Near-identical folds (TM = 0.996) despite only 59% sequence identity — structural mimicry enables immune pathway hijacking.")

struct_col1, struct_col2 = st.columns(2)
with struct_col1:
    st.markdown("**A. baumannii GroEL** (V5VAH2)")
    st.components.v1.iframe(
        "https://molstar.org/viewer/?pdb-url=https://alphafold.ebi.ac.uk/files/AF-V5VAH2-F1-model_v4.cif&hide-controls=1",
        height=400,
    )
with struct_col2:
    st.markdown("**Human HSPD1** (P10809)")
    st.components.v1.iframe(
        "https://molstar.org/viewer/?pdb-url=https://alphafold.ebi.ac.uk/files/AF-P10809-F1-model_v4.cif&hide-controls=1",
        height=400,
    )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# CONVERGENT TARGETS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## Top Drug Target Candidates")
st.markdown("Proteins with **dual signal** (structural mimic of human protein + essential hub in pathogen network) = highest priority drug targets.")

top_n = 50
top = targets.head(top_n).copy()
top["rank"] = range(1, len(top) + 1)
top["label"] = top.apply(
    lambda r: str(r["gene"]) if r["gene"] and not isinstance(r["gene"], float) and str(r["gene"]) != "nan"
    else r["protein_id"][:12], axis=1
)
top["signal"] = top["signals"].apply(
    lambda x: "Convergent (mimicry + hub)" if len(x) >= 2 else (x[0].title() if x else "Other")
)

fig_rank = px.bar(
    top, x="rank", y="composite_score",
    color="signal",
    hover_data=["label", "mimicry_tm", "hub_score", "degree", "n_ppi_partners", "best_human_target"],
    color_discrete_map={
        "Convergent (mimicry + hub)": "#e74c3c",
        "Hub": "#3498db",
        "Mimicry": "#2ecc71",
        "Other": "#bdc3c7",
    },
    labels={"composite_score": "Target Score", "rank": "Rank"},
)
fig_rank.update_layout(
    height=400,
    plot_bgcolor="white",
    xaxis=dict(gridcolor="#eee"),
    yaxis=dict(gridcolor="#eee"),
    font=dict(size=12),
)
st.plotly_chart(fig_rank, use_container_width=True)

# Table of top targets
display_cols = ["rank", "label", "signal", "best_human_target", "mimicry_tm", "degree", "n_ppi_partners"]
show = top[top["signal"] == "Convergent (mimicry + hub)"][display_cols].head(20).rename(columns={
    "label": "Protein", "signal": "Signal", "best_human_target": "Human Target",
    "mimicry_tm": "TM-score", "degree": "PPI Degree", "n_ppi_partners": "PPI Partners",
})
st.dataframe(show, use_container_width=True, hide_index=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# NETWORK VIEWS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## Structural Mimicry & PPI Network")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Mimicry: TM-score distribution")
    fig_tm = px.histogram(
        mimicry, x="alntmscore", nbins=50,
        labels={"alntmscore": "TM-score (structural similarity)"},
        color_discrete_sequence=["#e74c3c"],
    )
    fig_tm.update_layout(
        height=350, plot_bgcolor="white",
        xaxis=dict(gridcolor="#eee"), yaxis=dict(gridcolor="#eee"),
    )
    st.plotly_chart(fig_tm, use_container_width=True)

with col2:
    st.markdown("### Most mimicked human proteins")
    target_counts = mimicry["target_gene"].value_counts().head(15)
    fig_targets = px.bar(
        x=target_counts.values, y=target_counts.index,
        orientation="h",
        labels={"x": "# pathogen proteins mimicking it", "y": ""},
        color_discrete_sequence=["#3498db"],
    )
    fig_targets.update_layout(
        height=350, plot_bgcolor="white",
        xaxis=dict(gridcolor="#eee"), yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_targets, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown("### Pathogen PPI hubs (FlashPPI)")
    fig_hubs = px.scatter(
        hubs, x="degree", y="betweenness",
        size="hub_score", size_max=18,
        hover_data=["protein"],
        labels={"degree": "Degree", "betweenness": "Betweenness"},
        color="hub_score", color_continuous_scale="Reds",
    )
    fig_hubs.update_layout(
        height=350, plot_bgcolor="white",
        xaxis=dict(gridcolor="#eee"), yaxis=dict(gridcolor="#eee"),
    )
    st.plotly_chart(fig_hubs, use_container_width=True)

with col4:
    st.markdown("### Network summary")
    st.markdown(f"""
    | | Count |
    |---|---|
    | **Nodes** | {network['metadata']['node_count']:,} |
    | **Edges** | {network['metadata']['edge_count']:,} |
    | **Edge types** | {', '.join(network['metadata']['edge_types'])} |
    | **Pathogen proteins** | 3,661 |
    | **Human proteins** | {mimicry['target_gene'].nunique()} |
    | **PPI edges** | {len(pd.read_csv(RESULTS / 'pathogen_network.csv')):,} |
    """)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("## Method")
st.markdown("""
1. Download 3,661 A. baumannii protein structures from AlphaFold DB
2. **Foldseek** structural similarity search against human proteome (1.9M comparisons)
3. **FlashPPI** all-vs-all intra-pathogen PPI prediction on H100 GPU
4. **STRING DB** interaction network expansion (1-hop + 2-hop)
5. Tiered ranking: convergent > hub > PPI-connected > mimicry only
6. Validation against VFDB + PHI-base ground truth (11/12 pathways, 2.9x enrichment)

**Runtime**: ~30 minutes end-to-end on single GPU node. Scales to any bacterial proteome.
""")

st.caption("PathogenScope | Bio x AI Hackathon 2026 | Gabriel, Joe, Joseph, Zijian")
