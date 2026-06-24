"""
Network Metrics Module for Social Network Analysis System
Computes structural graph metrics and supports visualization.
"""

import os
import pandas as pd # type: ignore
import networkx as nx
import plotly.express as px # type: ignore
import logging

# Configure logger (works with main pipeline logging)
logger = logging.getLogger(__name__)

# ===============================
# ---- Load + Filter Data -------
# ===============================

def load_edges_csv(path: str) -> pd.DataFrame:
    """Load cleaned CSV robustly."""
    try:
        df = pd.read_csv(
            path,
            engine="c",
            low_memory=False,
            on_bad_lines="warn",
            encoding="utf-8",
            encoding_errors="ignore"
        )
        logger.info(f"Loaded data: {len(df):,} rows from {path}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        raise


def filter_valid_emails(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows where sender & recipient look like valid emails (contain '@')."""
    required_cols = [c for c in df.columns if c.lower() in ["sender", "from", "email_from"]]
    recipient_cols = [c for c in df.columns if c.lower() in ["recipient", "to", "email_to"]]
    sender_col = required_cols[0] if required_cols else "sender"
    recipient_col = recipient_cols[0] if recipient_cols else "recipient"

    if sender_col not in df.columns or recipient_col not in df.columns:
        raise ValueError("Could not find sender/recipient columns in dataset.")

    df = df.rename(columns={sender_col: "sender", recipient_col: "recipient"})

    mask = df["sender"].astype(str).str.contains("@", na=False) & \
           df["recipient"].astype(str).str.contains("@", na=False)

    subset_cols = [c for c in ["sender", "recipient", "recipient_type", "subject", "date"] if c in df.columns]
    filtered = df.loc[mask, subset_cols].copy()

    logger.info(f"Filtered valid email pairs: {len(filtered):,} rows remain")
    return filtered


# ===============================
# ----- Build Communication -----
# ===============================

def build_comm_graph(df_edges: pd.DataFrame,
                     directed: bool = True,
                     weight_by_recipient_type: bool = False):
    """
    Build a communication graph from rows (sender, recipient[, recipient_type]).
    - directed=True -> DiGraph with edges sender -> recipient
    - weight_by_recipient_type=True  -> To=1.0, Cc=0.5, Bcc=0.5
    """
    logger.info("Building communication graph...")
    if weight_by_recipient_type:
        wmap = {"to": 1.0, "cc": 0.5, "bcc": 0.5}
        df_edges = df_edges.assign(w=df_edges["recipient_type"].str.lower().map(wmap).fillna(1.0))
    else:
        df_edges = df_edges.assign(w=1.0)

    agg = (
        df_edges
        .groupby(["sender", "recipient"], as_index=False)["w"]
        .sum()
        .rename(columns={"w": "weight"})
    )

    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(pd.unique(agg[["sender", "recipient"]].values.ravel("K")))

    for _, r in agg.iterrows():
        u, v, w = r["sender"], r["recipient"], float(r["weight"])
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)

    logger.info(f"Graph built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    return G, agg


# ===============================
# --------- SNA Metrics ---------
# ===============================

def compute_metrics(
    G: nx.DiGraph,
    betweenness_sample_k: int | None = 800,
    seed: int = 42,
    include_pagerank: bool = True
) -> pd.DataFrame:
    """Compute core SNA metrics and return as DataFrame."""
    logger.info("Computing network metrics...")

    indeg = dict(G.in_degree()) if G.is_directed() else dict(G.degree())
    outdeg = dict(G.out_degree()) if G.is_directed() else dict(G.degree())
    wdeg = dict(G.degree(weight="weight"))
    und = G.to_undirected()

    clustering = nx.clustering(und, weight="weight")

    k_val = min(betweenness_sample_k, G.number_of_nodes()) if betweenness_sample_k else None
    betw = nx.betweenness_centrality(G, k=k_val, normalized=True, weight="weight", seed=seed)

    pr = {}
    if include_pagerank:
        pr = nx.pagerank(G, alpha=0.85, weight="weight")

    rows = []
    for n in G.nodes():
        rows.append({
            "node": n,
            "in_degree": indeg.get(n, 0),
            "out_degree": outdeg.get(n, 0),
            "weighted_degree": wdeg.get(n, 0.0),
            "clustering": clustering.get(n, 0.0),
            "betweenness": betw.get(n, 0.0),
            **({"pagerank": pr.get(n, 0.0)} if include_pagerank else {}),
        })

    metrics_df = pd.DataFrame(rows).sort_values("weighted_degree", ascending=False).reset_index(drop=True)
    logger.info(f"Metrics computed for {len(metrics_df):,} nodes")
    return metrics_df


# ===============================
# -------- Visualization --------
# ===============================

def plot_top_subgraph(
    G: nx.DiGraph,
    metrics: pd.DataFrame,
    top_n: int = 120,
    layout: str = "spring",
    title: str = "Communication Network (Top by Weighted Degree)"
):
    """Plot top-N nodes with Plotly for interactivity."""
    keep = set(metrics.head(top_n)["node"].tolist())
    H = G.subgraph(keep).copy()

    logger.info(f"Plotting top {top_n} nodes...")

    if layout == "spring":
        pos = nx.spring_layout(H, seed=42, weight="weight")
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(H, weight="weight")
    else:
        pos = nx.random_layout(H, seed=42)

    xs, ys = [], []
    for u, v in H.edges():
        xs += [pos[u][0], pos[v][0], None]
        ys += [pos[u][1], pos[v][1], None]

    m_sub = metrics.set_index("node").loc[list(H.nodes())]
    nodes_df = pd.DataFrame({
        "node": list(H.nodes()),
        "x": [pos[n][0] for n in H.nodes()],
        "y": [pos[n][1] for n in H.nodes()],
        "weighted_degree": m_sub["weighted_degree"].values,
        "betweenness": m_sub["betweenness"].values,
        "pagerank": m_sub.get("pagerank", pd.Series(0, index=m_sub.index)).values,
        "in_degree": m_sub["in_degree"].values,
        "out_degree": m_sub["out_degree"].values,
    })

    fig = px.scatter(
        nodes_df, x="x", y="y",
        size="weighted_degree",
        hover_name="node",
        hover_data={
            "in_degree": True, "out_degree": True,
            "betweenness": ':.4f', "pagerank": ':.4f',
            "x": False, "y": False
        },
        title=title
    )
    fig.add_scatter(x=xs, y=ys, mode="lines", name="edges", hoverinfo="skip")
    fig.update_layout(showlegend=False, xaxis_visible=False, yaxis_visible=False,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ===============================
# -------- Standalone Run -------
# ===============================

if __name__ == "__main__":
    print("🧠 Running standalone Network Metrics Analysis...")

    INPUT = "enron_cleaned_final.csv"
    OUTPUT = "output/network_metrics.csv"

    os.makedirs("output", exist_ok=True)

    try:
        df = load_edges_csv(INPUT)
        df = filter_valid_emails(df)
        G, _ = build_comm_graph(df, directed=True, weight_by_recipient_type=False)
        metrics = compute_metrics(G, betweenness_sample_k=800, include_pagerank=True)
        metrics.to_csv(OUTPUT, index=False)
        print(f"✅ Saved network metrics to {OUTPUT}")
    except Exception as e:
        print(f"❌ Error: {e}")
