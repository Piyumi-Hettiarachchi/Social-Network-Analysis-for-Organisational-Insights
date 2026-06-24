"""
Feature Analysis Module
Computes weekly communication and network metrics from enron_cleaned_final.csv.
Exports results to output/features.csv.
"""

import os
import pandas as pd # type: ignore
import numpy as np
import networkx as nx
from datetime import datetime

INPUT = "enron_cleaned_final.csv"
OUTPUT_DIR = "output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "features.csv")


def run_feature_analysis(input_file: str = INPUT, output_file: str = OUTPUT_FILE):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(input_file)

    # --- Auto-map columns ---
    def auto_map(colnames, candidates):
        for cand in candidates:
            for col in colnames:
                if cand == col.lower():
                    return col
        for cand in candidates:
            for col in colnames:
                if cand in col.lower():
                    return col
        return None

    sender_candidates = ["sender_id", "from", "email_from", "from_email", "sender"]
    receiver_candidates = ["receiver_id", "to", "email_to", "to_email", "recipient"]
    timestamp_candidates = ["ts_utc", "date", "timestamp", "datetime", "sent"]

    sender_col = auto_map(df.columns, sender_candidates)
    receiver_col = auto_map(df.columns, receiver_candidates)
    timestamp_col = auto_map(df.columns, timestamp_candidates)

    if not sender_col or not receiver_col or not timestamp_col:
        raise ValueError(
            f"Missing columns. Found: sender={sender_col}, receiver={receiver_col}, timestamp={timestamp_col}"
        )

    df = df.rename(
        columns={
            sender_col: "sender_id",
            receiver_col: "receiver_id",
            timestamp_col: "ts_utc",
        }
    )

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    df["week_start"] = df["ts_utc"].dt.to_period("W-MON").dt.start_time

    # --- Helper functions ---
    def is_after_hours(dt):
        hour = dt.hour
        weekday = dt.weekday()
        return (hour < 9 or hour > 18) or (weekday >= 5)

    def burstiness(times):
        if len(times) < 3:
            return 0.0
        t = np.sort(times.view(np.int64) / 1e9 / 60.0)
        diffs = np.diff(t)
        return float(np.std(diffs) / (np.mean(diffs) + 1e-9)) if np.mean(diffs) > 0 else 0.0

    # --- Weekly features ---
    features = []
    for (emp, week), group in df.groupby(["sender_id", "week_start"]):
        msgs_sent = len(group)
        uniq_contacts = group["receiver_id"].nunique()
        pct_after = np.mean([is_after_hours(t) for t in group["ts_utc"]]) if msgs_sent > 0 else 0.0
        burst = burstiness(group["ts_utc"])
        features.append({
            "employee_id": emp,
            "week_start": week,
            "msgs_sent": msgs_sent,
            "uniq_contacts": uniq_contacts,
            "pct_after_hours": pct_after,
            "burstiness": burst,
        })
    feat_df = pd.DataFrame(features)

    # --- Received counts ---
    recv = (
        df.groupby(["receiver_id", "week_start"])["ts_utc"]
        .count()
        .reset_index()
        .rename(columns={"receiver_id": "employee_id", "ts_utc": "msgs_recv"})
    )
    feat_df = feat_df.merge(recv, on=["employee_id", "week_start"], how="left").fillna({"msgs_recv": 0})

    # --- Network metrics ---
    graph_rows = []
    for week, week_df in df.groupby("week_start"):
        G = nx.DiGraph()
        for _, r in week_df.iterrows():
            G.add_edge(r["sender_id"], r["receiver_id"])
        nodes = list(G.nodes())
        degree_in = dict(G.in_degree())
        degree_out = dict(G.out_degree())
        try:
            betweenness = nx.betweenness_centrality(G)
        except Exception:
            betweenness = {n: 0.0 for n in nodes}
        try:
            eigenvector = nx.eigenvector_centrality_numpy(G)
        except Exception:
            eigenvector = {n: 0.0 for n in nodes}
        try:
            clustering = nx.clustering(G.to_undirected())
        except Exception:
            clustering = {n: 0.0 for n in nodes}
        for n in nodes:
            graph_rows.append({
                "employee_id": n,
                "week_start": week,
                "degree_in": degree_in.get(n, 0),
                "degree_out": degree_out.get(n, 0),
                "betweenness": betweenness.get(n, 0.0),
                "eigenvector": eigenvector.get(n, 0.0),
                "clustering_coeff": clustering.get(n, 0.0),
            })

    graph_df = pd.DataFrame(graph_rows)
    feat_df = feat_df.merge(graph_df, on=["employee_id", "week_start"], how="left").fillna(0)

    # --- Export ---
    feat_df.to_csv(output_file, index=False)
    print(f"✅ Features exported to {output_file}")


if __name__ == "__main__":
    run_feature_analysis()
