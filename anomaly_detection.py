"""
Anomaly Detection (weekly user behavior)
- Reads edges CSV/XLSX (sender, recipient, timestamp)
- Builds user-week features
- Computes per-user rolling z-scores + IsolationForest
- Writes anomalies to output/processed/anomalies.csv
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd # type: ignore
from sklearn.ensemble import IsolationForest # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore

# Defaults (relative to project)
DEFAULT_EDGES_PATH = "enron_cleaned_final.csv"
OUT_INTERIM_DIR = Path("output/interim")
OUT_PROCESSED_DIR = Path("output/processed")

WEEK_FREQ = "W-MON"
MIN_EMAILS_GUARD = 5
ROLL_WEEKS = 8
IF_CONTAMINATION = 0.03
IF_ESTIMATORS = 300
RANDOM_STATE = 42

MODEL_FEATURES = [
    "emails_sent",
    "emails_received",
    "unique_contacts",
    "ooh_ratio",
    "weekend_ratio",
]


def load_edges(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Edges file not found: {p.resolve()}")

    df = pd.read_excel(p) if p.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(p)

    candidates = {
        "timestamp": ["timestamp", "date", "datetime", "time", "sent_at", "ts_utc"],
        "sender_email": ["sender_email", "from", "from_email", "sender"],
        "recipient_email": ["recipient_email", "to", "to_email", "recipient", "receiver"],
        "subject": ["subject", "subj"],
        "body": ["body", "text", "content"],
    }

    def pick(name_list):
        lower = {c.lower(): c for c in df.columns}
        for n in name_list:
            if n in lower:
                return lower[n]
        return None

    ts_col = pick(candidates["timestamp"])
    s_col = pick(candidates["sender_email"])
    r_col = pick(candidates["recipient_email"])
    subj = pick(candidates["subject"])
    body = pick(candidates["body"])

    missing = [lab for lab, col in [("timestamp", ts_col), ("sender_email", s_col), ("recipient_email", r_col)] if col is None]
    if missing:
        raise ValueError(f"Required columns not found: {missing}. Found: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df[ts_col], errors="coerce", utc=True),
            "sender_email": df[s_col].astype(str).str.lower().str.strip(),
            "recipient_email": df[r_col].astype(str).str.lower().str.strip(),
        }
    )
    if subj is not None:
        out["subject"] = df[subj].astype(str)
    if body is not None:
        out["body"] = df[body].astype(str)

    out = out.dropna(subset=["timestamp", "sender_email", "recipient_email"])
    return out


def build_user_week_features(edges: pd.DataFrame) -> pd.DataFrame:
    df = edges.copy()
    df["week_start"] = df["timestamp"].dt.to_period(WEEK_FREQ).dt.start_time

    sent = (
        df.groupby(["sender_email", "week_start"])
        .size()
        .rename("emails_sent")
        .reset_index()
        .rename(columns={"sender_email": "user"})
    )

    recv = (
        df.groupby(["recipient_email", "week_start"])
        .size()
        .rename("emails_received")
        .reset_index()
        .rename(columns={"recipient_email": "user"})
    )

    a = df[["sender_email", "recipient_email", "week_start"]].rename(columns={"sender_email": "user", "recipient_email": "contact"})
    b = df[["recipient_email", "sender_email", "week_start"]].rename(columns={"recipient_email": "user", "sender_email": "contact"})
    uniq = (
        pd.concat([a, b], ignore_index=True)
        .groupby(["user", "week_start"])["contact"]
        .nunique()
        .rename("unique_contacts")
        .reset_index()
    )

    s = df[["sender_email", "timestamp", "week_start"]].rename(columns={"sender_email": "user"})
    s["weekday"] = s["timestamp"].dt.weekday
    s["hour"] = s["timestamp"].dt.hour
    s["is_working_hour"] = ((s["weekday"] < 5) & (s["hour"] >= 9) & (s["hour"] < 18)).astype(int)
    s["is_weekend"] = (s["weekday"] >= 5).astype(int)
    agg = (
        s.groupby(["user", "week_start"])
        .agg(total=("timestamp", "count"), wh=("is_working_hour", "sum"), wknd=("is_weekend", "sum"))
        .reset_index()
    )
    agg["ooh_ratio"] = np.where(agg["total"] > 0, (agg["total"] - agg["wh"]) / agg["total"], 0.0)
    agg["weekend_ratio"] = np.where(agg["total"] > 0, agg["wknd"] / agg["total"], 0.0)
    time_feats = agg.rename(columns={"total": "sent_events"})[["user", "week_start", "ooh_ratio", "weekend_ratio", "sent_events"]]

    feat = (
        sent.merge(recv, on=["user", "week_start"], how="outer")
        .merge(uniq, on=["user", "week_start"], how="outer")
        .merge(time_feats, on=["user", "week_start"], how="outer")
    )
    for c in ["emails_sent", "emails_received", "unique_contacts", "ooh_ratio", "weekend_ratio", "sent_events"]:
        if c in feat.columns:
            feat[c] = feat[c].fillna(0)

    feat["emails_sent"] = feat["emails_sent"].astype(int)
    feat["emails_received"] = feat["emails_received"].astype(int)
    feat["unique_contacts"] = feat["unique_contacts"].astype(int)
    return feat


def add_personal_zscores(user_week: pd.DataFrame, cols=("emails_sent", "unique_contacts", "ooh_ratio"), roll: int = ROLL_WEEKS) -> pd.DataFrame:
    df = user_week.sort_values(["user", "week_start"]).copy()

    def roll_mean(s: pd.Series) -> pd.Series:
        return s.shift().rolling(roll, min_periods=max(2, roll // 2)).mean()

    def roll_std(s: pd.Series) -> pd.Series:
        return s.shift().rolling(roll, min_periods=max(2, roll // 2)).std()

    for col in cols:
        if col not in df.columns:
            continue
        mean_ = df.groupby("user")[col].transform(roll_mean)
        std_ = df.groupby("user")[col].transform(roll_std)
        df[f"z_{col}"] = (df[col] - mean_) / std_.replace(0, np.nan)
    return df


def isolation_forest_scores(df: pd.DataFrame, feature_cols) -> pd.Series:
    X = df[feature_cols].fillna(0.0).values
    X = StandardScaler().fit_transform(X)
    iso = IsolationForest(
        n_estimators=IF_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X)
    return -iso.score_samples(X)  # higher = more anomalous


def compute_top_reasons(df: pd.DataFrame, z_cols):
    def mk(row):
        pairs = []
        for c in z_cols:
            v = row.get(c)
            if pd.notnull(v):
                pairs.append((c.replace("z_", ""), abs(float(v))))
        if not pairs:
            return ""
        pairs.sort(key=lambda x: x[1], reverse=True)
        return "; ".join([f"{name}={row['z_'+name]:+.2f}σ" for name, _ in pairs[:3]])

    return df.apply(mk, axis=1)


def run_pipeline(edges_path: str, out_interim: Path = OUT_INTERIM_DIR, out_processed: Path = OUT_PROCESSED_DIR):
    out_interim.mkdir(parents=True, exist_ok=True)
    out_processed.mkdir(parents=True, exist_ok=True)

    edges = load_edges(edges_path)

    uw = build_user_week_features(edges)
    uw = uw[uw["emails_sent"] >= MIN_EMAILS_GUARD].copy()

    uw = add_personal_zscores(uw, cols=["emails_sent", "unique_contacts", "ooh_ratio"], roll=ROLL_WEEKS)

    feat_path = out_interim / "user_features_week.csv"
    uw.to_csv(feat_path, index=False)

    features = [c for c in MODEL_FEATURES if c in uw.columns]
    if not features:
        raise ValueError("No model features found for IsolationForest.")

    uw["if_score"] = isolation_forest_scores(uw, features)

    uw["univar_count"] = (
        (uw.get("z_emails_sent").abs() >= 3).astype(int).fillna(0)
        + (uw.get("z_unique_contacts").abs() >= 3).astype(int).fillna(0)
        + (uw.get("z_ooh_ratio").abs() >= 3).astype(int).fillna(0)
    )

    if_thresh = uw["if_score"].quantile(0.97) if len(uw) > 20 else uw["if_score"].quantile(0.95)
    uw["flag"] = (uw["if_score"] >= if_thresh) | (uw["univar_count"] >= 2)

    z_cols = [c for c in uw.columns if c.startswith("z_")]
    uw["top_reasons"] = compute_top_reasons(uw, z_cols)

    keep = [
        "user",
        "week_start",
        "if_score",
        "flag",
        "top_reasons",
        "emails_sent",
        "emails_received",
        "unique_contacts",
        "ooh_ratio",
        "weekend_ratio",
    ]
    keep = [c for c in keep if c in uw.columns]
    anomalies = uw.loc[uw["flag"], keep].sort_values(["week_start", "if_score"], ascending=[True, False])

    out_path = out_processed / "anomalies.csv"
    anomalies.to_csv(out_path, index=False)
    return out_path


def parse_args():
    ap = argparse.ArgumentParser(description="Anomaly Detection Pipeline")
    ap.add_argument("--edges", type=str, default=DEFAULT_EDGES_PATH, help="Path to edges file (.csv or .xlsx)")
    ap.add_argument("--interim_dir", type=str, default=str(OUT_INTERIM_DIR))
    ap.add_argument("--processed_dir", type=str, default=str(OUT_PROCESSED_DIR))
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = run_pipeline(args.edges, Path(args.interim_dir), Path(args.processed_dir))
    print(f"✅ Anomalies → {path}")
