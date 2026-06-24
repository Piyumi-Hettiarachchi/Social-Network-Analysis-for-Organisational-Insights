import pandas as pd # type: ignore
import numpy as np
import json
from pathlib import Path

# ---- Project paths ----
OUTPUT_DIR = Path("output")
CSV_DIR    = OUTPUT_DIR / "csv"
INTERIM_DIR = OUTPUT_DIR / "interim"
PROCESSED_DIR = OUTPUT_DIR / "processed"

# Inputs produced earlier in your pipeline
ANOMALIES           = PROCESSED_DIR / "anomalies.csv"              # anomaly_detection.py output
USER_SENTIMENT      = CSV_DIR / "user_sentiment.csv"               # csv_exporter.py output
DOCUMENT_SENTIMENT  = CSV_DIR / "document_sentiment.csv"           # csv_exporter.py output
COMMUNICATION_RISKS = CSV_DIR / "communication_risks.csv"          # (optional; may not exist)

# Output
OUTPUT = PROCESSED_DIR / "burnout_risk.csv"

def robust_minmax(x: pd.Series) -> pd.Series:
    if x.isna().all():
        return pd.Series(0.0, index=x.index)
    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    rng = max(hi - lo, 1e-6)
    return (x.clip(lower=lo, upper=hi) - lo) / rng

def first_non_null(a: pd.Series | None, b: pd.Series | None, default=0.0) -> pd.Series:
    if a is None and b is None:
        return pd.Series(default, index=[])
    if a is None:
        return b.fillna(default)
    if b is None:
        return a.fillna(default)
    out = a.copy()
    out[a.isna()] = b[a.isna()]
    return out.fillna(default)

def build_burnout() -> pd.DataFrame:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[burnout] Building burnout risk ...")

    # ---- Load base (weekly anomalies) ----
    anom = pd.read_csv(ANOMALIES, parse_dates=["week_start"])

    # ---- Load per-user sentiment/stress (strong source) ----
    usent = pd.read_csv(USER_SENTIMENT) if USER_SENTIMENT.exists() else pd.DataFrame()
    if not usent.empty and "user" not in usent.columns:
        raise ValueError("user_sentiment.csv must contain a 'user' column.")
    usent = usent[[c for c in ["user", "avg_sentiment", "avg_stress"] if c in usent.columns]]

    # ---- Fallback: aggregate doc-level sentiment/stress to user-level ----
    if DOCUMENT_SENTIMENT.exists():
        doc = pd.read_csv(DOCUMENT_SENTIMENT, usecols=["sender", "sentiment_score", "stress_score"])
        nlp_user = (
            doc.rename(columns={"sender": "user"})
               .groupby("user", as_index=False)
               .agg(sentiment_fallback=("sentiment_score", "mean"),
                    stress_fallback=("stress_score", "mean"))
        )
    else:
        nlp_user = pd.DataFrame(columns=["user", "sentiment_fallback", "stress_fallback"])

    # ---- Merge ----
    df = anom.merge(usent, on="user", how="left").merge(nlp_user, on="user", how="left")

    df["sentiment_score"] = first_non_null(df.get("avg_sentiment"), df.get("sentiment_fallback"))
    df["stress_score"]    = first_non_null(df.get("avg_stress"),    df.get("stress_fallback"))

    for col in ["sentiment_score", "stress_score", "if_score"]:
        if col in df.columns:
            med = df[col].median() if not df[col].dropna().empty else 0.0
            df[col] = df[col].fillna(med)
    df["sentiment_score"] += 1e-6
    df["stress_score"]    += 1e-6
    df["if_score"]        = df["if_score"].fillna(0.0) + 1e-6

    # Optional enrichment
    if COMMUNICATION_RISKS.exists():
        cr = pd.read_csv(COMMUNICATION_RISKS)
        if "node_user" in cr.columns and "user" not in cr.columns:
            cr = cr.rename(columns={"node_user": "user"})
        keep = ["user", "risk_type", "risk_level", "degree_centrality",
                "betweenness_centrality", "total_emails", "avg_stress",
                "avg_sentiment", "unique_contacts_count", "domain"]
        cr = cr[[c for c in keep if c in cr.columns]]
        df = df.merge(cr, on="user", how="left", suffixes=("", "_cr"))

    # ---- Normalise & score ----
    df["anomaly_norm"]   = robust_minmax(df["if_score"])
    df["stress_norm"]    = robust_minmax(df["stress_score"])
    sent_scaled          = robust_minmax(df["sentiment_score"])
    df["sentiment_norm"] = 1.0 - sent_scaled  # lower sentiment ⇒ higher risk

    if "betweenness_centrality" in df.columns:
        df["betw_norm"] = robust_minmax(pd.to_numeric(df["betweenness_centrality"], errors="coerce").fillna(0.0))
    else:
        df["betw_norm"] = 0.0

    df["burnout_risk"] = (
        0.45 * df["anomaly_norm"] +
        0.35 * df["stress_norm"] +
        0.18 * df["sentiment_norm"] +
        0.02 * df["betw_norm"]
    )

    df["risk_level"] = pd.cut(
        df["burnout_risk"], bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["Low", "Medium", "High"]
    )

    def mk_factors(row):
        return {
            "if_score": round(float(row["if_score"]), 4),
            "sentiment_score": round(float(row["sentiment_score"]), 4),
            "stress_score": round(float(row["stress_score"]), 4),
            "betweenness_centrality_norm": float(row["betw_norm"]),
            "weights": {"anomaly": 0.45, "stress": 0.35, "sentiment": 0.18, "centrality": 0.02},
            "risk_type": (row.get("risk_type") if pd.notna(row.get("risk_type")) else None),
            "risk_level_comm": (row.get("risk_level_cr") if pd.notna(row.get("risk_level_cr")) else None),
        }
    df["factors_json"] = df.apply(lambda r: json.dumps(mk_factors(r)), axis=1)

    def mk_reason(row):
        bits = [
            f"anomaly={row.if_score:.2f}",
            f"stress={row.stress_score:.2f}",
            f"sentiment={row.sentiment_score:.2f} (lower=worse)"
        ]
        if pd.notna(row.get("risk_type")):
            bits.append(f"comm_flag={row.get('risk_type')}")
        return " | ".join(bits)
    df["rationale"] = df.apply(mk_reason, axis=1)

    base_cols = ["user", "week_start", "if_score", "flag", "top_reasons",
                 "emails_sent", "emails_received", "unique_contacts", "ooh_ratio", "weekend_ratio"]
    comp_cols = ["sentiment_score", "stress_score",
                 "anomaly_norm", "sentiment_norm", "stress_norm", "betw_norm",
                 "burnout_risk", "risk_level", "factors_json", "rationale"]
    keep = [c for c in base_cols + comp_cols if c in df.columns]
    df = df[keep].copy()

    df.to_csv(OUTPUT, index=False)
    print(f"[burnout] ✓ Saved → {OUTPUT} (rows={len(df)})")
    return df

if __name__ == "__main__":
    build_burnout()
