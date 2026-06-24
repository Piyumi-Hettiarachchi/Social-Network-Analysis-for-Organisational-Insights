🕸️ Social Network Analysis System — Main Pipeline

====================================================

📘 Overview

The Social Network Analysis (SNA) System is a complete analytical pipeline integrating NLP, Feature Engineering, Network Metrics, Anomaly Detection, and Burnout Risk Analysis.
It processes organizational communication datasets (like Enron emails) to produce insights on collaboration patterns, sentiment, and potential employee burnout.

====================================================

⚙️ Core Components
Step	Module	Description	Output
1	nlp_topic_modeling.py	Extracts latent topics from email content.	output/topics.json
2	nlp_sentiment_analysis.py	Computes sentiment and stress levels using NLP models.	output/sentiment_analysis.json, output/csv/user_sentiment.csv
3	nlp_semantic_similarity.py	Builds semantic similarity networks.	output/semantic_similarity.json
3b	feature_Analysis.py	Generates behavioral and structural features.	output/features.csv
3c	network_metrics.py	Computes communication graph metrics.	output/network_metrics.csv
3d	anomaly_detection.py	Detects abnormal user behaviors using Isolation Forest.	output/processed/anomalies.csv
3e	burnout_pipeline.py	Aggregates anomaly, sentiment, and network data to estimate burnout risk.	output/processed/burnout_risk.csv
4	csv_exporter.py	Exports all analysis results into CSVs for dashboards.	output/csv/

====================================================

🧩 Directory Structure

Social_Network_Analysis/
│
├── main_pipeline.py
├── nlp_topic_modeling.py
├── nlp_sentiment_analysis.py
├── nlp_semantic_similarity.py
├── feature_Analysis.py
├── network_metrics.py
├── anomaly_detection.py
├── burnout_pipeline.py
├── csv_exporter.py
│
├── output/
│ ├── csv/
│ ├── interim/
│ ├── processed/
│ ├── features.csv
│ ├── network_metrics.csv
│ ├── processed/anomalies.csv
│ ├── processed/burnout_risk.csv
│ └── pipeline_summary.json / .txt
│
└── logs/
└── pipeline.log

====================================================

🧰 Requirements

Activate your virtual environment and install dependencies:

pip install -r requirements.txt


Key Libraries:

pandas, numpy

networkx, matplotlib

nltk, scikit-learn

transformers, vaderSentiment

gensim, spacy

openpyxl, seaborn

====================================================

🚀 Running the Pipeline

Activate the virtual environment:
.venv\Scripts\activate

Run the main pipeline:
python main_pipeline.py

The pipeline will automatically:

Process “enron_cleaned_final.csv”

Run all analytical steps

Save outputs to the /output/ directory

====================================================

📊 Output Files
File	Description
output/features.csv	Extracted behavioral features
output/network_metrics.csv	Graph-level metrics
output/processed/anomalies.csv	Detected anomalies
output/processed/burnout_risk.csv	Weekly burnout risk levels
output/pipeline_summary.json	Structured report
output/pipeline_summary.txt	Human-readable summary

====================================================

🧠 Burnout Risk Scoring Formula

burnout_risk = 0.45 * anomaly_norm
+ 0.35 * stress_norm
+ 0.18 * (1 - sentiment_norm)
+ 0.02 * centrality_norm

Each user-week is classified as Low, Medium, or High risk.

====================================================

🧾 Logging

All logs are saved in:
logs/pipeline.log

Any pipeline errors are timestamped with detailed traces.

====================================================

🧪 Testing Individual Modules

Each component can run independently:
python anomaly_detection.py
python burnout_pipeline.py

====================================================

🧩 Integration Order

NLP → 2. Feature Analysis → 3. Network Metrics →

Anomaly Detection → 5. Burnout Risk

Burnout analysis depends on:

output/processed/anomalies.csv

output/csv/user_sentiment.csv

====================================================

🏁 Summary

The main pipeline acts as a single automation layer combining:

NLP topic and sentiment modeling

Communication network analytics

ML-based anomaly detection

Early burnout risk prediction

Designed for organizational insight, HR analytics, and behavioral monitoring based on email communication data.

====================================================