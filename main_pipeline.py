"""
Main Pipeline for Social Network Analysis System
(NLP + Feature + Network Metrics + Anomaly + Burnout)
"""
import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict
from pathlib import Path

# Ensure folders
os.makedirs('output', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Active modules
from nlp_topic_modeling import SimpleTopicModeler
from nlp_sentiment_analysis import SentimentAnalyzer
from nlp_semantic_similarity import SemanticSimilarityAnalyzer
from csv_exporter import CSVExporter
from feature_Analysis import run_feature_analysis
from network_metrics import load_edges_csv, filter_valid_emails, build_comm_graph, compute_metrics
from anomaly_detection import run_pipeline as run_anomaly_detection
from burnout_pipeline import build_burnout  # <-- NEW

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/pipeline.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class SocialNetworkAnalysisPipeline:
    def __init__(self, data_file: str = 'enron_cleaned_final.csv', max_records: int = 5000):
        self.data_file = data_file
        self.max_records = max_records
        self.results: Dict = {}
        self.execution_times: Dict = {}
        os.makedirs('output', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

    def run_complete_analysis(self) -> Dict:
        start_time = time.time()
        self._step_1_topic_modeling()
        self._step_2_sentiment_analysis()
        self._step_3_semantic_similarity()
        self._step_3b_feature_analysis()
        self._step_3c_network_metrics()
        self._step_3d_anomaly_detection()
        self._step_3e_burnout()           # <-- NEW
        self._step_4_export_csv()

        self.execution_times['total'] = time.time() - start_time
        self._generate_summary_report()
        logger.info(f"Pipeline completed successfully in {self.execution_times['total']:.2f} seconds")
        return self.results

    def _step_1_topic_modeling(self):
        logger.info("Step 1: Topic modeling")
        t0 = time.time()
        modeler = SimpleTopicModeler(self.data_file)
        modeler.load_documents(max_docs=self.max_records)
        topics = modeler.extract_topics_simple(num_topics=8)
        modeler.save_topics()
        self.results['topics'] = {'num_topics': len(topics), 'total_documents': len(modeler.documents)}
        self.execution_times['step_1_topics'] = time.time() - t0

    def _step_2_sentiment_analysis(self):
        logger.info("Step 2: Sentiment analysis")
        t0 = time.time()
        analyzer = SentimentAnalyzer(self.data_file)
        analyzer.load_documents(max_docs=self.max_records)
        results = analyzer.analyze_all_documents()
        analyzer.save_analysis(results)
        summary = analyzer.get_sentiment_summary(results)
        self.results['sentiment'] = {
            'total_documents': summary['total_documents'],
            'avg_sentiment': summary['avg_sentiment_score'],
            'avg_stress': summary['avg_stress_score'],
        }
        self.execution_times['step_2_sentiment'] = time.time() - t0

    def _step_3_semantic_similarity(self):
        logger.info("Step 3: Semantic similarity")
        t0 = time.time()
        analyzer = SemanticSimilarityAnalyzer(self.data_file)
        analyzer.load_documents(max_docs=self.max_records)
        semantic_net = analyzer.build_semantic_network(similarity_threshold=0.2)
        analyzer.save_semantic_analysis()
        self.results['semantic'] = {
            'num_nodes': semantic_net['stats'].get('num_nodes'),
            'num_edges': semantic_net['stats'].get('num_edges'),
            'avg_similarity': semantic_net['stats'].get('avg_similarity'),
        }
        self.execution_times['step_3_semantic'] = time.time() - t0

    def _step_3b_feature_analysis(self):
        logger.info("Step 3b: Feature analysis")
        t0 = time.time()
        run_feature_analysis(input_file=self.data_file)
        self.results["features"] = {"generated": os.path.exists("output/features.csv"), "file": "output/features.csv"}
        self.execution_times["step_3b_features"] = time.time() - t0

    def _step_3c_network_metrics(self):
        logger.info("Step 3c: Network metrics")
        t0 = time.time()
        df = load_edges_csv(self.data_file)
        df = filter_valid_emails(df)
        G, _ = build_comm_graph(df, directed=True)
        metrics = compute_metrics(G, betweenness_sample_k=800, include_pagerank=True)
        metrics.to_csv("output/network_metrics.csv", index=False)
        self.results["network_metrics"] = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "output": "output/network_metrics.csv",
        }
        self.execution_times["step_3c_network"] = time.time() - t0

    def _step_3d_anomaly_detection(self):
        logger.info("Step 3d: Anomaly detection")
        t0 = time.time()
        out_interim = Path("output/interim"); out_interim.mkdir(parents=True, exist_ok=True)
        out_processed = Path("output/processed"); out_processed.mkdir(parents=True, exist_ok=True)
        run_anomaly_detection(edges_path=self.data_file, out_interim=out_interim, out_processed=out_processed)
        self.results["anomalies"] = {"generated": (out_processed / "anomalies.csv").exists(),
                                     "file": "output/processed/anomalies.csv"}
        self.execution_times["step_3d_anomalies"] = time.time() - t0

    def _step_3e_burnout(self):
        logger.info("Step 3e: Burnout risk")
        t0 = time.time()
        df = build_burnout()                   # reads outputs from previous steps
        self.results["burnout"] = {"generated": True, "file": "output/processed/burnout_risk.csv",
                                   "rows": len(df)}
        self.execution_times["step_3e_burnout"] = time.time() - t0

    def _step_4_export_csv(self):
        logger.info("Step 4: CSV export (NLP CSVs)")
        t0 = time.time()
        CSVExporter().export_all_to_csv()
        csv_dir = os.path.join('output', 'csv')
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')] if os.path.exists(csv_dir) else []
        self.results['csv_export'] = {'exported': True, 'num_files': len(csv_files), 'directory': 'output/csv/'}
        self.execution_times['step_4_csv'] = time.time() - t0

    def _generate_summary_report(self):
        logger.info("Summary report")
        report = {
            'pipeline_info': {
                'data_file': self.data_file,
                'max_records': self.max_records,
                'execution_date': datetime.now().isoformat(),
                'total_execution_time': self.execution_times.get('total', 0)
            },
            'step_execution_times': self.execution_times,
            'analysis_results': self.results,
            'output_files': [
                'output/topics.json',
                'output/sentiment_analysis.json',
                'output/semantic_similarity.json',
                'output/features.csv',
                'output/network_metrics.csv',
                'output/processed/anomalies.csv',
                'output/processed/burnout_risk.csv',
            ],
            'csv_files': [
                'output/csv/user_sentiment.csv',
                'output/csv/document_sentiment.csv',
                'output/csv/topics.csv',
            ]
        }

        import json
        with open('output/pipeline_summary.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        text_summary = f"""
=== PIPELINE SUMMARY (NLP + Feature + Network + Anomaly + Burnout) ===
Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data File: {self.data_file}
Records Processed: {self.max_records}
Total Execution Time: {self.execution_times.get('total', 0):.2f} seconds

Outputs:
  - Features: output/features.csv
  - Network Metrics: output/network_metrics.csv
  - Anomalies: output/processed/anomalies.csv
  - Burnout Risk: output/processed/burnout_risk.csv
"""
        with open('output/pipeline_summary.txt', 'w') as f:
            f.write(text_summary)
        print(text_summary)


def main():
    print("🕸️ Social Network Analysis System — NLP + Feature + Network + Anomaly + Burnout")
    print("=" * 50)
    pipeline = SocialNetworkAnalysisPipeline(data_file='enron_cleaned_final.csv', max_records=5000)
    try:
        pipeline.run_complete_analysis()
        print("\n✅ Pipeline completed successfully!")
        print("📁 Outputs in 'output/' and 'output/processed/'")
        print("📈 CSVs available in 'output/csv/'")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
