import json
import csv
import os
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVExporter:
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = output_dir
        self.csv_dir = os.path.join(output_dir, 'csv')
        os.makedirs(self.csv_dir, exist_ok=True)

    def export_all_to_csv(self, clean: bool = True):
        if clean and os.path.exists(self.csv_dir):
            for f in os.listdir(self.csv_dir):
                if f.endswith(".csv"):
                    try:
                        os.remove(os.path.join(self.csv_dir, f))
                    except Exception:
                        pass
        self.export_topic_data()
        self.export_sentiment_data()
        self.export_semantic_data()
        logger.info(f"CSV export complete. Files in: {self.csv_dir}")

    def export_topic_data(self):
        try:
            path = os.path.join(self.output_dir, 'topics.json')
            if not os.path.exists(path):
                return
            with open(path, 'r', encoding='utf-8') as f:
                topic_data = json.load(f)
            topics_list: List[Dict[str, Any]] = []
            topics = topic_data.get('topics', {})
            for topic_id, topic_info in topics.items():
                top_words = ', '.join([w for w, _ in topic_info.get('words', [])[:10]])
                topics_list.append({
                    'topic_id': topic_id,
                    'top_words': top_words,
                    'num_documents': len(topic_info.get('documents', [])),
                    'top_senders': ', '.join([doc.get('sender', '') for doc in topic_info.get('documents', [])[:5]])
                })
            self._write_csv('topics.csv', topics_list)
            user_topics = topic_data.get('user_topics', {})
            user_topic_list: List[Dict[str, Any]] = []
            for user, info in user_topics.items():
                user_topic_list.append({
                    'user': user,
                    'dominant_topic': info.get('dominant_topic', ''),
                    'total_emails': info.get('total_emails', 0),
                    'dominance_ratio': info.get('dominance_ratio', 0)
                })
            self._write_csv('user_topics.csv', user_topic_list)
        except Exception as e:
            logger.error(f"Error exporting topic data: {e}")

    def export_sentiment_data(self):
        try:
            path = os.path.join(self.output_dir, 'sentiment_analysis.json')
            if not os.path.exists(path):
                return
            with open(path, 'r', encoding='utf-8') as f:
                sentiment_data = json.load(f)
            user_analysis = sentiment_data.get('user_analysis', {})
            user_rows: List[Dict[str, Any]] = []
            for user, data in user_analysis.items():
                user_rows.append({
                    'user': user,
                    'avg_sentiment': data.get('avg_sentiment', 0),
                    'avg_stress': data.get('avg_stress', 0),
                    'total_emails': data.get('total_emails', 0),
                    'positive_emails': data.get('sentiment_distribution', {}).get('positive', 0),
                    'negative_emails': data.get('sentiment_distribution', {}).get('negative', 0),
                    'neutral_emails': data.get('sentiment_distribution', {}).get('neutral', 0),
                    'high_stress_emails': data.get('stress_distribution', {}).get('high', 0),
                    'medium_stress_emails': data.get('stress_distribution', {}).get('medium', 0),
                    'low_stress_emails': data.get('stress_distribution', {}).get('low', 0),
                    'no_stress_emails': data.get('stress_distribution', {}).get('none', 0),
                })
            self._write_csv('user_sentiment.csv', user_rows)
            docs = sentiment_data.get('document_analysis', [])
            doc_rows: List[Dict[str, Any]] = []
            for doc in docs[:1000]:
                doc_rows.append({
                    'doc_id': doc.get('doc_id', ''),
                    'sender': doc.get('sender', ''),
                    'recipient': doc.get('recipient', ''),
                    'date': doc.get('date', ''),
                    'subject': doc.get('subject', ''),
                    'sentiment_score': doc.get('sentiment', {}).get('sentiment_score', 0),
                    'sentiment_label': doc.get('sentiment', {}).get('sentiment_label', ''),
                    'stress_score': doc.get('stress', {}).get('stress_score', 0),
                    'stress_level': doc.get('stress', {}).get('stress_level', ''),
                })
            self._write_csv('document_sentiment.csv', doc_rows)
        except Exception as e:
            logger.error(f"Error exporting sentiment data: {e}")

    def export_semantic_data(self):
        try:
            path = os.path.join(self.output_dir, 'semantic_similarity.json')
            if not os.path.exists(path):
                return
            with open(path, 'r', encoding='utf-8') as f:
                semantic_data = json.load(f)
            semantic_network = semantic_data.get('semantic_network', {})
            edges = semantic_network.get('edges', {})
            edge_rows: List[Dict[str, Any]] = []
            for edge_str, sim in edges.items():
                if '<->' in edge_str:
                    u1, u2 = edge_str.split('<->')
                    edge_rows.append({'user1': u1, 'user2': u2, 'similarity': sim})
            self._write_csv('semantic_edges.csv', edge_rows)
            communities = semantic_data.get('communities', {})
            comm_rows: List[Dict[str, Any]] = []
            for cid, members in communities.items():
                for m in members:
                    comm_rows.append({'community_id': cid, 'member': m, 'community_size': len(members)})
            self._write_csv('semantic_communities.csv', comm_rows)
            pairs = semantic_data.get('top_similar_pairs', [])
            pair_rows: List[Dict[str, Any]] = []
            for p in pairs:
                if len(p) >= 3:
                    pair_rows.append({'user1': p[0], 'user2': p[1], 'similarity': p[2]})
            self._write_csv('top_similar_pairs.csv', pair_rows)
        except Exception as e:
            logger.error(f"Error exporting semantic data: {e}")

    def _write_csv(self, filename: str, data: List[Dict[str, Any]]):
        if not data:
            return
        filepath = os.path.join(self.csv_dir, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)

def main():
    exporter = CSVExporter()
    exporter.export_all_to_csv()
    print("✅ CSV export completed")

if __name__ == "__main__":
    main()
