"""
Topic Modeling Module for Social Network Analysis
Implements basic topic identification using simple algorithms
"""
#Extract lightweight topics from email text and attach representative documents and senders.
import csv
import re
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import logging
import math

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTopicModeler:
    """
    Simple topic modeling implementation using TF-IDF and clustering
    """
    
    def __init__(self, data_file: str = 'enron_cleaned_final.csv'):
        self.data_file = data_file
        self.documents = []
        self.vocabulary = set()
        self.tf_idf_matrix = {}
        self.topics = {}
        self.stop_words = self._get_stop_words()
        
    def _get_stop_words(self) -> Set[str]:
        """Get common English stop words"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'can', 'shall', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
            'its', 'our', 'their', 'this', 'that', 'these', 'those', 'what', 'which',
            'who', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'now', 'here', 'there', 'then'
        }
    
    def load_documents(self, max_docs: int = 5000) -> List[Dict]:
        """Load email documents for topic modeling"""
        logger.info(f"Loading documents from {self.data_file}")
        
        self.documents = []
        
        with open(self.data_file, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            for i, row in enumerate(reader):
                if i >= max_docs:
                    break
                    
                if len(row) >= 7:
                    doc = {
                        'id': i,
                        'message_id': row[0],
                        'date': row[1],
                        'sender': row[2].lower().strip(),
                        'recipient': row[3].lower().strip(),
                        'subject': row[5].strip(),
                        'body': row[6].strip(),
                        'text': f"{row[5]} {row[6]}".strip()  # Combined subject and body
                    }
                    
                    if doc['text'] and len(doc['text']) > 20:  # Filter very short texts
                        self.documents.append(doc)
        
        logger.info(f"Loaded {len(self.documents)} documents")
        return self.documents
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for topic modeling"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stop words
        words = text.split()
        words = [word for word in words if len(word) > 2 and word not in self.stop_words]
        
        return words
    
    def build_vocabulary(self):
        """Build vocabulary from all documents"""
        logger.info("Building vocabulary")
        
        self.vocabulary = set()
        for doc in self.documents:
            words = self.preprocess_text(doc['text'])
            self.vocabulary.update(words)
        
        # Filter vocabulary (keep words that appear in at least 2 documents)
        word_doc_count = defaultdict(int)
        for doc in self.documents:
            words = set(self.preprocess_text(doc['text']))
            for word in words:
                word_doc_count[word] += 1
        
        self.vocabulary = {word for word, count in word_doc_count.items() if count >= 2}
        logger.info(f"Vocabulary size: {len(self.vocabulary)}")
    
    def calculate_tf_idf(self):
        """Calculate TF-IDF matrix"""
        logger.info("Calculating TF-IDF matrix")
        
        if not self.vocabulary:
            self.build_vocabulary()
        
        # Calculate document frequency for each word
        df = defaultdict(int)
        for doc in self.documents:
            words = set(self.preprocess_text(doc['text']))
            for word in words:
                if word in self.vocabulary:
                    df[word] += 1
        
        # Calculate TF-IDF for each document
        self.tf_idf_matrix = {}
        total_docs = len(self.documents)
        
        for i, doc in enumerate(self.documents):
            words = self.preprocess_text(doc['text'])
            word_count = Counter(words)
            total_words = len(words)
            
            self.tf_idf_matrix[i] = {}
            
            for word in self.vocabulary:
                # Term frequency
                tf = word_count[word] / total_words if total_words > 0 else 0
                
                # Inverse document frequency
                idf = math.log(total_docs / df[word]) if df[word] > 0 else 0
                
                # TF-IDF
                self.tf_idf_matrix[i][word] = tf * idf
    
    def extract_topics_simple(self, num_topics: int = 10, words_per_topic: int = 10) -> Dict:
        """Extract topics using simple clustering approach"""
        logger.info(f"Extracting {num_topics} topics")
        
        if not self.tf_idf_matrix:
            self.calculate_tf_idf()
        
        # Find most important words across all documents
        word_importance = defaultdict(float)
        for doc_id, word_scores in self.tf_idf_matrix.items():
            for word, score in word_scores.items():
                word_importance[word] += score
        
        # Sort words by importance
        sorted_words = sorted(word_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Create topics by grouping related words
        topics = {}
        words_used = set()
        
        for topic_id in range(num_topics):
            topic_words = []
            
            # Start with the most important unused word
            for word, score in sorted_words:
                if word not in words_used and len(topic_words) < words_per_topic:
                    topic_words.append((word, score))
                    words_used.add(word)
            
            if topic_words:
                topics[f"topic_{topic_id}"] = {
                    'words': topic_words,
                    'top_words': [word for word, _ in topic_words[:5]],
                    'documents': []
                }
        
        # Assign documents to topics
        for doc_id, doc in enumerate(self.documents):
            if doc_id in self.tf_idf_matrix:
                doc_scores = {}
                
                for topic_id, topic_info in topics.items():
                    score = 0
                    for word, _ in topic_info['words']:
                        score += self.tf_idf_matrix[doc_id].get(word, 0)
                    doc_scores[topic_id] = score
                
                # Assign to best topic
                if doc_scores:
                    best_topic = max(doc_scores, key=doc_scores.get)
                    topics[best_topic]['documents'].append({
                        'doc_id': doc_id,
                        'sender': doc['sender'],
                        'subject': doc['subject'][:100],
                        'score': doc_scores[best_topic]
                    })
        
        self.topics = topics
        logger.info("Topic extraction completed")
        return topics
    
    def get_user_topics(self) -> Dict[str, Dict]:
        """Get dominant topics for each user"""
        if not self.topics:
            self.extract_topics_simple()
        
        user_topics = defaultdict(lambda: defaultdict(int))
        
        # Count topic assignments per user
        for topic_id, topic_info in self.topics.items():
            for doc_info in topic_info['documents']:
                sender = doc_info['sender']
                user_topics[sender][topic_id] += 1
        
        # Find dominant topic for each user
        user_dominant_topics = {}
        for user, topic_counts in user_topics.items():
            if topic_counts:
                dominant_topic = max(topic_counts, key=topic_counts.get)
                total_emails = sum(topic_counts.values())
                
                user_dominant_topics[user] = {
                    'dominant_topic': dominant_topic,
                    'topic_distribution': dict(topic_counts),
                    'total_emails': total_emails,
                    'dominance_ratio': topic_counts[dominant_topic] / total_emails
                }
        
        return user_dominant_topics
    
    def get_topic_summary(self) -> Dict:
        """Get summary of all topics"""
        if not self.topics:
            return {}
        
        summary = {
            'num_topics': len(self.topics),
            'total_documents': len(self.documents),
            'topics': {}
        }
        
        for topic_id, topic_info in self.topics.items():
            summary['topics'][topic_id] = {
                'top_words': topic_info['top_words'],
                'num_documents': len(topic_info['documents']),
                'top_senders': self._get_top_senders_for_topic(topic_info['documents'])
            }
        
        return summary
    
    def _get_top_senders_for_topic(self, documents: List[Dict], top_n: int = 5) -> List[Tuple[str, int]]:
        """Get top senders for a specific topic"""
        sender_counts = Counter([doc['sender'] for doc in documents])
        return sender_counts.most_common(top_n)
    
    def save_topics(self, filename: str = 'topics.json'):
        """Save topics to JSON file"""
        import os
        os.makedirs('output', exist_ok=True)
        
        output_data = {
            'topics': self.topics,
            'user_topics': self.get_user_topics(),
            'summary': self.get_topic_summary()
        }
        
        output_path = f"output/{filename}"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Topics saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    modeler = SimpleTopicModeler()
    
    # Load documents
    docs = modeler.load_documents(max_docs=2000)
    
    # Extract topics
    topics = modeler.extract_topics_simple(num_topics=8)
    
    # Print topic summary
    summary = modeler.get_topic_summary()
    print("=== TOPIC MODELING RESULTS ===")
    print(f"Number of topics: {summary['num_topics']}")
    print(f"Total documents: {summary['total_documents']}")
    
    for topic_id, topic_info in summary['topics'].items():
        print(f"\n{topic_id.upper()}:")
        print(f"  Top words: {', '.join(topic_info['top_words'])}")
        print(f"  Documents: {topic_info['num_documents']}")
        print(f"  Top senders: {topic_info['top_senders'][:3]}")
    
    # Get user topics
    user_topics = modeler.get_user_topics()
    print(f"\n=== USER TOPIC ANALYSIS ===")
    print(f"Users analyzed: {len(user_topics)}")
    
    # Show top users by email volume
    top_users = sorted(user_topics.items(), key=lambda x: x[1]['total_emails'], reverse=True)[:5]
    for user, info in top_users:
        print(f"{user}: {info['total_emails']} emails, dominant topic: {info['dominant_topic']}")
    
    # Save results
    modeler.save_topics()
