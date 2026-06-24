"""
Semantic Similarity Networks Module
Implements simple semantic similarity using TF-IDF and cosine similarity
"""
#Build a content-based network of people whose email text (subject+body) is similar. Useful to spot topic-aligned clusters and natural knowledge brokers
import csv
import re
import json
import math
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSimilarityAnalyzer:
    """
    Semantic similarity analysis using TF-IDF and cosine similarity
    """
    
    def __init__(self, data_file: str = 'enron_cleaned_final.csv'):
        self.data_file = data_file
        self.documents = []
        self.user_profiles = {}
        self.vocabulary = set()
        self.tf_idf_matrix = {}
        self.similarity_matrix = {}
        self.semantic_network = {}
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
    
    def load_documents(self, max_docs: int = 3000) -> List[Dict]:
        """Load email documents"""
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
                        'text': f"{row[5]} {row[6]}".strip()
                    }
                    
                    if doc['text'] and len(doc['text']) > 20:
                        self.documents.append(doc)
        
        logger.info(f"Loaded {len(self.documents)} documents")
        return self.documents
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for semantic analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove email addresses and URLs
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize and remove stop words
        words = text.split()
        words = [word for word in words if len(word) > 2 and word not in self.stop_words]
        
        return words
    
    def build_user_profiles(self):
        """Build semantic profiles for each user based on their communications"""
        logger.info("Building user profiles")
        
        if not self.documents:
            self.load_documents()
        
        # Aggregate text by user
        user_texts = defaultdict(list)
        for doc in self.documents:
            user_texts[doc['sender']].append(doc['text'])
        
        # Create combined text profile for each user
        self.user_profiles = {}
        for user, texts in user_texts.items():
            combined_text = ' '.join(texts)
            self.user_profiles[user] = {
                'combined_text': combined_text,
                'email_count': len(texts),
                'words': self.preprocess_text(combined_text)
            }
        
        logger.info(f"Built profiles for {len(self.user_profiles)} users")
    
    def build_vocabulary(self):
        """Build vocabulary from all user profiles"""
        logger.info("Building vocabulary")
        
        if not self.user_profiles:
            self.build_user_profiles()
        
        self.vocabulary = set()
        for user, profile in self.user_profiles.items():
            self.vocabulary.update(profile['words'])
        
        # Filter vocabulary (keep words that appear for at least 2 users)
        word_user_count = defaultdict(int)
        for user, profile in self.user_profiles.items():
            unique_words = set(profile['words'])
            for word in unique_words:
                word_user_count[word] += 1
        
        self.vocabulary = {word for word, count in word_user_count.items() if count >= 2}
        logger.info(f"Vocabulary size: {len(self.vocabulary)}")
    
    def calculate_tf_idf_profiles(self):
        """Calculate TF-IDF vectors for user profiles"""
        logger.info("Calculating TF-IDF profiles")
        
        if not self.vocabulary:
            self.build_vocabulary()
        
        # Calculate document frequency for each word
        df = defaultdict(int)
        for user, profile in self.user_profiles.items():
            unique_words = set(profile['words'])
            for word in unique_words:
                if word in self.vocabulary:
                    df[word] += 1
        
        # Calculate TF-IDF for each user
        self.tf_idf_matrix = {}
        total_users = len(self.user_profiles)
        
        for user, profile in self.user_profiles.items():
            word_count = Counter(profile['words'])
            total_words = len(profile['words'])
            
            self.tf_idf_matrix[user] = {}
            
            for word in self.vocabulary:
                # Term frequency
                tf = word_count[word] / total_words if total_words > 0 else 0
                
                # Inverse document frequency
                idf = math.log(total_users / df[word]) if df[word] > 0 else 0
                
                # TF-IDF
                self.tf_idf_matrix[user][word] = tf * idf
    
    def calculate_cosine_similarity(self, user1: str, user2: str) -> float:
        """Calculate cosine similarity between two users"""
        if user1 not in self.tf_idf_matrix or user2 not in self.tf_idf_matrix:
            return 0.0
        
        vector1 = self.tf_idf_matrix[user1]
        vector2 = self.tf_idf_matrix[user2]
        
        # Calculate dot product
        dot_product = 0.0
        for word in self.vocabulary:
            dot_product += vector1.get(word, 0) * vector2.get(word, 0)
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(score ** 2 for score in vector1.values()))
        magnitude2 = math.sqrt(sum(score ** 2 for score in vector2.values()))
        
        # Calculate cosine similarity
        if magnitude1 > 0 and magnitude2 > 0:
            similarity = dot_product / (magnitude1 * magnitude2)
        else:
            similarity = 0.0
        
        return similarity
    
    def build_similarity_matrix(self, min_similarity: float = 0.1):
        """Build similarity matrix between all users"""
        logger.info("Building similarity matrix")
        
        if not self.tf_idf_matrix:
            self.calculate_tf_idf_profiles()
        
        users = list(self.user_profiles.keys())
        self.similarity_matrix = {}
        
        for i, user1 in enumerate(users):
            self.similarity_matrix[user1] = {}
            for j, user2 in enumerate(users):
                if i != j:  # Don't calculate self-similarity
                    similarity = self.calculate_cosine_similarity(user1, user2)
                    if similarity >= min_similarity:
                        self.similarity_matrix[user1][user2] = similarity
        
        logger.info("Similarity matrix completed")
    
    def build_semantic_network(self, similarity_threshold: float = 0.3) -> Dict:
        """Build semantic similarity network"""
        logger.info(f"Building semantic network with threshold {similarity_threshold}")
        
        if not self.similarity_matrix:
            self.build_similarity_matrix()
        
        # Extract edges above threshold
        edges = {}
        nodes = set()
        
        for user1, similarities in self.similarity_matrix.items():
            for user2, similarity in similarities.items():
                if similarity >= similarity_threshold:
                    edge_key = f"{user1}<->{user2}"
                    edges[edge_key] = similarity
                    nodes.add(user1)
                    nodes.add(user2)
        
        # Calculate node attributes
        node_attributes = {}
        for node in nodes:
            # Calculate average similarity with all connected nodes
            similarities = []
            for user2, sim in self.similarity_matrix.get(node, {}).items():
                if sim >= similarity_threshold:
                    similarities.append(sim)
            
            # Add reverse connections
            for user1, user_sims in self.similarity_matrix.items():
                if node in user_sims and user_sims[node] >= similarity_threshold:
                    similarities.append(user_sims[node])
            
            node_attributes[node] = {
                'avg_similarity': sum(similarities) / len(similarities) if similarities else 0,
                'num_connections': len(similarities),
                'email_count': self.user_profiles[node]['email_count'],
                'domain': node.split('@')[1] if '@' in node else 'unknown'
            }
        
        # Network statistics
        stats = {
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'avg_similarity': sum(edges.values()) / len(edges) if edges else 0,
            'similarity_threshold': similarity_threshold,
            'density': len(edges) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0
        }
        
        self.semantic_network = {
            'type': 'semantic_similarity',
            'nodes': list(nodes),
            'edges': edges,
            'node_attributes': node_attributes,
            'stats': stats
        }
        
        logger.info(f"Built semantic network: {len(nodes)} nodes, {len(edges)} edges")
        return self.semantic_network
    
    def find_semantic_communities(self, min_similarity: float = 0.4) -> Dict[str, List[str]]:
        """Find semantic communities using simple clustering"""
        if not self.similarity_matrix:
            self.build_similarity_matrix()
        
        # Simple community detection based on high similarity
        communities = {}
        assigned_users = set()
        community_id = 0
        
        for user1, similarities in self.similarity_matrix.items():
            if user1 not in assigned_users:
                # Start new community
                community = [user1]
                assigned_users.add(user1)
                
                # Add highly similar users
                for user2, similarity in similarities.items():
                    if similarity >= min_similarity and user2 not in assigned_users:
                        community.append(user2)
                        assigned_users.add(user2)
                
                if len(community) > 1:  # Only keep communities with multiple members
                    communities[f"community_{community_id}"] = community
                    community_id += 1
        
        return communities
    
    def get_most_similar_pairs(self, top_n: int = 10) -> List[Tuple[str, str, float]]:
        """Get most similar user pairs"""
        if not self.similarity_matrix:
            self.build_similarity_matrix()
        
        pairs = []
        for user1, similarities in self.similarity_matrix.items():
            for user2, similarity in similarities.items():
                pairs.append((user1, user2, similarity))
        
        # Sort by similarity and return top N
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:top_n]
    
    def save_semantic_analysis(self, filename: str = 'semantic_similarity.json'):
        """Save semantic analysis results"""
        import os
        os.makedirs('output', exist_ok=True)
        
        # Get communities and top pairs
        communities = self.find_semantic_communities()
        top_pairs = self.get_most_similar_pairs()
        
        output_data = {
            'semantic_network': self.semantic_network,
            'communities': communities,
            'top_similar_pairs': [(pair[0], pair[1], pair[2]) for pair in top_pairs],
            'user_profiles_summary': {
                user: {
                    'email_count': profile['email_count'],
                    'word_count': len(profile['words']),
                    'unique_words': len(set(profile['words']))
                }
                for user, profile in self.user_profiles.items()
            }
        }
        
        output_path = f"output/{filename}"
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        logger.info(f"Semantic analysis saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    analyzer = SemanticSimilarityAnalyzer()
    
    # Load documents and build profiles
    docs = analyzer.load_documents(max_docs=2000)
    analyzer.build_user_profiles()
    
    # Build semantic network
    semantic_net = analyzer.build_semantic_network(similarity_threshold=0.2)
    
    # Print results
    print("=== SEMANTIC SIMILARITY ANALYSIS ===")
    print(f"Network nodes: {semantic_net['stats']['num_nodes']}")
    print(f"Network edges: {semantic_net['stats']['num_edges']}")
    print(f"Average similarity: {semantic_net['stats']['avg_similarity']:.3f}")
    print(f"Network density: {semantic_net['stats']['density']:.4f}")
    
    # Find communities
    communities = analyzer.find_semantic_communities(min_similarity=0.3)
    print(f"\nSemantic communities found: {len(communities)}")
    for comm_id, members in communities.items():
        print(f"  {comm_id}: {len(members)} members")
        print(f"    {members[:3]}{'...' if len(members) > 3 else ''}")
    
    # Top similar pairs
    top_pairs = analyzer.get_most_similar_pairs(5)
    print(f"\nTop similar user pairs:")
    for user1, user2, similarity in top_pairs:
        print(f"  {user1} <-> {user2}: {similarity:.3f}")
    
    # Save results
    analyzer.save_semantic_analysis()
