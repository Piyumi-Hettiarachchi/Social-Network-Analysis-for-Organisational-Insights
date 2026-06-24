"""
Configuration file for Social Network Analysis System
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data files
ENRON_DATA_FILE = "enron_cleaned_final.csv"

# NLP Configuration
TOPIC_MODELING = {
    'lda_num_topics': 10,
    'lda_passes': 10,
    'lda_iterations': 100,
    'bertopic_min_topic_size': 10,
    'bertopic_nr_topics': 'auto'
}

SENTIMENT_ANALYSIS = {
    'vader_threshold': 0.05,
    'transformer_model': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'stress_keywords': [
        'urgent', 'asap', 'immediately', 'crisis', 'emergency', 'deadline',
        'pressure', 'stress', 'worried', 'concerned', 'problem', 'issue',
        'critical', 'important', 'rush', 'hurry'
    ]
}

SEMANTIC_SIMILARITY = {
    'sentence_transformer_model': 'all-MiniLM-L6-v2',
    'similarity_threshold': 0.7,
    'max_sequence_length': 512
}

# Network Analysis Configuration
NETWORK_ANALYSIS = {
    'min_edge_weight': 1,
    'community_detection_algorithm': 'louvain',
    'centrality_measures': ['degree', 'betweenness', 'eigenvector', 'closeness'],
    'ego_network_radius': 2
}

# Visualization Configuration
VISUALIZATION = {
    'node_size_range': (10, 100),
    'edge_width_range': (0.5, 5),
    'color_palette': 'viridis',
    'layout_algorithm': 'spring',
    'figure_size': (12, 8)
}


# Logging Configuration
LOGGING = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'sna_system.log'
}
