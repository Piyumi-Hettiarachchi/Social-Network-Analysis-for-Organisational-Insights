"""
Sentiment Analysis and Stress Detection Module
"""
#Score each email for sentiment (positive/neutral/negative) and stress/urgency,
import csv
import re
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Simple sentiment analysis and stress detection implementation
    """
    
    def __init__(self, data_file: str = 'enron_cleaned_final.csv'):
        self.data_file = data_file
        self.documents = []
        self.sentiment_lexicon = self._build_sentiment_lexicon()
        self.stress_keywords = self._get_stress_keywords()
        self.urgency_keywords = self._get_urgency_keywords()
        
    def _build_sentiment_lexicon(self) -> Dict[str, float]:
        """Build a simple sentiment lexicon"""
        positive_words = {
            'good': 1.0, 'great': 1.5, 'excellent': 2.0, 'amazing': 2.0, 'wonderful': 1.5,
            'fantastic': 2.0, 'awesome': 1.5, 'perfect': 2.0, 'outstanding': 2.0, 'superb': 1.5,
            'pleased': 1.0, 'happy': 1.5, 'delighted': 1.5, 'satisfied': 1.0, 'glad': 1.0,
            'thank': 0.5, 'thanks': 0.5, 'appreciate': 1.0, 'grateful': 1.0, 'congratulations': 1.5,
            'success': 1.0, 'successful': 1.0, 'win': 1.0, 'winner': 1.0, 'achieve': 0.5,
            'opportunity': 0.5, 'benefit': 0.5, 'advantage': 0.5, 'positive': 1.0, 'approve': 0.5,
            'agree': 0.5, 'support': 0.5, 'recommend': 0.5, 'endorse': 0.5, 'love': 1.5
        }
        
        negative_words = {
            'bad': -1.0, 'terrible': -2.0, 'awful': -2.0, 'horrible': -2.0, 'disgusting': -2.0,
            'hate': -2.0, 'dislike': -1.0, 'angry': -1.5, 'mad': -1.5, 'furious': -2.0,
            'disappointed': -1.5, 'frustrated': -1.5, 'annoyed': -1.0, 'irritated': -1.0,
            'problem': -1.0, 'issue': -0.5, 'trouble': -1.0, 'difficulty': -0.5, 'concern': -0.5,
            'worry': -1.0, 'worried': -1.0, 'fear': -1.0, 'afraid': -1.0, 'scared': -1.5,
            'fail': -1.5, 'failure': -1.5, 'mistake': -1.0, 'error': -0.5, 'wrong': -0.5,
            'sorry': -0.5, 'apologize': -0.5, 'regret': -1.0, 'unfortunate': -1.0, 'sad': -1.0,
            'crisis': -2.0, 'emergency': -1.5, 'urgent': -0.5, 'critical': -1.0, 'serious': -0.5,
            'reject': -1.0, 'deny': -0.5, 'refuse': -1.0, 'decline': -0.5, 'cancel': -0.5
        }
        
        # Combine lexicons
        lexicon = {}
        lexicon.update(positive_words)
        lexicon.update(negative_words)
        
        return lexicon
    
    def _get_stress_keywords(self) -> Set[str]:
        """Get stress-related keywords"""
        return {
            'stress', 'stressed', 'pressure', 'pressured', 'overwhelmed', 'overworked',
            'exhausted', 'tired', 'burnout', 'burnt', 'deadline', 'deadlines',
            'rush', 'rushing', 'hurry', 'hurried', 'panic', 'panicked', 'frantic',
            'crisis', 'emergency', 'urgent', 'critical', 'serious', 'trouble',
            'problem', 'problems', 'issue', 'issues', 'concern', 'concerns',
            'worry', 'worried', 'anxious', 'anxiety', 'nervous', 'tense'
        }
    
    def _get_urgency_keywords(self) -> Set[str]:
        """Get urgency-related keywords"""
        return {
            'asap', 'immediately', 'urgent', 'urgently', 'emergency', 'critical',
            'deadline', 'rush', 'hurry', 'quick', 'quickly', 'fast', 'soon',
            'now', 'today', 'tonight', 'tomorrow', 'priority', 'important',
            'crucial', 'essential', 'must', 'need', 'required', 'necessary'
        }
    
    def load_documents(self, max_docs: int = 5000) -> List[Dict]:
        """Load email documents for sentiment analysis"""
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
                    
                    if doc['text'] and len(doc['text']) > 10:
                        self.documents.append(doc)
        
        logger.info(f"Loaded {len(self.documents)} documents")
        return self.documents
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for sentiment analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove email addresses and URLs
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Keep letters, numbers, and basic punctuation
        text = re.sub(r'[^\w\s!?.]', ' ', text)
        
        # Tokenize
        words = text.split()
        
        return words
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text using lexicon-based approach"""
        words = self.preprocess_text(text)
        
        sentiment_score = 0.0
        positive_count = 0
        negative_count = 0
        total_sentiment_words = 0
        
        for word in words:
            if word in self.sentiment_lexicon:
                score = self.sentiment_lexicon[word]
                sentiment_score += score
                total_sentiment_words += 1
                
                if score > 0:
                    positive_count += 1
                elif score < 0:
                    negative_count += 1
        
        # Normalize sentiment score
        if total_sentiment_words > 0:
            normalized_score = sentiment_score / total_sentiment_words
        else:
            normalized_score = 0.0
        
        # Classify sentiment
        if normalized_score > 0.1:
            sentiment_label = 'positive'
        elif normalized_score < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        return {
            'sentiment_score': normalized_score,
            'sentiment_label': sentiment_label,
            'positive_words': positive_count,
            'negative_words': negative_count,
            'total_sentiment_words': total_sentiment_words
        }
    
    def detect_stress_indicators(self, text: str) -> Dict[str, any]:
        """Detect stress indicators in text"""
        words = self.preprocess_text(text)
        word_set = set(words)
        
        # Count stress keywords
        stress_words_found = word_set.intersection(self.stress_keywords)
        urgency_words_found = word_set.intersection(self.urgency_keywords)
        
        # Check for excessive punctuation (stress indicator)
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_words = sum(1 for word in words if word.isupper() and len(word) > 2)
        
        # Calculate stress score
        stress_score = (
            len(stress_words_found) * 2.0 +
            len(urgency_words_found) * 1.5 +
            exclamation_count * 0.5 +
            caps_words * 0.3
        )
        
        # Normalize by text length
        text_length = len(words)
        if text_length > 0:
            stress_score = stress_score / (text_length / 100)  # Per 100 words
        
        # Classify stress level
        if stress_score > 2.0:
            stress_level = 'high'
        elif stress_score > 1.0:
            stress_level = 'medium'
        elif stress_score > 0.5:
            stress_level = 'low'
        else:
            stress_level = 'none'
        
        return {
            'stress_score': stress_score,
            'stress_level': stress_level,
            'stress_keywords': list(stress_words_found),
            'urgency_keywords': list(urgency_words_found),
            'exclamation_count': exclamation_count,
            'caps_words_count': caps_words
        }
    
    def analyze_all_documents(self) -> Dict:
        """Analyze sentiment and stress for all documents"""
        if not self.documents:
            self.load_documents()
        
        logger.info("Analyzing sentiment and stress for all documents")
        
        results = {
            'document_analysis': [],
            'user_analysis': defaultdict(lambda: {
                'emails': [],
                'avg_sentiment': 0.0,
                'avg_stress': 0.0,
                'sentiment_distribution': {'positive': 0, 'negative': 0, 'neutral': 0},
                'stress_distribution': {'high': 0, 'medium': 0, 'low': 0, 'none': 0}
            })
        }
        
        for doc in self.documents:
            # Analyze sentiment
            sentiment_result = self.analyze_sentiment(doc['text'])
            
            # Detect stress
            stress_result = self.detect_stress_indicators(doc['text'])
            
            # Combine results
            doc_analysis = {
                'doc_id': doc['id'],
                'sender': doc['sender'],
                'recipient': doc['recipient'],
                'date': doc['date'],
                'subject': doc['subject'][:100],
                'sentiment': sentiment_result,
                'stress': stress_result
            }
            
            results['document_analysis'].append(doc_analysis)
            
            # Update user analysis
            sender = doc['sender']
            user_data = results['user_analysis'][sender]
            user_data['emails'].append(doc_analysis)
            user_data['sentiment_distribution'][sentiment_result['sentiment_label']] += 1
            user_data['stress_distribution'][stress_result['stress_level']] += 1
        
        # Calculate user averages
        for user, data in results['user_analysis'].items():
            if data['emails']:
                sentiments = [email['sentiment']['sentiment_score'] for email in data['emails']]
                stresses = [email['stress']['stress_score'] for email in data['emails']]
                
                data['avg_sentiment'] = sum(sentiments) / len(sentiments)
                data['avg_stress'] = sum(stresses) / len(stresses)
                data['total_emails'] = len(data['emails'])
        
        logger.info("Analysis completed")
        return results
    
    def get_sentiment_summary(self, results: Dict) -> Dict:
        """Get summary of sentiment analysis"""
        total_docs = len(results['document_analysis'])
        
        sentiment_counts = Counter()
        stress_counts = Counter()
        
        for doc in results['document_analysis']:
            sentiment_counts[doc['sentiment']['sentiment_label']] += 1
            stress_counts[doc['stress']['stress_level']] += 1
        
        # Top users by sentiment extremes
        user_sentiments = [(user, data['avg_sentiment']) for user, data in results['user_analysis'].items()]
        most_positive = sorted(user_sentiments, key=lambda x: x[1], reverse=True)[:5]
        most_negative = sorted(user_sentiments, key=lambda x: x[1])[:5]
        
        # Top users by stress
        user_stress = [(user, data['avg_stress']) for user, data in results['user_analysis'].items()]
        most_stressed = sorted(user_stress, key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_documents': total_docs,
            'sentiment_distribution': dict(sentiment_counts),
            'stress_distribution': dict(stress_counts),
            'most_positive_users': most_positive,
            'most_negative_users': most_negative,
            'most_stressed_users': most_stressed,
            'avg_sentiment_score': sum(doc['sentiment']['sentiment_score'] for doc in results['document_analysis']) / total_docs,
            'avg_stress_score': sum(doc['stress']['stress_score'] for doc in results['document_analysis']) / total_docs
        }
    
    def save_analysis(self, results: Dict, filename: str = 'sentiment_analysis.json'):
        """Save analysis results to JSON file"""
        import os
        os.makedirs('output', exist_ok=True)
        
        # Convert defaultdict to regular dict for JSON serialization
        results_copy = {
            'document_analysis': results['document_analysis'],
            'user_analysis': dict(results['user_analysis']),
            'summary': self.get_sentiment_summary(results)
        }
        
        output_path = f"output/{filename}"
        with open(output_path, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        logger.info(f"Analysis saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()
    
    # Load documents
    docs = analyzer.load_documents(max_docs=2000)
    
    # Analyze all documents
    results = analyzer.analyze_all_documents()
    
    # Get summary
    summary = analyzer.get_sentiment_summary(results)
    
    print("=== SENTIMENT ANALYSIS RESULTS ===")
    print(f"Total documents analyzed: {summary['total_documents']}")
    print(f"Average sentiment score: {summary['avg_sentiment_score']:.3f}")
    print(f"Average stress score: {summary['avg_stress_score']:.3f}")
    
    print(f"\nSentiment distribution:")
    for sentiment, count in summary['sentiment_distribution'].items():
        percentage = (count / summary['total_documents']) * 100
        print(f"  {sentiment}: {count} ({percentage:.1f}%)")
    
    print(f"\nStress distribution:")
    for stress, count in summary['stress_distribution'].items():
        percentage = (count / summary['total_documents']) * 100
        print(f"  {stress}: {count} ({percentage:.1f}%)")
    
    print(f"\nMost positive users:")
    for user, score in summary['most_positive_users']:
        print(f"  {user}: {score:.3f}")
    
    print(f"\nMost negative users:")
    for user, score in summary['most_negative_users']:
        print(f"  {user}: {score:.3f}")
    
    print(f"\nMost stressed users:")
    for user, score in summary['most_stressed_users']:
        print(f"  {user}: {score:.3f}")
    
    # Save results
    analyzer.save_analysis(results)
