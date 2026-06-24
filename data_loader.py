"""
Data loading and preprocessing utilities for Enron dataset
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime
import config

# Set up logging
logging.basicConfig(level=getattr(logging, config.LOGGING['level']))
logger = logging.getLogger(__name__)

class EnronDataLoader:
    """
    Class for loading and preprocessing Enron email dataset
    """
    
    def __init__(self, data_file: str = config.ENRON_DATA_FILE):
        self.data_file = data_file
        self.df = None
        self.processed_df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the Enron dataset"""
        try:
            logger.info(f"Loading data from {self.data_file}")
            self.df = pd.read_csv(self.data_file)
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            logger.info(f"Columns: {list(self.df.columns)}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def explore_data(self) -> Dict:
        """Explore the dataset and return summary statistics"""
        if self.df is None:
            self.load_data()
            
        exploration = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'unique_senders': self.df['From'].nunique() if 'From' in self.df.columns else 'N/A',
            'unique_recipients': self.df['To'].nunique() if 'To' in self.df.columns else 'N/A',
            'date_range': self._get_date_range(),
            'sample_data': self.df.head().to_dict()
        }
        
        logger.info("Data exploration completed")
        return exploration
    
    def _get_date_range(self) -> Dict:
        """Get date range from the dataset"""
        date_columns = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if not date_columns:
            return {'error': 'No date columns found'}
        
        try:
            date_col = date_columns[0]
            dates = pd.to_datetime(self.df[date_col], errors='coerce')
            return {
                'column': date_col,
                'min_date': dates.min(),
                'max_date': dates.max(),
                'date_range_days': (dates.max() - dates.min()).days
            }
        except Exception as e:
            return {'error': f'Error processing dates: {e}'}
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the data for network analysis"""
        if self.df is None:
            self.load_data()
        
        logger.info("Starting data preprocessing")
        self.processed_df = self.df.copy()
        
        # Clean email addresses
        self._clean_email_addresses()
        
        # Process dates
        self._process_dates()
        
        # Clean text content
        self._clean_text_content()
        
        # Remove duplicates
        self._remove_duplicates()
        
        # Filter valid communications
        self._filter_valid_communications()
        
        logger.info(f"Data preprocessing completed. Final shape: {self.processed_df.shape}")
        return self.processed_df
    
    def _clean_email_addresses(self):
        """Clean and standardize email addresses"""
        email_columns = ['From', 'To', 'Cc', 'Bcc']
        
        for col in email_columns:
            if col in self.processed_df.columns:
                # Convert to lowercase and strip whitespace
                self.processed_df[col] = self.processed_df[col].astype(str).str.lower().str.strip()
                
                # Remove invalid email addresses
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                valid_emails = self.processed_df[col].str.match(email_pattern, na=False)
                self.processed_df.loc[~valid_emails, col] = np.nan
    
    def _process_dates(self):
        """Process and standardize date columns"""
        date_columns = [col for col in self.processed_df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        for col in date_columns:
            try:
                self.processed_df[col] = pd.to_datetime(self.processed_df[col], errors='coerce')
                # Add derived date features
                self.processed_df[f'{col}_year'] = self.processed_df[col].dt.year
                self.processed_df[f'{col}_month'] = self.processed_df[col].dt.month
                self.processed_df[f'{col}_day_of_week'] = self.processed_df[col].dt.dayofweek
                self.processed_df[f'{col}_hour'] = self.processed_df[col].dt.hour
            except Exception as e:
                logger.warning(f"Error processing date column {col}: {e}")
    
    def _clean_text_content(self):
        """Clean text content for NLP processing"""
        text_columns = ['Subject', 'Body', 'Content']
        
        for col in text_columns:
            if col in self.processed_df.columns:
                # Remove HTML tags
                self.processed_df[col] = self.processed_df[col].astype(str).apply(self._remove_html_tags)
                
                # Remove extra whitespace
                self.processed_df[col] = self.processed_df[col].str.replace(r'\s+', ' ', regex=True).str.strip()
                
                # Remove very short content (likely not meaningful)
                self.processed_df.loc[self.processed_df[col].str.len() < 10, col] = np.nan
    
    def _remove_html_tags(self, text: str) -> str:
        """Remove HTML tags from text"""
        if pd.isna(text):
            return text
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def _remove_duplicates(self):
        """Remove duplicate emails"""
        # Define columns to check for duplicates
        duplicate_columns = ['From', 'To', 'Subject']
        available_columns = [col for col in duplicate_columns if col in self.processed_df.columns]
        
        if available_columns:
            initial_count = len(self.processed_df)
            self.processed_df = self.processed_df.drop_duplicates(subset=available_columns, keep='first')
            removed_count = initial_count - len(self.processed_df)
            logger.info(f"Removed {removed_count} duplicate emails")
    
    def _filter_valid_communications(self):
        """Filter out invalid or incomplete communications"""
        # Remove rows with missing sender or recipient
        required_columns = ['From', 'To']
        for col in required_columns:
            if col in self.processed_df.columns:
                self.processed_df = self.processed_df.dropna(subset=[col])
        
        # Remove self-communications (emails to oneself)
        if 'From' in self.processed_df.columns and 'To' in self.processed_df.columns:
            self.processed_df = self.processed_df[self.processed_df['From'] != self.processed_df['To']]
    
    def get_communication_pairs(self) -> List[Tuple[str, str]]:
        """Extract communication pairs for network construction"""
        if self.processed_df is None:
            self.preprocess_data()
        
        pairs = []
        for _, row in self.processed_df.iterrows():
            sender = row.get('From')
            recipient = row.get('To')
            
            if pd.notna(sender) and pd.notna(recipient):
                pairs.append((sender, recipient))
        
        logger.info(f"Extracted {len(pairs)} communication pairs")
        return pairs
    
    def save_processed_data(self, filename: str = "processed_enron_data.csv"):
        """Save processed data to file"""
        if self.processed_df is None:
            self.preprocess_data()
        
        output_path = config.OUTPUT_DIR / filename
        self.processed_df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    loader = EnronDataLoader()
    
    # Load and explore data
    data = loader.load_data()
    exploration = loader.explore_data()
    print("Data Exploration Results:")
    for key, value in exploration.items():
        if key != 'sample_data':
            print(f"{key}: {value}")
    
    # Preprocess data
    processed_data = loader.preprocess_data()
    
    # Get communication pairs
    pairs = loader.get_communication_pairs()
    print(f"Communication pairs: {len(pairs)}")
    
    # Save processed data
    loader.save_processed_data()
