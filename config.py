"""
Configuration Settings
"""

import os

class Config:
    """Application configuration"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here-change-in-production'
    
    # Database settings
    SQLALCHEMY_DATABASE_URI = 'sqlite:///fake_news_db.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Model paths
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'models')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    # Model files
    GENERAL_MODEL = os.path.join(BASE_DIR, 'fake_news_model.pkl')
    GENERAL_VECTORIZER = os.path.join(BASE_DIR, 'vectorizer.pkl')
    
    # Upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'txt', 'csv'}
    
    # Application settings
    PREDICTIONS_PER_PAGE = 20
    MAX_TEXT_LENGTH = 10000  # characters

    