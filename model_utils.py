"""
Enhanced Model Utilities - Multi-Domain Support
Handles Political, Health, and General news models
"""

import re
import string
import pickle
import numpy as np
from lime.lime_text import LimeTextExplainer
import os
import warnings
warnings.filterwarnings('ignore')

class MultiDomainDetector:
    """Fake news detector with domain-specific models"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.lime_explainers = {}
        self.domains = ['general', 'political', 'health']
        
    def load_all_models(self):
        """Load all domain-specific models"""
        loaded = []
        
        for domain in self.domains:
            model_path = f'models/{domain}_model.pkl'
            vectorizer_path = f'models/{domain}_vectorizer.pkl'
            
            if self.load_model(domain, model_path, vectorizer_path):
                loaded.append(domain)
        
        # Fallback to old general model if new ones don't exist
        if not loaded:
            if self.load_model('general', 'fake_news_model.pkl', 'vectorizer.pkl'):
                loaded.append('general (legacy)')
        
        return loaded
    
    def load_model(self, domain, model_path, vectorizer_path):
        """Load a specific domain model"""
        try:
            if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                return False
            
            with open(model_path, 'rb') as f:
                self.models[domain] = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizers[domain] = pickle.load(f)
            
            # Create LIME explainer for this domain
            self.lime_explainers[domain] = LimeTextExplainer(
                class_names=['Real', 'Fake']
            )
            
            return True
        except Exception as e:
            print(f"Error loading {domain} model: {e}")
            return False
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not text:
            return ""
        
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        text = re.sub(r'\d+', '', text)
        text = ' '.join(text.split())
        
        return text
    
    def detect_domain(self, text):
        """Detect if article is political, health, or general"""
        text_lower = text.lower()
        
        political_keywords = [
            'government', 'president', 'election', 'vote', 'political',
            'congress', 'senate', 'democrat', 'republican', 'policy',
            'minister', 'parliament', 'legislation', 'campaign', 'politician',
            'white house', 'federal', 'administration', 'supreme court'
        ]
        
        health_keywords = [
            'covid', 'vaccine', 'virus', 'pandemic', 'disease', 'health',
            'medical', 'doctor', 'hospital', 'symptom', 'treatment',
            'infection', 'cure', 'medicine', 'patient', 'coronavirus',
            'cdc', 'who', 'fda', 'healthcare', 'drug'
        ]
        
        political_count = sum(1 for kw in political_keywords if kw in text_lower)
        health_count = sum(1 for kw in health_keywords if kw in text_lower)
        
        if political_count >= 3 and political_count > health_count:
            return 'political'
        elif health_count >= 3 and health_count > political_count:
            return 'health'
        else:
            return 'general'
    
    def predict(self, text, domain='auto'):
        """Make prediction using appropriate domain model"""
        
        # Auto-detect domain if needed
        if domain == 'auto':
            domain = self.detect_domain(text)
        
        # Check if model exists, fallback to general
        if domain not in self.models:
            domain = 'general'
        
        if domain not in self.models:
            return None
        
        # Clean text
        cleaned = self.clean_text(text)
        
        # Transform and predict
        model = self.models[domain]
        vectorizer = self.vectorizers[domain]
        
        text_tfidf = vectorizer.transform([cleaned])
        prediction = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]
        
        result = {
            'prediction': 'Fake' if prediction == 1 else 'Real',
            'prediction_class': int(prediction),
            'confidence': float(probabilities[prediction] * 100),
            'real_probability': float(probabilities[0] * 100),
            'fake_probability': float(probabilities[1] * 100),
            'domain': domain,
            'original_text': text,
            'cleaned_text': cleaned
        }
        
        return result
    
    def predict_all_domains(self, text):
        """Predict using all available models for comparison"""
        cleaned = self.clean_text(text)
        results = {}
        
        for domain in self.models.keys():
            try:
                model = self.models[domain]
                vectorizer = self.vectorizers[domain]
                
                text_tfidf = vectorizer.transform([cleaned])
                prediction = model.predict(text_tfidf)[0]
                probabilities = model.predict_proba(text_tfidf)[0]
                
                results[domain] = {
                    'prediction': 'Fake' if prediction == 1 else 'Real',
                    'confidence': float(probabilities[prediction] * 100),
                    'fake_probability': float(probabilities[1] * 100)
                }
            except Exception as e:
                print(f"Error predicting with {domain} model: {e}")
        
        return results
    
    def explain_prediction(self, text, domain='auto', num_features=10):
        """Generate explanation using LIME"""
        
        # Auto-detect or use specified domain
        if domain == 'auto':
            domain = self.detect_domain(text)
        
        if domain not in self.models:
            domain = 'general'
        
        if domain not in self.models:
            return None
        
        cleaned = self.clean_text(text)
        model = self.models[domain]
        vectorizer = self.vectorizers[domain]
        explainer = self.lime_explainers[domain]
        
        # Prediction function for LIME
        def predict_proba_fn(texts):
            texts_cleaned = [self.clean_text(t) for t in texts]
            texts_tfidf = vectorizer.transform(texts_cleaned)
            return model.predict_proba(texts_tfidf)
        
        try:
            # Generate explanation
            explanation = explainer.explain_instance(
                cleaned,
                predict_proba_fn,
                num_features=num_features,
                top_labels=2
            )
            
            # Get prediction
            prediction = self.predict(text, domain)
            pred_class = prediction['prediction_class']
            
            # Extract feature importance
            features = explanation.as_list(label=pred_class)
            
            important_words = []
            for word, weight in features:
                important_words.append({
                    'word': word,
                    'weight': float(weight),
                    'impact': 'positive' if weight > 0 else 'negative'
                })
            
            return {
                'prediction': prediction,
                'important_words': important_words,
                'explanation_text': self._generate_explanation_text(
                    important_words, 
                    prediction['prediction'],
                    domain
                )
            }
            
        except Exception as e:
            print(f"Explanation error: {e}")
            return None
    
    def _generate_explanation_text(self, important_words, prediction, domain):
        """Generate human-readable explanation"""
        if not important_words:
            return "Unable to generate explanation."
        
        top_words = sorted(important_words, 
                          key=lambda x: abs(x['weight']), 
                          reverse=True)[:3]
        
        domain_text = f" in {domain} news" if domain != 'general' else ""
        
        if prediction == 'Fake':
            explanation = f"This article is predicted as FAKE NEWS{domain_text} primarily because of words like: "
            words = [f"'{w['word']}'" for w in top_words if w['impact'] == 'positive']
            explanation += ", ".join(words) if words else "suspicious patterns"
            explanation += ". These terms are commonly found in misleading or sensationalized content"
            explanation += f" within the {domain} domain." if domain != 'general' else "."
        else:
            explanation = f"This article appears to be REAL NEWS{domain_text}. Key indicators include words like: "
            words = [f"'{w['word']}'" for w in top_words if w['impact'] == 'positive']
            explanation += ", ".join(words) if words else "factual language"
            explanation += ". These terms suggest credible, fact-based reporting"
            explanation += f" typical of reliable {domain} news sources." if domain != 'general' else "."
        
        return explanation
    
    def get_word_highlighting(self, text, important_words):
        """Create HTML with highlighted words"""
        words = text.split()
        highlighted = []
        
        word_weights = {}
        for item in important_words:
            word_weights[item['word'].lower()] = item['weight']
        
        for word in words:
            clean_word = re.sub(r'[^\w\s]', '', word.lower())
            
            if clean_word in word_weights:
                weight = word_weights[clean_word]
                if weight > 0:
                    intensity = min(abs(weight) * 100, 100)
                    color = f"rgba(239, 68, 68, {intensity/100})"
                else:
                    intensity = min(abs(weight) * 100, 100)
                    color = f"rgba(34, 197, 94, {intensity/100})"
                
                highlighted.append(
                    f'<span style="background-color: {color}; '
                    f'padding: 2px 4px; border-radius: 3px;" '
                    f'title="Weight: {weight:.3f}">{word}</span>'
                )
            else:
                highlighted.append(word)
        
        return ' '.join(highlighted)
    
    def get_available_models(self):
        """Return list of loaded models"""
        return list(self.models.keys())