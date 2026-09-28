"""
ADVANCED FAKE NEWS DETECTOR
Uses real Kaggle dataset with 44,000+ news articles
Achieves 90%+ accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ADVANCED FAKE NEWS DETECTOR - REAL DATASET VERSION")
print("="*70)

# STEP 1: Load Real Dataset
print("\n[STEP 1] Loading dataset...")

try:
    # Load fake and real news
    fake_df = pd.read_csv('Fake.csv')
    real_df = pd.read_csv('True.csv')
    
    print(f"   ✓ Loaded {len(fake_df):,} FAKE news articles")
    print(f"   ✓ Loaded {len(real_df):,} REAL news articles")
    
    # Add labels
    fake_df['label'] = 1  # Fake = 1
    real_df['label'] = 0  # Real = 0
    
    # Combine datasets
    df = pd.concat([fake_df, real_df], ignore_index=True)
    
    # Combine title and text for better accuracy
    df['content'] = df['title'] + " " + df['text']
    
    print(f"   ✓ Total articles: {len(df):,}")
    print(f"   ✓ Dataset balanced: {len(fake_df)} fake, {len(real_df)} real")
    
except FileNotFoundError:
    print("\n❌ ERROR: Dataset files not found!")
    print("\nPlease make sure 'Fake.csv' and 'True.csv' are in this folder:")
    print("C:\\Users\\HP\\OneDrive\\Desktop\\fake-news-detector\\")
    print("\nDownload from: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset")
    exit()

# STEP 2: Text Preprocessing
print("\n[STEP 2] Preprocessing text data...")

def clean_text(text):
    """Advanced text cleaning"""
    if pd.isna(text):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Apply cleaning
df['cleaned_content'] = df['content'].apply(clean_text)

print("   ✓ Text preprocessing complete")

# STEP 3: Shuffle and Sample Data (for faster training)
print("\n[STEP 3] Preparing data...")

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Use subset for faster training (you can increase this later)
df_sample = df.head(10000)  # Using 10,000 articles

print(f"   ✓ Using {len(df_sample):,} articles for training")

# STEP 4: Split Data
print("\n[STEP 4] Splitting data into train/test sets...")

X = df_sample['cleaned_content']
y = df_sample['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,  # 20% for testing
    random_state=42,
    stratify=y  # Keep same ratio of fake/real in both sets
)

print(f"   ✓ Training set: {len(X_train):,} articles")
print(f"   ✓ Testing set: {len(X_test):,} articles")
print(f"   ✓ Fake news in training: {sum(y_train):,}")
print(f"   ✓ Real news in training: {len(y_train) - sum(y_train):,}")

# STEP 5: Vectorization (Convert text to numbers)
print("\n[STEP 5] Converting text to numerical features...")

vectorizer = TfidfVectorizer(
    max_features=5000,  # Use top 5000 words
    ngram_range=(1, 2),  # Use single words and pairs of words
    min_df=2  # Word must appear at least 2 times
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"   ✓ Created {X_train_tfidf.shape[1]:,} features")
print(f"   ✓ Training matrix shape: {X_train_tfidf.shape}")

# STEP 6: Train the Model
print("\n[STEP 6] Training Logistic Regression model...")
print("   (This may take 30-60 seconds...)")

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

print("   ✓ Model training complete!")

# STEP 7: Evaluate Model
print("\n[STEP 7] Evaluating model performance...")

# Training accuracy
y_train_pred = model.predict(X_train_tfidf)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Testing accuracy
y_test_pred = model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)
print(f"\nTraining Accuracy: {train_accuracy*100:.2f}%")
print(f"Testing Accuracy:  {test_accuracy*100:.2f}%")

print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_test_pred, 
                          target_names=['Real News', 'Fake News']))

# STEP 8: Create Confusion Matrix Visualization
print("\n[STEP 8] Creating confusion matrix visualization...")

cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - Accuracy: {test_accuracy*100:.1f}%', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("   ✓ Saved confusion matrix as 'confusion_matrix.png'")

# STEP 9: Save Model
print("\n[STEP 9] Saving trained model...")

with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("   ✓ Model saved as 'fake_news_model.pkl'")
print("   ✓ Vectorizer saved as 'vectorizer.pkl'")

# STEP 10: Interactive Testing
print("\n" + "="*70)
print("INTERACTIVE TESTING MODE")
print("="*70)
print("\nTest the detector with your own headlines!")
print("(Type 'quit' to exit)\n")

def predict_news(text):
    """Predict if news is fake or real"""
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])
    
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0]
    
    return prediction, probability

# Interactive loop
while True:
    user_input = input("\nEnter news headline/text: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\n" + "="*70)
        print("Thanks for using Advanced Fake News Detector!")
        print("="*70)
        break
    
    if user_input:
        prediction, prob = predict_news(user_input)
        
        print("\n" + "-"*70)
        if prediction == 1:
            print("⚠️  PREDICTION: FAKE NEWS")
            print(f"   Confidence: {prob[1]*100:.2f}%")
        else:
            print("✓  PREDICTION: REAL NEWS")
            print(f"   Confidence: {prob[0]*100:.2f}%")
        
        print(f"\n   Probability Breakdown:")
        print(f"   • Real News: {prob[0]*100:.2f}%")
        print(f"   • Fake News: {prob[1]*100:.2f}%")
        print("-"*70)

        