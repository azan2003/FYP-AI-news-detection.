"""
FAKE NEWS DETECTOR - YOUR FIRST VERSION
A simple machine learning model to detect fake news
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import re

print("="*60)
print("FAKE NEWS DETECTOR - STARTING VERSION")
print("="*60)

# STEP 1: Create sample training data
print("\n[1] Creating sample training data...")

# Sample headlines (mix of real and fake)
headlines = [
    "Scientists discover new planet in solar system",
    "You won't believe this one weird trick doctors hate!",
    "Government announces new climate policy measures",
    "Shocking: Miracle cure found that fixes everything instantly",
    "University researchers publish findings on AI safety",
    "Breaking: Aliens contacted world leaders secretly",
    "Economic report shows steady GDP growth this quarter",
    "Celebrity reveals secret to losing 50 pounds in 3 days",
    "New study links exercise to improved mental health",
    "Unbelievable: This food can make you immortal"
]

# Labels: 0 = Real, 1 = Fake
labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

print(f"   Loaded {len(headlines)} sample headlines")

# STEP 2: Preprocess the text
print("\n[2] Preprocessing text...")

def clean_text(text):
    """Clean the text by removing special characters and converting to lowercase"""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters
    text = ' '.join(text.split())  # Remove extra spaces
    return text

cleaned_headlines = [clean_text(h) for h in headlines]
print("   Text cleaned successfully")

# STEP 3: Split data into training and testing
print("\n[3] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    cleaned_headlines, 
    labels, 
    test_size=0.3,  # 30% for testing
    random_state=42
)

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# STEP 4: Convert text to numbers (TF-IDF)
print("\n[4] Converting text to numerical features...")

vectorizer = TfidfVectorizer(max_features=100)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"   Created {X_train_tfidf.shape[1]} features")

# STEP 5: Train the AI model
print("\n[5] Training the AI model...")

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("   Model trained successfully!")

# STEP 6: Test the model
print("\n[6] Testing the model...")

predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

print(f"   Accuracy: {accuracy * 100:.1f}%")

# STEP 7: Try it yourself!
print("\n" + "="*60)
print("TEST YOUR OWN HEADLINES")
print("="*60)

while True:
    print("\nEnter a news headline (or type 'quit' to exit):")
    user_input = input("> ")
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\nThanks for using Fake News Detector!")
        break
    
    if user_input.strip():
        # Clean and predict
        cleaned_input = clean_text(user_input)
        input_tfidf = vectorizer.transform([cleaned_input])
        prediction = model.predict(input_tfidf)[0]
        probabilities = model.predict_proba(input_tfidf)[0]
        
        # Show results
        if prediction == 1:
            print(f"\n⚠️  PREDICTION: FAKE NEWS")
            print(f"   Confidence: {probabilities[1]*100:.1f}%")
        else:
            print(f"\n✓  PREDICTION: REAL NEWS")
            print(f"   Confidence: {probabilities[0]*100:.1f}%")
        
        print("-"*60)

        