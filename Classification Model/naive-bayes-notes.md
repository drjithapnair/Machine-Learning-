# Naive Bayes Algorithm: Comprehensive Guide

## 1. Introduction to Naive Bayes

### Basic Concepts
- Probabilistic classifier based on Bayes' Theorem
- "Naive" assumes feature independence
- Fast training and prediction
- Works well with high-dimensional data
- Particularly effective for text classification

### Types of Naive Bayes
1. Gaussian Naive Bayes (continuous data)
2. Multinomial Naive Bayes (discrete counts)
3. Bernoulli Naive Bayes (binary features)

### Bayes' Theorem
```
P(A|B) = P(B|A) * P(A) / P(B)

For classification:
P(class|features) = P(features|class) * P(class) / P(features)
```

## 2. Implementation Examples

### Example 1: Email Spam Classification

```python
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Sample email data
emails = [
    "Get rich quick! Buy now!",
    "Meeting at 3pm tomorrow",
    "WINNER! Claim prize now!!!",
    "Project deadline reminder",
    "FREE DISCOUNT LIMITED TIME",
    "Quarterly report attached",
    "Buy medications cheap!",
    "Team lunch next week"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for not spam

# Create text vectorizer
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(emails)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42
)

# Train model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Make predictions
y_pred = nb_model.predict(X_test)

# Print results
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to classify new emails
def classify_email(email_text):
    email_vectorized = vectorizer.transform([email_text])
    prediction = nb_model.predict(email_vectorized)
    probability = nb_model.predict_proba(email_vectorized)
    return {
        'is_spam': bool(prediction[0]),
        'spam_probability': probability[0][1]
    }

# Test the function
new_email = "URGENT! You've won a prize worth $1000000!"
result = classify_email(new_email)
print(f"Email classification: {'Spam' if result['is_spam'] else 'Not Spam'}")
print(f"Spam probability: {result['spam_probability']:.2%}")
```

### Example 2: Weather Prediction using Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Sample weather data
weather_data = {
    'temperature': [75, 68, 82, 65, 70, 85, 72, 69],
    'humidity': [65, 75, 55, 70, 60, 50, 65, 80],
    'windy': [0, 1, 1, 1, 0, 0, 1, 1],
    'play': ['Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No']
}

# Create DataFrame
df = pd.DataFrame(weather_data)

# Encode categorical variable
le = LabelEncoder()
df['play_encoded'] = le.fit_transform(df['play'])

# Prepare features and target
X = df[['temperature', 'humidity', 'windy']]
y = df['play_encoded']

# Create and train model
gnb = GaussianNB()
gnb.fit(X, y)

# Function to predict weather conditions
def predict_weather(temperature, humidity, windy):
    prediction = gnb.predict([[temperature, humidity, windy]])
    probability = gnb.predict_proba([[temperature, humidity, windy]])
    return {
        'prediction': le.inverse_transform(prediction)[0],
        'probability': max(probability[0])
    }

# Test prediction
result = predict_weather(temperature=72, humidity=60, windy=1)
print(f"Weather prediction: {result['prediction']}")
print(f"Prediction confidence: {result['probability']:.2%}")
```

### Example 3: Document Classification

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "Python is a great programming language",
    "Machine learning is fascinating",
    "Data science uses statistics",
    "Programming in Python is fun",
    "Statistics and probability in ML",
    "Web development with Django"
]
categories = ['programming', 'ml', 'stats', 'programming', 'ml', 'programming']

# Create TF-IDF vectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(documents)

# Train model
doc_classifier = MultinomialNB()
doc_classifier.fit(X, categories)

# Function to classify new documents
def classify_document(text):
    text_vectorized = tfidf.transform([text])
    prediction = doc_classifier.predict(text_vectorized)
    probabilities = doc_classifier.predict_proba(text_vectorized)
    return {
        'category': prediction[0],
        'confidence': max(probabilities[0])
    }

# Test classification
new_doc = "Neural networks and deep learning concepts"
result = classify_document(new_doc)
print(f"Document category: {result['category']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 3. Advanced Techniques

### Handling Imbalanced Data

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# Create pipeline with SMOTE
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('smote', SMOTE()),
    ('classifier', MultinomialNB())
])

# Use pipeline
pipeline.fit(X_train, y_train)
```

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, chi2

# Create pipeline with feature selection
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('feature_selection', SelectKBest(chi2, k=100)),
    ('classifier', MultinomialNB())
])
```

## 4. Model Evaluation and Tuning

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# Perform cross-validation
cv_scores = cross_val_score(nb_model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.2f}")
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
    'fit_prior': [True, False]
}

# Perform grid search
grid_search = GridSearchCV(
    MultinomialNB(),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
```

## 5. Visualization and Interpretation

### Feature Importance

```python
def plot_feature_importance(vectorizer, model, n_top_features=10):
    import matplotlib.pyplot as plt
    
    # Get feature names
    feature_names = vectorizer.get_feature_names_out()
    
    # Calculate feature importance
    importance = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    
    # Create DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    # Sort and plot top features
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_top_features), 
            feature_importance['importance'][:n_top_features])
    plt.xticks(range(n_top_features), 
               feature_importance['feature'][:n_top_features], 
               rotation=45)
    plt.title('Top Features by Importance')
    plt.tight_layout()
    plt.show()
```

### Confusion Matrix Visualization

```python
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
```

## 6. Best Practices and Tips

1. **Data Preprocessing**
```python
# Handle missing values
def preprocess_text(text):
    if pd.isna(text):
        retu