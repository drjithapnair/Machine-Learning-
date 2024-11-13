# Logistic Regression: A Comprehensive Guide

## 1. Introduction to Logistic Regression

### Definition
Logistic Regression is a statistical method for predicting binary outcomes by estimating the probability that an instance belongs to a particular class.

### Key Characteristics
- Binary output (0 or 1)
- Uses sigmoid function
- Linear decision boundary
- Outputs probabilities between 0 and 1

## 2. Mathematical Foundation

### The Sigmoid Function
```
σ(z) = 1 / (1 + e^(-z))
where z = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

### Probability Estimation
```
P(y=1|X) = σ(WᵀX + b)
```

### Cost Function
```
J(θ) = -1/m Σ[y⁽ⁱ⁾log(h_θ(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h_θ(x⁽ⁱ⁾))]
```

## 3. Practical Implementation

### Example 1: Credit Card Fraud Detection

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Sample dataset
data = {
    'transaction_amount': [100, 25000, 450, 8000, 250, 90000, 300, 150],
    'time_of_day': [14, 2, 15, 3, 13, 1, 16, 14],
    'unusual_location': [0, 1, 0, 1, 0, 1, 0, 0],
    'is_fraud': [0, 1, 0, 1, 0, 1, 0, 0]
}

# Create DataFrame
df = pd.DataFrame(data)

# Prepare features and target
X = df[['transaction_amount', 'time_of_day', 'unusual_location']]
y = df['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print results
print("Model Coefficients:", model.coef_)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

### Example 2: Email Spam Classification

```python
# Example with text data
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample email data
emails = [
    "Special offer! Buy now!",
    "Meeting at 3pm tomorrow",
    "WINNER! Claim prize now!!!",
    "Project deadline reminder",
    "FREE DISCOUNT LIMITED TIME",
    "Quarterly report attached"
]
labels = [1, 0, 1, 0, 1, 0]  # 1 for spam, 0 for not spam

# Convert text to features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

# Create and train model
spam_model = LogisticRegression()
spam_model.fit(X, labels)

# Test new emails
new_emails = ["FREE WINNER! CLAIM NOW!", "Team meeting at 2pm"]
new_X = vectorizer.transform(new_emails)
predictions = spam_model.predict(new_X)
probabilities = spam_model.predict_proba(new_X)
```

## 4. Advantages and Disadvantages

### Advantages
1. Simple and interpretable
2. Fast training and prediction
3. Provides probability scores
4. Works well for linearly separable data
5. Less prone to overfitting
6. Easy to update with new data

### Disadvantages
1. Assumes linear relationship
2. May underperform with non-linear data
3. Sensitive to outliers
4. Requires feature scaling
5. Assumes independence of features

## 5. Best Practices and Tips

### Feature Engineering
```python
# Example of feature engineering
def engineer_features(df):
    # Create interaction terms
    df['amount_time'] = df['transaction_amount'] * df['time_of_day']
    
    # Binning continuous variables
    df['amount_category'] = pd.qcut(df['transaction_amount'], q=5, labels=['VL', 'L', 'M', 'H', 'VH'])
    
    # One-hot encoding
    df = pd.get_dummies(df, columns=['amount_category'])
    
    return df
```

### Model Evaluation
```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# ROC Curve visualization
def plot_roc_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
```

## 6. Real-world Applications

### Healthcare Example
```python
# Patient Diabetes Risk Prediction
health_data = {
    'glucose': [140, 85, 200, 91, 160],
    'bmi': [32, 24, 35, 26, 31],
    'age': [45, 35, 55, 25, 50],
    'has_diabetes': [1, 0, 1, 0, 1]
}

health_df = pd.DataFrame(health_data)

# Prepare and scale data
X_health = health_df[['glucose', 'bmi', 'age']]
y_health = health_df['has_diabetes']

# Scale features
X_health_scaled = StandardScaler().fit_transform(X_health)

# Train model
diabetes_model = LogisticRegression()
diabetes_model.fit(X_health_scaled, y_health)

# Make prediction for new patient
new_patient = np.array([[150, 28, 40]])  # glucose, bmi, age
new_patient_scaled = StandardScaler().fit(X_health).transform(new_patient)
risk_probability = diabetes_model.predict_proba(new_patient_scaled)[0][1]
print(f"Diabetes Risk Probability: {risk_probability:.2%}")
```

## 7. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Create grid search object
grid_search = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

# Fit grid search
grid_search.fit(X_train_scaled, y_train)

# Print best parameters
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

## 8. Common Pitfalls and Solutions

1. **Class Imbalance**
```python
from imblearn.over_sampling import SMOTE

# Handle imbalanced data
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_train_scaled, y_train)
```

2. **Multicollinearity**
```python
# Check for multicollinearity
correlation_matrix = X_train.corr()
high_correlation = correlation_matrix > 0.8
```

3. **Feature Scaling**
```python
# Proper scaling pipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
```

## 9. Interpretation Methods

```python
# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_[0]
})
feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)

# Odds ratios
odds_ratios = np.exp(model.coef_[0])
```
