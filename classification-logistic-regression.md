# Introduction to Classification and Logistic Regression
## Comprehensive Lecture Notes with Examples

## 1. Introduction to Classification

### 1.1 What is Classification?

Classification is a supervised learning task where the model learns to predict categorical outcomes (classes) based on input features.

Key Characteristics:
- Discrete output classes
- Labeled training data
- Probability-based predictions
- Multiple possible algorithms

### 1.2 Types of Classification Problems

1. **Binary Classification**
   - Two possible classes
   - Examples: 
     - Spam detection (spam/not spam)
     - Medical diagnosis (positive/negative)
     - Customer churn (will churn/won't churn)

2. **Multiclass Classification**
   - More than two classes
   - Examples:
     - Image recognition (cat/dog/bird)
     - Document categorization
     - Product classification

3. **Multilabel Classification**
   - Multiple labels per instance
   - Examples:
     - Movie genre classification
     - Image tagging
     - Document keywords

### 1.3 Common Classification Algorithms

1. Logistic Regression
2. Decision Trees
3. Random Forests
4. Support Vector Machines
5. Neural Networks
6. k-Nearest Neighbors

## 2. Logistic Regression

### 2.1 Basic Concepts

Logistic Regression is a statistical model that uses a logistic function to model binary dependent variables.

The logistic function (sigmoid):
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Visualize sigmoid function
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid Function')
plt.grid(True)
plt.show()
```

### 2.2 Mathematical Foundation

1. **Linear Combination**:
   z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

2. **Probability Transformation**:
   P(y=1) = 1 / (1 + e^(-z))

3. **Log Odds (Logit)**:
   log(P(y=1)/(1-P(y=1))) = z

### 2.3 Implementation with scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Sample binary classification dataset
def create_sample_data(n_samples=1000):
    np.random.seed(42)
    
    # Generate two features
    X1 = np.random.normal(0, 1, n_samples)
    X2 = np.random.normal(0, 1, n_samples)
    
    # Generate target based on feature combination
    z = 1.5 * X1 - 2 * X2
    prob = sigmoid(z)
    y = (np.random.random(n_samples) < prob).astype(int)
    
    return pd.DataFrame({'feature1': X1, 'feature2': X2}), y

# Create and prepare data
X, y = create_sample_data()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)
```

### 2.4 Model Evaluation

1. **Confusion Matrix**
```python
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

plot_confusion_matrix(y_test, y_pred)
```

2. **Classification Metrics**
```python
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

3. **ROC Curve**
```python
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

plot_roc_curve(y_test, y_pred_prob)
```

### 2.5 Complete Example: Customer Churn Prediction

```python
# Create sample customer churn dataset
def create_churn_data(n_samples=1000):
    np.random.seed(42)
    
    data = {
        'usage_minutes': np.random.normal(600, 200, n_samples),
        'contract_length': np.random.randint(1, 24, n_samples),
        'monthly_charges': np.random.normal(70, 30, n_samples),
        'service_calls': np.random.poisson(3, n_samples)
    }
    
    # Generate churn based on features
    z = (
        -0.003 * data['usage_minutes'] +
        -0.1 * data['contract_length'] +
        0.03 * data['monthly_charges'] +
        0.3 * data['service_calls']
    )
    
    prob = sigmoid(z)
    data['churn'] = (np.random.random(n_samples) < prob).astype(int)
    
    return pd.DataFrame(data)

# Load and prepare data
df = create_churn_data()

# Split features and target
X = df.drop('churn', axis=1)
y = df['churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)

# Analyze feature importance
def plot_feature_importance(model, feature_names):
    importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': np.abs(model.coef_[0])
    })
    importance = importance.sort_values('coefficient', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='coefficient', y='feature', data=importance)
    plt.title('Feature Importance')
    plt.show()

plot_feature_importance(model, X.columns)

# Print model performance
print("\nModel Performance:")
print("\nConfusion Matrix:")
plot_confusion_matrix(y_test, y_pred)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC Curve:")
plot_roc_curve(y_test, y_pred_prob)
```

### 2.6 Model Interpretation

1. **Coefficients**
```python
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")
```

2. **Odds Ratios**
```python
odds_ratios = np.exp(model.coef_[0])
for feature, odds in zip(X.columns, odds_ratios):
    print(f"{feature} odds ratio: {odds:.4f}")
```

3. **Probability Predictions**
```python
# Example of predicting probability for a new customer
new_customer = np.array([[500, 12, 65, 2]])  # Example values
new_customer_scaled = scaler.transform(new_customer)
prob = model.predict_proba(new_customer_scaled)[0][1]
print(f"Churn probability: {prob:.2%}")
```

## 3. Best Practices and Tips

### 3.1 Data Preprocessing

1. **Handle Missing Values**
```python
# Check missing values
print(df.isnull().sum())

# Handle missing values
df.fillna(df.mean(), inplace=True)  # For numerical
df.fillna(df.mode().iloc[0], inplace=True)  # For categorical
```

2. **Handle Class Imbalance**
```python
from sklearn.utils.class_weight import compute_class_weight

# Calculate class weights
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(y_train),
    y=y_train
)

# Use in model
model = LogisticRegression(class_weight='balanced')
```

### 3.2 Feature Engineering

1. **Feature Scaling**
```python
# Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

2. **Handling Categorical Variables**
```python
# One-hot encoding
X_encoded = pd.get_dummies(X, columns=['categorical_column'])
```

### 3.3 Model Selection and Validation

1. **Cross-Validation**
```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"Cross-validation scores: {scores}")
print(f"Average CV score: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
```

2. **Hyperparameter Tuning**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

grid_search = GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    scoring='roc_auc'
)

grid_search.fit(X_train_scaled, y_train)
print(f"Best parameters: {grid_search.best_params_}")
```

### 3.4 Model Deployment Considerations

1. **Save and Load Model**
```python
import joblib

# Save model
joblib.dump(model, 'logistic_regression_model.pkl')

# Load model
loaded_model = joblib.load('logistic_regression_model.pkl')
```

2. **Model Pipeline**
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

