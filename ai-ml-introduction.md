# Introduction to Artificial Intelligence and Machine Learning
## Comprehensive Lecture Notes with Examples

## 1. Understanding Artificial Intelligence (AI)

### 1.1 What is Artificial Intelligence?

Artificial Intelligence refers to the simulation of human intelligence by machines programmed to think and learn like humans. It encompasses systems that can:

- Mimic human cognitive functions
- Learn from experience
- Adapt to new inputs
- Perform human-like tasks

### 1.2 Key Components of AI:

1. **Learning**: Acquiring information and rules for using the information
2. **Reasoning**: Using rules to reach approximate or definite conclusions
3. **Self-correction**: Learning from mistakes and adjusting approaches
4. **Perception**: Processing and analyzing inputs from the environment
5. **Language Understanding**: Processing and engaging with natural language

### 1.3 Types of AI:

1. **Narrow/Weak AI**
   - Designed for specific tasks
   - Examples: Siri, chess programs, recommendation systems

2. **General/Strong AI**
   - Capable of performing any intellectual task
   - Currently theoretical and not yet achieved

3. **Super AI**
   - Surpasses human intelligence across all domains
   - Theoretical concept for future development

## 2. Machine Learning: A Subset of AI

### 2.1 Definition and Core Concepts

Machine Learning is a subset of AI that focuses on developing systems that can learn from and make decisions based on data. It enables computers to:

- Learn without explicit programming
- Improve from experience
- Find patterns in data
- Make data-driven predictions

### 2.2 Types of Machine Learning

1. **Supervised Learning**
   ```python
   # Example of supervised learning
   from sklearn.linear_model import LinearRegression
   
   # Create and train model
   model = LinearRegression()
   model.fit(X_train, y_train)  # X contains features, y contains target
   
   # Make predictions
   predictions = model.predict(X_test)
   ```

   Common applications:
   - Classification
   - Regression
   - Image recognition
   - Spam detection

2. **Unsupervised Learning**
   ```python
   # Example of unsupervised learning
   from sklearn.cluster import KMeans
   
   # Create and train clustering model
   kmeans = KMeans(n_clusters=3)
   kmeans.fit(X)  # Only features, no target variable
   
   # Get cluster assignments
   clusters = kmeans.predict(X)
   ```

   Common applications:
   - Clustering
   - Dimensionality reduction
   - Association rules
   - Anomaly detection

3. **Reinforcement Learning**
   - Learning through trial and error
   - Agents learn optimal actions through rewards/penalties
   - Examples: Game playing AI, robotics

## 3. Machine Learning Workflow

### 3.1 Overview of ML Pipeline

1. Data Collection
2. Data Preprocessing
3. Data Splitting
4. Model Training
5. Model Evaluation
6. Model Deployment

### 3.2 Train-Test Split: In-Depth Analysis

#### Why Split the Data?

The train-test split is crucial for:
- Evaluating model performance on unseen data
- Preventing overfitting
- Providing unbiased evaluation of model performance

#### Implementation using sklearn:

```python
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Sample dataset
data = pd.DataFrame({
    'feature1': np.random.random(1000),
    'feature2': np.random.random(1000),
    'target': np.random.randint(0, 2, 1000)
})

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Basic train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    # 20% for testing
    random_state=42,  # for reproducibility
    stratify=y        # maintain class distribution
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
```

#### Advanced Train-Test Split Techniques:

1. **Stratified Split**
```python
# Ensuring balanced class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y
)
```

2. **Multiple Splits (Cross-Validation)**
```python
from sklearn.model_selection import KFold

# Create 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate through folds
for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

3. **Time-Based Split (for Time Series)**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

### 3.3 Best Practices for Train-Test Splits

1. **Choosing Split Ratio**
   - Common ratios: 80:20, 70:30, 60:40
   - Factors to consider:
     - Dataset size
     - Problem complexity
     - Model complexity

2. **Data Independence**
   ```python
   # Shuffle data before splitting
   X_train, X_test, y_train, y_test = train_test_split(
       X, y,
       test_size=0.2,
       shuffle=True,
       random_state=42
   )
   ```

3. **Handling Imbalanced Data**
   ```python
   # Check class distribution
   print("Class distribution before split:")
   print(y.value_counts(normalize=True))
   
   # Use stratified split
   X_train, X_test, y_train, y_test = train_test_split(
       X, y,
       test_size=0.2,
       stratify=y
   )
   
   print("\nClass distribution in training set:")
   print(y_train.value_counts(normalize=True))
   ```

4. **Complete Example with Validation Set**
```python
# Split into train+validation and test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Split remaining data into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")
```

### 3.4 Common Pitfalls to Avoid

1. **Data Leakage**
   - Perform feature scaling after splitting
   - Handle missing values after splitting
   - Create feature engineering pipelines after splitting

2. **Temporal Dependency**
   - Respect time order for time series data
   - Use appropriate splitting techniques for temporal data

3. **Small Test Sets**
   - Ensure test set is large enough for reliable evaluation
   - Consider cross-validation for small datasets

4. **Non-Representative Splits**
   - Use stratification for categorical targets
   - Ensure all classes are represented in both sets

## 4. Practical Implementation Guide

### 4.1 Complete ML Workflow Example
```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 2. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 4. Make predictions
y_pred = model.predict(X_test_scaled)

# 5. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")
```

### 4.2 Visualization of Train-Test Split
```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_split_distribution(y_train, y_test):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(y_train)
    plt.title('Training Set Distribution')
    
    plt.subplot(1, 2, 2)
    sns.countplot(y_test)
    plt.title('Test Set Distribution')
    
    plt.tight_layout()
    plt.show()
```

