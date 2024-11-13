# Support Vector Machine (SVM) - Comprehensive Guide

## 1. Introduction to SVM

### Basic Concepts
- Support Vector Machine (SVM) is a supervised learning algorithm
- Creates optimal hyperplane to separate classes
- Maximizes margin between classes
- Can handle both linear and non-linear classification
- Supports binary and multi-class classification

### Key Components
- Support Vectors
- Kernel Functions
- Margin
- Hyperplane
- Decision Boundary

## 2. Mathematical Foundation

### Linear SVM
```
f(x) = wᵀx + b
Decision Rule: sign(wᵀx + b)
```

### Kernel Trick
```
K(x, x') = φ(x)ᵀφ(x')
Common kernels:
- Linear: K(x,x') = xᵀx'
- RBF: K(x,x') = exp(-γ||x-x'||²)
- Polynomial: K(x,x') = (γxᵀx' + r)^d
```

## 3. Practical Implementation

### Example 1: Iris Classification

```python
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load iris dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train SVM model
svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svm_model.predict(X_test_scaled)

# Print results
print("Classification Report:\n", classification_report(y_test, y_pred))

# Visualization of decision boundary (for 2 features)
def plot_decision_boundary(X, y, model, feature_names):
    h = 0.02  # step size in the mesh
    
    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on mesh points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('SVM Decision Boundary')
    plt.show()

# Plot for first two features
X_subset = X_train_scaled[:, [0, 1]]
model_2d = SVC(kernel='rbf', C=1.0)
model_2d.fit(X_subset, y_train)
plot_decision_boundary(X_subset, y_train, model_2d, 
                      [iris.feature_names[0], iris.feature_names[1]])
```

### Example 2: Text Classification

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Sample text data
texts = [
    "This movie is fantastic and amazing",
    "The movie was terrible and boring",
    "Great film, highly recommended",
    "Waste of time, very disappointing",
    "Excellent storyline and acting"
]
labels = [1, 0, 1, 0, 1]  # 1 for positive, 0 for negative

# Create pipeline
text_classifier = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', SVC(kernel='linear'))
])

# Train model
text_classifier.fit(texts, labels)

# Test new reviews
new_reviews = ["This film was wonderful", "I didn't like this movie at all"]
predictions = text_classifier.predict(new_reviews)
```

## 4. Kernel Selection and Parameter Tuning

### Different Kernels and Their Use Cases

```python
# Linear Kernel
svm_linear = SVC(kernel='linear')

# RBF Kernel
svm_rbf = SVC(kernel='rbf', gamma='scale')

# Polynomial Kernel
svm_poly = SVC(kernel='poly', degree=3)

# Custom Kernel
def custom_kernel(X1, X2):
    return np.dot(X1, X2.T) ** 2

svm_custom = SVC(kernel=custom_kernel)
```

### Grid Search for Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'kernel': ['rbf', 'linear', 'poly']
}

# Create grid search object
grid_search = GridSearchCV(
    SVC(),
    param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

# Fit grid search
grid_search.fit(X_train_scaled, y_train)

# Print best parameters
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
```

## 5. Real-world Application: Image Classification

```python
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load digits dataset
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
svm_digits = SVC(kernel='rbf', C=10, gamma='scale')
svm_digits.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = svm_digits.predict(X_test_scaled)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
```

## 6. Advanced Techniques

### Multi-class Classification Strategies

```python
# One-vs-Rest
from sklearn.multiclass import OneVsRestClassifier
ovr_svm = OneVsRestClassifier(SVC(kernel='rbf'))

# One-vs-One
from sklearn.multiclass import OneVsOneClassifier
ovo_svm = OneVsOneClassifier(SVC(kernel='rbf'))
```

### Handling Imbalanced Data

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Create pipeline with SMOTE
imbalanced_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('sampler', SMOTE()),
    ('classifier', SVC(kernel='rbf'))
])
```

## 7. Performance Optimization

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif

# Create pipeline with feature selection
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=5)),
    ('classifier', SVC(kernel='rbf'))
])
```

### Cross-Validation Strategy

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Perform stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=skf)
print("Cross-validation scores:", cv_scores)
print("Mean CV score:", cv_scores.mean())
```

## 8. Model Interpretation

```python
# For linear kernel only
def plot_feature_importance(model, feature_names):
    if model.kernel == 'linear':
        importance = np.abs(model.coef_[0])
        feat_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        })
        feat_importance = feat_importance.sort_values('Importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feat_importance, x='Importance', y='Feature')
        plt.title('Feature Importance (Linear SVM)')
        plt.show()
```

## 9. Common Pitfalls and Solutions

1. **Scaling Issues**
```python
# Proper scaling pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])
```

2. **Overfitting**
```python
# Use cross-validation and regularization
svm_regularized = SVC(C=0.1, kernel='rbf')
```

3. **Long Training Time**
```python
# Use LinearSVC for large datasets
from sklearn.svm import LinearSVC
fast_svm = LinearSVC(dual=False)
```
