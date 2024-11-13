# Classification Evaluation Metrics
## Introduction
Evaluation metrics are essential tools for assessing the performance of classification models. Different metrics are suitable for different scenarios, particularly when dealing with various types of class imbalance and different error costs.

## Confusion Matrix: The Foundation
### Basic Structure
```
                  Predicted
Actual    | Positive | Negative |
-------------------------------
Positive  |    TP    |    FN    |
Negative  |    FP    |    TN    |
```

- **True Positives (TP)**: Correctly identified positive cases
- **True Negatives (TN)**: Correctly identified negative cases
- **False Positives (FP)**: Incorrectly identified positive cases (Type I error)
- **False Negatives (FN)**: Incorrectly identified negative cases (Type II error)

```python
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print("----------------")
    print(f"            Predicted")
    print(f"Actual     {labels[0]}  {labels[1]}")
    print(f"{labels[0]}      {cm[0][0]}      {cm[0][1]}")
    print(f"{labels[1]}      {cm[1][0]}      {cm[1][1]}")
```

## Basic Metrics
### 1. Accuracy
- **Definition**: Proportion of correct predictions among total predictions
- **Formula**: (TP + TN) / (TP + TN + FP + FN)
- **Use Case**: Balanced datasets with equal error costs
- **Limitations**: Misleading for imbalanced datasets

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true, y_pred)
```

### 2. Precision (Positive Predictive Value)
- **Definition**: Proportion of correct positive predictions
- **Formula**: TP / (TP + FP)
- **Use Case**: When false positives are costly
- **Example**: Spam detection (avoiding marking legitimate emails as spam)

```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
```

### 3. Recall (Sensitivity, True Positive Rate)
- **Definition**: Proportion of actual positives correctly identified
- **Formula**: TP / (TP + FN)
- **Use Case**: When false negatives are costly
- **Example**: Disease detection (avoiding missing actual cases)

```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
```

### 4. F1-Score
- **Definition**: Harmonic mean of precision and recall
- **Formula**: 2 * (Precision * Recall) / (Precision + Recall)
- **Use Case**: When balance between precision and recall is needed
```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
```

## Advanced Metrics
### 1. ROC Curve and AUC
- **ROC**: Plot of True Positive Rate vs False Positive Rate
- **AUC**: Area Under the ROC Curve
- **Interpretation**: 
  - AUC = 1.0: Perfect classifier
  - AUC = 0.5: Random classifier
  - AUC > 0.7: Generally considered good

```python
from sklearn.metrics import roc_curve, auc

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc
```

### 2. Precision-Recall Curve
- **Definition**: Plot of Precision vs Recall at different thresholds
- **Use Case**: Imbalanced datasets where ROC curves might be misleading
- **AUC-PR**: Area under the Precision-Recall curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_pr_curve(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    return precision, recall, ap
```

### 3. Cohen's Kappa
- **Definition**: Agreement between predicted and actual labels, accounting for chance
- **Range**: -1 to 1 (1 being perfect agreement)
- **Use Case**: When you want to account for chance agreement

```python
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(y_true, y_pred)
```

## Metrics for Multi-class Classification
### 1. Micro-averaging
- Aggregate TP, FP, FN across all classes
- Treats all instances equally

### 2. Macro-averaging
- Calculate metric for each class and average
- Treats all classes equally

### 3. Weighted-averaging
- Similar to macro-averaging but weighted by class frequency

```python
# Example of different averaging methods
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, 
    y_pred,
    average='micro'  # or 'macro', 'weighted'
)
```

## Practical Implementation
```python
def comprehensive_evaluation(y_true, y_pred, y_pred_proba=None):
    """
    Comprehensive evaluation of classification results
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # ROC-AUC (if probabilities available)
    roc_auc = None
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'cohen_kappa': kappa,
        'roc_auc': roc_auc
    }
```

## Best Practices
1. **Choose Appropriate Metrics**
   - Consider class distribution
   - Consider cost of different types of errors
   - Use multiple metrics for comprehensive evaluation

2. **Use Cross-Validation**
   - Evaluate metrics across multiple folds
   - Report mean and standard deviation

3. **Statistical Significance**
   - Use confidence intervals
   - Perform statistical tests when comparing models

4. **Visualization**
   - Plot confusion matrices
   - Create ROC and PR curves
   - Show metric distributions across CV folds

## Common Pitfalls
1. **Relying Solely on Accuracy**
   - Can be misleading for imbalanced datasets
   - Doesn't capture all aspects of performance

2. **Ignoring Class Distribution**
   - Different metrics needed for imbalanced cases
   - Consider class weights in evaluation

3. **Not Using Probability Thresholds**
   - Fixed 0.5 threshold may not be optimal
   - Consider threshold tuning

4. **Inappropriate Averaging**
   - Wrong choice of micro/macro/weighted averaging
   - Not considering class importance

## Conclusion
Proper evaluation is crucial for:
- Model selection
- Performance assessment
- Business decision making
- Model monitoring and maintenance

Choose metrics based on:
- Problem characteristics
- Business requirements
- Data distribution
- Cost of different types of errors
