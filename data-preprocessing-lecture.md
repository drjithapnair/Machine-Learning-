# Data Preprocessing and Feature Engineering
## Comprehensive Lecture Notes with Examples

## Topic 1: Basic Preprocessing Techniques

### 1.1 Handling Missing Values

Missing values are a common challenge in real-world datasets. There are several strategies to handle them:

#### Methods for Handling Missing Values:

1. **Deletion Methods**
   ```python
   # Drop rows with any missing values
   df.dropna()
   
   # Drop rows where all values are missing
   df.dropna(how='all')
   
   # Drop columns with >50% missing values
   df.dropna(thresh=len(df)*0.5, axis=1)
   ```

2. **Imputation Methods**
   ```python
   # Mean imputation
   df['column'].fillna(df['column'].mean())
   
   # Median imputation (better for skewed data)
   df['column'].fillna(df['column'].median())
   
   # Mode imputation (for categorical data)
   df['category'].fillna(df['category'].mode()[0])
   
   # Forward fill
   df['column'].fillna(method='ffill')
   
   # Backward fill
   df['column'].fillna(method='bfill')
   ```

Example:
```python
import pandas as pd
import numpy as np

# Create sample dataset
data = {
    'age': [25, np.nan, 30, 35, np.nan],
    'salary': [50000, 60000, np.nan, 75000, 80000],
    'department': ['IT', 'HR', np.nan, 'IT', 'Finance']
}
df = pd.DataFrame(data)

# Apply different imputation strategies
df['age'].fillna(df['age'].mean(), inplace=True)  # Mean for numerical
df['department'].fillna(df['department'].mode()[0], inplace=True)  # Mode for categorical
```

### 1.2 Handling Duplicates

Duplicate records can skew analysis and need to be handled appropriately:

```python
# Check for duplicates
df.duplicated().sum()

# Remove duplicate rows
df.drop_duplicates()

# Remove duplicates based on specific columns
df.drop_duplicates(subset=['column1', 'column2'])

# Keep last occurrence instead of first
df.drop_duplicates(keep='last')
```

Example:
```python
# Create sample dataset with duplicates
data = {
    'name': ['John', 'Alice', 'John', 'Bob', 'Alice'],
    'age': [25, 30, 25, 35, 30],
    'city': ['NY', 'LA', 'NY', 'SF', 'LA']
}
df = pd.DataFrame(data)

# Remove duplicates keeping first occurrence
df_clean = df.drop_duplicates()
```

## Topic 2: Handling Outliers and Skewed Data

### 2.1 Outlier Detection and Treatment

#### Using Box Plots and IQR Method
```python
import numpy as np

def detect_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers

# Example usage
outliers = detect_outliers_iqr(df['salary'])
clean_data = df[~outliers]
```

#### Using Z-Score Method
```python
from scipy import stats

def detect_outliers_zscore(data, threshold=3):
    z_scores = np.abs(stats.zscore(data))
    outliers = z_scores > threshold
    return outliers

# Example usage
outliers = detect_outliers_zscore(df['salary'])
clean_data = df[~outliers]
```

### 2.2 Treating Skewed Data

#### Square Root Transformation
```python
# Apply square root transformation
df['salary_sqrt'] = np.sqrt(df['salary'])

# For negative values
df['value_sqrt'] = np.sqrt(df['value'] - df['value'].min())
```

#### Log Transformation
```python
# Natural log transformation
df['salary_log'] = np.log(df['salary'])

# Log1p (log(1+x)) for data with zeros
df['value_log1p'] = np.log1p(df['value'])
```

Example:
```python
import matplotlib.pyplot as plt

# Create skewed data
skewed_data = np.random.lognormal(0, 1, 1000)

# Plot original vs transformed data
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.hist(skewed_data, bins=30)
ax1.set_title('Original Data')

ax2.hist(np.sqrt(skewed_data), bins=30)
ax2.set_title('Square Root Transformed')

ax3.hist(np.log1p(skewed_data), bins=30)
ax3.set_title('Log Transformed')
```

## Topic 3: Feature Engineering

### 3.1 Label Encoding

Used for ordinal categorical variables:
```python
from sklearn.preprocessing import LabelEncoder

# Example usage
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])

# Example mapping: 
# 'High School' -> 0
# 'Bachelor's' -> 1
# 'Master's' -> 2
# 'PhD' -> 3
```

### 3.2 One-Hot Encoding

Used for nominal categorical variables:
```python
# Using pandas
pd.get_dummies(df['department'])

# Using scikit-learn
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder(sparse=False)
dept_encoded = onehot.fit_transform(df[['department']])
```

### 3.3 Feature Scaling

#### Standardization (Z-score scaling)
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['age', 'salary']])

# Results in features with mean=0 and std=1
```

#### Min-Max Normalization
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df[['age', 'salary']])

# Results in features scaled to range [0,1]
```

## Topic 4: Feature Selection

### 4.1 SelectKBest
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top k features based on ANOVA F-value
selector = SelectKBest(score_func=f_classif, k=5)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
```

### 4.2 Correlation Analysis
```python
# Calculate correlation matrix
correlation_matrix = df.corr()

# Plot correlation heatmap
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')

# Remove highly correlated features
def remove_correlated_features(df, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(to_drop, axis=1)
```

### Best Practices and Tips:

1. **Order of Operations**:
   - Handle missing values first
   - Remove duplicates
   - Handle outliers
   - Transform skewed features
   - Encode categorical variables
   - Scale features
   - Select relevant features

2. **Cross-Validation**:
   - Always fit preprocessing steps on training data only
   - Transform validation/test data using parameters from training data

3. **Documentation**:
   - Keep track of all preprocessing steps
   - Document the rationale for each decision
   - Save preprocessing parameters for future use

