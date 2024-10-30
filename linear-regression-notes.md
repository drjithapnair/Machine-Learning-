# Introduction to Supervised Learning: Linear Regression
## Comprehensive Lecture Notes with Case Study

## 1. Introduction to Supervised Learning

### 1.1 What is Supervised Learning?

Supervised Learning is a type of machine learning where the model learns from labeled data. The algorithm learns to map input features to known output values, allowing it to make predictions on new, unseen data.

Key components:
- Input features (X): Independent variables
- Target variable (y): Dependent variable/outcome
- Training data: Labeled dataset used for learning
- Testing data: Used to evaluate model performance

### 1.2 Types of Supervised Learning:

1. **Classification**
   - Predicts categorical outcomes
   - Example: Spam detection (spam/not spam)

2. **Regression**
   - Predicts continuous numerical values
   - Example: House price prediction

## 2. Linear Regression

### 2.1 Basic Concepts

Linear Regression models the relationship between:
- Dependent variable (y)
- One or more independent variables (x)

The basic equation:
y = β₀ + β₁x + ε

Where:
- β₀: y-intercept
- β₁: slope coefficient
- ε: error term

### 2.2 Types of Linear Regression

1. **Simple Linear Regression**
   - One independent variable
   ```python
   # Example
   from sklearn.linear_model import LinearRegression
   X = df[['square_feet']]
   y = df['price']
   model = LinearRegression()
   model.fit(X, y)
   ```

2. **Multiple Linear Regression**
   - Multiple independent variables
   ```python
   # Example
   X = df[['square_feet', 'bedrooms', 'age']]
   y = df['price']
   model = LinearRegression()
   model.fit(X, y)
   ```

### 2.3 Assumptions of Linear Regression

1. **Linearity**
   - Relationship between X and y is linear
   ```python
   import seaborn as sns
   
   # Check linearity
   sns.scatterplot(data=df, x='square_feet', y='price')
   plt.title('Checking Linearity Assumption')
   plt.show()
   ```

2. **Independence**
   - Observations are independent

3. **Homoscedasticity**
   - Constant variance of residuals
   ```python
   # Check homoscedasticity
   plt.scatter(model.predict(X), model.predict(X) - y)
   plt.axhline(y=0, color='r', linestyle='--')
   plt.title('Residual Plot')
   plt.show()
   ```

4. **Normality**
   - Residuals are normally distributed
   ```python
   # Check normality of residuals
   from scipy import stats
   
   residuals = model.predict(X) - y
   stats.probplot(residuals, dist="norm", plot=plt)
   plt.title('Q-Q Plot of Residuals')
   plt.show()
   ```

### 2.4 Model Evaluation Metrics

1. **R-squared (Coefficient of Determination)**
```python
r2 = model.score(X, y)
print(f"R-squared: {r2:.4f}")
```

2. **Mean Squared Error (MSE)**
```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y, model.predict(X))
print(f"MSE: {mse:.4f}")
```

3. **Root Mean Squared Error (RMSE)**
```python
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")
```

4. **Mean Absolute Error (MAE)**
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y, model.predict(X))
print(f"MAE: {mae:.4f}")
```

## 3. Case Study: House Price Prediction

### 3.1 Problem Statement
Predict house prices based on various features like square footage, number of bedrooms, location, etc.

### 3.2 Complete Implementation

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load and Prepare Data
def load_house_data():
    # Sample house data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'square_feet': np.random.normal(2000, 500, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age': np.random.randint(0, 50, n_samples),
        'location_score': np.random.uniform(1, 10, n_samples)
    }
    
    # Generate price with some noise
    price = (
        200 * data['square_feet'] +
        50000 * data['bedrooms'] +
        75000 * data['bathrooms'] -
        1000 * data['age'] +
        25000 * data['location_score'] +
        np.random.normal(0, 50000, n_samples)
    )
    
    data['price'] = price
    
    return pd.DataFrame(data)

# Load data
df = load_house_data()

# 2. Exploratory Data Analysis
def perform_eda(df):
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlations')
    plt.show()
    
    # Distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], bins=50)
    plt.title('Distribution of House Prices')
    plt.show()
    
    # Feature relationships
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    sns.scatterplot(data=df, x='square_feet', y='price', ax=axes[0,0])
    sns.scatterplot(data=df, x='bedrooms', y='price', ax=axes[0,1])
    sns.scatterplot(data=df, x='age', y='price', ax=axes[1,0])
    sns.scatterplot(data=df, x='location_score', y='price', ax=axes[1,1])
    plt.tight_layout()
    plt.show()

# 3. Data Preprocessing
def preprocess_data(df):
    # Split features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# 4. Model Training and Evaluation
def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'R2 Score (Train)': r2_score(y_train, y_pred_train),
        'R2 Score (Test)': r2_score(y_test, y_pred_test),
        'RMSE (Train)': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'RMSE (Test)': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'MAE (Train)': mean_absolute_error(y_train, y_pred_train),
        'MAE (Test)': mean_absolute_error(y_test, y_pred_test)
    }
    
    return model, metrics, y_pred_test

# 5. Model Diagnostics
def plot_diagnostics(y_test, y_pred):
    # Residual plot
    residuals = y_test - y_pred
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.subplot(1, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    plt.tight_layout()
    plt.show()

# Execute the analysis
df = load_house_data()
perform_eda(df)

X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(df)
model, metrics, y_pred_test = train_and_evaluate_model(
    X_train_scaled, X_test_scaled, y_train, y_test
)

# Print results
print("\nModel Coefficients:")
for feature, coef in zip(df.drop('price', axis=1).columns, model.coef_):
    print(f"{feature}: {coef:.2f}")

print("\nModel Performance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.2f}")

# Plot diagnostics
plot_diagnostics(y_test, y_pred_test)

# 6. Example Prediction
def make_prediction(model, scaler, features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    return prediction

# Example house
sample_house = [2500, 3, 2, 10, 8]  # square_feet, bedrooms, bathrooms, age, location_score
predicted_price = make_prediction(model, scaler, sample_house)
print(f"\nPredicted price for sample house: ${predicted_price:,.2f}")
```

### 3.3 Interpreting Results

1. **Model Performance**
   - R-squared value interpretation
   - RMSE in context of house prices
   - MAE for average prediction error

2. **Feature Importance**
   - Coefficient interpretation
   - Most influential features
   - Feature relationship analysis

3. **Model Diagnostics**
   - Residual analysis
   - Assumption verification
   - Potential improvements

### 3.4 Business Insights

1. **Key Price Drivers**
   - Square footage impact
   - Location importance
   - Age effect

2. **Practical Applications**
   - Price estimation tool
   - Market analysis
   - Investment decisions

3. **Limitations**
   - Linear assumptions
   - Market variability
   - External factors

## 4. Best Practices and Tips

1. **Data Preprocessing**
   - Handle missing values
   - Remove outliers
   - Scale features appropriately

2. **Model Development**
   - Start simple
   - Add complexity gradually
   - Cross-validate results

3. **Model Evaluation**
   - Use multiple metrics
   - Compare with baseline
   - Consider business context

4. **Documentation**
   - Document assumptions
   - Record preprocessing steps
   - Note model limitations

