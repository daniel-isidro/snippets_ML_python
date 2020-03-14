# Starting Point
```python
# Input
X = df[['TotalSF']] # pandas DataFrame
# Label
y = df["SalePrice"] # pandas Series
```

# Models
## Linear Regression

```python
# Load the library
from sklearn.linear_model import LinearRegression
# Create an instance of the model
reg = LinearRegression()
# Fit the regressor
reg.fit(X,y)
# Do predictions
reg.predict([[2540],[3500],[4000]])
```
## K-near Neighbors

## Decision Tree

# Metrics
## MAE
## MAPE
## RMSE
## Correlation
## Bias

# Evaluation

# Train/Test Split
```python
# Load the library
from sklearn.model_selection import train_test_split

# Create 2 groups each with input and labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# Fit only with training data
reg.fit(X_train,y_train)
```

## Cross Validation
## Grid Search
## Randomized Search
