import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns  # Assuming seaborn is installed

# Load the dataset
dataset = pd.read_csv('C:/Users/tannu/student_sleep_patterns.csv')

# Data Cleaning
#   - Identify missing values
print(dataset.isnull().sum())

# Option 1: Handle missing values before outlier detection
cleaned_dataset = dataset.dropna(subset=['Sleep_Duration', 'Weekday_Sleep_End', 'Weekday_Sleep_Start'])

# Option 2: Handle missing values with appropriate strategies (e.g., mean, median)

#   - Imputation
for col in ['Sleep_Duration', 'Weekday_Sleep_End', 'Weekday_Sleep_Start']:
    if dataset[col].dtype == 'object':  # Handle categorical columns
        # Use the recommended way to fill missing values
        dataset[col] = dataset[col].fillna(dataset[col].mode()[0])
    else:
        # Use the recommended way to fill missing values
        dataset[col] = dataset[col].fillna(dataset[col].median())

# # Outlier Detection and Handling (using cleaned data)
# Uncomment if you need to handle outliers in 'Sleep_Duration'
Q1 = cleaned_dataset['Sleep_Duration'].quantile(0.25)
Q3 = cleaned_dataset['Sleep_Duration'].quantile(0.75)
IQR = Q3 - Q1
cleaned_dataset = cleaned_dataset[~((cleaned_dataset['Sleep_Duration'] < (Q1 - 1.5 * IQR)) | (cleaned_dataset['Sleep_Duration'] > (Q3 + 1.5 * IQR)))]

# Data Preprocessing
# Assuming 'Weekday_Sleep_End' and 'Weekday_Sleep_Start' are in a format that can be subtracted to get hours slept.
# If not, ensure they are datetime objects first.
cleaned_dataset['Sleep_Efficiency'] = cleaned_dataset['Sleep_Duration'] / (cleaned_dataset['Weekday_Sleep_End'] - cleaned_dataset['Weekday_Sleep_Start'])

# Feature Scaling
scaler = StandardScaler()
# List of columns to scale
columns_to_scale = ['Sleep_Duration', 'Study_Hours', 'Screen_Time', 'Caffeine_Intake', 'Physical_Activity']
# Scale the selected columns
cleaned_dataset[columns_to_scale] = scaler.fit_transform(cleaned_dataset[columns_to_scale])

# Visualization
plt.hist(cleaned_dataset['Sleep_Duration'], bins=20)
plt.xlabel('Sleep_Duration')
plt.ylabel('Frequency')
plt.title('Histogram of Sleep Duration')
plt.show()

sns.scatterplot(x='Sleep_Duration', y='Study_Hours', data=cleaned_dataset)
plt.xlabel('Sleep_Duration')
plt.ylabel('Study_Hours')
plt.title('Sleep_Duration vs. Study_Hours')
plt.show()

# Analysis
print(cleaned_dataset.head())  # Print the first 5 rows
print(cleaned_dataset.describe())  # Statistical summary

print("\n\n SIMPLE LINEAR REGRESSION")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data from the image (replace with your actual data loading method)
# Assuming the data is in a CSV file named 'data.csv'
data = pd.read_csv('C:/Users/tannu/student_sleep_patterns.csv')

# Select the features and target variable
X = data['Screen_Time']  # Independent variable (feature)
y = data['Study_Hours']  # Dependent variable (target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train.values.reshape(-1, 1), y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test.values.reshape(-1, 1))

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Evaluation of Model Performance for SImple Linear Regression Model")
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualize the model
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Screen_Time')
plt.ylabel('Study_Hours')
plt.title('Simple Linear Regression Model')
plt.show()

print("\n\n MULTIPLE LINEAR REGRESSION")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data from the image (replace with your actual data loading method)
# Assuming the data is in a CSV file named 'student_sleep_patterns.csv'
data = pd.read_csv('student_sleep_patterns.csv')

# Select the features and target variable
X = data[['Screen_Time', 'Caffeine_Intake', 'Physical_Activity', 'Sleep_Quality', 'Weekday_Sleep_Start', 'Weekend_Sleep_Start', 'Weekday_Sleep_End', 'Weekend_Sleep_End']]  # Independent variables (features)
y = data['Study_Hours']  # Dependent variable (target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a multiple linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Evaluation of Model Performance for Multiple Linear Regression Model")
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Visualize the relationship between predicted and actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Study_Hours')
plt.ylabel('Predicted Study_Hours')
plt.title('Actual vs. Predicted Study_ Hours(Multiple Linear Regression)')
plt.show()

# Visualize the residuals (difference between actual and predicted values)
plt.scatter(y_pred, y_test - y_pred)
plt.xlabel('Predicted Study_Hours')
plt.ylabel('Residuals')
plt.title('Residual Plot(Multiple Linear Regression)')
plt.show()

# Plotting the results
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs. predicted values
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Study_Hours')
plt.scatter(range(len(y_test)), y_pred, color='red', label='Predicted Study_Hours')

# Plotting the best fit line
plt.plot(range(len(y_test)), y_pred, color='green', linestyle='--', label='Best Fit Line')

plt.xlabel('Index')
plt.ylabel('Study_Hours')
plt.title('Actual vs. Predicted Study_ Hours with Best Fit Line (Multiple Linear Regression)')
plt.legend()
plt.show()

#POLYNOMIAL REGRESSION
print("\n\n POLYNOMIAL REGRESSION")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
# Replace with the correct file path if needed
dataset = pd.read_csv('C:/Users/tannu/student_sleep_patterns.csv')

# Data Preprocessing
# Drop rows with missing values in the target variable 'Sleep_Duration'
dataset.dropna(subset=['Sleep_Duration'], inplace=True)

# Select numerical features for polynomial regression
features = ['Age', 'Study_Hours', 'Screen_Time', 'Caffeine_Intake', 'Physical_Activity', 'Sleep_Quality']
X = dataset[features]
y = dataset['Sleep_Duration']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Polynomial Regression (degree can be adjusted, e.g., 2 or 3)
degree = 2  # You can change the degree for experimentation
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predictions
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Model Performance Metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')
print(f'Train R^2: {train_r2}')
print(f'Test R^2: {test_r2}')

# Visualization of Model Performance
plt.figure(figsize=(10, 6))

# Scatter plot of actual vs predicted values for the test set
plt.scatter(y_test, y_test_pred, color='blue', edgecolor='k', alpha=0.6, label='Test Data')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linewidth=2, linestyle='--', label='Best Fit Line')
plt.xlabel('Actual Sleep_Duration')
plt.ylabel('Predicted Sleep_Duration')
plt.title('Actual vs Predicted Sleep_Duration (Polynomial Regression)')
plt.legend()
plt.show()

# Visualization of residuals
plt.figure(figsize=(10, 6))
residuals = y_test - y_test_pred
plt.scatter(y_test, residuals, color='purple', edgecolor='k', alpha=0.6)
plt.axhline(y=0, color='red', linewidth=2, linestyle='--')
plt.xlabel('Actual Sleep_Duration')
plt.ylabel('Residuals')
plt.title('Residual Plot (Polynomial Regression)')
plt.show()



