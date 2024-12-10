'''
Predicting Credit Card Approvals
Author: Henry Ha
'''

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load the dataset
cc_apps = pd.read_csv("cc_approvals.csv", header=None)
cc_apps.head()

# Replace the '?'s with NaN in dataset
cc_apps.replace("?", np.NaN, inplace=True)

# Iterate over each column of cc_apps_nans_replaced and impute the most frequent value for object data types and the mean for numeric data types
for col in cc_apps.columns:
    if cc_apps[col].dtype == 'object':
        cc_apps[col].fillna(cc_apps[col].mode()[0], inplace=True)
    else:
        cc_apps[col].fillna(cc_apps[col].mean(), inplace=True)

# Summary statistics
cc_apps.describe()

# Plot the distribution of the categorical column (column 0)
ax = cc_apps[0].value_counts().plot(kind='bar', color='skyblue')

# Add a title and axis labels
plt.title('Distribution of Column 0', fontsize=14)
plt.xlabel('Categories', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Add value labels on top of each bar
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10)

# Show the plot
plt.show()

#TODO: Data preprocessing

# Perform one-hot encoding
cc_apps_encoded = pd.get_dummies(cc_apps, drop_first=True)

# Feature scaling

# Extract features and target variable
X = cc_apps_encoded.iloc[:, :-1].values
y = cc_apps_encoded.iloc[:, -1].values

# Initialize and apply StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)

#TODO: Build the model

# Instantiate the logistic regression model
logreg = LogisticRegression()

# Train the model on the training data
logreg.fit(X_train, y_train)

# Predict labels for the training set
y_train_pred = logreg.predict(X_train)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_train, y_train_pred)

print("Confusion Matrix:\n", conf_matrix)

# Define the parameter grid
param_grid = {
    'tol': [0.01, 0.001, 0.0001],
    'max_iter': [100, 150, 200]
}

# Set up GridSearchCV
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Train the grid search model
grid_model_result = grid_model.fit(X_train, y_train)

# Extract the best score and parameters
best_train_score = grid_model_result.best_score_
best_train_params = grid_model_result.best_params_

print(f"Best Training Score: {best_train_score}")
print(f"Best Parameters: {best_train_params}")

# Extract the best model
best_model = grid_model_result.best_estimator_

# Evaluate the best model on the test set
best_model = grid_model_result.best_estimator_
test_accuracy = best_model.score(X_test, y_test)

print(f"Test Accuracy: {test_accuracy:.2f}")
