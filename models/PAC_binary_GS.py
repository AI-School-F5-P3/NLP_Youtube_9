from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')
data_path = os.path.join(data_dir, 'preprocessed_data.csv')

# Load the preprocessed data
df = pd.read_csv(data_path)

# Extract the features and target
X = df.drop(columns=['IsHateSpeech'])
y = df['IsHateSpeech']

# Split the data into train and test sets
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization strength
    'max_iter': [1000, 2000, 3000],  # Maximum iterations
    'tol': [1e-3, 1e-4, 1e-5],  # Tolerance for stopping criterion
    'early_stopping': [True, False],  # Whether to stop early when convergence is reached
}

# Initialize the PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=pac, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding cross-validation score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f'Test set accuracy: {accuracy:.2f}')