import os
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')
data_path = os.path.join(data_dir, 'preprocessed_data.csv')

models_dir = os.path.join(current_dir, '..', 'models', 'trained')
model_path = os.path.join(models_dir, 'nn_hatespeech_model.keras')  # Saving as HDF5 format for TensorFlow/Keras
scaler_path = os.path.join(models_dir, 'nn_hatespeech_scaler.joblib')

# Load the dataset
df = pd.read_csv(data_path)

# Extract features and target
X = df.drop(columns=['IsHateSpeech'])
y = df['IsHateSpeech']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the scaler and scale the features
scaler = StandardScaler()

# Fit the scaler on the training data and transform both train and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_params = {
    'n_layers': 3,
    'n_units': 93,
    'activation': 'relu',
    'dropout_rate': 0.22,
    'batch_size': 223, 
    'epochs': 123
}

fold_accuracies = []
fold_conf_matrices = []

# StratifiedKFold Cross-validation loop
for train_index, val_index in skf.split(X, y):
    # Split the data for this fold
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Scale the data
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Define the model architecture
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(X_train_scaled.shape[1],)))  # Adjust input shape for scaled data

    for _ in range(best_params['n_layers']):
        model.add(keras.layers.Dense(best_params['n_units'], activation=best_params['activation']))
        model.add(keras.layers.Dropout(best_params['dropout_rate']))

    model.add(keras.layers.Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model for this fold
    model.fit(X_train_scaled, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'], verbose=0)

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val, verbose=0)
    fold_accuracies.append(val_accuracy)

    # Make predictions and generate confusion matrix
    y_val_pred = (model.predict(X_val_scaled) > 0.5).astype("int32")
    cm = confusion_matrix(y_val, y_val_pred)
    fold_conf_matrices.append(cm)

# Calculate average accuracy across all folds
avg_accuracy = np.mean(fold_accuracies)
print(f"Average Accuracy from Cross-Validation: {avg_accuracy:.2f}")

# Average confusion matrix across folds (you can average them, or choose to display individual ones)
avg_conf_matrix = np.mean(fold_conf_matrices, axis=0).astype(int)

# Display average confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=avg_conf_matrix, display_labels=['Not Hate Speech', 'Hate Speech'])
disp.plot(cmap='Blues')

model.save(model_path)  

# Save the scaler
joblib.dump(scaler, scaler_path)

print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")