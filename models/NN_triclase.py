import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Dense, Dropout

# Define directories and paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')
data_path = os.path.join(data_dir, 'preprocessed_data_triclase.csv')
models_dir = os.path.join(current_dir, '..', 'models', 'trained')
model_path = os.path.join(models_dir, 'nn_hatespeech_model_triclase.keras')
scaler_path = os.path.join(models_dir, 'nn_hatespeech_scaler_triclase.joblib')

# Load and prepare the data
df = pd.read_csv(data_path)

# Map categorical labels to numeric
category_mapping = {'Neither': 0, 'Hate Speech': 1, 'Offensive Language': 2}
df['Category'] = df['Category'].map(category_mapping)

X = df.drop(columns=['Category'])
y = df['Category']

# Define parameters
best_params = {
    'dense_units_1': 349,
    'dense_units_2': 140,
    'dropout_rate': 0.384,
    'batch_size': 20,
    'epochs': 45
}

# Set up cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize scaler
scaler = StandardScaler()

fold_accuracies = []
fold_conf_matrices = []

# Compute class weights to handle class imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = dict(zip(np.unique(y), class_weights))

# StratifiedKFold Cross-validation loop
for train_index, val_index in skf.split(X, y):
    # Split the data for this fold
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # Scale the data
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # One-hot encode the labels for multi-class classification
    y_train_onehot = to_categorical(y_train, num_classes=3)
    y_val_onehot = to_categorical(y_val, num_classes=3)

    # Define the model architecture
    model = keras.Sequential([
        Dense(best_params['dense_units_1'], activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(best_params['dropout_rate']),
        Dense(best_params['dense_units_2'], activation='relu'),
        Dropout(best_params['dropout_rate']),
        Dense(3, activation='softmax')  # 3 output classes with softmax activation
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(
        X_train_scaled, y_train_onehot,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        validation_data=(X_val_scaled, y_val_onehot),
        verbose=0,
        class_weight=class_weight_dict
    )

    # Evaluate the model on the validation set
    val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val_onehot, verbose=0)
    fold_accuracies.append(val_accuracy)

    # Make predictions and generate confusion matrix
    y_val_pred = np.argmax(model.predict(X_val_scaled), axis=1)
    cm = confusion_matrix(y_val, y_val_pred)
    fold_conf_matrices.append(cm)

# Calculate average accuracy across all folds
avg_accuracy = np.mean(fold_accuracies)
print(f"Average Accuracy from Cross-Validation: {avg_accuracy:.2f}")

# Average confusion matrix across folds
avg_conf_matrix = np.mean(fold_conf_matrices, axis=0).astype(int)

# Display average confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=avg_conf_matrix, display_labels=['Neither', 'Hate Speech', 'Offensive Language'])
disp.plot(cmap='Blues')

# Save the model and scaler
model.save(model_path)
joblib.dump(scaler, scaler_path)

print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")
