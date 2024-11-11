import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import optuna
import os

# Set up paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', 'data')
data_path = os.path.join(data_dir, 'preprocessed_data.csv')

models_dir = os.path.join(current_dir, '..', 'models')
model_path = os.path.join(models_dir, 'nn_hatespeech_best_model.joblib')  # Best model from Grid Search
scaler_path = os.path.join(models_dir, 'nn_hatespeech_scaler.joblib')

# Load the dataset
df = pd.read_csv(data_path)

# Extract features and target
X = df.drop(columns=['IsHateSpeech'])
y = df['IsHateSpeech']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def objective(trial):
    n_layers = trial.suggest_int('n_layers', 3, 7)  # Número de capas ocultas
    n_units = trial.suggest_int('n_units', 16, 512)   # Número de neuronas por capa
    activation = trial.suggest_categorical('activation', ['relu'])  # Funciones de activación
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)  # Tasa de dropout
    batch_size = trial.suggest_int('batch_size', 16, 512)  # Tamaño del lote
    epochs = trial.suggest_int('epochs', 100, 300)  # Número de épocas

    # Crear el modelo
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(X_train.shape[1],)))

    for _ in range(n_layers):
        model.add(keras.layers.Dense(n_units, activation=activation))
        model.add(keras.layers.Dropout(dropout_rate))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    accuracy = model.evaluate(X_test, y_test, verbose=0)[1]
    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Mejores hiperparámetros: ", study.best_params)
print("Mejor precisión: ", study.best_value)