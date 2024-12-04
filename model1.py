import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, classification_report
import optuna

# Cargar el dataset preprocesado
df = pd.read_csv('dataset_encoded.csv')

# Separar las variables predictoras y la variable objetivo
X = df.drop('custcat', axis=1)
y = df['custcat']

# Dividir en conjuntos de entrenamiento, validación y prueba
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Función de evaluación mejorada
def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    # Entrenamiento
    model.fit(X_train, y_train)
    
    # Predicciones
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Métricas detalladas
    print(f"\nDetailed Report for {model_name}:")
    print("Classification Report (Test Set):")
    print(classification_report(y_test, y_test_pred))
    
    # Métricas tradicionales
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'val_accuracy': accuracy_score(y_val, y_val_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'train_hamming': hamming_loss(y_train, y_train_pred),
        'val_hamming': hamming_loss(y_val, y_val_pred),
        'test_hamming': hamming_loss(y_test, y_test_pred),
        'train_f1': f1_score(y_train, y_train_pred, average='weighted'),
        'val_f1': f1_score(y_val, y_val_pred, average='weighted'),
        'test_f1': f1_score(y_test, y_test_pred, average='weighted')
    }
    
    # Validación cruzada
    cv_scores = cross_val_score(model, X_train_val, y_train_val, cv=5, scoring='accuracy')
    metrics['cv_mean'] = np.mean(cv_scores)
    metrics['cv_std'] = np.std(cv_scores)
    
    # Detección de overfitting más sofisticada
    overfitting_threshold = 0.05
    if metrics['train_accuracy'] - metrics['val_accuracy'] > overfitting_threshold:
        print(f"⚠️ Posible overfitting en {model_name}")
    
    return metrics

# Funciones objetivo para Optuna
def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 1, 32),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }
    model = RandomForestClassifier(**params, random_state=42)
    return np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))

def objective_gb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
        'max_depth': trial.suggest_int('max_depth', 1, 32),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 4)
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    return np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))

def objective_nn(trial):
    params = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50,50), (100,50)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
        'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-1),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive'])
    }
    model = MLPClassifier(**params, max_iter=1000, random_state=42)
    return np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))

def objective_knn(trial):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 1, 20),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
        'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])
    }
    model = KNeighborsClassifier(**params)
    return np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy'))

# Función para optimizar modelos con Optuna
def optimize_model(objective, n_trials=100, model_name=""):
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    print(f"Best parameters for {model_name}:")
    print(study.best_params)
    print(f"Best value: {study.best_value:.4f}")
    
    return study.best_params

# Optimizar cada modelo
print("Optimizing Random Forest...")
best_params_rf = optimize_model(objective_rf, n_trials=100, model_name="Random Forest")

print("\nOptimizing Gradient Boosting...")
best_params_gb = optimize_model(objective_gb, n_trials=100, model_name="Gradient Boosting")

print("\nOptimizing Neural Network...")
best_params_nn = optimize_model(objective_nn, n_trials=100, model_name="Neural Network")

print("\nOptimizing KNN...")
best_params_knn = optimize_model(objective_knn, n_trials=100, model_name="KNN")

# Crear modelos con los mejores parámetros
best_models = {
    'Random Forest': RandomForestClassifier(**best_params_rf, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(**best_params_gb, random_state=42),
    'Neural Network': MLPClassifier(**best_params_nn, max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(**best_params_knn)
}

# Evaluar los modelos optimizados
results = {}
for name, model in best_models.items():
    print(f"\nEvaluating optimized {name}...")
    results[name] = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, name)

# Imprimir resultados resumidos
for name, result in results.items():
    print(f"\nResumen de {name} optimizado:")
    for metric, value in result.items():
        print(f"{metric}: {value:.4f}")