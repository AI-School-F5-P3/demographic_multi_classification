import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sdv.tabular import CTGAN

relu = np.maximum

# Distribuciones de hiperparámetros para búsqueda
rf_param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

gb_param_dist = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.3],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

nn_param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'adaptive']
}

knn_param_dist = {
    'n_neighbors': [3, 5, 7, 9, 11],  # Número de vecinos
    'weights': ['uniform', 'distance'],  # Ponderación de vecinos
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],  # Algoritmo de búsqueda
    'metric': ['euclidean', 'manhattan', 'minkowski']  # Métricas de distancia
}

# Cargar el dataset preprocesado
df = pd.read_csv('umap_results.csv')
df.head()


# Asumiendo que 'df' es tu DataFrame original
model = CTGAN()
model.fit(df)

# Genera el mismo número de muestras que el original
synthetic_data = model.sample(num_rows=len(df))
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

# Modelos con búsqueda de hiperparámetros
models = {
    'Random Forest': GridSearchCV(
        RandomForestClassifier(random_state=42), 
        rf_param_dist,  
        cv=5,
        verbose=1,  
        n_jobs=-1
    ),
    'Gradient Boosting': GridSearchCV(
        GradientBoostingClassifier(random_state=42), 
        gb_param_dist, 
        cv=5, 
        verbose=1,  
        n_jobs=-1
    ),
    'Neural Network': GridSearchCV(
        MLPClassifier(
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ), 
        nn_param_dist,  
        cv=5, 
        verbose=1,  
        n_jobs=-1
    ),
    'KNN': GridSearchCV(
        KNeighborsClassifier(), 
        knn_param_dist, 
        cv=5, 
        verbose=1,  
        n_jobs=-1
    )
}

# Evaluación de modelos
results = {}
for name, model in models.items():
    print(f"Evaluating {name}...")
    results[name] = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, name)

# Imprimir resultados resumidos
for name, result in results.items():
    print(f"\nResumen de {name}:")
    for metric, value in result.items():
        print(f"{metric}: {value:.4f}")


# Callback para imprimir los mejores parámetros en cada iteración
class PrintBestParams:
    def __init__(self):
        self.best_params_ = None

    def __call__(self, search, iteration, total):
        print(f"Iteration {iteration}/{total}:")
        print("Current Best Parameters:")
        print(search.best_params_)
        print(f"Best Score so far: {search.best_score_:.4f}\n")

# Función de búsqueda y evaluación con impresión
def perform_random_search(model, param_dist, X_train, y_train, model_name, n_iter=20, cv=5):
    # Crear el callback
    callback = PrintBestParams()
    
    # Configurar RandomizedSearchCV
    search = GridSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        random_state=42,
        verbose=1,
        n_jobs=-1
    )
    
    # Ejecutar la búsqueda
    print(f"Starting Randomized Search for {model_name}...")
    search.fit(X_train, y_train)
    
    # Imprimir los mejores parámetros y métricas finales
    print(f"Final Best Parameters for {model_name}:")
    print(search.best_params_)
    print(f"Best CV Score: {search.best_score_:.4f}\n")
    
    # Retornar el modelo optimizado y los resultados
    return search.best_estimator_, search.best_params_

# Llamar la función de búsqueda para cada modelo
best_models = {}

for model_name, (model, param_dist) in {
    'Random Forest': (RandomForestClassifier(random_state=42), rf_param_dist),
    'Gradient Boosting': (GradientBoostingClassifier(random_state=42), gb_param_dist),
    'Neural Network': (MLPClassifier(max_iter=1000, random_state=42), nn_param_dist),
    'KNN': (KNeighborsClassifier(), knn_param_dist)
}.items():
    best_models[model_name], best_params = perform_random_search(
        model, param_dist, X_train, y_train, model_name
    )


