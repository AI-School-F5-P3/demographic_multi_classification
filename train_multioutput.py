import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sklearn.multioutput import MultiOutputClassifier
import joblib
import umap
from multioutput_functions import CustomMultiOutputClassifier, train_and_save_model, get_top_prediction, get_top_2_predictions

# Cargar el dataset preprocesado
df = pd.read_csv('umap_results.csv')

# Separar las variables predictoras y la variable objetivo
X = df.drop('custcat', axis=1)
y = df['custcat']

# Implementación de CTGAN
metadata = Metadata.detect_from_dataframe(data=df, table_name='customer_data')

synthesizer = CTGANSynthesizer(
    metadata,
    enforce_rounding=False,
    epochs=500,
    verbose=True
)

synthesizer.fit(df)

# Generar nuevos puntos
new_samples = synthesizer.sample(len(df))

# Combinar los nuevos puntos con los datos originales
X_augmented = pd.concat([X, new_samples.drop('custcat', axis=1)], axis=0)
y_augmented = pd.concat([y, new_samples['custcat']], axis=0)

# Función para obtener las dos mejores etiquetas
def get_top_2_labels(y, n_classes):
    y_top_2 = np.zeros((len(y), 2), dtype=int)
    for i, label in enumerate(y):
        y_top_2[i, 0] = label
        probs = np.random.dirichlet(np.ones(n_classes))
        probs[int(label)] = 0  # Convertir label a entero y excluir la etiqueta principal
        y_top_2[i, 1] = np.random.choice(np.arange(n_classes), p=probs/np.sum(probs))
    return y_top_2

n_classes = len(np.unique(y_augmented))
y_top_2 = get_top_2_labels(y_augmented, n_classes)

# Dividir en conjuntos de entrenamiento, validación y prueba
X_train_val, X_test, y_train_val, y_test = train_test_split(X_augmented, y_top_2, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

def custom_hamming_loss(y_true, y_pred):
    return np.mean([np.any(y_true[i] != y_pred[i]) for i in range(len(y_true))])

# Función de evaluación mejorada para top 2 predicciones
def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    def top_2_accuracy(y_true, y_pred):
        return np.mean([y_true[i] in y_pred[i] for i in range(len(y_true))])
    
    def top_1_accuracy(y_true, y_pred):
        return np.mean([y_true[i] == y_pred[i][0] for i in range(len(y_true))])
    
    metrics = {
        'top_2_accuracy': (top_2_accuracy(y_train[:, 0], y_train_pred), 
                           top_2_accuracy(y_val[:, 0], y_val_pred), 
                           top_2_accuracy(y_test[:, 0], y_test_pred)),
        'top_1_accuracy': (top_1_accuracy(y_train[:, 0], y_train_pred), 
                           top_1_accuracy(y_val[:, 0], y_val_pred), 
                           top_1_accuracy(y_test[:, 0], y_test_pred)),
        'hamming_loss': (custom_hamming_loss(y_train, y_train_pred), 
                         custom_hamming_loss(y_val, y_val_pred), 
                         custom_hamming_loss(y_test, y_test_pred)),
    }

    
    overfitting = {}
    for metric, (train, val, test) in metrics.items():
        overfitting[f'{metric}_train_val'] = train - val
        overfitting[f'{metric}_train_test'] = train - test
    
    print(f"\nDetailed Report for {model_name}:")
    print(f"Top 2 Accuracy (Test Set): {metrics['top_2_accuracy'][2]:.4f}")
    print(f"Top 1 Accuracy (Test Set): {metrics['top_1_accuracy'][2]:.4f}")
    print(f"Hamming Loss (Test Set): {metrics['hamming_loss'][2]:.4f}")
    
    print(f"\nOverfitting metrics for {model_name}:")
    for metric, value in overfitting.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics, overfitting, model

# Distribuciones de hiperparámetros para búsqueda

neural_network_param_dist = {
    'estimator__hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
    'estimator__activation': ['relu', 'tanh'],
    'estimator__solver': ['adam', 'sgd'],
    'estimator__alpha': [0.0001, 0.001, 0.01],
    'estimator__learning_rate': ['constant', 'adaptive']
}

# Función de búsqueda y evaluación con impresión
def perform_grid_search(model, param_dist, X_train, y_train, model_name, cv=5):
    from sklearn.model_selection import GridSearchCV
    
    search = GridSearchCV(
        model,
        param_grid=param_dist,
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    
    print(f"Starting Grid Search for {model_name}...")
    search.fit(X_train, y_train)
    
    print(f"Final Best Parameters for {model_name}:")
    print(search.best_params_)
    print(f"Best CV Score: {search.best_score_:.4f}\n")
    
    return search.best_estimator_, search.best_params_

# Definir modelos base
models = {
    'Neural Network': MultiOutputClassifier(MLPClassifier(max_iter=1000, random_state=42))
}

# Realizar búsqueda de hiperparámetros y entrenar modelos optimizados
best_models = {}
best_params = {}

for model_name, model in models.items():
    param_dist = eval(f"{model_name.lower().replace(' ', '_')}_param_dist")
    best_models[model_name], best_params[model_name] = perform_grid_search(
        model, param_dist, X_train, y_train, model_name
    )

# Evaluación de modelos optimizados
results = {}
overfitting_results = {}
low_overfitting_models = {}

for name, model in best_models.items():
    print(f"Evaluating optimized {name}...")
    results[name], overfitting_results[name], trained_model = evaluate_model(model,
                                                                           X_train,
                                                                           y_train,
                                                                           X_val,
                                                                           y_val,
                                                                           X_test,
                                                                           y_test,
                                                                           name)
    
    # Verificar si el overfitting es menor a 0.05 para top 2 accuracy
    if overfitting_results[name]['top_2_accuracy_train_val'] < 0.05:
        low_overfitting_models[name] = trained_model
        print(f"Model {name} has low overfitting. Saving...")
        
        # Guardar el modelo después del entrenamiento usando el nombre correcto del modelo.
        joblib.dump(trained_model.model if hasattr(trained_model,'model') else trained_model,
                    f'{name}_low_overfitting.joblib')

# Imprimir resultados resumidos
for name, result in results.items():
    print(f"\nResumen de {name}:")
    for metric, values in result.items():
        print(f"{metric}: Train={values[0]:.4f}, Val={values[1]:.4f}, Test={values[2]:.4f}")

print("\nModelos con bajo overfitting guardados:")
for name in low_overfitting_models.keys():
    print(f"- {name}")

# Imprimir los mejores parámetros encontrados
print("\nMejores parámetros encontrados:")
for name, params in best_params.items():
    print(f"\n{name}:")
    for param, value in params.items():
        print(f"  {param}: {value}")
