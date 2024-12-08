import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
import joblib


class CustomMultiOutputClassifier:
    def __init__(self, estimator=MLPClassifier(max_iter=1000, random_state=42)):
        self.model = MultiOutputClassifier(estimator)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)  # Esto devuelve un array de predicciones

    def predict_proba(self, X):
        try:
            probas = [estimator.predict_proba(X) for estimator in self.model.estimators_]
            return np.mean(probas, axis=0)
        except AttributeError:
            # Fallback method if predict_proba is not available
            return self.model.predict(X)

def train_and_save_model(X, y, param_dist):
    from sklearn.model_selection import GridSearchCV
    
    model = CustomMultiOutputClassifier()
    
    search = GridSearchCV(
        model.model,
        param_grid=param_dist,
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    
    search.fit(X, y)
    
    best_model = CustomMultiOutputClassifier(search.best_estimator_.estimator)
    best_model.fit(X, y)
    
    joblib.dump(best_model, 'modelo_multioutput.joblib')
    return best_model, search.best_params_

def get_top_prediction(model, X):
    probabilities = model.predict_proba(X)
    return np.argmax(probabilities, axis=1)

def get_top_2_predictions(model, X):
    probabilities = model.predict_proba(X)
    return np.argsort(probabilities, axis=1)[:, -2:][:, ::-1]


# best_model, best_params = train_and_save_model(X, y, param_dist)