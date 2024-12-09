import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib

df = pd.read_csv('data/teleCust1000t.csv')
df.head()

df.dtypes

# Convertir float a int
df['income'] = df['income'].astype(int)
df['retire'] = df['retire'].astype(int)

df.dtypes

missing_values = df.isnull().sum()
print(missing_values[missing_values > 0])

# Crear el imputador KNN
knn_imputer = KNNImputer(n_neighbors=5)

# Imputar los valores nulos
df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)

# Verificar que no hay más valores nulos
print("Valores nulos después de la imputación:")
print(df_imputed.isnull().sum())

# Definir variables categóricas y numéricas
categorical_vars = ['region', 'marital', 'ed', 'retire', 'gender']
numerical_vars = ['tenure', 'age', 'address', 'income', 'employ', 'reside']

# Mostrar las primeras filas para verificar
print("Variables categóricas:")
print(df[categorical_vars].head())
print("\nVariables numéricas:")
print(df[numerical_vars].head())

# Estadísticas descriptivas para variables numéricas
print("Estadísticas descriptivas para variables numéricas:")
print(df[numerical_vars].describe())

# Gráficas de distribución para variables numéricas
for var in numerical_vars:
    plt.figure(figsize=(6,4))
    sns.histplot(df[var], kde=True)
    plt.title(f'Distribución de {var}')
    plt.show()

# Tablas de frecuencia para variables categóricas
for var in categorical_vars:
    print(f'Tabla de frecuencia para {var}:')
    print(df[var].value_counts())
    print()
    # Gráficas de barras para variables categóricas
for var in categorical_vars:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[var])
    plt.title(f'Frecuencia de {var}')
    plt.show()

# Estadísticas descriptivas para variables numéricas
print("Estadísticas descriptivas para variables numéricas:")
print(df[numerical_vars].describe())

# Gráficas de distribución para variables numéricas
for var in numerical_vars:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[var], kde=True)
    plt.title(f'Distribución de {var}')
    plt.show()

#Capar outliers 
def cap_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    #Capar valores por debajo del límite inferior
    data[column] = np.where(data[column] < lower_bound, lower_bound, data[column])
    
    #Capar valores por encima del límite superior
    data[column] = np.where(data[column] > upper_bound, upper_bound, data[column])

#Aplicar capado a las columnas relevantes 
cap_outliers(df, 'income')
cap_outliers(df, 'employ')

print("Columnas actuales en df:", df_imputed.columns.tolist())
df_imputed.columns = df_imputed.columns.str.strip()

df = df_imputed.copy()

# Ajustar la variable custcat para que empiece en 0
df['custcat'] -= 1  # Ahora será 0, 1, 2, 3

# Convertir a logarítmico para reducir el sesgo positivo
df['income'] = np.log1p(df['income'])
df['address'] = np.log1p(df['address'])
df['employ'] = np.log1p(df['employ'])


# Normalizar variables numéricas
scaler = StandardScaler()
df[numerical_vars] = scaler.fit_transform(df[numerical_vars])

# Guardar el scaler
joblib.dump(scaler, 'scaler.joblib') 

# Codificación one-hot para variables categóricas (excepto binarias)
one_hot_vars = ['region', 'ed']
encoder = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' para evitar colinealidad
encoded_features = encoder.fit_transform(df[one_hot_vars])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(one_hot_vars))

# Guardar el encoder
joblib.dump(encoder, 'encoder.joblib')

# Crear un nuevo DataFrame codificado sin modificar el original
dataset_encoded = pd.concat([df, encoded_df], axis=1)
dataset_encoded.to_csv('dataset_encoded.csv', index=False)

# Asignar el DataFrame codificado a df para mantener la consistencia
df = dataset_encoded

# Mostrar las primeras filas para verificar
print("Variables numéricas normalizadas:")
print(df[numerical_vars].head())
print("\nVariables categóricas codificadas:")
print(encoded_df.head())

# Matriz de correlación
correlation_matrix = df[numerical_vars].corr()

# Visualización de correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Matriz de Correlación de Variables Numéricas')
plt.tight_layout()
plt.show()

# Identificar correlaciones significativas
print("Correlaciones significativas:")
for i in range(len(numerical_vars)):
    for j in range(i+1, len(numerical_vars)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.5:  # Umbral de correlación significativa
            print(f"{numerical_vars[i]} - {numerical_vars[j]}: {corr_value}")

print(df.columns.tolist())

df.isnull().sum()

# Separar las variables predictoras y la variable objetivo
X = df.drop('custcat', axis=1)
y = df['custcat']

# Asumiendo que X es tu conjunto de datos y y es la variable objetivo
reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
embedding = reducer.fit_transform(X)

# Crear y guardar un DataFrame con los componentes UMAP
df_umap = pd.DataFrame(data=embedding, columns=['UMAP 1', 'UMAP 2'])
df_umap['custcat'] = y  # Añadir la variable objetivo
df_umap.to_csv('umap_results.csv', index=False)

# Guardar el modelo UMAP
joblib.dump(reducer, 'umap_transformer.joblib')

# Visualizar los resultados
plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_umap['UMAP 1'], df_umap['UMAP 2'], c=y, cmap='viridis')
plt.colorbar(scatter)
plt.title('UMAP Projection')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()

