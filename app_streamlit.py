import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sqlite3

# Cargar el modelo, scaler, encoder y el transformador UMAP
model = joblib.load('Neural Network_low_overfitting.joblib')
scaler = joblib.load('scaler.joblib')
encoder = joblib.load('encoder.joblib')
umap_transformer = joblib.load('umap_transformer.joblib')  # Cargar el modelo UMAP

# Conectar a la base de datos SQLite (se crea si no existe)
conn = sqlite3.connect('clientes_recomendaciones.db')
c = conn.cursor()

# Crear tablas si no existen
c.execute('''
CREATE TABLE IF NOT EXISTS clientes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    region INTEGER,
    tenure INTEGER,
    age INTEGER,
    marital INTEGER,
    address INTEGER,
    income REAL,
    ed INTEGER,
    employ INTEGER,
    retire INTEGER,
    gender INTEGER,
    reside INTEGER
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS recomendaciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cliente_id INTEGER,
    servicio_recomendado TEXT,
    probabilidad REAL,
    FOREIGN KEY(cliente_id) REFERENCES clientes(id)
)
''')

def get_service_name(service_id):
    services = {
        0: "Servicio básico",
        1: "Servicio electrónico",
        2: "Servicio Plus", 
        3: "Servicio Total"
    }
    return services.get(service_id, "Servicio desconocido")  # Manejo de error

def insert_cliente(conn, cliente_data):
    c = conn.cursor()
    c.execute("INSERT INTO clientes (region, tenure, age, marital, address, income, ed, employ, retire, gender, reside) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", cliente_data)
    conn.commit()
    return c.lastrowid

def insert_recomendacion(conn, cliente_id, servicio_recomendado, probabilidad):
    c = conn.cursor()
    c.execute("INSERT INTO recomendaciones (cliente_id, servicio_recomendado, probabilidad) VALUES (?, ?, ?)", (cliente_id, servicio_recomendado, probabilidad))
    conn.commit()

st.title('Predicción de Servicios')

# Preguntar si quiere una o dos recomendaciones
num_recommendations = st.radio("¿Cuántas recomendaciones desea?", (1, 2))

# Crear inputs para cada variable
region = st.selectbox('Región', [1, 2, 3])
tenure = st.slider('Tiempo como cliente (meses)', 0, 100)
age = st.slider('Edad', 18, 100)
marital = st.selectbox('Estado civil', [0, 1])
address = st.slider('Años en la dirección actual', 0, 50)
income = st.slider('Ingresos anuales (en miles)', 0, 200)
ed = st.selectbox('Nivel educativo', [1, 2, 3, 4, 5])
employ = st.slider('Años empleado', 0, 50)
retire = st.selectbox('¿Está jubilado?', [0, 1])
gender = st.selectbox('Género', [0, 1])
reside = st.slider('Número de residentes', 1, 10)

if st.button('Predecir'):
    # Crear el DataFrame con los datos de entrada
    input_data = pd.DataFrame({
        'region': [region],
        'tenure': [tenure],
        'age': [age],
        'marital': [marital],
        'address': [address],
        'income': [income],
        'ed': [ed],
        'employ': [employ],
        'retire': [retire],
        'gender': [gender],
        'reside': [reside]
    })

    # Aplicar transformaciones logarítmicas
    input_data['income'] = np.log1p(input_data['income'])
    input_data['address'] = np.log1p(input_data['address'])
    input_data['employ'] = np.log1p(input_data['employ'])

    # Normalizar variables numéricas
    numeric_columns = ['tenure', 'age', 'address', 'income', 'employ', 'reside']
    input_data[numeric_columns] = scaler.transform(input_data[numeric_columns])

    # Aplicar one-hot encoding a 'region' y 'ed'
    categorical_columns = ['region', 'ed']
    encoded_features = encoder.transform(input_data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Combinar datos normalizados y codificados
    input_data = pd.concat([input_data.drop(columns=categorical_columns), encoded_df], axis=1)

    # Aplicar UMAP a los datos transformados
    umap_features = umap_transformer.transform(input_data)
    
    # Crear un DataFrame para las características UMAP
    umap_df = pd.DataFrame(umap_features, columns=['UMAP 1', 'UMAP 2'])
    
    # Combinar las características UMAP con los datos originales
    input_data = pd.concat([input_data.reset_index(drop=True), umap_df], axis=1)

    # Asegurarse de que todas las columnas necesarias estén presentes y en el orden correcto
    expected_columns = model.feature_names_in_

    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reordenar las columnas para que coincidan con el orden del conjunto de entrenamiento
    input_data = input_data.reindex(columns=expected_columns)

    # Realizar la predicción
    try:
        if num_recommendations == 1:
            # Para modelo MultiOutputClassifier, predict devuelve un array de arrays
            prediction_array = model.predict(input_data)
            
            # Tomar el primer elemento del primer array
            prediction = prediction_array[0][0]
            service_name = get_service_name(int(prediction))
            
            # Guardar datos del cliente en la base de datos
            cliente_id = insert_cliente(conn, (region, tenure, age, marital, address, income, ed, employ, retire, gender, reside))
            
            # Guardar recomendación en la base de datos
            insert_recomendacion(conn, cliente_id, service_name, 1.0)  # Probabilidad 1.0 para una sola recomendación
            
            st.write(f"Servicio recomendado: {service_name}")
        
        elif num_recommendations == 2:
            # Obtener las probabilidades usando predict_proba
            probabilities = model.predict_proba(input_data)[0]
            
            # Obtener los dos índices con mayores probabilidades
            top_2_indices = np.argsort(probabilities[0])[-2:][::-1]
            
            # Guardar datos del cliente en la base de datos
            cliente_id = insert_cliente(conn, (region, tenure, age, marital, address, income, ed, employ, retire, gender, reside))
            
            st.write(f"Servicios recomendados:")
            for i, service_index in enumerate(top_2_indices, 1):
                service_name = get_service_name(service_index)
                probabilidad = probabilities[0][service_index]
                st.write(f"{i}. {service_name}")
                st.write(f"   Probabilidad: {probabilidad:.2%}")
                
                # Guardar recomendación en la base de datos
                insert_recomendacion(conn, cliente_id, service_name, probabilidad)
    
    except Exception as e:
        st.error(f"Ha ocurrido un error durante la predicción: {e}")
        # Información adicional de depuración
        st.write("Detalles del error:")
        st.write(str(e))

# Cerrar conexión a la base de datos al finalizar el script
conn.close()