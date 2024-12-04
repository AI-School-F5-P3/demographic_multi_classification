
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('data/teleCust1000t.csv')

# Preparar las variables numéricas
numeric_vars = ['tenure', 'age', 'address', 'income', 'employ', 'reside']

# Escalar las variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_vars])

# Calcular VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = numeric_vars
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

# Ordenar de mayor a menor
vif_data = vif_data.sort_values("VIF", ascending=False)
print(vif_data)


