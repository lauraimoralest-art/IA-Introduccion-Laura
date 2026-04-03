import numpy as np
from sklearn.ensemble import IsolationForest
import random

def generar_caso_de_uso_detectar_anomalias_red():
    n_normal = 100
    n_outliers = 5
    # Datos normales (cluster centrado)
    X_normal = np.random.randn(n_normal, 2)
    # Datos anomalos (alejados)
    X_outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, 2))
    X = np.vstack([X_normal, X_outliers])
    
    contam = 0.05
    input_data = {'X': X, 'contaminacion': contam}
    
    # Lógica esperada
    model = IsolationForest(contamination=contam, random_state=42)
    output_data = model.fit_predict(X)
    
    return input_data, output_data
