import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
import random

def generar_caso_de_uso_detectar_anomalias_red():
    """
    Genera un caso de prueba aleatorio para detectar_anomalias_red.
    """
    # 1. Configuración aleatoria
    n_normal = random.randint(40, 60)
    n_outliers = random.randint(2, 5)
    contam = n_outliers / (n_normal + n_outliers)
    
    # 2. Generar datos (Normales vs Anomalías)
    X_normal = np.random.randn(n_normal, 2)
    X_outliers = np.random.uniform(low=-10, high=10, size=(n_outliers, 2))
    X = np.vstack([X_normal, X_outliers])
    
    # 3. Construir INPUT
    input_data = {
        'X': X,
        'contaminacion': contam
    }
    
    # 4. Calcular OUTPUT esperado
    model = IsolationForest(contamination=contam, random_state=42)
    output_data = model.fit_predict(X)
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_detectar_anomalias_red()
    
    print("=== INPUT (Diccionario) ===")
    print(f"Shape de la matriz X: {entrada['X'].shape}")
    print(f"Nivel de contaminación: {entrada['contaminacion']:.4f}")
    
    print("\n=== OUTPUT ESPERADO (Array de predicciones) ===")
    print(f"Tipo de objeto: {type(salida_esperada)}")
    print(f"Primeras 10 predicciones (1=Normal, -1=Anomalía):")
    print(salida_esperada[:10])
