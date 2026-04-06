import numpy as np
import random
from sklearn.ensemble import IsolationForest

# FUNCIÓN DE SOLUCIÓN (Esta faltaba dentro del archivo)
def detectar_fugas_energia(X, contaminacion):
    model = IsolationForest(contamination=contaminacion, random_state=42)
    return model.fit_predict(X)

# GENERADOR
def generar_caso_de_uso_detectar_fugas_energia():
    n_normal = random.randint(50, 70)
    n_outliers = random.randint(3, 6)
    contam = n_outliers / (n_normal + n_outliers)
    X_normal = np.random.randn(n_normal, 2)
    X_outliers = np.random.uniform(low=-15, high=15, size=(n_outliers, 2))
    X = np.vstack([X_normal, X_outliers])
    
    input_data = {'X': X, 'contaminacion': contam}
    # Aquí se llama a la función de arriba
    output_data = detectar_fugas_energia(X, contam)
    return input_data, output_data

if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_detectar_fugas_energia()
    print("Ejercicio 1: OK")
