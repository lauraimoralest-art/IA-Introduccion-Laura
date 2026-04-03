import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import random

def generar_caso_de_uso_imputar_knn_medico():
    """
    Genera un caso de prueba aleatorio para imputar_knn_medico.
    """
    # 1. Configuración
    n_rows = random.randint(8, 12)
    n_vecinos = 2
    
    # 2. Generar datos con NaNs
    data = {
        'edad': np.random.randint(18, 70, size=n_rows).astype(float),
        'presion': np.random.randint(110, 160, size=n_rows).astype(float)
    }
    df = pd.DataFrame(data)
    
    # Introducir NaNs aleatorios
    indices_nan = random.sample(range(n_rows), 2)
    df.loc[indices_nan, 'presion'] = np.nan
    
    # 3. Construir INPUT
    input_data = {
        'df': df.copy(),
        'n_vecinos': n_vecinos
    }
    
    # 4. Calcular OUTPUT esperado
    imputer = KNNImputer(n_neighbors=n_vecinos)
    X_imputed = imputer.fit_transform(df)
    output_data = pd.DataFrame(X_imputed, columns=df.columns)
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_imputar_knn_medico()
    
    print("=== INPUT (Diccionario) ===")
    print(f"Número de vecinos (K): {entrada['n_vecinos']}")
    print("DataFrame original (con NaNs):")
    print(entrada['df'])
    
    print("\n=== OUTPUT ESPERADO (DataFrame completo) ===")
    print(salida_esperada)
