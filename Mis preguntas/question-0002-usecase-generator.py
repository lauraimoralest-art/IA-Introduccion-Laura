import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import random

def generar_caso_de_uso_reparar_sensores_logistica():
    """
    Genera un caso de prueba aleatorio para reparar_sensores_logistica.
    """
    n_rows = random.randint(10, 15)
    data = {
        'temp_ambiente': np.random.uniform(20, 30, size=n_rows),
        'temp_cava': np.random.uniform(-5, 5, size=n_rows)
    }
    df = pd.DataFrame(data)
    
    # Simular fallos en sensores (NaN)
    indices_nan = random.sample(range(n_rows), 2)
    df.loc[indices_nan, 'temp_cava'] = np.nan
    
    n = 2
    input_data = {'df': df.copy(), 'n_vecinos': n}
    
    # Lógica esperada
    imputer = KNNImputer(n_neighbors=n)
    res = imputer.fit_transform(df)
    output_data = pd.DataFrame(res, columns=df.columns)
    
    return input_data, output_data

# --- Ejemplo de uso completo ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_reparar_sensores_logistica()
    
    print("=== INPUT (Diccionario con DataFrame sucio) ===")
    print(f"Número de vecinos a usar: {entrada['n_vecinos']}")
    print("Datos originales (busque los valores NaN):")
    print(entrada['df'])
    
    print("\n=== OUTPUT ESPERADO (DataFrame Reparado) ===")
    print("Datos después de la imputación KNN:")
    print(salida_esperada)
