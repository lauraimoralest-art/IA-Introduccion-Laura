import pandas as pd
import numpy as np
import random
from sklearn.impute import KNNImputer

def generar_caso_de_uso_reparar_sensores_logistica():
    """Genera un caso de prueba para imputación de sensores."""
    n_rows = random.randint(10, 15)
    df = pd.DataFrame({
        'temp_cava': np.random.uniform(-5, 5, size=n_rows),
        'humedad_rel': np.random.uniform(70, 90, size=n_rows)
    })
    
    # Insertar fallos aleatorios
    df.iloc[random.sample(range(n_rows), 2), 0] = np.nan
    
    n = 2
    input_data = {'df': df.copy(), 'n_vecinos': n}
    
    imputer = KNNImputer(n_neighbors=n)
    output_data = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    return input_data, output_data

# === EJEMPLO DE USO COMPLETO ===
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_reparar_sensores_logistica()
    
    print("=== DATOS DE ENTRADA (CON FALLOS) ===")
    print(f"Número de vecinos a usar: {entrada['n_vecinos']}")
    print(entrada['df'].head())
    
    print("\n=== RESULTADO ESPERADO (REPARADO) ===")
    print(salida_esperada.head())
