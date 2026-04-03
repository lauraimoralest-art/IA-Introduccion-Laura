import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_analizar_correlacion_rangos():
    """
    Genera un caso de prueba aleatorio para analizar_correlacion_rangos.
    """
    n_rows = random.randint(5, 10)
    x = np.linspace(1, 100, n_rows)
    y = np.log(x) + np.random.normal(0, 0.1, n_rows)
    df = pd.DataFrame({'estudio': x, 'nota': y})
    
    input_data = {'df': df.copy()}
    output_data = df.corr(method='spearman')
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_analizar_correlacion_rangos()
    print("=== INPUT (DataFrame de datos) ===")
    print(entrada['df'].head())
    print("\n=== OUTPUT ESPERADO (Matriz de Spearman) ===")
    print(salida_esperada)
