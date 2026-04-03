import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_balancear_clases_duplicado():
    """
    Genera un caso de prueba aleatorio para balancear_clases_duplicado.
    """
    # 1. Crear dataset desbalanceado
    n_major = random.randint(10, 15)
    n_minor = random.randint(2, 4)
    df_major = pd.DataFrame({'val': np.random.rand(n_major), 'clase': 'Legal'})
    df_minor = pd.DataFrame({'val': np.random.rand(n_minor), 'clase': 'Fraude'})
    df = pd.concat([df_major, df_minor]).sample(frac=1).reset_index(drop=True)
    target_col = 'clase'
    
    # 2. Construir INPUT
    input_data = {'df': df.copy(), 'target_col': target_col}
    
    # 3. Calcular OUTPUT esperado
    counts = df[target_col].value_counts()
    clase_major = counts.idxmax()
    clase_minor = counts.idxmin()
    df_m_major = df[df[target_col] == clase_major]
    df_m_minor = df[df[target_col] == clase_minor]
    
    df_minor_resampled = df_m_minor.sample(len(df_m_major), replace=True, random_state=42)
    output_data = pd.concat([df_m_major, df_minor_resampled]).sample(frac=1).reset_index(drop=True)
    
    return input_data, output_data

# --- Ejemplo de uso ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_balancear_clases_duplicado()
    print("=== INPUT (Distribución original) ===")
    print(entrada['df']['clase'].value_counts())
    print("\n=== OUTPUT ESPERADO (Distribución balanceada) ===")
    print(salida_esperada['clase'].value_counts())
