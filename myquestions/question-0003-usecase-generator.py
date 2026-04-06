import pandas as pd
import numpy as np

# FUNCIÓN DE SOLUCIÓN
def balancear_fallas_criticas(df, target_col):
    counts = df[target_col].value_counts()
    major, minor = counts.idxmax(), counts.idxmin()
    df_major = df[df[target_col] == major]
    df_minor = df[df[target_col] == minor]
    df_minor_up = df_minor.sample(len(df_major), replace=True, random_state=42)
    return pd.concat([df_major, df_minor_up]).sample(frac=1).reset_index(drop=True)

# GENERADOR
def generar_caso_de_uso_balancear_fallas_criticas():
    df = pd.DataFrame({'v': np.random.rand(5), 'estado': ['Normal']*4 + ['Falla']})
    input_data = {'df': df.copy(), 'target_col': 'estado'}
    output_data = balancear_fallas_criticas(df, 'estado')
    return input_data, output_data

if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_balancear_fallas_criticas()
    print("Ejercicio 3: OK")
