import pandas as pd
import numpy as np
import random
from sklearn.impute import KNNImputer

# FUNCIÓN DE SOLUCIÓN
def reparar_sensores_logistica(df, n_vecinos):
    imputer = KNNImputer(n_neighbors=n_vecinos)
    res = imputer.fit_transform(df)
    return pd.DataFrame(res, columns=df.columns)

# GENERADOR
def generar_caso_de_uso_reparar_sensores_logistica():
    df = pd.DataFrame({'temp': [25.0, np.nan, 24.5], 'hum': [80, 81, np.nan]})
    n = 2
    input_data = {'df': df.copy(), 'n_vecinos': n}
    output_data = reparar_sensores_logistica(df, n)
    return input_data, output_data

if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_reparar_sensores_logistica()
    print("Ejercicio 2: OK")
