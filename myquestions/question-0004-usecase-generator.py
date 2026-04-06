import pandas as pd
import numpy as np

# FUNCIÓN DE SOLUCIÓN
def analizar_correlacion_rangos(df):
    return df.corr(method='spearman')

# GENERADOR
def generar_caso_de_uso_analizar_correlacion_rangos():
    df = pd.DataFrame({'horas': [1, 2, 3], 'nota': [10, 30, 90]})
    input_data = {'df': df.copy()}
    output_data = analizar_correlacion_rangos(df)
    return input_data, output_data

if __name__ == "__main__":
    entrada, salida = generar_caso_de_uso_analizar_correlacion_rangos()
    print("Ejercicio 4: OK")
