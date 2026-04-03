import pandas as pd
import numpy as np

def generar_caso_de_uso_analizar_correlacion_rangos():
    # Relación no lineal pero monótona
    x = np.arange(10)
    y = x**3 
    df = pd.DataFrame({'estudio': x, 'nota': y})
    
    input_data = {'df': df.copy()}
    
    # Lógica esperada
    output_data = df.corr(method='spearman')
    
    return input_data, output_data
