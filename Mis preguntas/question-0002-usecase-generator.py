import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import random

def generar_caso_de_uso_imputar_knn_medico():
    data = {
        'edad': [20, 21, 19, 45, 46, 50],
        'presion': [120, 121, np.nan, 140, np.nan, 150]
    }
    df = pd.DataFrame(data)
    n = 2
    input_data = {'df': df.copy(), 'n_vecinos': n}
    
    # Lógica esperada
    imputer = KNNImputer(n_neighbors=n)
    res = imputer.fit_transform(df)
    output_data = pd.DataFrame(res, columns=df.columns)
    
    return input_data, output_data
