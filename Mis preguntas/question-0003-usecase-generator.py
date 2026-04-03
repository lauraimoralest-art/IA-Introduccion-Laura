import pandas as pd
import numpy as np

def generar_caso_de_uso_balancear_clases_duplicado():
    # 10 filas clase A, 2 filas clase B
    df = pd.DataFrame({
        'feature': np.random.rand(12),
        'clase': ['A']*10 + ['B']*2
    })
    target = 'clase'
    input_data = {'df': df.copy(), 'target_col': target}
    
    # Lógica esperada
    count_a = len(df[df[target] == 'A'])
    count_b = len(df[df[target] == 'B'])
    
    major = 'A' if count_a > count_b else 'B'
    minor = 'B' if major == 'A' else 'A'
    
    df_major = df[df[target] == major]
    df_minor = df[df[target] == minor]
    
    df_minor_upsampled = df_minor.sample(len(df_major), replace=True, random_state=42)
    output_data = pd.concat([df_major, df_minor_upsampled])
    
    return input_data, output_data
