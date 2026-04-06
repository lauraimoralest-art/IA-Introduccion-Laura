import pandas as pd
import numpy as np

def generar_caso_de_uso_balancear_fallas_criticas():
    """Genera un caso de prueba para balanceo de clases."""
    df = pd.DataFrame({
        'vibracion': np.random.rand(12),
        'estado': ['Normal']*10 + ['Falla']*2
    })
    target = 'estado'
    input_data = {'df': df.copy(), 'target_col': target}
    
    counts = df[target].value_counts()
    major, minor = counts.idxmax(), counts.idxmin()
    
    df_major = df[df[target] == major]
    df_minor = df[df[target] == minor]
    
    df_minor_up = df_minor.sample(len(df_major), replace=True, random_state=42)
    output_data = pd.concat([df_major, df_minor_up]).sample(frac=1).reset_index(drop=True)
    
    return input_data, output_data

# === EJEMPLO DE USO COMPLETO ===
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_balancear_fallas_criticas()
    
    print("=== DISTRIBUCIÓN ORIGINAL ===")
    print(entrada['df']['estado'].value_counts())
    
    print("\n=== DISTRIBUCIÓN BALANCEADA (RESULTADO) ===")
    print(salida_esperada['estado'].value_counts())
    print("\nVista previa de datos balanceados:")
    print(salida_esperada.head())
