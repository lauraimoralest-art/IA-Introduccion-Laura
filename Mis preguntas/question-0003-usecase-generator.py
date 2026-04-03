import pandas as pd
import numpy as np

def generar_caso_de_uso_balancear_fallas_criticas():
    """
    Genera un caso de prueba aleatorio para balancear_fallas_criticas.
    """
    df = pd.DataFrame({
        'vibracion': np.random.rand(15),
        'estado': ['Normal']*12 + ['Falla']*3
    })
    target = 'estado'
    input_data = {'df': df.copy(), 'target_col': target}
    
    # Lógica de balanceo esperada
    counts = df[target].value_counts()
    major_label = counts.idxmax()
    minor_label = counts.idxmin()
    
    df_major = df[df[target] == major_label]
    df_minor = df[df[target] == minor_label]
    
    # Duplicar minoría
    df_minor_resampled = df_minor.sample(len(df_major), replace=True, random_state=42)
    output_data = pd.concat([df_major, df_minor_resampled]).sample(frac=1).reset_index(drop=True)
    
    return input_data, output_data

# --- Ejemplo de uso completo ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_balancear_fallas_criticas()
    
    print("=== INPUT (Dataset Desbalanceado) ===")
    print(f"Columna objetivo: {entrada['target_col']}")
    print("Distribución inicial de clases:")
    print(entrada['df']['estado'].value_counts())
    
    print("\n=== OUTPUT ESPERADO (Dataset Balanceado) ===")
    print("Nueva distribución de clases:")
    print(salida_esperada['estado'].value_counts())
    print("\nPrimeras filas del resultado balanceado:")
    print(salida_esperada.head())
