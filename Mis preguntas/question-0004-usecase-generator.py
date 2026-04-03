import pandas as pd
import numpy as np

def generar_caso_de_uso_analizar_correlacion_rangos():
    """Genera un caso de prueba para correlación de Spearman."""
    horas = np.arange(1, 11)
    notas = np.exp(horas / 2) # Relación monótona exponencial
    df = pd.DataFrame({'horas_estudio': horas, 'calificacion': notas})
    
    input_data = {'df': df.copy()}
    output_data = df.corr(method='spearman')
    
    return input_data, output_data

# === EJEMPLO DE USO COMPLETO ===
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_analizar_correlacion_rangos()
    
    print("=== DATOS DE ESTUDIO (HEAD) ===")
    print(entrada['df'].head())
    
    print("\n=== MATRIZ DE SPEARMAN (RESULTADO) ===")
    print(salida_esperada)
