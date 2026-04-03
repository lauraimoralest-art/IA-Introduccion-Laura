import pandas as pd
import numpy as np

def generar_caso_de_uso_analizar_correlacion_rangos():
    """
    Genera un caso de prueba aleatorio para analizar_correlacion_rangos.
    """
    # Generar relación monótona pero no lineal (exponencial)
    x = np.arange(1, 11)
    y = x**3 
    df = pd.DataFrame({'estudio_horas': x, 'nota_final': y})
    
    input_data = {'df': df.copy()}
    
    # Lógica esperada
    output_data = df.corr(method='spearman')
    
    return input_data, output_data

# --- Ejemplo de uso completo ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_analizar_correlacion_rangos()
    
    print("=== INPUT (DataFrame de datos) ===")
    print("Datos de ejemplo (horas vs nota):")
    print(entrada['df'].head())
    
    print("\n=== OUTPUT ESPERADO (Matriz de Correlación de Spearman) ===")
    print(salida_esperada)
    correlacion_final = salida_esperada.iloc[0,1]
    print(f"\nValor de correlación entre estudio y nota: {correlacion_final:.2f}")
