import numpy as np
from sklearn.ensemble import IsolationForest
import random

def generar_caso_de_uso_detectar_fugas_energia():
    """
    Genera un caso de prueba aleatorio para detectar_fugas_energia.
    """
    n_normal = random.randint(50, 70)
    n_outliers = random.randint(3, 6)
    contam = n_outliers / (n_normal + n_outliers)
    
    # Generar datos de consumo normal (alrededor de un promedio)
    X_normal = np.random.randn(n_normal, 2)
    # Generar consumos extremos (posibles fugas)
    X_outliers = np.random.uniform(low=-15, high=15, size=(n_outliers, 2))
    
    X = np.vstack([X_normal, X_outliers])
    input_data = {'X': X, 'contaminacion': contam}
    
    # Lógica esperada
    model = IsolationForest(contamination=contam, random_state=42)
    output_data = model.fit_predict(X)
    
    return input_data, output_data

# --- Ejemplo de uso completo ---
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_detectar_fugas_energia()
    
    print("=== INPUT (Matriz de Consumo X) ===")
    print(f"Registros totales generados: {entrada['X'].shape[0]}")
    print(f"Porcentaje de contaminación definido: {entrada['contaminacion']:.4f}")
    print("Primeras 5 filas de X:")
    print(entrada['X'][:5])
    
    print("\n=== OUTPUT ESPERADO (Predicciones del modelo) ===")
    print(f"Tipo de objeto devuelto: {type(salida_esperada)}")
    print(f"Primeras 10 etiquetas (1: Normal, -1: Fuga):")
    print(salida_esperada[:10])
