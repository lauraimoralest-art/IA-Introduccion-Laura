import numpy as np
import random
from sklearn.ensemble import IsolationForest

def generar_caso_de_uso_detectar_fugas_energia():
    """Genera un caso de prueba para detección de anomalías."""
    n_normal = random.randint(50, 70)
    n_outliers = random.randint(3, 6)
    contam = n_outliers / (n_normal + n_outliers)
    
    X_normal = np.random.randn(n_normal, 2)
    X_outliers = np.random.uniform(low=-15, high=15, size=(n_outliers, 2))
    X = np.vstack([X_normal, X_outliers])
    
    input_data = {'X': X, 'contaminacion': contam}
    
    model = IsolationForest(contamination=contam, random_state=42)
    output_data = model.fit_predict(X)
    
    return input_data, output_data

# === EJEMPLO DE USO COMPLETO ===
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_detectar_fugas_energia()
    
    print("=== DATOS DE ENTRADA (CONSUMO) ===")
    print(f"Registros generados: {entrada['X'].shape[0]}")
    print(f"Contaminación definida: {entrada['contaminacion']:.4f}")
    
    print("\n=== RESULTADO ESPERADO (ETIQUETAS) ===")
    print(f"Primeras 10 predicciones: {list(salida_esperada[:10])}")
    print("(Nota: 1 es normal, -1 es posible fuga)")
