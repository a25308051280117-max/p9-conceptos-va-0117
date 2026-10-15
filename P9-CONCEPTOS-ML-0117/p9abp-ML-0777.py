import pandas as pd
from sklearn.linear_model import LinearRegression

# Crear datos para el modelo
datos = {
    "distancia_km": [1, 2, 3, 4, 5],
    "trafico_nivel": [1, 2, 3, 2, 4],
    "edad_repartidor": [22, 28, 30, 25, 29],
    "tiempo_entrega_min": [15, 22, 30, 40, 45]
}

# Crear el DataFrame
df = pd.DataFrame(datos)

# Mostrar los datos
print("DATOS DEL MODELO")
print(df)

# Informacion del dataset
print("\nINFORMACION DEL DATASET")
df.info()

# Primeras filas
print("\nPRIMERAS FILAS")
print(df.head())

# Estadisticas descriptivas
print("\nESTADISTICAS DESCRIPTIVAS")
print(df.describe())

# Variables de entrada X
X = df[["distancia_km", "trafico_nivel", "edad_repartidor"]]

# Variable objetivo y
y = df["tiempo_entrega_min"]

print("\nVARIABLES DE ENTRADA X:")
print(X)

print("\nVARIABLE OBJETIVO y")
print(y)

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Predicciones
predicciones = modelo.predict(X)

print("\nPREDICCIONES DEL MODELO")
print(pd.DataFrame({
    "Real": y,
    "Prediccion": predicciones
}))

# Coeficientes
print("\nCOEFICIENTES DEL MODELO")
print("Intercepto:", modelo.intercept_)
print("Coeficientes:", modelo.coef_)

# Evaluacion
print("\nEVALUACION DEL MODELO")
print("R^2:", modelo.score(X, y))

# Prediccion de una nueva variable
nuevo_pedido = pd.DataFrame({
    "distancia_km": [3],
    "trafico_nivel": [2],
    "edad_repartidor": [25]
})

tiempo_estimado = modelo.predict(nuevo_pedido)

print("\nNUEVA PREDICCION")
print("Tiempo estimado:", tiempo_estimado[0], "minutos")