#🧠 PARTE 1: MLP con Scikit-learn

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el dataset
digits = load_digits()
X, y = digits.data, digits.target

# 2. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Crear el modelo MLP
mlp = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', max_iter=5)

# 4. Entrenar
mlp.fit(X_train, y_train)

# 5. Predecir
y_pred = mlp.predict(X_test)

# 6. Evaluar
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# 7. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# 8. Graficar
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusión MLP")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.show()













# 🧠 PARTE 1: MLP con Scikit-learn

from sklearn.datasets import load_digits  # 📥 Carga un dataset de imágenes de dígitos escritos a mano (del 0 al 9)
from sklearn.model_selection import train_test_split  # 🔀 Sirve para dividir los datos en entrenamiento y prueba
from sklearn.neural_network import MLPClassifier  # 🧠 Importa el modelo Perceptrón Multicapa (MLP)
from sklearn.metrics import confusion_matrix, accuracy_score  # 📊 Funciones para evaluar el modelo
import matplotlib.pyplot as plt  # 📈 Librería para graficar
import seaborn as sns  # 🎨 Librería para graficar más bonito (usa colores y cuadrículas)

# 1. Cargar el dataset
digits = load_digits()  # 📥 Cargamos los datos y los guardamos en una variable llamada 'digits'
X, y = digits.data, digits.target  # 🔣 X = imágenes en forma de números (64 columnas), y = número real (de 0 a 9)

# 2. Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 🧪 Divide los datos:
#  - 80% para entrenar (X_train, y_train)
#  - 20% para probar (X_test, y_test)
#  - random_state=42 hace que la división sea siempre igual (para repetir resultados)

# 3. Crear el modelo MLP
mlp = MLPClassifier(hidden_layer_sizes=(64,), activation='relu', solver='adam', max_iter=5)
# 🧠 Creamos el modelo MLP (Red neuronal simple)
# - hidden_layer_sizes=(64,) → 1 capa oculta con 64 neuronas
# - activation='relu' → Función que activa las neuronas
# - solver='adam' → Algoritmo que optimiza el aprendizaje
# - max_iter=5 → Entrena por 5 épocas (pocas veces para que sea rápido)

# 4. Entrenar
mlp.fit(X_train, y_train)  # 🔧 Entrena el modelo con los datos de entrenamiento

# 5. Predecir
y_pred = mlp.predict(X_test)  # 🎯 El modelo intenta adivinar los números del conjunto de prueba

# 6. Evaluar
acc = accuracy_score(y_test, y_pred)  # ✅ Compara las predicciones con los valores reales
print("Accuracy:", acc)  # 🖨️ Muestra el porcentaje de aciertos

# 7. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)  # 🔍 Muestra los aciertos y errores por clase (ej. cuántos 2 confundió con 3)

# 8. Graficar
plt.figure(figsize=(8, 6))  # 🖼️ Tamaño del gráfico
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # 📊 Dibuja la matriz con números y color
plt.title("Matriz de Confusión MLP")  # Título del gráfico
plt.xlabel("Predicción")  # Eje X
plt.ylabel("Real")  # Eje Y
plt.show()  # 👀 Muestra la figura
