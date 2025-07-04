#  🧠 PARTE 2: CNN con Keras + Fashion MNIST

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Cargar datos
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 2. Normalizar y reshape
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 3. One-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 4. Crear modelo
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, kernel_size=(3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# 5. Compilar
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. Entrenar
history = model.fit(x_train, y_train_cat, epochs=5, validation_data=(x_test, y_test_cat))

# 7. Graficar precisión y pérdida
plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión validación')
plt.title("Precisión del modelo CNN")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.title("Pérdida del modelo CNN")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()
plt.show()





















# 🧠 PARTE 2: CNN con Keras + Fashion MNIST

from tensorflow.keras.datasets import fashion_mnist  # Dataset de imágenes de ropa
from tensorflow.keras.models import Sequential  # Modelo secuencial
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense  # Capas necesarias para la red CNN
from tensorflow.keras.utils import to_categorical  # Convierte etiquetas a formato one-hot
import matplotlib.pyplot as plt  # Para graficar

# 1. Cargar datos
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 2. Normalizar y reshape
x_train = x_train / 255.0  # Normaliza los valores de píxeles a rango [0,1]
x_test = x_test / 255.0
x_train = x_train.reshape(-1, 28, 28, 1)  # Cambia forma a (cantidad, alto, ancho, canales)
x_test = x_test.reshape(-1, 28, 28, 1)

# 3. One-hot encoding
y_train_cat = to_categorical(y_train, 10)  # Convierte etiquetas (0–9) a vectores binarios
y_test_cat = to_categorical(y_test, 10)

# 4. Crear modelo CNN
model = Sequential([
    Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),  # 1ra capa conv: 32 filtros 3x3
    MaxPooling2D(pool_size=(2,2)),  # Reduce tamaño a la mitad
    Conv2D(64, kernel_size=(3,3), activation='relu'),  # 2da capa conv: 64 filtros
    MaxPooling2D(pool_size=(2,2)),  # Pooling otra vez
    Flatten(),  # Aplana todo en una sola capa
    Dense(128, activation='relu'),  # Capa densa con 128 neuronas
    Dense(10, activation='softmax')  # Capa de salida para 10 clases
])

# 5. Compilar modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 6. Entrenar modelo
history = model.fit(x_train, y_train_cat, epochs=5, validation_data=(x_test, y_test_cat))

# 7. Graficar precisión
plt.plot(history.history['accuracy'], label='Precisión entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión validación')
plt.title("Precisión del modelo CNN")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend()
plt.show()

# 8. Graficar pérdida
plt.plot(history.history['loss'], label='Pérdida entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida validación')
plt.title("Pérdida del modelo CNN")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()
plt.show()

