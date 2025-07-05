import cv2 # Importa la librería OpenCV, esencial para todas las operaciones de visión por computadora.

# --- Carga del Clasificador de Rostros ---
# Carga el clasificador Haar Cascade pre-entrenado para la detección de rostros frontales.
# Estos archivos XML contienen información sobre patrones característicos de rostros.
face_cascade = cv2.CascadeClassifier(cv2.data.haascades + 'haarcascade_frontalface_default.xml')

# --- Acceso a la Cámara ---
# Inicializa la captura de video desde la cámara web predeterminada (generalmente la cámara 0).
cap = cv2.VideoCapture(0)

# --- Bucle Principal de Procesamiento de Video ---
# Este bucle se ejecuta continuamente, procesando fotograma a fotograma el video de la cámara.
while True:
    # Lee un fotograma de la cámara.
    # 'ret' (boolean) es True si el fotograma se leyó correctamente, False de lo contrario.
    # 'frame' es la imagen actual capturada por la cámara.
    ret, frame = cap.read()

    # Si no se pudo leer el fotograma (por ejemplo, la cámara se desconectó), se sale del bucle.
    if not ret:
        break

    # --- Preprocesamiento para Detección ---
    # Convierte el fotograma a escala de grises. La detección de rostros con Haar Cascades
    # es más eficiente y precisa en imágenes en escala de grises.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Detección de Rostros ---
    # Realiza la detección de rostros en el fotograma en escala de grises.
    # 'scaleFactor': Reduce la imagen en un 10% en cada escala para detectar rostros de diferentes tamaños.
    # 'minNeighbors': Parámetro para la calidad de la detección. Un valor más alto requiere más "vecinos"
    #                 para confirmar una detección, reduciendo falsos positivos.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # --- Lógica de Procesamiento Condicional ---
    # Verifica si se detectaron exactamente 2 rostros.
    if len(faces) == 2:
        # Si se detectan 2 rostros, dibuja un rectángulo verde alrededor de cada uno.
        for (x, y, w, h) in faces:
            # Dibuja un rectángulo en el fotograma original (a color).
            # (x, y): esquina superior izquierda del rostro.
            # (x + w, y + h): esquina inferior derecha del rostro.
            # (0, 255, 0): color verde en formato BGR.
            # 2: grosor de la línea del rectángulo.
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Define y muestra un mensaje en la consola.
        mensaje = "Exactamente 2 rostros detectados"
        print(mensaje)
        # Dibuja el mensaje en el fotograma.
        # (10, 25): posición del texto en el fotograma.
        # (255, 0, 0): color azul para el texto.
        cv2.putText(frame, mensaje, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        # Si no se detectan exactamente 2 rostros, define un mensaje de ignorancia.
        mensaje = f"{len(faces)} rostros detectados - NO EXISTE :) "
        print(mensaje)
        # Dibuja este mensaje en el fotograma.
        # (0, 0, 255): color rojo para el texto.
        cv2.putText(frame, mensaje, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- Mostrar el Fotograma ---
    # Muestra el fotograma procesado en una ventana llamada "Detector - Solo 2 rostros".
    cv2.imshow("Detector - Solo 2 rostros", frame)

    # --- Salida y Guardado ---
    # Espera 1 milisegundo por una pulsación de tecla.
    # Si la tecla presionada es 'q' (quit), se ejecuta el bloque.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Antes de salir, verifica si en el momento de presionar 'q' había 2 rostros.
        if len(faces) == 2:
            # Si sí, guarda el fotograma actual como 'dos.jpg'.
            cv2.imwrite("dos.jpg", frame)
        break # Sale del bucle principal, terminando el programa.

# --- Liberación de Recursos ---
# Libera la cámara para que otras aplicaciones puedan usarla.
cap.release()
# Cierra todas las ventanas de OpenCV que se abrieron.
cv2.destroyAllWindows()