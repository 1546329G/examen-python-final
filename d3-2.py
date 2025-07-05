import cv2

# Cargar clasificador Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Acceso a la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Dibujar rectángulos
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar número de rostros detectados
    cv2.putText(frame, f'Rostros detectados: {len(faces)}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Mostrar resultados
    cv2.imshow("Haar Face Detector", frame)

    # Salir con 'q' y guardar imagen
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("haar_resultado.jpg", frame)
        print("Imagen guardada como haar_resultado.jpg")
        break

cap.release()
cv2.destroyAllWindows()
