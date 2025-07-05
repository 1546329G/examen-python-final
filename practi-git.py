import cv2

# Cargar clasificadores Haar Cascade (corregir la ruta)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Capturar video desde la cámara por defecto
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capturar frame del video
    if not ret:
        break

    # Convertir frame a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.9, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]  # Área del rostro en escala de grises
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors=20)  # Detectar sonrisas

        if len(smiles) > 0:
            cv2.putText(frame, 'Sonriendo', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)  # Rectángulo verde
        else:
            cv2.putText(frame, 'Serio', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)  # Rectángulo rojo

    cv2.imshow('Detector de sonrisas', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas (fuera del while)
cap.release()
cv2.destroyAllWindows()