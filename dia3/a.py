import cv2
import numpy as np

# ---------- Función para dibujar con el mouse ----------
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(dibujo, (x, y), 20, (0, 255, 0), -1)

# ---------- Cargar imagen ----------
imagen = cv2.imread('foto.jpg')
if imagen is None:
    print("No se pudo cargar la imagen.")
    exit()

# Copia para dibujo
dibujo = imagen.copy()
cv2.namedWindow('Dibujo')
cv2.setMouseCallback('Dibujo', draw_circle)

# ---------- Convertir a escala de grises ----------
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# ---------- Desenfoque ----------
blur = cv2.GaussianBlur(gris, (15, 15), 0)

# ---------- Detección de bordes ----------
bordes = cv2.Canny(blur, 50, 150)

# ---------- Detección de caras ----------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
caras = face_cascade.detectMultiScale(gris, 1.3, 5)
img_caras = imagen.copy()
for (x, y, w, h) in caras:
    cv2.rectangle(img_caras, (x, y), (x+w, y+h), (255, 0, 0), 2)

# ---------- Segmentación por color (rojo) ----------
hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(hsv, lower_red, upper_red)
resultado_rojo = cv2.bitwise_and(imagen, imagen, mask=mask)

# ---------- Contornos ----------
contornos, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contornos_img = imagen.copy()
cv2.drawContours(contornos_img, contornos, -1, (0, 255, 255), 2)

# ---------- Mostrar todo ----------
cv2.imshow('Original', imagen)
cv2.imshow('Grises', gris)
cv2.imshow('Desenfoque', blur)
cv2.imshow('Bordes', bordes)
cv2.imshow('Caras detectadas', img_caras)
cv2.imshow('Rojo detectado', resultado_rojo)
cv2.imshow('Contornos', contornos_img)

# ---------- Ventana de dibujo con el mouse ----------
while True:
    cv2.imshow('Dibujo', dibujo)
    if cv2.waitKey(20) & 0xFF == 27:  # ESC para salir
        break

# ---------- Captura de cámara con Canny en tiempo real ----------
cap = cv2.VideoCapture(0)
print("Presiona 'q' para cerrar la cámara.")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    gris_cam = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bordes_cam = cv2.Canny(gris_cam, 100, 200)
    cv2.imshow('Webcam - Bordes', bordes_cam)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
