import cv2
import imutils
import numpy as np

cap = cv2.VideoCapture("coches.mp4")  # Asegúrate de usar la ruta correcta
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# Definir el kernel para las operaciones morfológicas
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
cont = -1
in_area = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensiona el frame
    frame = imutils.resize(frame, width=640)

    # Desplazamiento en el eje Y para mover el área hacia arriba
    offset_y = 40  # Ajusta este valor para mover el área más arriba o más abajo

    # Desplazamiento en el eje X para mover el área hacia la izquierda
    offset_x = -80  # Ajusta este valor para mover el área más a la izquierda o a la derecha

    # Especificamos los puntos extremos del área a analizar (más estrecho)
    area_pts = np.array([[270 - offset_x, 240 - offset_y], 
                         [370 - offset_x, 240 - offset_y], 
                         [370 - offset_x, 326 - offset_y], 
                         [270 - offset_x, 326 - offset_y]])
    
    imAux = np.zeros(frame.shape[:2], dtype="uint8")
    imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    image_area = cv2.bitwise_and(frame, frame, mask=imAux)
    
    # Subtracción de fondo
    fgmask = fgbg.apply(image_area)
    
    # Aplicar operaciones morfológicas para mejorar la visibilidad de los contornos
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=1)

    # Detección de contornos
    cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    detected = False
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > 400:  # Ajusta este valor para detectar contornos más pequeños
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "COCHE", (x, y-5), 1, 1, (0, 255, 0), 1, cv2.LINE_AA)
            detected = True

    # Verificar si el coche entra y sale del área
    if detected and not in_area:
        in_area = True
    elif not detected and in_area:
        cont += 1
        in_area = False

    # Mostrar el conteo en el frame
    cv2.putText(frame, f"TOTAL COCHES DENTRO: {cont}", (10, 20), 1, 1, (255,0,0), 1, cv2.LINE_AA)

    # Dibujamos el área a analizar usando cv2.polylines
    
    
    # Mostrar el frame con el área dibujada
    cv2.imshow("Video", frame)
    cv2.imshow("imAux", fgmask)

    # Salir si presionas la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()