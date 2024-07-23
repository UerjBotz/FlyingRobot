import cv2 #lib pro processamento imagem e video
import numpy as np #lib pra manipular os arrays
from pyzbar.pyzbar import decode #lib pra decodificar qrcode e codigo de barras

cap = cv2.VideoCapture(0) #objeto para a captura e webcam id
cap.set(3,640) #largura(640 pixels)
cap.set(4,480) #altura(480 pixels)

# ===================================================================================================================================================================
# Limiarização
# ===================================================================================================================================================================
def preProcess(img): #funçao pro pre processamento da imagem

    imgPre = cv2.GaussianBlur(img,(5,5),3) #desfoque para reduzir os ruidos
    imgPre = cv2.Canny(imgPre,90,140) #detector de bordas
    kernel = np.ones((4,4),np.uint8)
    imgPre = cv2.dilate(imgPre,kernel,iterations = 2) #dilata a imagem para preencher pequenos buracos
    imgPre = cv2.erode(imgPre,kernel,iterations = 1) #erode a imagem para diminuir o tamanho dos objetos
    return imgPre #retorna a imagem processada


while True: #loop

    success,img = cap.read() #faz a leitura

    imgPre = preProcess(img)

# ===================================================================================================================================================================
# Contornos nos circulos
# ===================================================================================================================================================================
    circles = cv2.HoughCircles(imgPre, cv2.HOUGH_GRADIENT, dp=1, minDist=50, param1=50, param2=30, minRadius=5,maxRadius=60) #melhorar essas variaveis

    if circles is not None: #detecçao para cada circulo e seu centro
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 4)  # desenha o círculo externo
            cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)  # desenha o centro do círculo

# ===================================================================================================================================================================
# QRcode / codigo de barras
# ===================================================================================================================================================================

    for barcode in decode(img): #decodifica qualquer qrcode ou codigo de barras
        print(barcode.data)
        myData = barcode.data.decode('utf_8') #conversao e armazenamento do conteudo
        print(myData)
        pts = np.array([barcode.polygon], np.int32) #obtem os pontos e cria o array
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(0,255,0),5) #define a imagem, os pontos, a cor e espessura
        pts2 = barcode.rect #posiçao do poligono
        cv2.putText(img,myData,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.9,(0,255,0),2) #colocando o conteudo na borda

    cv2.imshow('Result', img) #mostra a imagem
    cv2.waitKey(1) #delay

    cv2.imshow('IMG', imgPre)
