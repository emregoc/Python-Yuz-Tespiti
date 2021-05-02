import cv2

# Yüz tanıma için oluşturulan kütüphane
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# video dosyası kullanmak için bu şekilde çalıştırırız. 
cap = cv2.VideoCapture('hababam.mp4')
# bilgisayarın kamerası için bu şekilde kullanırız

#cap = cv2.VideoCapture(0)# bu bilgisayar kamerayı çalıştırır harici kamera varsa i=1,2,3 olabilir


while True:
    # Çerçeveyi okur görüntü yakalar
    _, img = cap.read()
    # gri tonlamaya dönüştürür. Bunu yapması görüntü işleme için çok önemlidir.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Yüzleri tespit eder
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Bütün yüzlerin etrafına dikdörtgen şekli çizer yüz tanır
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    # Görüntüle
    cv2.imshow('img', img)
    # Escape tuşuna basılınca durur.
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Kamerayı serbest bırakır yoksa opencv programına özel hatalar çıkabilir
cap.release()
