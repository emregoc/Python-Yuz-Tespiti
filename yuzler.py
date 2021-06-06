import numpy as np
from cv2 import cv2
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime
import time


cred = credentials.Certificate('pythonbitirme.json')

firebase_admin.initialize_app(cred, {


    'databaseURL':'https://python-a1d82-default-rtdb.firebaseio.com/'

})

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels ={"person_name": 1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
       # print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]  #(ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]

        id_, conf =recognizer.predict(roi_gray)
        if conf>=45: #and conf<= 85:
            print(id_)
            print(labels[id_])
            name = labels[id_]
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

            
            #a=datetime.now()
            #b=a.replace(day=a.day+0, hour=0, minute=5, second=0, microsecond=0)
            #delta_t=b-a
            #secs=delta_t.seconds+1
            #minut =(secs /60) %60
           # print(minut)
            #ref = db.reference('Kalan Süresi')
            #ref.set(minut)

            
            ref = db.reference('algilananKisi')
            sonkisi=ref.get()

            print(sonkisi)
            if (sonkisi!=name):
                ref = db.reference('algilananKisi')
                ref.set(name)
                ref = db.reference('giris saati')
                ref.set(str(datetime.now()))
                

            sureRef = db.reference('giris saati')
            sonGirisStr = sureRef.get()
            sonGiris = datetime.strptime(sonGirisStr, '%Y-%m-%d %H:%M:%S.%f')
            besDakika = 60 * 5
            suan = datetime.now()
            
            gecenSure = suan - sonGiris
            gecenSureSaniye = gecenSure.total_seconds()

            print(sonGiris)
            print(suan)
            print("----")
            print("Geçen süresi: " + str(gecenSureSaniye))

            if(gecenSureSaniye > besDakika):
                print("Beş dk.yı geçti")
                ref = db.reference('Bes Dakika Süre Doldu:')
                ref.set(str(gecenSureSaniye))
            else:
                print("5dk olmadı")
            ref = db.reference('Gecen Süresi:')
            ref.set(str(gecenSureSaniye))

            
        img_item = "7.png"
        cv2.imwrite(img_item, roi_color)

        color = (255,0, 0) #BGR 0-255
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y+ h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        
    cv2.imshow('Yuz Tespiti',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
