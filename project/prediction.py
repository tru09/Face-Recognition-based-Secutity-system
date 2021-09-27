import cv2
import numpy as np
from numpy import asarray

vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

while True:
    ret,frame = vid.read()
    faces = face_cascade.detectMultiScale(frame,1.1,4)

    # for (x, y, w, h) in faces:
    #     center = (x + w//2, y + h//2)
    #     cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),4)

    if faces is None:
        continue
    elif count<=200:
        for (x, y, w, h) in faces:
            center = (x + w//2, y + h//2)
            #frame = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)
            crop = img[y:y+h,x:x+w]
            resize_img = cv2.resize(crop,(256,256))
            img = asarray(img)/255
            img = np.resize(img,(48,48,3))
            img = img.reshape(-1,48,48,3)
            pred = int(model.predict_classes(img))
            print(labels[pred])
            




    cv2.imshow('video',frame)
    
    if cv2.waitKey(10)==27:
        break


vid.release()
cv2.destroyAllWindows()