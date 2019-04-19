import cv2
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX # fontface

facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)

rec=cv2.face.LBPHFaceRecognizer_create();
rec.read('train.yml')
id=0

while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])#predict your id
        if (id==1):
            id="owen"#myname
        cv2.putText(img,str(id),(x,y+h),font,1,(0,255,0))
        #put your id under rectangle
    cv2.imshow("Face",img)
    if(cv2.waitKey(1)==ord('q')):
       break;
cam.release()
cv2.destroyAllWindows()
