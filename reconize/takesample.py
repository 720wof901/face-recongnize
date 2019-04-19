import cv2
import numpy as np 
capture = cv2.VideoCapture(0)#if you use laptop ,then (0) will use your laptop cam
face_haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#just opencv provide ,here you need to copy all you need in the same folder,or "disk:\file's path"
id=input('enter your uid') 
#classify people you well put in
samplenum=0
if capture.isOpened():
    while True:
        ret, img = capture.read() #read img
        if ret==True:
            gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #grayscale
            faces = face_haar.detectMultiScale(gray,1.3,5) #detect
            for (x,y,w,h) in faces: #draw rectangle and output image that it detect 
                samplenum +=1
                cv2.rectangle( img,(x,y),(x+w,y+h),(0,255,0),2)#just draw rectangle 
                cv2.imwrite("dataset/user."+str(id)+"."+str(samplenum)+".jpg",gray[y:y+h,x:x+w])
                #gray[y:y+h,x:x+w] crap your detect rectangle , or you will have whole image that your cam captured
                
                cv2.waitKey(100)
            cv2.imshow('video', gray)#if you wont use is ok,
            cv2.waitKey(1)
            if samplenum >20: #(x,y,w,h) in faces ,right? so it will detect and crap until image=21
                break
capture.release()        
cv2.destroyAllWindows()

