import cv2
import numpy as np
import os
from PIL import Image

recon =cv2.face.LBPHFaceRecognizer_create()
#here you need to download opencv-contrib-python or you can't use "face" module 
#pip install opencv-contrib-python 
path='dataset'#your image path

def getImageWithID(path):#write function
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]#join your path,and catch everyImage
    faces=[]
    IDs=[]
    #create list
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNP =np.array(faceImg,'uint8')#you can use print(faceNP) and (ID) to know what it catch
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNP)#append what you array
        IDs.append(ID)#same
        cv2.imshow("training",faceNP)
        cv2.waitKey(10)
    return np.array(IDs),faces

Ids,faces=getImageWithID(path)
recon.train(faces,Ids)
recon.save('reconizer/train.yml')
cv2.destroyAllWindows()