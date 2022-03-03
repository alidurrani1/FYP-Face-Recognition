import cv2
import numpy as np
import face_recognition

image = face_recognition.load_image_file('images/ali.jpg')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

image_test = face_recognition.load_image_file('images/ali.jpg')
image_test = cv2.cvtColor(image_test,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(image)[0]
encode_img = face_recognition.face_encodings(image)[0]
cv2.rectangle(image,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(0,255,0),1)

faceloc_test = face_recognition.face_locations(image_test)[0]
encode_imgtest = face_recognition.face_encodings(image_test)[0]
cv2.rectangle(image_test,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(0,255,0),1)

results = face_recognition.compare_faces([encode_img],encode_imgtest)
faceDis = face_recognition.face_distance([encode_img],encode_imgtest)
cv2.putText(image_test,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_TRIPLEX,1,(0,0,255),2)
print(results,faceDis)


#cv2.imshow('Ali Durrani',image)
cv2.imshow('Ali Durrani',image_test)

cv2.waitKey(0)