import cv2
import numpy as np
import face_recognition
import os

path = 'images'
images = []
classnames = []

mylist = os.listdir(path)
print(mylist)

for cls in mylist:
    imgs = cv2.imread(f'{path}/{cls}')
    images.append(imgs)
    classnames.append(os.path.splitext(cls)[0])

print(classnames)

def encodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)

    return encodelist

encodelistknown = encodings(images)
print("Encoding Completes")

realtime = cv2.VideoCapture(0)

while True:
    ret, frame = realtime.read()
    #check = cv2.resize(frame,(0,0),None,0.25,0.25)
    check = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(check)
    encode = face_recognition.face_encodings(check,faces)

    for encodeface, location in zip(encode,faces):
        matches = face_recognition.compare_faces(encodelistknown,encodeface)
        face_ds = face_recognition.face_distance(encodelistknown, encodeface)
        print(face_ds)
        matchIndex = np.argmin(face_ds)

        if matches[matchIndex]:
            name = classnames[matchIndex].upper()
            print(name)

    cv2.imshow(frame)

realtime.release()

cv2.destroyAllWindows()


# faceloc = face_recognition.face_locations(image)[0]
# encode_img = face_recognition.face_encodings(image)[0]
# cv2.rectangle(image,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(0,255,0),1)
#
# faceloc_test = face_recognition.face_locations(image_test)[0]
# encode_imgtest = face_recognition.face_encodings(image_test)[0]
# cv2.rectangle(image_test,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(0,255,0),1)


