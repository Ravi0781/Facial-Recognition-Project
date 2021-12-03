import numpy as np
import cv2
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("D:\haarcascade_frontalface_alt.xml")
X = []
face_section = 0
padding = 10
while True:
    ret, image = cap.read()
    if ret == False:
        continue
    faces = face_cascade.detectMultiScale(image)
    print(faces)
    for face in faces:
        x,y,w,h = face
        face_section = image[y-padding:y+h+padding, x-padding:x+h+padding]
        face_section = cv2.resize(face_section, (100,100))
    cv2.imshow("face section", face_section)
    X.append(face_section.reshape(1,-1))
    print(len(X), X[-1].shape)
    key_pressed = cv2.waitKey(25)
    if key_pressed == ord('q'):
        print(key_pressed)
        print('Q was pressed')
        break
cap.release()
cv2.destroyAllWindows()
X = np.array(X)
filename = input("Enter the name of the person:")
np.save(filename, X)



