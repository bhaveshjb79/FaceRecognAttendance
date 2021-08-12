import  cv2
import numpy as np
import face_recognition

#original image
imgemilia=face_recognition.load_image_file('emilia1.jpg')
#convert image into rgb
imgemilia=cv2.cvtColor(imgemilia,cv2.COLOR_BGR2RGB)

#test image
imgemiliatest=face_recognition.load_image_file('emilia.jpg')
#convert image into rgb
imgemiliatest=cv2.cvtColor(imgemiliatest,cv2.COLOR_BGR2RGB)

imgelon=face_recognition.load_image_file('elon musk.png')
imgelon=cv2.cvtColor(imgelon,cv2.COLOR_BGR2RGB);


faceLoc = face_recognition.face_locations(imgemilia)[0]
encodeemilia = face_recognition.face_encodings(imgemilia)[0]
cv2.rectangle(imgemilia, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)  # top, right, bottom, left

faceLocTest = face_recognition.face_locations(imgemiliatest)[0]
encodeTest = face_recognition.face_encodings(imgemiliatest)[0]
cv2.rectangle(imgemiliatest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results=face_recognition.compare_faces([encodeemilia],encodeTest)
facedis= face_recognition.face_distance([encodeemilia],encodeTest)
print(results,facedis)
cv2.putText(imgemiliatest,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

cv2.imshow('emilia clarke',imgemilia)
cv2.imshow('emilia clarke Test',imgemiliatest)
cv2.waitKey(0)