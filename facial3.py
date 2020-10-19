import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()
cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # perform face detection
    bboxes = classifier.detectMultiScale(gray)

    # detect faces in the image
    faces = detector.detect_faces(frame)
    for face in faces:
        print(face)
        x, y, width, height = face['box']
        x2, y2 = x + width, y + height
        cv2.rectangle(gray, (x, y), (x2, y2), (255,0,0), 5)


    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()