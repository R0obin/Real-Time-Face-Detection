import cv2 

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _, frame= cap.read()

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.1, 2)
    for (x, y, w, h) in faces: 
     cv2.rectangle(frame, (x,y), (x+w , y+h), (0, 255, 0),3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
    cv2.imshow('frame',frame)

cap.release()

