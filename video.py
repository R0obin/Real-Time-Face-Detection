import cv2

read_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture('WhatsApp Video 2024-01-28 at 3.03.33 PM.mp4')

while video.isOpened:        
    _, vid = video.read()

    grey = cv2.cvtColor(vid,cv2.COLOR_BGR2GRAY)
    frame = read_cascade.detectMultiScale(grey,1.1,3)


    for (x, y, w, h) in frame:
        cv2.rectangle(vid,(x, y),(x+w , y+h),(0, 255, 0),3)
    cv2.imshow('vid',vid)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()