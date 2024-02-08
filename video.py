import cv2

read_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
video = cv2.VideoCapture('WhatsApp Video 2024-01-28 at 3.03.33 PM.mp4')

while video.isOpened:        
    _, vid = video.read()
    gray = cv2.cvtColor(vid,cv2.COLOR_BGR2GRAY)
    frame = read_cascade.detectMultiScale(gray,1.1,3)
    for (x, y, w, h) in frame:
        cv2.rectangle(vid,(x, y),(x+w , y+h),(0, 255, 0),3)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = vid[y:y+h,x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_color)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color,(ex, ey),(ex + ew , ey + eh),(255, 0, 0),3)

    cv2.imshow('vid',vid)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()