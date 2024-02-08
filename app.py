import cv2 

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

# Open the video capture (0 for webcam)
cap = cv2.VideoCapture(0)

# Loop through the frames
while cap.isOpened():
    _, frame= cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 2)
    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces: 
        cv2.rectangle(frame, (x,y), (x+w , y+h), (0, 255, 0),3)
        roi_gray= gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255, 0, 0),3)
        # Check for 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('frame',frame)

# Release the video capture
cap.release()
cv2.destroyAllWindows()