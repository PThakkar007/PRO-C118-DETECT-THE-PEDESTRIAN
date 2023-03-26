import cv2

# Create our body classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    # Read first frame
    ret, frame = cap.read()

    # Convert each frame into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame with rectangles drawn around detected bodies
    cv2.imshow('Pedestrians', frame)

    # Exit loop if the space key is pressed
    if cv2.waitKey(1) == 32:
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
