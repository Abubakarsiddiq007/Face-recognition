import cv2 # type: ignore
from deepface import DeepFace # type: ignore

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Check if the cascade file loaded properly
if face_cascade.empty():
    raise IOError("Failed to load face cascade classifier.")

# Start video capture from the default camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Process each detected face
    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]

        try:
            # Analyze the face region for emotion
            result = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
        except Exception as e:
            emotion = "Unknown"
            print("Emotion detection error:", e)

        # Draw rectangle and label on each face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,
                    emotion,
                    (x, y - 10),
                    font,
                    0.9,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA)

    # Show the video frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(3) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
