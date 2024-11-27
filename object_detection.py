import torch
import cv2
from gtts import gTTS
import os
import time

# Load the YOLOv5 model from PyTorch Hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()  # Set to evaluation mode

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use '0' for the default camera

def detect_objects(frame):
    # Resize and prepare the frame for YOLOv5
    img_resized = cv2.resize(frame, (640, 640))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Run object detection
    results = model(img_rgb)
    return results.pandas().xyxy[0]  # Returns detected objects in a DataFrame

def speak(text):
    # Use Google Text-to-Speech (gTTS) to convert text to audio
    tts = gTTS(text=text, lang='en')
    tts.save("detected_object.mp3")
    os.system("start detected_object.mp3")  # 'start' for Windows; 'afplay' for Mac; 'mpg321' for Linux

def main_loop():
    last_object_spoken = ""  # Keep track of the last object to avoid repetition

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        # Detect objects in the current frame
        detected_objects = detect_objects(frame)

        # Annotate frame and give voice feedback for detected objects
        for _, obj in detected_objects.iterrows():
            obj_name = obj['name']
            confidence = obj['confidence']
            x1, y1, x2, y2 = int(obj['xmin']), int(obj['ymin']), int(obj['xmax']), int(obj['ymax'])
            
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{obj_name} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Speak the name of the object if it's new
            if confidence > 0.5 and obj_name != last_object_spoken:
                speak(f"I see a {obj_name}")
                last_object_spoken = obj_name  # Update last spoken object
                time.sleep(2)  # Delay to prevent overlapping audio

        # Display the frame with annotations
        cv2.imshow("Object Detection", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Run the main detection and feedback loop
if __name__ == "__main__":
    main_loop()

