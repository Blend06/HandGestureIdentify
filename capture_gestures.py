import cv2
import mediapipe as mp
import numpy as np
import csv
import os

CSV_FILE = "gesture_data.py"

GESTURE_KEYS = {
    ord("1"): "Thumbs_Up",
    ord("2"): "Peace",
    ord("3"): "OK",
    ord("4"): "Fist",
    ord("5"): "Stop"
}

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["fx{i}" for i in range(1,22)] +[f"y{i}" for i in range(1, 22)] + ["label"]
        writer.writerow(header)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils
    
cap = cv2.VideoCapture(0)

while True:
    succes, frame = cap.read()
    if not succes:
        continue
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        #landmarks to flat list
        landmark_list = []
        for lm in hand_landmarks.landmark:
            landmark_list.append(lm.x)
        for lm in hand_landmarks.landmark:
            landmark_list.append(lm.y)
            
        
         #check if any gesture key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key in GESTURE_KEYS:
            label = GESTURE_KEYS[key]
            with open(CSV_FILE, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(landmark_list + [label])
            print(f"Saved sample for gesture: {label}")

    else:
        key = cv2.waitKey(1) & 0xFF

    # Display the frame
    cv2.imshow("Gesture Capture", frame)

    # Quit
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()    