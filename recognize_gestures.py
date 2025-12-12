import cv2
import mediapipe as mp
import joblib
import numpy as np

MODEL_FILE = "gesture.pkl"

model = joblib.load(MODEL_FILE)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)as hands:
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                x_list = [lm.x for lm in hand_landmarks.landmark]
                y_list = [lm.y for lm in hand_landmarks.landmark]
                
                # Flatten: 21 x, followed by 21 y = 42 features
                landmark_vector = np.array(x_list + y_list).reshape(1, -1)
                
                prediction = model.predict(landmark_vector)[0]
                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                
                # Display prediction text
                cv2.putText(
                    frame,
                    f"Gesture: {prediction}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
        
        # Always show the frame (whether hand is detected or not)
        cv2.imshow("Real-Time Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()