import cv2
import mediapipe as mp

#mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7
)
mp_draw = mp.solutions.drawing_utils

#webcam open
cap = cv2.VideoCapture(0)

while True:
    succes, frame = cap.read()
    if not succes:
        continue
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb)
    
    #is hand detected check
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            
    #show results
    cv2.imshow("Hand Detection", frame)
    
    #'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()