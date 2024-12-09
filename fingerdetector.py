import cv2
import time
import mediapipe as mp

# Constants for colors and camera size
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
wCam, hCam = 640, 480

# Camera configuration
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Function to sum values in a dictionary
def sum_dict(dict):
    result = 0
    for key in dict:
        result += dict[key]
    return result

while True:
    fingers = {"4": 0, "8": 0, "12": 0, "16": 0, "20": 0}  # For first hand
    fingers2 = {"4": 0, "8": 0, "12": 0, "16": 0, "20": 0}  # For second hand

    success, img = cap.read()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Finger recognition
    if results.multi_hand_landmarks:
        for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw hand landmarks and connections
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Determine which hand is left and which is right
            hand_label = results.multi_handedness[hand_no].classification[0].label

            # Extract landmark positions
            landmarks = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((id, cx, cy))

            if len(landmarks) != 0:
                # Thumb
                if hand_label == 'Left':  # Sol əl
                    if landmarks[4][1] > landmarks[3][1]:
                        fingers["4"] = 1
                else:  # Sağ əl
                    if landmarks[4][1] < landmarks[3][1]:  # Right hand logic
                        fingers2["4"] = 1

                # Index finger
                if landmarks[8][2] < landmarks[6][2]:
                    if hand_label == 'Left':
                        fingers["8"] = 1
                    else:
                        fingers2["8"] = 1

                # Middle finger
                if landmarks[12][2] < landmarks[10][2]:
                    if hand_label == 'Left':
                        fingers["12"] = 1
                    else:
                        fingers2["12"] = 1

                # Ring finger
                if landmarks[16][2] < landmarks[14][2]:
                    if hand_label == 'Left':
                        fingers["16"] = 1
                    else:
                        fingers2["16"] = 1

                # Pinky
                if landmarks[20][2] < landmarks[18][2]:
                    if hand_label == 'Left':
                        fingers["20"] = 1
                    else:
                        fingers2["20"] = 1

    # Sum the number of detected fingers for both hands
    numberOfFingers1 = sum_dict(fingers)
    numberOfFingers2 = sum_dict(fingers2)

    # Total number of fingers
    totalFingers = numberOfFingers1 + numberOfFingers2

    # Display the number of fingers with enhanced text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    font_thickness = 8

    # Create a shadow effect
    cv2.putText(img, f'{totalFingers}', (40, 310), font, font_scale, BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, f'{totalFingers}', (35, 305), font, font_scale, WHITE, 5, cv2.LINE_AA)

    # Display the FPS 
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (15, 65), font, 1.5, WHITE, 2, cv2.LINE_AA)

    # Show the image
    cv2.imshow("Finger counting", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
