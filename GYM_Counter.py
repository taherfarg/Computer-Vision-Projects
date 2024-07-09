import cv2
import mediapipe as mp
import numpy as np
import time

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set the camera window size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize variables for counting reps
right_countr = 0
left_countr = 0
start_time = time.time()
display_text = False
text_display_time = 0

# Initialize MediaPipe Pose
with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # Process the frame with MediaPipe Pose
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = pose.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Extract landmarks and calculate angles
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Right arm landmarks
            r_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                          landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            r_elbow = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].x,
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value].y]
            r_wrist = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x,
                       landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Draw lines and points for right arm
            if all(r_shoulder) and all(r_elbow):
                cv2.line(frame, (int(r_shoulder[0] * frame.shape[1]), int(r_shoulder[1] * frame.shape[0])),
                         (int(r_elbow[0] * frame.shape[1]), int(r_elbow[1] * frame.shape[0])), (0, 0, 255), 2)
                cv2.circle(frame, (int(r_shoulder[0] * frame.shape[1]), int(r_shoulder[1] * frame.shape[0])), 5, (0, 0, 255), -1)
                cv2.circle(frame, (int(r_elbow[0] * frame.shape[1]), int(r_elbow[1] * frame.shape[0])), 5, (0, 0, 255), -1)
            if all(r_elbow) and all(r_wrist):
                cv2.line(frame, (int(r_elbow[0] * frame.shape[1]), int(r_elbow[1] * frame.shape[0])),
                         (int(r_wrist[0] * frame.shape[1]), int(r_wrist[1] * frame.shape[0])), (0, 0, 255), 2)
                cv2.circle(frame, (int(r_wrist[0] * frame.shape[1]), int(r_wrist[1] * frame.shape[0])), 5, (0, 0, 255), -1)

            # Left arm landmarks
            l_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                          landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
            l_elbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x,
                       landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y]
            l_wrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x,
                       landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y]

            # Draw lines and points for left arm
            if all(l_shoulder) and all(l_elbow):
                cv2.line(frame, (int(l_shoulder[0] * frame.shape[1]), int(l_shoulder[1] * frame.shape[0])),
                         (int(l_elbow[0] * frame.shape[1]), int(l_elbow[1] * frame.shape[0])), (0, 0, 255), 2)
                cv2.circle(frame, (int(l_shoulder[0] * frame.shape[1]), int(l_shoulder[1] * frame.shape[0])), 5, (0, 0, 255), -1)
                cv2.circle(frame, (int(l_elbow[0] * frame.shape[1]), int(l_elbow[1] * frame.shape[0])), 5, (0, 0, 255), -1)
            if all(l_elbow) and all(l_wrist):
                cv2.line(frame, (int(l_elbow[0] * frame.shape[1]), int(l_elbow[1] * frame.shape[0])),
                         (int(l_wrist[0] * frame.shape[1]), int(l_wrist[1] * frame.shape[0])), (0, 0, 255), 2)
                cv2.circle(frame, (int(l_wrist[0] * frame.shape[1]), int(l_wrist[1] * frame.shape[0])), 5, (0, 0, 255), -1)

            # Curl counter logic for right arm
            r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
            if r_angle > 160:
                right_stage = "down"
            if r_angle < 30 and right_stage == 'down':
                right_stage = "up"
                right_countr += 1
                print("Right count:", right_countr)
            
            # Curl counter logic for left arm
            l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
            if l_angle > 160:
                left_stage = "down"
            if l_angle < 30 and left_stage == 'down':
                left_stage = "up"
                left_countr += 1
                print("Left count:", left_countr)

            # Restart counter if arms are down for 30 seconds
            if (right_stage == 'down' or left_stage == 'down') and time.time() - start_time > 30:
                right_countr = 0
                left_countr = 0
                start_time = time.time()
                print("Counter reset")
                
            if (right_countr >= 10 or left_countr >= 10) and not display_text:
                display_text = True
                text_display_time = time.time()
                additional_text = "3ash Ya Wa7sh"
                text_size = cv2.getTextSize(additional_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                cv2.putText(frame, additional_text, (text_x, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
            if display_text and time.time() - text_display_time > 5:
                display_text = False

        except Exception as e:
            print("Error:", e)

        # Draw UI elements
        cv2.rectangle(frame, (0, 0), (300, 150), (52, 73, 94), -1)
        cv2.putText(frame, 'REPS', (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Right: {right_countr}  Left: {left_countr}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('MediaPipe Pose', frame)
        
        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
