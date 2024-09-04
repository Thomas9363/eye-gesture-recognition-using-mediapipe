import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
from adafruit_servokit import ServoKit

# servo channels initization
kit = ServoKit(channels=16,address=0x40 )
LID_CHANNEL_L, PAN_CHANNEL_L,TILT_CHANNEL_L = 0, 1, 2
LID_CHANNEL_R, PAN_CHANNEL_R,TILT_CHANNEL_R = 4, 5, 6

lid_initial_angle=140
pan_initial_angle=90
tilt_initial_angle=90
eye_min, eye_max =-25, 25
lid_min, lid_max=90, 140
eye_center=90
kit.servo[PAN_CHANNEL_L].angle = pan_initial_angle
kit.servo[TILT_CHANNEL_L].angle = tilt_initial_angle
kit.servo[LID_CHANNEL_L].angle = lid_initial_angle
kit.servo[PAN_CHANNEL_R].angle = pan_initial_angle
kit.servo[TILT_CHANNEL_R].angle = tilt_initial_angle
kit.servo[LID_CHANNEL_R].angle = lid_initial_angle

def extract_and_plot_eye_indices(frame, face_landmarks):
    all_eye_points = np.array([[int(face_landmarks.landmark[i].x * frame.shape[1]),
                                int(face_landmarks.landmark[i].y * frame.shape[0])]
                               for i in all_eye_indices])

    ex, ey, ew, eh = cv2.boundingRect(all_eye_points)
    margin = 5
    ex = max(0, ex - margin)
    ey = max(0, ey - margin)
    ew = min(frame.shape[1], ew + 2 * margin)
    eh = min(frame.shape[0], eh + 2 * margin)
    eye_region = frame[ey:ey + eh, ex:ex + ew]
    new_height = int(frame.shape[1] * eh / ew)
    eye_enlarged = cv2.resize(eye_region, (frame.shape[1], new_height), interpolation=cv2.INTER_CUBIC)

    enlarged_frame = np.zeros((new_height, frame.shape[1], 3), dtype=np.uint8)
    enlarged_frame[:new_height, :frame.shape[1]] = eye_enlarged
    points = []
    for i in all_eye_indices:
        nx = int((face_landmarks.landmark[i].x * frame.shape[1] - ex) * frame.shape[1] / ew)
        ny = int((face_landmarks.landmark[i].y * frame.shape[0] - ey) * new_height / eh)
        points.append((nx, ny))
    points = np.array(points, dtype=np.int32)
    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(points[16:20])
    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(points[37:41])
    center_right = np.array([r_cx, r_cy], dtype=np.int32)
    center_left = np.array([l_cx, l_cy], dtype=np.int32)
    cv2.polylines(enlarged_frame, [points[:15]], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.circle(enlarged_frame, center_right, int(r_radius), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(enlarged_frame, points[20], 3, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.polylines(enlarged_frame, [points[21:36]], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.circle(enlarged_frame, center_left, int(l_radius), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(enlarged_frame, points[41], 3, (255, 255, 255), 2, cv2.LINE_AA)
    # cv2.line(enlarged_frame, points[0], points[8], (0, 255, 255), 1, cv2.LINE_AA)
    # cv2.line(enlarged_frame, points[4], points[12], (0, 255, 255), 1, cv2.LINE_AA)

    return enlarged_frame

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='iris_gesture_model.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Define the eye and iris landmark indices, including iris centers
right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
left_iris_indices = [474, 475, 476, 477, 473]  # Include iris center point 468
right_iris_indices = [469, 470, 471, 472, 468]  # Include iris center point 473

# Combine all the indices
eye_iris_indices = left_eye_indices + left_iris_indices +right_eye_indices +  right_iris_indices
all_eye_indices = eye_iris_indices

# Function to normalize landmarks
def normalize_landmarks(landmarks, indices):
    # Compute the midpoint between the two eyes
    left_eye_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in left_eye_indices], axis=0)
    right_eye_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in right_eye_indices], axis=0)
    mid_point = (left_eye_center + right_eye_center) / 2.0

    normalized = np.array([[landmarks[i].x - mid_point[0], landmarks[i].y - mid_point[1]] for i in indices])
    return normalized.flatten()

def calculate_fps(prev_time, prev_fps):
    current_time = time.time()
    fps = 0.9*prev_fps+ 0.1*(1 / (current_time - prev_time))
    return fps, current_time


# Labels for gestures
# labels = ['up', 'down', 'left', 'right', 'center', 'close']
labels = ['up', 'down', 'right', 'left', 'center', 'close', 'l_close', 'r_close']

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

prev_time = time.time()
prev_fps= 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB as MediaPipe expects RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find faces and iris landmarks
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            enlarged_frame = extract_and_plot_eye_indices(frame, face_landmarks)
            cv2.imshow('Enlarged Eyes', enlarged_frame)
            # Draw the face landmarks on the frame
            # mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_IRISES)

            # Normalize the landmarks
            normalized_landmarks = normalize_landmarks(face_landmarks.landmark, eye_iris_indices)
#             input_data = np.array(normalized_landmarks, dtype=np.float32).reshape(1, -1)
            input_data = np.array(normalized_landmarks, dtype=np.float32).reshape(input_details[0]['shape'])

            # Set the tensor to point to the input data to be inferred
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            interpreter.invoke()

            # Get the result
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_label = np.argmax(output_data[0])
            gesture_name = labels[predicted_label]
            if gesture_name == "up":
                kit.servo[TILT_CHANNEL_L].angle = eye_center+eye_min
                kit.servo[TILT_CHANNEL_R].angle = eye_center+eye_min
                kit.servo[PAN_CHANNEL_L].angle = eye_center
                kit.servo[PAN_CHANNEL_R].angle = eye_center
                kit.servo[LID_CHANNEL_L].angle = lid_max
                kit.servo[LID_CHANNEL_R].angle = lid_max
            elif gesture_name == "down":
                kit.servo[TILT_CHANNEL_L].angle = eye_center+eye_max
                kit.servo[TILT_CHANNEL_R].angle = eye_center+eye_max
                kit.servo[PAN_CHANNEL_L].angle = eye_center
                kit.servo[PAN_CHANNEL_R].angle = eye_center
                kit.servo[LID_CHANNEL_L].angle = lid_max
                kit.servo[LID_CHANNEL_R].angle = lid_max
            elif gesture_name == "left":
                kit.servo[PAN_CHANNEL_L].angle = eye_center+eye_min
                kit.servo[PAN_CHANNEL_R].angle = eye_center+eye_min
                kit.servo[TILT_CHANNEL_L].angle = eye_center
                kit.servo[TILT_CHANNEL_R].angle = eye_center
                kit.servo[LID_CHANNEL_L].angle = lid_max
                kit.servo[LID_CHANNEL_R].angle = lid_max
            elif gesture_name == "right":
                kit.servo[PAN_CHANNEL_L].angle = eye_center+eye_max
                kit.servo[PAN_CHANNEL_R].angle = eye_center+eye_max
                kit.servo[TILT_CHANNEL_L].angle = eye_center
                kit.servo[TILT_CHANNEL_R].angle = eye_center
                kit.servo[LID_CHANNEL_L].angle = lid_max
                kit.servo[LID_CHANNEL_R].angle = lid_max
            elif gesture_name == "center":
                kit.servo[PAN_CHANNEL_L].angle = eye_center
                kit.servo[PAN_CHANNEL_R].angle = eye_center
                kit.servo[TILT_CHANNEL_L].angle = eye_center
                kit.servo[TILT_CHANNEL_R].angle = eye_center
                kit.servo[LID_CHANNEL_L].angle = lid_max
                kit.servo[LID_CHANNEL_R].angle = lid_max
            elif gesture_name == "close":
                kit.servo[PAN_CHANNEL_L].angle = eye_center
                kit.servo[PAN_CHANNEL_R].angle = eye_center
                kit.servo[TILT_CHANNEL_L].angle = eye_center
                kit.servo[TILT_CHANNEL_R].angle = eye_center
                kit.servo[LID_CHANNEL_L].angle = lid_min
                kit.servo[LID_CHANNEL_R].angle = lid_min
            elif gesture_name == "l_close":
                kit.servo[PAN_CHANNEL_L].angle = eye_center
                kit.servo[PAN_CHANNEL_R].angle = eye_center
                kit.servo[TILT_CHANNEL_L].angle = eye_center
                kit.servo[TILT_CHANNEL_R].angle = eye_center
                kit.servo[LID_CHANNEL_L].angle = lid_min
                kit.servo[LID_CHANNEL_R].angle = lid_max
            elif gesture_name == "r_close":
                kit.servo[PAN_CHANNEL_L].angle = eye_center
                kit.servo[PAN_CHANNEL_R].angle = eye_center
                kit.servo[TILT_CHANNEL_L].angle = eye_center
                kit.servo[TILT_CHANNEL_R].angle = eye_center
                kit.servo[LID_CHANNEL_L].angle = lid_max
                kit.servo[LID_CHANNEL_R].angle = lid_min
                
               
            # Display the predicted gesture
    cv2.putText(frame, f'Gesture: {labels[predicted_label]}', (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)
    fps, prev_time = calculate_fps(prev_time, prev_fps)  # Calculate and display FPS
    prev_fps = fps
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Iris Gesture Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break
kit.servo[PAN_CHANNEL_L].angle = pan_initial_angle 
kit.servo[TILT_CHANNEL_L].angle = tilt_initial_angle
kit.servo[LID_CHANNEL_L].angle = lid_initial_angle
kit.servo[PAN_CHANNEL_R].angle = pan_initial_angle 
kit.servo[TILT_CHANNEL_R].angle = tilt_initial_angle
kit.servo[LID_CHANNEL_R].angle = lid_initial_angle
time.sleep(1)
# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
