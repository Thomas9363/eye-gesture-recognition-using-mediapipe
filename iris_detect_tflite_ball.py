import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

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
eye_iris_indices = left_eye_indices + left_iris_indices + right_eye_indices + right_iris_indices
all_eye_indices = eye_iris_indices
width = 640 # 960
height = 360 # 540

# Load the iris image with alpha channel
iris_img = cv2.imread('iris_image.png', cv2.IMREAD_UNCHANGED)  # Ensure this image has an alpha channel

# Resize the iris image
iris_radius = 48
iris_img = cv2.resize(iris_img, (2 * iris_radius, 2 * iris_radius))

# Load the eye image with alpha channel
eye_img = cv2.imread('eye_image.png', cv2.IMREAD_UNCHANGED)  # Ensure this image has an alpha channel
eye_radius = 80
# Resize the eye image
eye_img = cv2.resize(eye_img, (2 * eye_radius, 2 * eye_radius))

# Center of the eyes
eye_centers = [(width // 3, height // 2), (2 * width // 3, height // 2)]


# Function to extract and plot eye indices
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


# Function to normalize landmarks
def normalize_landmarks(landmarks, indices):
    # Compute the midpoint between the two eyes
    left_eye_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in left_eye_indices], axis=0)
    right_eye_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in right_eye_indices], axis=0)
    mid_point = (left_eye_center + right_eye_center) / 2.0

    normalized = np.array([[landmarks[i].x - mid_point[0], landmarks[i].y - mid_point[1]] for i in indices])
    return normalized.flatten()


# Labels for gestures
labels = ['up', 'down', 'right', 'left', 'center', 'both close', 'left close', 'right close']
# Ball properties for two eyes
ball_positions = [eye_centers[0], eye_centers[1]]
ball_speed = 20


def move_ball(gesture):
    global ball_positions

    if gesture == "close":
        return  # Do nothing if the gesture is "close"

    for i in range(len(ball_positions)):
        ball_x, ball_y = ball_positions[i]

        if gesture == "center":
            ball_positions[i] = eye_centers[i]
            continue

        if gesture == "up":
            ball_y -= ball_speed
        elif gesture == "down":
            ball_y += ball_speed
        elif gesture == "left":
            ball_x += ball_speed
        elif gesture == "right":
            ball_x -= ball_speed

        # Calculate the distance from the eye center to the ball center
        dx = ball_x - eye_centers[i][0]
        dy = ball_y - eye_centers[i][1]
        distance = np.sqrt(dx ** 2 + dy ** 2)

        # If the ball is outside the eye circle, move it back inside
        if distance > (eye_radius - iris_radius):
            angle = np.arctan2(dy, dx)
            ball_x = int(eye_centers[i][0] + (eye_radius - iris_radius) * np.cos(angle))
            ball_y = int(eye_centers[i][1] + (eye_radius - iris_radius) * np.sin(angle))

        ball_positions[i] = (ball_x, ball_y)


# Overlay function
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]

    alpha = alpha_mask[y1o:y2o, x1o:x2o, np.newaxis]
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop


def draw_half_closed_eyelid(img, center, radius):
    # Draw a half-circle (semi-transparent black) over the left eye to simulate half-closed eyelid
    overlay = img.copy()
    cv2.ellipse(overlay, center, (radius, radius // 2), 0, 0, 180, (0, 0, 0), -1)  # Black half-ellipse
    alpha = 0.4  # Transparency factor
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # frame = cv2.flip(frame, 1)
    # Convert the frame to RGB as MediaPipe expects RGB images
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to find faces and iris landmarks
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Draw the face landmarks on the frame
            # mp.solutions.drawing_utils.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_IRISES)
            enlarged_frame = extract_and_plot_eye_indices(frame, face_landmarks)
            cv2.imshow('Enlarged Eyes', enlarged_frame)

            # Normalize the landmarks
            normalized_landmarks = normalize_landmarks(face_landmarks.landmark, eye_iris_indices)
            input_data = np.array(normalized_landmarks, dtype=np.float32).reshape(1, -1)

            # Set the tensor to point to the input data to be inferred
            interpreter.set_tensor(input_details[0]['index'], input_data)

            # Run inference
            interpreter.invoke()

            # Get the result
            output_data = interpreter.get_tensor(output_details[0]['index'])
            predicted_label = np.argmax(output_data[0])
            gesture_name = labels[predicted_label]
            move_ball(gesture_name)
            # Display the predicted gesture
            cv2.putText(frame, f'Gesture: {labels[predicted_label]}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Iris Gesture Detection', frame)
    # Create a blank image for the eye window
    eye_window = np.zeros((height, width, 3), np.uint8) * 255

    for i, center in enumerate(eye_centers):
        # Overlay the eye image
        overlay_image_alpha(eye_window, eye_img[:, :, :3],
                            (center[0] - eye_radius, center[1] - eye_radius),
                            eye_img[:, :, 3] / 255.0)

        # Draw the appropriate eyelid or iris based on the gesture
        if i == 0 and gesture_name in ["both close", "right close"]:
            draw_half_closed_eyelid(eye_window, eye_centers[0], eye_radius)
        elif i == 1 and gesture_name in ["both close", "left close"]:
            draw_half_closed_eyelid(eye_window, eye_centers[1], eye_radius)
        else:
            # Overlay the iris image only if the eye is not closed
            overlay_image_alpha(eye_window, iris_img[:, :, :3],
                                (ball_positions[i][0] - iris_radius, ball_positions[i][1] - iris_radius),
                                iris_img[:, :, 3] / 255.0)
    cv2.putText(eye_window, f'Gesture: {labels[predicted_label]}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 125, 255), 2, cv2.LINE_AA)
    cv2.imshow('Eyeball Movement', eye_window)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any open windows
cap.release()
cv2.destroyAllWindows()
