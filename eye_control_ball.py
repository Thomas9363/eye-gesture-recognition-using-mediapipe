import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
eye = ""
debounce_time = 1  # seconds
last_event_time = 0

# Ball properties
width = 640
height = 480
ball_radius = 20
ball_color = (255, 255, 255)
ball_speed = 5
ball_x = width // 2
ball_y = height//4*3
def move_ball(gesture):
    global ball_x, ball_y, ball_speed, ball_color
    if gesture == "eye move left":
        ball_x += ball_speed
    elif gesture == "eye move right":
        ball_x -= ball_speed
    elif gesture == "left eye close":
        ball_color = (255, 0, 0)
    elif gesture == "right eye close":
        ball_color = (0, 255, 0)
    elif gesture == "both eye close":
        ball_color = (0, 0, 255)
    else:
        ball_color = (255, 255, 255)

    # Ensure the ball stays within the window boundaries
    ball_x = np.clip(ball_x, ball_radius, width - ball_radius)
    ball_y = np.clip(ball_y, ball_radius, height - ball_radius)

# Function to calculate Euclidean distance between two points
def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance


# Function to calculate moving ratio
def iris_position(iris_center, point1, point2):
    center_to_point1 = euclidean_distance(iris_center, point1)
    center_to_point2 = euclidean_distance(iris_center, point2)
    point1_to_point2 = euclidean_distance(point1, point2)
    ratio = center_to_point1 / point1_to_point2
    return ratio

def calculate_fps(prev_time, prev_fps):
    current_time = time.time()
    fps = 0.9 * prev_fps + 0.1 * (1 / (current_time - prev_time))
    return fps, current_time

# Initialize Video Capture
cap = cv2.VideoCapture(0)

prev_time = time.time()
prev_fps = 0

with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame =cv2.flip(frame,1)
        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('output', 640, 480)
        cv2.moveWindow('output', 300, 100)
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        current_time = time.time()
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark  # get the landmarks of the first face
            points = [(landmark.x, landmark.y) for landmark in landmarks]  # extract the x and y coordinates
            p = np.array(  # convert landmarks to a numpy array  and scale it using np.multiply
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in landmarks])
            if len(points) > 2:  # key landmarks around right eye. They are used to calculate movement
                ratioRH = iris_position(p[468], p[33], p[133])
                #                 print ("{:.2f}".format(ratioH))
                eye_width_R = euclidean_distance(p[33], p[133])
                eye_width_L = euclidean_distance(p[362], p[263])
                eye_height_R = euclidean_distance(p[159], p[145])
                eye_height_L = euclidean_distance(p[386], p[374])
                ratioROC = eye_height_R / eye_width_R
                ratioLOC = eye_height_L / eye_width_L
                if ratioRH > 0.6:  # horizontal movement to left
                    if current_time - last_event_time > debounce_time:
                        eye_move = "eye move left"
                        print("eye move left  ", ",", "{:.2f}".format(ratioRH))
                        last_event_time = current_time
                #                     print("eye move left  ", ",", "{:.2f}".format(ratioRH))
                elif ratioRH <= 0.6 and ratioRH >= 0.4:  # eye at center, eyelid open/ close
                    if (ratioROC < 0.15) and (ratioLOC > 0.15):
                        if current_time - last_event_time > debounce_time:
                            eye_move = "right eye close"
                            print("right eye close")
                            last_event_time = current_time
                    #                         print("right eye close")
                    elif (ratioROC > 0.15) and (ratioLOC < 0.15):
                        if current_time - last_event_time > debounce_time:
                            eye_move = "left eye close"
                            print("left eye close")
                            last_event_time = current_time
                    #                         print("left eye close")
                    elif (ratioROC < 0.15) and (ratioLOC < 0.15):
                        if current_time - last_event_time > debounce_time:
                            eye_move = "both eye close"
                            print("both eye close")
                            last_event_time = current_time
                    #                         print("both eye close")
                    else:
                        if current_time - last_event_time > debounce_time:
                            eye_move = "eye move center"
                            print("eye move center")
                            last_event_time = current_time
                 #                         print("eye move center")
                elif ratioRH < 0.4:
                    if current_time - last_event_time > debounce_time:
                        eye_move = "eye move right"
                        print("eye move right  ", ",", "{:.2f}".format(ratioRH))
                        last_event_time = current_time
                #                     print("eye move right  ", ",", "{:.2f}".format(ratioRH))
                cv2.circle(frame, p[159], 2, (0, 255, 0), -1)
                cv2.circle(frame, p[145], 2, (0, 255, 0), -1)
                cv2.circle(frame, p[33], 2, (255, 255, 0), -1)
                cv2.circle(frame, p[133], 2, (255, 255, 0), -1)
                cv2.circle(frame, p[468], 2, (0, 255, 255), -1)
                cv2.line(frame, p[159], p[145], (255, 255, 255), 1)
        cv2.putText(frame, f'Gesture: {eye_move}', (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)
        fps, prev_time = calculate_fps(prev_time, prev_fps)  # Calculate and display FPS
        prev_fps = fps
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(frame, (ball_x, ball_y), ball_radius, ball_color, -1)
        move_ball(eye_move)
        cv2.imshow('output', frame)
        # Create a blank image for the ball window
        # move_ball(eye_move)
        # ball_window = np.zeros((height, width, 3), np.uint8) * 255
        # cv2.circle(ball_window, (ball_x, ball_y), ball_radius, ball_color, -1)
        # cv2.imshow('Ball Movement', ball_window)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
