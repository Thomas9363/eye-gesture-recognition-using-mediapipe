import cv2
import numpy as np
import mediapipe as mp
import csv

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Define the eye and iris landmark indices, including iris centers
right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
left_iris_indices = [474, 475, 476, 477, 473]  # Include iris center point 468
right_iris_indices = [469, 470, 471, 472, 468]  # Include iris center point 473
all_eye_indices = left_eye_indices + left_iris_indices + right_eye_indices + right_iris_indices

# CSV file path
csv_file_path = 'iris_gesture_data.csv'

# Initialize the count dictionary
data_count = {i: 0 for i in range(10)}  # Assuming labels are 0-9

# Function to normalize landmarks
def normalize_landmarks(landmarks, indices):
    left_eye_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in left_eye_indices], axis=0)
    right_eye_center = np.mean([[landmarks[i].x, landmarks[i].y] for i in right_eye_indices], axis=0)
    mid_point = (left_eye_center + right_eye_center) / 2.0
    normalized = np.array([[landmarks[i].x - mid_point[0], landmarks[i].y - mid_point[1]] for i in indices])
    return normalized.flatten()

# Function to read the existing data from CSV
def read_csv_data(file_path):
    data = []
    with open(file_path, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Skip header
        for row in csv_reader:
            if row and row[0].isdigit():
                data.append(row)
                label = int(row[0])
                if label in data_count:
                    data_count[label] += 1
    return data

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

    return enlarged_frame

# Read the existing data
existing_data = read_csv_data(csv_file_path)

# Print existing data count for each label
print("Existing points per class:")
start_data_points_str = ", ".join([f"Class {class_label}:({count})" for class_label, count in data_count.items() if count > 0])
print(start_data_points_str)

# Start capturing video from the camera
cap = cv2.VideoCapture(0)
cv2.namedWindow('Data Collection', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('Data Collection', 640, 360)
cv2.namedWindow('Enlarged Eyes', cv2.WINDOW_NORMAL)
cv2.moveWindow('Data Collection', 100, 100)
cv2.moveWindow('Enlarged Eyes', 100, 460)

print("Press 0, 1, 2, 3, ... to record iris gesture data with the respective label.")
print("Press 'q' to quit.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                enlarged_frame = extract_and_plot_eye_indices(frame, face_landmarks)
                cv2.imshow('Enlarged Eyes', enlarged_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    break
                elif key in [ord(str(i)) for i in range(10)]:
                    label = int(chr(key))
                    normalized_landmarks = normalize_landmarks(face_landmarks.landmark, all_eye_indices)
                    row = [label] + normalized_landmarks.tolist()
                    existing_data.append(row)
                    data_count[label] += 1
                    print(f"Recorded gesture with label {label}. Data points per class:")
                    data_points_str = ", ".join([f"Class {class_label}: {count} data points" for class_label, count in data_count.items() if count > 0])
                    print(data_points_str)

        cv2.imshow('Data Collection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Sort the data by class label
    sorted_data = sorted(existing_data, key=lambda x: int(x[0]))

    # Write the sorted data back to the CSV file
    with open(csv_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        header = ['label']
        for i in range(len(all_eye_indices)):
            header.append(f'x{i}')
            header.append(f'y{i}')
        csv_writer.writerow(header)
        # Write the sorted data
        csv_writer.writerows(sorted_data)

    # Release the capture and close any open windows
    cap.release()
    cv2.destroyAllWindows()

    print("Final data points per class:")
    final_data_points_str = ", ".join([f"Class {class_label}:({count})" for class_label, count in data_count.items() if count > 0])
    print(final_data_points_str)
