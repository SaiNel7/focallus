import numpy as np
import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mpFaceMesh = mp.solutions.face_mesh
face_mesh = mpFaceMesh.FaceMesh(max_num_faces=1)

# Drawing utilities
mpDrawing = mp.solutions.drawing_utils
drawingSpec = mpDrawing.DrawingSpec(thickness=1, circle_radius=1)

# Start webcam
cap = cv2.VideoCapture(0)

# Eye landmark indices (improved)
LEFT_EYE = [33, 133, 160, 159, 158, 157, 173]
RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398]

# Distraction counter
distraction_counter = 0  # Tracks distraction duration

# Get bounding box around the eye
def get_eye_bbox(eye_landmarks):
    x_coords = [point[0] for point in eye_landmarks]
    y_coords = [point[1] for point in eye_landmarks]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

# Check for circular contours
def is_circle(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)
    if perimeter == 0:
        return False
    circularity = 4 * np.pi * (area / (perimeter ** 2))
    return 0.5 < circularity < 1.2  # Range for circular shapes

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mpDrawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mpFaceMesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawingSpec,
                connection_drawing_spec=drawingSpec)

            h, w, _ = frame.shape
            left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayFrame = cv2.equalizeHist(grayFrame)
            grayFrame = cv2.GaussianBlur(grayFrame, (7, 7), 0)

            for eye in [left_eye, right_eye]:
                x1, y1, x2, y2 = get_eye_bbox(eye)
                eye_region = grayFrame[y1:y2, x1:x2]

                # Adaptive thresholding
                threshold = cv2.adaptiveThreshold(
                    eye_region, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    11, 2
                )

                contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 30 < area < 300 and is_circle(contour):
                        (px, py), radius = cv2.minEnclosingCircle(contour)
                        center = (int(px) + x1, int(py) + y1)
                        cv2.circle(frame, center, int(radius), (255, 0, 0), 2)

            # ✅ Distraction Detection Logic (Integrated Here)
            center_x = (left_eye[0][0] + right_eye[0][0]) // 2  
            frame_center = frame.shape[1] // 2                 

            # Detect if gaze shifts significantly from center
            if abs(center_x - frame_center) > 50:  
                distraction_counter += 1           
            else:
                distraction_counter = 0          

            # Trigger distraction alert (e.g., buzz the bracelet)
            if distraction_counter > 60:  # ~1 second if running at 30 FPS
                print("Distracted! Buzz the bracelet.")

    frame = cv2.resize(frame, (640, 360))
    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
