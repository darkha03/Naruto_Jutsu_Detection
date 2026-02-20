import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import draw_landmarks_on_image


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# --- OPEN THE WEBCAM ---
cap = cv2.VideoCapture(0)
print("Opening webcam... Press 'q' to quit!")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    # Flip the image (mirror effect) and convert BGR to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- STEP 3: Load the input image (From webcam instead of .jpg) ---
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # --- STEP 4: Detect hand landmarks ---
    detection_result = detector.detect(mp_image)

    # --- STEP 5: Process and visualize the result ---
    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
    
    # Convert back to BGR so OpenCV can display it correctly
    bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Jutsu Tracker - Official API Style', bgr_annotated_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()