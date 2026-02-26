import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import sys
from pathlib import Path
from time import sleep

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset.utils import draw_landmarks_on_image

# --- 1. SETUP THE TWO-HANDED CSV FILE ---
csv_file = ROOT_DIR / "dataset" / "jutsu_dataset_hands.csv"
if not csv_file.exists():
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Create headers for Hand 1 and Hand 2
        h1_headers = [f"h1_joint_{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']]
        h2_headers = [f"h2_joint_{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']]
        writer.writerow(h1_headers + h2_headers + ['label', 'image_file'])

# Create images folder if it doesn't exist
images_dir = ROOT_DIR / "images"
images_dir.mkdir(exist_ok=True)

base_options = python.BaseOptions(model_asset_path=str(ROOT_DIR / "models" / "hand_landmarker.task"))
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# --- OPEN THE WEBCAM ---
cap = cv2.VideoCapture(0)
print("Opening webcam... Press 'q' to quit!")

img_counter = 0  # To keep track of saved images

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

    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('s'):
        print("Get ready! Countdown starting...")
        # Countdown for 3 seconds
        for countdown in range(3, 0, -1):
            # Capture a fresh frame for each countdown step
            success, frame = cap.read()
            if success:
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = detector.detect(mp_image)
                annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
                bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                
                # Display countdown text
                cv2.putText(bgr_annotated_image, str(countdown), (250, 250), 
                           cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 10)
                cv2.imshow('Jutsu Tracker - Official API Style', bgr_annotated_image)
                cv2.waitKey(1000)  # Wait 1 second
        
        # Capture final frame after countdown
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
            bgr_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            
            if detection_result.hand_landmarks:
                hand_data = []
                for hand_landmarks in detection_result.hand_landmarks:
                    for landmark in hand_landmarks:
                        hand_data.extend([landmark.x, landmark.y, landmark.z])
                # Pad with zeros if only one hand is detected
                while len(hand_data) < 126:  # 21 joints * 3 coordinates * 2 hands
                    hand_data.append(0.0)
                label = "Snake"
                img_path = images_dir / f"{label}_{img_counter}.jpg"
                cv2.imwrite(str(img_path), frame)  # Save original frame without landmarks
                with open(csv_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(hand_data + [label, str(img_path.relative_to(ROOT_DIR))])
                print(f"Data saved for {img_path} with label '{label}'")
                img_counter += 1
            else:
                print("No hands detected. Please try again.")
cap.release()
cv2.destroyAllWindows()