import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv
import pandas as pd
from pathlib import Path

# --- 1. SETUP THE TWO-HANDED CSV FILE ---
ROOT_DIR = Path(__file__).resolve().parents[1]

csv_file = ROOT_DIR / "dataset" / "jutsu_dataset_hands.csv"
if not csv_file.exists():
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Create headers for Hand 1 and Hand 2
        h1_headers = [f"h1_joint_{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']]
        h2_headers = [f"h2_joint_{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']]
        writer.writerow(h1_headers + h2_headers + ['label', 'image_file'])

base_options = python.BaseOptions(model_asset_path=str(ROOT_DIR / "models" / "hand_landmarker.task"))
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

annotation_file = ROOT_DIR / "Hands Seals Naruto.v1i.tensorflow" / "train" / "_annotations.csv"
image_dir = ROOT_DIR / "Hands Seals Naruto.v1i.tensorflow" / "train"

# --- 2. READ ANNOTATION FILE AND PROCESS IMAGES ---
if not annotation_file.exists():
    print(f"Annotation file not found: {annotation_file}")
    exit(1)

annotation_frame = pd.read_csv(annotation_file)
annotations = list(zip(annotation_frame['filename'], annotation_frame['class']))

# --- 3. PROCESS EACH IMAGE AND EXTRACT LANDMARKS ---
processed_count = 0
skipped_count = 0

for image_path, label in annotations:
    # Construct full path to image
    full_image_path = image_dir / image_path
    
    if not full_image_path.exists():
        print(f"Image not found: {full_image_path}")
        skipped_count += 1
        continue
    
    # Read image
    image = cv2.imread(str(full_image_path))
    if image is None:
        print(f"Failed to read image: {full_image_path}")
        skipped_count += 1
        continue
    
    # Convert BGR to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
    
    # Detect hand landmarks
    detection_result = detector.detect(mp_image)
    
    # Only process if EXACTLY 2 hands are detected
    if len(detection_result.hand_landmarks) == 2:
        hand_data = []
        for hand_landmarks in detection_result.hand_landmarks:
            for landmark in hand_landmarks:
                hand_data.extend([landmark.x, landmark.y, landmark.z])
        
        # Write to CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(hand_data + [label, image_path])
        
        #print(f"Processed: {image_path} -> {label}")
        processed_count += 1
    else:
        #print(f"Skipped: {image_path} (Expected 2 hands, found {len(detection_result.hand_landmarks)})")
        skipped_count += 1

print(f"\n=== Processing Complete ===")
print(f"Processed: {processed_count}")
print(f"Skipped: {skipped_count}")
print(f"Total: {processed_count + skipped_count}")

