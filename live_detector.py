import cv2
import time
from ultralytics import YOLO
from stabilizer import Stabilizer
from logger import Logger
from register import Register

# 1. INITIALIZE THE MODEL, STABILIZER, AND LOGGER
MODEL_PATH = 'bests.pt'
CLASS = "Tiger"

model = YOLO(MODEL_PATH)
stabilizer = Stabilizer(window_frames=10, uptime_threshold=0.5)
current_stable_class = None
logger = Logger(model_path=MODEL_PATH, logs_directory="logs", max_records=20)
register = Register(image_directory="images")

# 2. OPEN THE WEBCAM
cap = cv2.VideoCapture(0)
print("Sharingan Activated! Looking for Jutsus... Press 'q' to quit.")
print(f"Logging predictions to: {logger.log_file_path}")

prev_time = time.perf_counter()
fps = 0.0
frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame_idx += 1
    
    frame = cv2.flip(frame, 1) 
    
    # 3. RUN THE FRAME THROUGH YOLO
    
    results = model.predict(
        frame,  # The current video frame from the webcam
        stream=True,  # Stream mode for faster processing
        conf=0.5,  # Confidence threshold (adjust as needed)
        iou=0.5,   # IoU threshold for non-max suppression
        agnostic_nms=True  # Class-agnostic NMS to handle multiple classes better
    )  # Get predictions for the current frame

    # 4. DRAW THE PREDICTIONS
    annotated_frame = frame
    classes_in_frame = set()
    for r in results:
        annotated_frame = r.plot()

        if r.boxes is not None and len(r.boxes) > 0:
            names = r.names if hasattr(r, "names") else model.names
            for box in r.boxes:
                if logger.logged_records >= logger.max_records:
                    break

                class_id = int(box.cls[0].item())
                class_name = names[class_id] if class_id in names else str(class_id)
                classes_in_frame.add(class_name)

                logger.log_prediction(class_name, fps)

    stable_class = stabilizer.update(classes_in_frame)
    if stable_class is not None:
        current_stable_class = stable_class
        print(f"Stabilized class: {current_stable_class}")

    saved_image_path = register.update(frame, classes_in_frame)
    if saved_image_path is not None:
        print(f"Saved no-class frame: {saved_image_path}")

    logger.flush()

    # 5. CALCULATE AND DRAW FPS
    current_time = time.perf_counter()
    delta = current_time - prev_time
    prev_time = current_time

    if delta > 0:
        instant_fps = 1.0 / delta
        fps = (0.9 * fps) + (0.1 * instant_fps) if fps > 0 else instant_fps

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    # 6. SHOW THE VIDEO ON SCREEN
    cv2.imshow('Jutsu Detector - YOLOv8', annotated_frame)

    # Press 'q' to close safely
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
logger.close()

precision, correct_predictions, total_predictions = logger.calculate_precision(CLASS)
summary_file_path, decision = logger.save_run_decision(CLASS, precision)
print(f"Default class: {CLASS}")
print(f"Correct predictions: {correct_predictions}/{total_predictions}")
print(f"Precision: {precision:.4f}")
print(f"Decision: {decision}")
print(f"Saved summary to: {summary_file_path}")
print(f"Current stable class at end of run: {current_stable_class}")