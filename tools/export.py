from ultralytics import YOLO
from pathlib import Path

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    model = YOLO(str(root_dir / 'models' / 'best.pt'))
    model.export(format="engine", device=0, imgsz=640, half=True, workspace=2)

