from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('bests.pt')
    model.export(format="engine", device=0, imgsz=320, half=True, workspace=2)

