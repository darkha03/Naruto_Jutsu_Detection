import csv
from datetime import datetime
from pathlib import Path


class Logger:
    def __init__(self, model_path, logs_directory="logs", max_records=20):
        self.model_path = Path(model_path)
        self.logs_dir = Path(logs_directory)
        self.logs_dir.mkdir(exist_ok=True)

        run_started_at = datetime.now()
        self.log_file_path = self.logs_dir / (
            f"predictions_{self.model_path.stem}_{run_started_at.strftime('%Y%m%d_%H%M%S')}.csv"
        )

        self.log_file = open(self.log_file_path, mode="w", newline="", encoding="utf-8")
        self.log_writer = csv.writer(self.log_file)
        self.log_writer.writerow(["class_name", "timestamp", "fps", "confidence"])

        self.max_records = max_records
        self.logged_records = 0

    def log_prediction(self, class_name, fps, confidence=0.0):
        if self.logged_records >= self.max_records:
            return False

        self.log_writer.writerow([
            class_name,
            datetime.now().isoformat(timespec="milliseconds"),
            round(fps, 2),
            round(confidence, 2)
        ])
        self.logged_records += 1
        return True

    def flush(self):
        self.log_file.flush()

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()

    def calculate_precision(self, default_class):
        total_predictions = 0
        correct_predictions = 0

        with open(self.log_file_path, mode="r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                class_name = row.get("class_name", "").strip()
                if not class_name:
                    continue

                total_predictions += 1
                if class_name.lower() == default_class.lower():
                    correct_predictions += 1

        if total_predictions == 0:
            return 0.0, 0, 0

        precision = correct_predictions / total_predictions
        return precision, correct_predictions, total_predictions

    def save_run_decision(self, class_name, precision):
        summary_file_path = self.logs_dir / "model_decisions.csv"
        file_exists = summary_file_path.exists()

        with open(summary_file_path, mode="a", newline="", encoding="utf-8") as summary_file:
            writer = csv.writer(summary_file)
            if not file_exists:
                writer.writerow(["model", "class_name", "decision"])
            writer.writerow([self.model_path.name, class_name, precision])

        return summary_file_path, precision
