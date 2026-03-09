# 🌀 Naruto Jutsu Detection App

A real-time hand sign recognition system that uses a **two-stage vision pipeline** and an **intelligent sequence-matching engine** to detect Naruto jutsu from live camera input.

---

## 🚀 Project Summary

This project focuses on building a complete real-time recognition pipeline:

1. **YOLO detector** to localize hand regions in each frame
2. **MobileNetV3-Small classifier** to classify the detected hand sign from the cropped hand region
3. **Stateful matching engine** to validate and recognize full jutsu sequences

Instead of relying on strict exact matching, the system is designed to:

* Handle player mistakes
* Tolerate ML misclassifications
* Reduce possible matches progressively
* Maintain real-time responsiveness

The goal is to simulate a realistic recognition system where predictions are noisy and must be intelligently validated.

---

## 🎬 Demo

Demo video:

* https://github.com/user-attachments/assets/ee728095-e960-4cea-aca0-bf66d7e8e223

---

## 📁 Current Project Layout

* `app/` → executable runtime entrypoints
* `core/` → reusable detection, stabilization, chaining, logging, animation modules
* `dataset/` → dataset collection/preprocessing scripts and CSV
* `tools/` → utility scripts (e.g., export)
* `models/` → detector + classifier files (`bestn.engine`, `bests.pt`, `classifier.pt`, `classifier_labels.json`, etc.)
* `animations/` → jutsu animation videos
* `logs/` → prediction logs and summary decisions
* `images/` → captured frames/dataset images

---


## 🤖 Two-Stage Vision Pipeline

The vision system is split into two ML components with clear responsibilities:

### 1) Hand Detection (YOLO)

* Runs on full camera frames
* Detects the hand bounding box
* Provides the ROI (region of interest) for classification

### 2) Hand Sign Classification (MobileNetV3-Small)

* Takes cropped hand ROI from the detector output
* Predicts the hand-sign class with confidence
* Feeds predictions into stabilizer + sequence matcher

This design improves modularity and makes it easier to independently tune detection vs classification.

---

## 🧠 What I Built

### ✅ Jutsu Dictionary

A structured dictionary containing predefined valid hand-sign sequences.

### ✅ Temporary Input Buffer

Stores incoming predicted signs before validation to allow correction and flexible matching.

### ✅ Confusion Map

Defines commonly misclassified signs to compensate for ML prediction errors.

### ✅ Candidate Reduction System

Instead of checking all jutsu every time, the system:

* Initializes possible candidates from the first detected sign
* Reduces candidates step by step
* Eliminates invalid sequences progressively

### ✅ Error Tracking Per Candidate

Each possible jutsu maintains:

* Current position in sequence
* Error counter
* Validation state

This allows multiple potential matches to coexist while handling noisy ML outputs.

---

## 🔄 How It Works (High Level)

1. Camera captures frame
2. YOLO detects hand bounding box
3. Detected hand crop is sent to MobileNetV3-Small classifier
4. Predicted sign is stabilized over time
5. Candidate jutsu are updated and pruned
6. When a sequence completes successfully → Jutsu detected 🎯

---

## 🏗️ Key Engineering Concepts Demonstrated

* End-to-end ML + algorithm integration
* Real-time inference pipeline design
* Stateful algorithm design
* Error-tolerant sequence matching
* Handling noisy predictions
* Modular and scalable system architecture

---

## 🎯 Next Steps

* Improve model accuracy with more data
* Add confidence-based pruning
* Optimize inference performance
* Add visualization tools for debugging and demo

---

## 💡 Why This Project Matters

This project demonstrates the ability to:

* Train and integrate a machine learning model into a real-time system
* Design intelligent algorithms that compensate for ML imperfections
* Handle real-world uncertainty and noisy predictions
* Build modular architectures combining AI and classical software engineering

It showcases practical ML integration, system design thinking, and real-time processing skills relevant to AI-driven applications.

---

## ⚙️ Setup & Run

### Requirements

* Python 3.10+
* Webcam
* Optional: NVIDIA GPU (for faster inference)

### Install

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Run Main App

```bash
python -m app.live_detector
```

### Run Testbed

```bash
python -m app.live_detector_testbed
```

### Dataset Tools

```bash
python -m dataset.hand_track
python -m dataset.batch_process
```

### Model Export Tool

```bash
python -m tools.export
```

### Classifier Export Tool

```bash
python -m tools.export_classifier_tensorrt
```

---


## 📤 Runtime Outputs

* Prediction logs are written to `logs/predictions_*.csv`
* Run-level summaries are written to `logs/model_decisions.csv`
* Missing/unknown class snapshots are written to `images/` (via `Register`)

---

## 🛠️ Troubleshooting

* **`Import ... could not be resolved`**
	* Activate the same virtual environment where dependencies were installed:
		* `venv\Scripts\activate`
		* `pip install -r requirements.txt`

* **Camera does not open / black window**
	* Close other apps using the camera.
	* Try changing camera index in code (`cv2.VideoCapture(0, ...)` → `1` or `2`).

* **Model file errors (`.engine`, `.pt`, `.json`, or `.task` not found)**
	* Ensure files exist under `models/`:
		* detector: `bestn.engine` (or fallback `bests.pt`)
		* classifier: `classifier.pt` and `classifier_labels.json`
		* `hand_landmarker.task`

* **Slow inference / low FPS**
	* Reduce image size (`img_size`) in detector settings.
	* Confirm GPU is available (if expected) and CUDA-compatible environment is installed.

* **No animations shown after chain completion**
	* Verify `animations/` contains mapped videos (e.g., `fireball.mp4`, `chidori.mp4`).
	* Check chain names in `core/chainer.py` match animation mappings in `core/animator.py`.

