
# Basketball Shooting Motion Analyzer (YOLOv8 Pose)

A lightweight, video-based tool that analyzes basketball shooting form from a single clip. It uses **YOLOv8 Pose** to extract body keypoints frame-by-frame, segments the motion into phases, computes interpretable metrics, and generates a score, diagnostics, and an annotated output video.

---

## Features

This project performs pose-based shooting analysis using pretrained YOLOv8 Pose weights. It segments each shot into phases (e.g., setup → loading → release → follow-through), computes interpretable motion metrics (joint angles, alignment ratios, and vertical displacement proxies), and produces a final score (0–100) together with issue diagnostics, evidence, and coaching tips. For outputs, it generates an annotated video named `*_motion_analyzed.mp4`, as well as a machine-readable report `*_report.json` and a human-readable report `*_report.txt`.

---

## Example Output

After running on a short clip, you will typically see an overall motion score (for example, `82 / 100`) and a short list of main issues identified (for example, `Incomplete follow-through`). The report also includes key metrics such as phase scores, knee/arm angles, and follow-through angles.

---


shooting-motion-analyzer/
|-- main.py
|-- README.md
|-- requirements.txt
|-- sho_report.json
|-- sho_report.txt
|-- yolov8l-pose.pt
|-- yolov8m-pose.pt
|-- yolov8x-pose.pt
|-- yolov8n-pose.pt
|-- yolov8n.pt
|-- yolov8s.pt
|-- yolov8x.pt
|-- example.mp4
`-- example_motion_analyzed.mp4

````

> You can optionally place weights and videos into `weights/` and `videos/` folders, but the demo setup also works when everything is in one directory.

---

## Installation

### 1) Create a virtual environment (recommended)

```bash
python -m venv .venv
````

Activate it:

* Windows:

```bash
.venv\Scripts\activate
```

* macOS/Linux:

```bash
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Place your `.mp4` video in the project folder, then run:

```bash
python main.py
```

The CLI will:

* list available videos (excluding previously generated `*_motion_analyzed*` files)
* prompt you to select a pose model size (m / l / x)
* process the video and save outputs

---

## Output Files

* `*_motion_analyzed.mp4`
  Video with pose overlay, analysis progress, and final score screen.

* `*_report.json`
  Structured results including score, issues (problem/evidence/suggestion), metrics, and phase boundaries.

* `*_report.txt`
  A readable summary of the same results.

---

## Evaluation Style

This project uses an **explainable, metric-driven evaluation** (not dataset-level benchmarks like mAP).
Each detected issue is justified by specific metric values recorded in the report, making the feedback traceable and reproducible.

---

## Tips for Best Results

* Keep the camera stable and the shooter fully visible (especially arms during release/follow-through)
* Avoid multiple people in frame
* Higher resolution and better lighting improve keypoint stability

---

## Known Limitations

* Heuristic thresholds may vary with camera angle and personal shooting styles
* Occlusions (ball/hand/arm) can degrade keypoint quality
* Ball release estimation may be unreliable if the ball is not consistently visible

---

## Future Work

* More robust release detection using ball tracking + hand proximity
* Multi-person tracking to lock onto the shooter
* Data-driven calibration of scoring thresholds across viewpoints and players

---

## Acknowledgements

Built with **Ultralytics YOLOv8** and **OpenCV**.

```
```
