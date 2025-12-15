```md
# Basketball Shooting Motion Analyzer (YOLOv8 Pose)

A lightweight, video-based basketball shooting form analyzer.  
Given a single shooting clip, the pipeline runs **YOLOv8 Pose** to extract body keypoints frame-by-frame, segments the shot into phases, computes interpretable motion metrics, and outputs:

- a **final score (0–100)**
- **main issues** (with evidence + coaching tips)
- an **annotated output video**
- structured **JSON/TXT reports**

---

## Demo (What you get)
After running on a short clip, the system generates:

- `*_motion_analyzed.mp4`: overlayed skeleton + progress + final score screen
- `*_report.json`: machine-readable result
- `*_report.txt`: human-readable summary

Example (from a sample run):

- **Score:** 81 / 100  
- **Main issues:** elbow alignment during loading, limited vertical lift  
- **Metrics:** knee minimum angle, elbow peak angle, alignment ratios, jump pixels, phase boundaries

---

## How it works (Approach)
1. **Pose estimation (per frame)**  
   Uses **Ultralytics YOLOv8 Pose** pretrained weights to infer human keypoints.

2. **(Optional) ball detection**  
   A YOLOv8 detection model can be used to detect the basketball (e.g., “sports ball” class) to help locate release-related moments.

3. **Phase segmentation**  
   Converts the keypoint time series into phases such as:
   `setup → loading → release → follow-through`

4. **Metrics & scoring**  
   Computes phase-aware, interpretable proxies (angles, relative alignments, vertical displacement).  
   These metrics are mapped to:
   - a **final score**
   - **issue flags** (problem + evidence + suggestion)

---

## Repository layout (suggested)
(ASCII-only tree to avoid LaTeX/Unicode issues)

```

shooting-motion-analyzer/
|-- main.py
|-- README.md
|-- requirements.txt
|-- analysis/
|   |-- **init**.py
|   |-- analyzer.py
|   `-- renderer.py |-- weights/                # optional: store local .pt here |   |-- yolov8l-pose.pt |   |-- yolov8m-pose.pt |   `-- yolov8x-pose.pt
`-- videos/     |-- example.mp4
    `-- example_motion_analyzed.mp4

````

> Notes  
> - `main.py` is the CLI entry.  
> - `analysis/` contains the core pipeline (pose → phases → metrics → score) and visualization/overlay logic.  
> - You can keep model weights in the project folder (or let Ultralytics download them automatically, depending on your setup).

---

## Installation
### 1) Create environment (recommended)
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
````

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Quickstart (Run)

Put your input video (e.g., `.mp4`) in the project folder (or the configured `videos/` directory), then run:

```bash
python main.py
```

The CLI will:

1. list available videos
2. ask you to choose a **pose model size** (m / l / x)
3. process the clip and write outputs

---

## Output format

### JSON report (example fields)

* `ok`: run status
* `video`: input filename
* `output_video`: output filename
* `score`: final score (0–100)
* `issues`: list of `{part, problem, evidence, suggestion}`
* `metrics`: computed key metrics + `phases`

### TXT report

A readable summary with the same score/issues/metrics.

---

## Evaluation philosophy

This project is **explainable, metric-driven evaluation** (not dataset-level benchmarking like mAP).
Each diagnosis is backed by specific metric values (“evidence”), so users can trace feedback to measurable motion properties.

---

## Tips for best results

* Use a **single shooter** in frame
* Keep the camera relatively stable
* Ensure the full upper body is visible during release/follow-through
* Higher-resolution clips generally improve keypoint stability

---

## Limitations

* Heuristic scoring thresholds may need calibration for different camera angles / player styles.
* Ball visibility, occlusions, and multiple people can reduce reliability.
* Release moment estimation may be imperfect if the ball is not detected clearly.

---

## Future work

* More robust ball-release detection (trajectory + contact)
* Multi-person filtering (track the shooter consistently)
* Data-driven threshold tuning and per-user baseline comparison
* More detailed shot-type support (set shot vs jump shot vs step-back)

---

## Acknowledgements

Built on top of **Ultralytics YOLOv8** for pose estimation and **OpenCV** for video processing.

```
```
