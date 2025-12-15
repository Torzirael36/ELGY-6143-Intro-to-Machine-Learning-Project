# analysis/analyzer.py
import cv2
import math
import time
import numpy as np
import torch
from ultralytics import YOLO


# ============================================================
#  Kalman Filter (1D)
# ============================================================

class KalmanFilter1D:
    def __init__(self, process_noise=0.05, measurement_noise=2.5, motion_gain=0.4):
        self.x = 0.0
        self.P = 1.0
        self.base_q = process_noise
        self.base_r = measurement_noise
        self.motion_gain = motion_gain
        self.last_meas = None
        self.initialized = False

    def update(self, measurement):
        measurement = float(measurement)
        if not self.initialized:
            self.x = measurement
            self.last_meas = measurement
            self.initialized = True
            return self.x

        motion = abs(measurement - self.last_meas)
        Q = self.base_q + self.motion_gain * motion
        R = self.base_r

        self.P += Q
        K = self.P / (self.P + R)
        self.x += K * (measurement - self.x)
        self.P *= (1.0 - K)

        self.last_meas = measurement
        return self.x


# ============================================================
#  Keypoint Smoother
# ============================================================

class KeypointSmoother:
    def __init__(self, num_keypoints=17):
        self.fx = [KalmanFilter1D() for _ in range(num_keypoints)]
        self.fy = [KalmanFilter1D() for _ in range(num_keypoints)]

    def reset(self):
        for f in self.fx + self.fy:
            f.initialized = False

    def smooth(self, keypoints):
        out = np.array(keypoints, dtype=np.float32, copy=True)
        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:
                out[i, 0] = self.fx[i].update(x)
                out[i, 1] = self.fy[i].update(y)
        return out


# ============================================================
#  Motion Buffer
# ============================================================

class MotionBuffer:
    def __init__(self):
        self.data = []
        self.valid = []

    def add(self, frame_idx, metrics, is_valid):
        self.data.append({"t": int(frame_idx), **metrics})
        self.valid.append(bool(is_valid))

    def to_array(self, key):
        return np.array([d.get(key, np.nan) for d in self.data], dtype=np.float32)

    def __len__(self):
        return len(self.data)


# ============================================================
#  Phase Segmenter (simple knee-based)
# ============================================================

class PhaseSegmenter:
    def segment(self, buf: MotionBuffer):
        if len(buf) < 20:
            return None

        knee = buf.to_array("knee_angle")
        if np.all(np.isnan(knee)):
            return None

        t_min = int(np.nanargmin(knee))

        return {
            "setup": (0, max(0, t_min - 10)),
            "loading": (max(0, t_min - 10), t_min),
            "release": (t_min, min(len(buf) - 1, t_min + 15)),
            "follow": (min(len(buf) - 1, t_min + 15), len(buf) - 1),
            "t_min": t_min
        }


# ============================================================
#  Motion Evaluator  (now includes ball arc angle)
# ============================================================

class MotionEvaluator:
    def evaluate(self, buf: MotionBuffer, phases, ball_angle_deg=None):
        """
        Continuous (phase-level) scoring + ball release angle scoring.
        Returns: total_score, issues, metrics
        """

        # helpers
        def _slice(name):
            a, b = phases[name]
            return slice(a, b + 1)

        def _mean(x):
            x = np.asarray(x, dtype=np.float32)
            x = x[~np.isnan(x)]
            return float(np.mean(x)) if len(x) else np.nan

        def _min(x):
            x = np.asarray(x, dtype=np.float32)
            x = x[~np.isnan(x)]
            return float(np.min(x)) if len(x) else np.nan

        def _max(x):
            x = np.asarray(x, dtype=np.float32)
            x = x[~np.isnan(x)]
            return float(np.max(x)) if len(x) else np.nan

        knee = buf.to_array("knee_angle")
        elbow = buf.to_array("elbow_angle")

        setup_sl = _slice("setup")
        loading_sl = _slice("loading")
        release_sl = _slice("release")
        follow_sl = _slice("follow")

        # phase statistics
        knee_min_loading = _min(knee[loading_sl])
        knee_mean_loading = _mean(knee[loading_sl])

        elbow_peak_release = _max(elbow[release_sl])
        elbow_mean_follow = _mean(elbow[follow_sl])

        # phase scores
        phase_scores = {}

        # Loading (0–25)
        if not np.isnan(knee_min_loading):
            if knee_min_loading <= 110:
                phase_scores["loading"] = 25
            elif knee_min_loading <= 140:
                phase_scores["loading"] = 20
            else:
                phase_scores["loading"] = 12
        else:
            phase_scores["loading"] = 0

        # Release (0–35)
        if not np.isnan(elbow_peak_release):
            if elbow_peak_release >= 165:
                phase_scores["release"] = 35
            elif elbow_peak_release >= 150:
                phase_scores["release"] = 28
            else:
                phase_scores["release"] = 18
        else:
            phase_scores["release"] = 0

        # Follow-through (0–20)
        if not np.isnan(elbow_mean_follow):
            if elbow_mean_follow >= 150:
                phase_scores["follow"] = 20
            elif elbow_mean_follow >= 135:
                phase_scores["follow"] = 14
            else:
                phase_scores["follow"] = 8
        else:
            phase_scores["follow"] = 0

        # Setup (0–20)
        if not np.isnan(knee_mean_loading):
            if 120 <= knee_mean_loading <= 160:
                phase_scores["setup"] = 20
            else:
                phase_scores["setup"] = 14
        else:
            phase_scores["setup"] = 0

        # -------------------------
        # Ball arc score (0–15)
        # -------------------------
        ball_score = 0
        if ball_angle_deg is not None and not np.isnan(ball_angle_deg):
            a = float(ball_angle_deg)

            # typical good arc range ~40-55 deg in 2D view
            if 40 <= a <= 55:
                ball_score = 15
            elif 35 <= a < 40 or 55 < a <= 60:
                ball_score = 10
            elif 30 <= a < 35 or 60 < a <= 65:
                ball_score = 6
            else:
                ball_score = 2

        # total score
        total_score = int(min(100, sum(phase_scores.values()) + ball_score))

        # issues
        issues = []

        if phase_scores["loading"] < 18:
            issues.append({
                "problem": "Insufficient knee bend during loading",
                "advice": "Focus on deeper, smoother knee flexion before upward motion."
            })

        if phase_scores["release"] < 28:
            issues.append({
                "problem": "Limited elbow extension at release",
                "advice": "Emphasize full arm extension and avoid pushing the ball forward."
            })

        if phase_scores["follow"] < 14:
            issues.append({
                "problem": "Incomplete follow-through",
                "advice": "Hold the follow-through position for at least one second after release."
            })

        # ball angle issues
        if ball_angle_deg is not None and not np.isnan(ball_angle_deg):
            a = float(ball_angle_deg)
            if a < 35:
                issues.append({
                    "problem": "Flat shooting arc (low release angle)",
                    "advice": f"Ball release angle is {a:.1f}°. Try raising the release point and increasing arc."
                })
            elif a > 65:
                issues.append({
                    "problem": "Overly high shooting arc (too much loft)",
                    "advice": f"Ball release angle is {a:.1f}°. Reduce excessive loft and transfer more energy forward."
                })

        metrics = {
            "phase_scores": phase_scores,
            "ball_release_angle_deg": (float(ball_angle_deg) if ball_angle_deg is not None else None),
            "ball_score": ball_score,
            "knee_min_loading": knee_min_loading,
            "elbow_peak_release": elbow_peak_release,
            "elbow_mean_follow": elbow_mean_follow,
            "phases": phases
        }

        return total_score, issues, metrics


# ============================================================
#  Shooting Analyzer (MAIN)
# ============================================================

class ShootingAnalyzer:
    RIGHT_SHOULDER = 6
    RIGHT_ELBOW = 8
    RIGHT_WRIST = 10
    RIGHT_HIP = 12
    RIGHT_KNEE = 14
    RIGHT_ANKLE = 16

    def __init__(self, pose_size="l", ball_model="yolov8s.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("=" * 60)
        print("Shooting Motion Analyzer")
        print("=" * 60)
        print(f"Device: {self.device.upper()}")
        if self.device == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")

        self.pose_model_name = f"yolov8{pose_size}-pose.pt"
        self.ball_model_name = ball_model

        print(f"Pose model: {self.pose_model_name}")
        print(f"Ball model: {self.ball_model_name} (sports ball class=32)")
        print("=" * 60)

        self.pose_model = YOLO(self.pose_model_name).to(self.device)
        self.ball_model = YOLO(self.ball_model_name).to(self.device)

        self.smoother = KeypointSmoother()
        self.buf = MotionBuffer()
        self.segmenter = PhaseSegmenter()
        self.evaluator = MotionEvaluator()

        # ball trajectory storage
        self.ball_traj = []        # list of (frame_idx, (x,y))
        self.release_frame = None  # frame index where release phase starts
        self.phases = None         # segmented phases

        self.last_wrist = None  # 最近一帧的手腕位置
        self.ball_roi_size = 180  # ROI 半径（像素，可调）

    @staticmethod
    def _angle(p1, p2, p3):
        a, b, c = map(np.array, (p1, p2, p3))
        ba, bc = a - b, c - b
        cosv = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))

    # -------------------------
    # Ball detection
    # -------------------------
    def detect_ball(self, frame):
        """
        Detect basketball center (x,y) using ROI-expanded YOLO search.
        Priority: wrist-centered ROI -> fallback to full frame.
        """
        h, w, _ = frame.shape

        # -------------------------
        # 1. ROI search around wrist
        # -------------------------
        if self.last_wrist is not None:
            cx, cy = self.last_wrist
            r = self.ball_roi_size

            x1 = max(0, cx - r)
            y1 = max(0, cy - r)
            x2 = min(w, cx + r)
            y2 = min(h, cy + r)

            roi = frame[y1:y2, x1:x2]

            results = self.ball_model(
                roi,
                verbose=False,
                device=self.device,
                conf=0.15,  # 👈 球一定要低一点
                classes=[32],  # sports ball
            )

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                box = boxes[int(np.argmax(areas))]

                bx = int((box[0] + box[2]) / 2) + x1
                by = int((box[1] + box[3]) / 2) + y1
                return (bx, by)

        # -------------------------
        # 2. Fallback: full frame
        # -------------------------
        results = self.ball_model(
            frame,
            verbose=False,
            device=self.device,
            conf=0.15,
            classes=[32],
        )

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        box = boxes[int(np.argmax(areas))]

        cx = int((box[0] + box[2]) / 2)
        cy = int((box[1] + box[3]) / 2)
        return (cx, cy)

    def compute_ball_release_angle(self, release_frame, max_frames=8):
        """
        Compute release angle (deg) using early post-release trajectory points.
        Angle is with respect to ground (horizontal).
        """
        if release_frame is None:
            return None

        pts = [p for (t, p) in self.ball_traj if release_frame <= t < release_frame + max_frames]
        if len(pts) < 2:
            return None

        (x0, y0) = pts[0]
        (x1, y1) = pts[-1]

        dx = x1 - x0
        dy = y0 - y1  # invert image y-axis

        if abs(dx) < 1e-3:
            return None

        ang = math.degrees(math.atan2(dy, dx))
        # keep in [0, 90] for typical shots
        if ang < 0:
            ang = -ang
        return float(ang)

    def get_phase_name(self, frame_idx):
        if not self.phases:
            return None
        for name in ["setup", "loading", "release", "follow"]:
            a, b = self.phases.get(name, (None, None))
            if a is None:
                continue
            if a <= frame_idx <= b:
                return name
        return None

    # -------------------------
    # Pose analysis per frame
    # -------------------------
    def analyze_frame(self, frame, frame_idx):
        result = self.pose_model(frame, verbose=False, device=self.device)[0]

        if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
            self.buf.add(frame_idx, {}, False)
            return None

        kps = result.keypoints.xy[0].cpu().numpy()
        kps = self.smoother.smooth(kps)

        sh = kps[self.RIGHT_SHOULDER]
        el = kps[self.RIGHT_ELBOW]
        wr = kps[self.RIGHT_WRIST]

        self.last_wrist = (int(wr[0]), int(wr[1]))

        hp = kps[self.RIGHT_HIP]
        kn = kps[self.RIGHT_KNEE]
        an = kps[self.RIGHT_ANKLE]

        if (
            np.any(sh <= 0)
            or np.any(el <= 0)
            or np.any(wr <= 0)
            or np.any(hp <= 0)
            or np.any(kn <= 0)
            or np.any(an <= 0)
        ):
            self.buf.add(frame_idx, {}, False)
            return kps

        elbow_angle = self._angle(sh, el, wr)
        knee_angle = self._angle(hp, kn, an)

        self.buf.add(
            frame_idx,
            {"elbow_angle": float(elbow_angle), "knee_angle": float(knee_angle)},
            True
        )
        return kps

    # -------------------------
    # Main video processing
    # -------------------------
    def process_video(self, video_path):
        from render.renderer import Renderer

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return None

        # ================= Reset =================
        self.buf = MotionBuffer()
        self.smoother.reset()
        self.ball_traj = []
        self.release_frame = None
        self.phases = None

        # ================= Video info =================
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = video_path.rsplit(".", 1)[0] + "_motion_analyzed.mp4"
        out = cv2.VideoWriter(
            out_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h)
        )

        print(f"\nProcessing video: {video_path}")
        print(f"Resolution: {w}x{h}, FPS: {fps}, Total frames: {total}")
        print("-" * 60)

        start = time.time()
        frame_idx = 0

        # ================= MAIN LOOP =================
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # ---------- pose analysis ----------
            kps = self.analyze_frame(frame, frame_idx)

            # ---------- ball detection (only before / near release) ----------
            ball_pos = None
            if self.release_frame is None or frame_idx <= self.release_frame + 2:
                ball_pos = self.detect_ball(frame)
                if ball_pos is not None:
                    self.ball_traj.append((frame_idx, ball_pos))

            # ---------- draw base layers ----------
            Renderer.draw_skeleton(frame, kps)
            Renderer.draw_ball(frame, ball_pos)
            Renderer.draw_status_bar(frame)
            Renderer.draw_progress_bar(frame, frame_idx, total)

            # =========================================================
            # REAL-TIME PHASE ESTIMATION (NO SCORES YET)
            # =========================================================
            phase_name = None
            if len(self.buf) > 10:
                knee = self.buf.to_array("knee_angle")
                if not np.all(np.isnan(knee)):
                    t_min = int(np.nanargmin(knee))
                    if frame_idx < t_min - 10:
                        phase_name = "setup"
                    elif frame_idx < t_min:
                        phase_name = "loading"
                    elif frame_idx < t_min + 15:
                        phase_name = "release"
                    else:
                        phase_name = "follow"

            # ---------- LEFT PANEL (phase only, no score yet) ----------
            Renderer.draw_phase_and_score(
                frame,
                phase=phase_name,
                phase_scores=None,
                total_score=None
            )

            # ---------- ball trajectory ----------
            if hasattr(Renderer, "draw_ball_trajectory"):
                Renderer.draw_ball_trajectory(
                    frame,
                    self.ball_traj,
                    self.release_frame,
                    frame_idx,
                    max_len=40
                )

            out.write(frame)
            frame_idx += 1

            # ---------- TERMINAL PROGRESS BAR ----------
            if frame_idx % 10 == 0 or frame_idx == total:
                elapsed = time.time() - start
                fps_now = frame_idx / elapsed if elapsed > 0 else 0
                percent = frame_idx / max(1, total)

                bar_len = 40
                filled = int(bar_len * percent)
                bar = "█" * filled + "-" * (bar_len - filled)

                if fps_now > 0:
                    remaining = (total - frame_idx) / fps_now
                    eta = time.strftime("%M:%S", time.gmtime(remaining))
                else:
                    eta = "--:--"

                print(
                    f"\r[{bar}] {percent * 100:6.2f}% "
                    f"{frame_idx}/{total} "
                    f"{fps_now:5.1f} FPS "
                    f"ETA {eta}",
                    end=""
                )

        print()
        cap.release()

        # =========================================================
        # SEQUENCE-LEVEL ANALYSIS (NOW WE HAVE SCORES)
        # =========================================================
        self.phases = self.segmenter.segment(self.buf)
        if self.phases is None:
            out.release()
            print("\n⚠ Unable to extract a stable shooting motion sequence.")
            return None

        self.release_frame = self.phases["release"][0]

        ball_angle = self.compute_ball_release_angle(self.release_frame)

        score, issues, metrics = self.evaluator.evaluate(
            self.buf,
            self.phases,
            ball_angle_deg=ball_angle
        )

        # ---------- append summary ----------
        Renderer.append_summary(out, w, h, fps, score, issues, seconds=3)
        out.release()

        elapsed = time.time() - start
        print("\n" + "=" * 60)
        print("Video processing completed")
        print("=" * 60)
        print(f"Output video: {out_path}")
        print(f"Final score: {score} / 100")
        if ball_angle is not None:
            print(f"Ball release angle: {ball_angle:.1f} deg")
        else:
            print("Ball release angle: N/A")
        print(f"Processing time: {elapsed:.1f}s | Avg FPS: {total / elapsed:.1f}")

        if issues:
            print("\nMain issues:")
            for i, it in enumerate(issues, 1):
                print(f"  {i}. {it.get('problem', '')}")
        else:
            print("\nNo major issues detected.")

        print("=" * 60)

        return score, issues, metrics




