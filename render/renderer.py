# render/renderer.py
import cv2
import numpy as np


class Renderer:
    """
    Render layer: visualization only
    - Skeleton (adaptive smoothing, no drift)
    - Ball
    - Bottom status bar
    - Progress bar
    - Phase & score panel
    - Summary at the end
    """

    # ==============================
    # Class-level cache (DO NOT REMOVE)
    # ==============================
    _prev_keypoints = None

    # COCO skeleton definition
    SKELETON = [
        (5, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 11), (6, 12),
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]

    # =========================================================
    # Ball
    # =========================================================

    @staticmethod
    def draw_ball(img, ball_pos):
        if ball_pos is None:
            return
        cv2.circle(img, ball_pos, 18, (0, 165, 255), 3)

    # =========================================================
    # Skeleton (ADAPTIVE smoothing – FIXED)
    # =========================================================

    @staticmethod
    def draw_skeleton(img, keypoints, alpha=0.75, reset_thresh=80):
        """
        Adaptive temporal smoothing.
        - Smooth when motion is small
        - Reset immediately when jump is large
        """
        if keypoints is None:
            Renderer._prev_keypoints = None
            return

        kp = np.asarray(keypoints, dtype=np.float32)

        if Renderer._prev_keypoints is None:
            smooth = kp
        else:
            diff = np.linalg.norm(kp - Renderer._prev_keypoints, axis=1)
            jump = np.nanmax(diff)

            if jump > reset_thresh:
                smooth = kp
            else:
                smooth = alpha * Renderer._prev_keypoints + (1 - alpha) * kp

        Renderer._prev_keypoints = smooth.copy()

        # draw skeleton lines
        for s, e in Renderer.SKELETON:
            if (
                smooth[s][0] > 0 and smooth[s][1] > 0 and
                smooth[e][0] > 0 and smooth[e][1] > 0
            ):
                p1 = int(smooth[s][0]), int(smooth[s][1])
                p2 = int(smooth[e][0]), int(smooth[e][1])
                cv2.line(img, p1, p2, (0, 255, 0), 3)

        # draw joints
        for x, y in smooth:
            if x > 0 and y > 0:
                cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)

    # =========================================================
    # Bottom status bar
    # =========================================================

    @staticmethod
    def draw_status_bar(img, score=None, issues=None):
        h, w = img.shape[:2]
        bar_h = 60

        overlay = img.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

        if score is None:
            cv2.putText(
                img,
                "Analyzing shooting motion...",
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (220, 220, 220),
                2,
            )
            return

        color = (0, 255, 0) if score >= 75 else (0, 255, 255) if score >= 55 else (0, 0, 255)

        cv2.putText(
            img,
            f"Final Score: {score}/100",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )

        if issues:
            txt = ", ".join(it.get("problem", "") for it in issues[:2])
            cv2.putText(
                img,
                f"Issues: {txt}",
                (260, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )

    # =========================================================
    # Progress bar
    # =========================================================

    @staticmethod
    def draw_progress_bar(img, current, total):
        if total <= 0:
            return

        h, w = img.shape[:2]
        margin = 20
        bar_h = 14

        y1 = h - margin
        y0 = y1 - bar_h

        ratio = max(0.0, min(1.0, current / float(total)))

        overlay = img.copy()
        cv2.rectangle(overlay, (margin, y0), (w - margin, y1), (60, 60, 60), -1)

        fill_w = int((w - 2 * margin) * ratio)
        cv2.rectangle(
            overlay,
            (margin, y0),
            (margin + fill_w, y1),
            (0, 200, 255),
            -1,
        )

        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

        cv2.putText(
            img,
            f"{int(ratio * 100)}%",
            (w - margin - 50, y0 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
        )

    # =========================================================
    # Phase & score panel
    # =========================================================

    @staticmethod
    def draw_phase_and_score(img, phase=None, phase_scores=None, total_score=None):
        x0, y0 = 20, 120
        pw, ph = 320, 200

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (255, 255, 255), 2)

        phase_text = phase.upper() if phase else "N/A"
        cv2.putText(
            img,
            f"PHASE: {phase_text}",
            (x0 + 12, y0 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        y = y0 + 65
        if phase_scores:
            for name in ["setup", "loading", "release", "follow"]:
                val = phase_scores.get(name, 0)
                col = (0, 255, 0) if name == phase else (220, 220, 220)
                cv2.putText(
                    img,
                    f"{name.capitalize():<8}: {val:>2}",
                    (x0 + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    col,
                    2,
                )
                y += 28
        else:
            cv2.putText(
                img,
                "Collecting motion data...",
                (x0 + 12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                2,
            )

        if total_score is not None:
            cv2.putText(
                img,
                f"TOTAL: {total_score}/100",
                (x0 + 12, y0 + ph - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 255),
                2,
            )

    # =========================================================
    # Summary (FIXED – WAS MISSING)
    # =========================================================

    @staticmethod
    def make_summary_frame(w, h, score, issues):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)

        cv2.putText(
            frame,
            "MOTION SHOOTING ANALYSIS",
            (max(30, w // 2 - 360), 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (255, 255, 255),
            3,
        )

        color = (0, 255, 0) if score >= 75 else (0, 255, 255) if score >= 55 else (0, 0, 255)

        cv2.putText(
            frame,
            f"{int(score)}",
            (w // 2 - 90, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            5,
            color,
            10,
        )
        cv2.putText(
            frame,
            "/ 100",
            (w // 2 + 110, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (200, 200, 200),
            3,
        )

        y0 = h // 2 + 120
        cv2.putText(
            frame,
            "Coach Tips:",
            (60, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (220, 220, 220),
            2,
        )

        tips = [f"- {it.get('problem', '')}" for it in (issues or [])[:3]]
        if not tips:
            tips = ["- Form looks stable. Keep consistent follow-through."]

        for i, t in enumerate(tips):
            cv2.putText(
                frame,
                t,
                (60, y0 + 40 + i * 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (220, 220, 220),
                2,
            )

        return frame

    @staticmethod
    def append_summary(out_writer, w, h, fps, score, issues, seconds=3):
        frame = Renderer.make_summary_frame(w, h, score, issues)
        n = max(1, int(fps) * int(seconds))
        for _ in range(n):
            out_writer.write(frame)
