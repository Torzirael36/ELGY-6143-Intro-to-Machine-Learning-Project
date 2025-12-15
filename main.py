# main.py
import os
from analysis.analyzer import ShootingAnalyzer


def find_videos(folder):
    """Find video files in the given folder"""
    exts = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv")
    videos = []
    for f in os.listdir(folder):
        if f.lower().endswith(exts) and "_motion_analyzed" not in f:
            videos.append(f)
    return videos


def main():
    print("=" * 60)
    print("Basketball Shooting Motion Analyzer")
    print("YOLOv8 Pose + Motion Sequence Evaluation")
    print("=" * 60)

    # Working directory
    root = os.path.dirname(os.path.abspath(__file__))

    # Find videos
    videos = find_videos(root)
    if not videos:
        print("\n❌ No video files found in the current folder.")
        print(f"Please place your video files in:\n{root}")
        return

    print("\nAvailable videos:")
    for i, v in enumerate(videos):
        print(f"  [{i + 1}] {v}")

    if len(videos) == 1:
        choice = 0
    else:
        try:
            choice = int(input("\nSelect a video by number: ")) - 1
        except Exception:
            choice = 0

    if choice < 0 or choice >= len(videos):
        print("❌ Invalid selection.")
        return

    video_path = os.path.join(root, videos[choice])

    # Model selection
    print("\nSelect pose model size:")
    print("  [1] Medium   (yolov8m-pose)")
    print("  [2] Large    (yolov8l-pose)  Recommended")
    print("  [3] XLarge   (yolov8x-pose)")

    try:
        model_choice = int(input("Enter choice (default = 2): "))
    except Exception:
        model_choice = 2

    pose_map = {1: "m", 2: "l", 3: "x"}
    pose_size = pose_map.get(model_choice, "l")

    # Initialize analyzer
    print("\nInitializing analyzer...")
    analyzer = ShootingAnalyzer(pose_size=pose_size)

    # Run analysis
    print("\nProcessing video...\n")
    result = analyzer.process_video(video_path)

    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)

    if result is None:
        print("⚠ Unable to extract a stable shooting motion sequence.")
        return

    score, issues, metrics = result

    print(f"Overall Motion Score: {score} / 100")

    if issues:
        print("\nMain Issues Identified:")
        for i, it in enumerate(issues, 1):
            print(f"  {i}. {it.get('problem', '')}")
    else:
        print("\nNo major issues detected. The shooting motion appears stable.")

    print("\nKey Metrics:")
    for k, v in metrics.items():
        if k == "phases":
            continue
        print(f" - {k}: {v}")

    print("=" * 60)


if __name__ == "__main__":
    main()
