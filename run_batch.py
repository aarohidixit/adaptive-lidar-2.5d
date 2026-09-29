"""
run_batch.py  -  Adaptive 2.5D LiDAR Pipeline (batch mode)

Usage:
    python run_batch.py [--downsample] [--images <folder>] [--fps <float>]

Flags:
    --downsample   Use 30%% of points for faster processing
    --images DIR   Folder containing camera images (matched by frame name)
    --fps N        Target frames-per-second playback speed (default: as fast as possible)
                   e.g. --fps 0.5  -> one frame every 2 seconds (great for demos)
"""

import sys
import os
import glob
import shutil
import json
import time

# Force UTF-8 on Windows to avoid cp1252 encoding crashes
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# ── Project modules ───────────────────────────────────────────────────────────
from data_loader import load_kitti_bin
from segmentation import SegmentationEngine
from grid_engine import GridEngine
from visualizer import Visualizer
from metrics import MetricsTracker

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ── ANSI helpers (ASCII-only bar chars for Windows compat) ────────────────────
BAR_FULL  = "#"   # filled portion of bar
BAR_EMPTY = "."   # empty portion of bar
SEP_CHAR  = "-"


def _sep(n=62):
    return Fore.BLUE + Style.DIM + SEP_CHAR * n + Style.RESET_ALL


def _header(text):
    line = "=" * 62
    pad = (62 - 2 - len(text)) // 2
    mid = " " * pad + text
    return (
        "\n" + Fore.CYAN + Style.BRIGHT +
        line + "\n" +
        mid + "\n" +
        line + Style.RESET_ALL
    )


def _bar(label, value, max_value, width=28, color=Fore.CYAN):
    filled = int(width * value / max_value) if max_value > 0 else 0
    bar = color + BAR_FULL * filled + Style.DIM + BAR_EMPTY * (width - filled) + Style.RESET_ALL
    return f"  {label:<18} [{bar}]  {color}{value:>8,}{Style.RESET_ALL}"


def _memory_comparison_bar(uniform_mb, adaptive_mb, width=28):
    """Visual bar showing how tiny the adaptive grid is vs uniform."""
    lines = []
    lines.append(
        f"  {'Uniform 5cm':>18}  {Fore.RED}{BAR_FULL * width}{Style.RESET_ALL}"
        f"  {Fore.RED}{uniform_mb:>8.1f} MB{Style.RESET_ALL}"
    )
    filled = max(1, int(width * adaptive_mb / uniform_mb))
    lines.append(
        f"  {'Adaptive':>18}  {Fore.GREEN}{BAR_FULL * filled}{BAR_EMPTY * (width - filled)}{Style.RESET_ALL}"
        f"  {Fore.GREEN}{adaptive_mb:>8.2f} MB{Style.RESET_ALL}"
    )
    savings = ((uniform_mb - adaptive_mb) / uniform_mb * 100) if uniform_mb > 0 else 0
    lines.append(
        f"  {'Memory saved':>18}  {Fore.YELLOW + Style.BRIGHT}{savings:.1f}% reduction{Style.RESET_ALL}"
    )
    return "\n".join(lines)


# ── CLI arg parsing ───────────────────────────────────────────────────────────
def parse_args():
    downsample = "--downsample" in sys.argv
    images_dir = None
    target_fps = None

    if "--images" in sys.argv:
        idx = sys.argv.index("--images")
        if idx + 1 < len(sys.argv):
            images_dir = sys.argv[idx + 1]

    if "--fps" in sys.argv:
        idx = sys.argv.index("--fps")
        if idx + 1 < len(sys.argv):
            try:
                target_fps = float(sys.argv[idx + 1])
            except ValueError:
                print(Fore.RED + "[ERROR] --fps requires a numeric value" + Style.RESET_ALL)
                sys.exit(1)

    return downsample, images_dir, target_fps


# ── Per-frame pipeline ────────────────────────────────────────────────────────
def process_frame(bin_file, frame_idx, downsample, seg_engine, grid_engine, viz, outputs_dir):
    """Full pipeline for one .bin frame. Returns summary dict."""
    metrics = MetricsTracker()
    metrics.set_frame_index(frame_idx)

    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
    else:
        mem_before = 0.0

    metrics.start_timer()
    points = load_kitti_bin(bin_file)

    if downsample:
        n = points.shape[0]
        sample_size = int(n * 0.3)
        indices = np.random.choice(n, sample_size, replace=False)
        points = points[indices]

    labels = seg_engine.segment(points)
    grid_cells, _, zone_stats = grid_engine.build_grid(points)
    viz.plot_bev(points, labels, filename="bev_map.png")
    viz.plot_front_view(points, labels, filename="front_view.png")


    metrics.stop_timer()
    metrics.update(points, labels)
    metrics.update_zone_stats(zone_stats)
    metrics.calculate_memory_reduction(len(grid_cells))

    summary = metrics.get_summary()

    if HAS_PSUTIL:
        mem_after = process.memory_info().rss / (1024 * 1024)
        summary["memory_delta_mb"] = mem_after - mem_before
    else:
        summary["memory_delta_mb"] = 0.0

    ms = summary["processing_time_ms"]
    summary["fps"] = 1000.0 / ms if ms > 0 else 0.0

    # Write metrics.json (polled by dashboard every 1.2s)
    json_path = os.path.join(outputs_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)

    # Append to rolling history (last 60 frames) for sparklines
    history_path = os.path.join(outputs_dir, "metrics_history.json")
    history = []
    if os.path.exists(history_path):
        try:
            with open(history_path) as f:
                history = json.load(f)
        except Exception:
            history = []

    compact = {
        "frame": frame_idx,
        "fps": round(summary["fps"], 3),
        "ms": round(ms, 1),
        "terrain": summary["class_counts"]["terrain"],
        "vehicle": summary["class_counts"]["vehicle"],
        "pedestrian": summary["class_counts"]["pedestrian"],
        "static_obstacle": summary["class_counts"]["static_obstacle"],
        "adaptive_mb": round(summary["memory_reduction"]["adaptive_mb"], 3),
    }
    history.append(compact)
    history = history[-60:]
    with open(history_path, "w") as f:
        json.dump(history, f)

    return summary


# ── Print frame report ────────────────────────────────────────────────────────
def print_frame_report(frame_idx, total_frames, frame_name, summary):
    counts = summary["class_counts"]
    fps    = summary["fps"]
    ms     = summary["processing_time_ms"]
    mr     = summary["memory_reduction"]
    zs     = summary.get("zone_stats", {})

    fps_color = Fore.GREEN if fps >= 1.0 else Fore.YELLOW if fps >= 0.5 else Fore.RED

    print()
    print(
        Fore.CYAN + f"  Frame [{frame_idx}/{total_frames}]  " +
        Fore.WHITE + f"{frame_name}.bin  " +
        fps_color + f"{fps:.2f} FPS  " +
        Fore.WHITE + f"({ms:.0f} ms)" +
        Style.RESET_ALL
    )
    print(_sep())

    total = summary["total_points"]
    print(Fore.WHITE + Style.BRIGHT + "  CLASS BREAKDOWN" + Style.RESET_ALL)
    print(_bar("Terrain",         counts["terrain"],          total, color=Fore.GREEN))
    print(_bar("Static Obstacle", counts["static_obstacle"],  total, color=Fore.YELLOW))
    print(_bar("Vehicle",         counts["vehicle"],          total, color=Fore.CYAN))
    print(_bar("Pedestrian",      counts["pedestrian"],       total, color=Fore.MAGENTA))

    if zs:
        print()
        print(Fore.WHITE + Style.BRIGHT + "  ZONE GRID CELLS" + Style.RESET_ALL)
        total_cells = zs.get("total", 1)
        print(_bar("Zone 1 (0-10m)",   zs.get("zone1", 0), total_cells, color=Fore.CYAN))
        print(_bar("Zone 2 (10-50m)",  zs.get("zone2", 0), total_cells, color=Fore.BLUE))
        print(_bar("Zone 3 (50-100m)", zs.get("zone3", 0), total_cells, color=Fore.MAGENTA))
        print(f"  {'Total unique cells':<18}  {Fore.WHITE}{total_cells:>8,}{Style.RESET_ALL}")
        print(f"  {'Valid points':<18}  {Fore.WHITE}{zs.get('valid_points', 0):>8,}{Style.RESET_ALL}")

    if mr:
        print()
        print(Fore.WHITE + Style.BRIGHT + "  MEMORY COMPARISON" + Style.RESET_ALL)
        print(_memory_comparison_bar(mr["uniform_mb"], mr["adaptive_mb"]))

    print(_sep())


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(_header("LiDAR 2.5D Foveated Grid Pipeline"))

    downsample, images_dir, target_fps = parse_args()

    base_dir    = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(base_dir, "data")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    bin_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
    if not bin_files:
        print(Fore.RED + f"  [ERROR] No .bin files found in {data_dir}" + Style.RESET_ALL)
        sys.exit(1)

    print(f"  {Fore.CYAN}Found {Fore.WHITE + Style.BRIGHT}{len(bin_files)}{Style.RESET_ALL} {Fore.CYAN}frame(s){Style.RESET_ALL}")
    if downsample:
        print(f"  {Fore.YELLOW}Fast mode: 30% point sample{Style.RESET_ALL}")
    if images_dir:
        print(f"  {Fore.CYAN}Camera images folder: {images_dir}{Style.RESET_ALL}")
    if target_fps:
        delay = 1.0 / target_fps
        print(f"  {Fore.MAGENTA}Playback: {target_fps:.1f} FPS  (delay={delay:.2f}s/frame){Style.RESET_ALL}")

    seg_engine = SegmentationEngine()
    grid_engine = GridEngine()
    viz = Visualizer(output_dir="outputs")

    for i, bin_file in enumerate(bin_files, start=1):
        frame_name    = os.path.splitext(os.path.basename(bin_file))[0]
        t_frame_start = time.time()

        summary = process_frame(
            bin_file, i, downsample,
            seg_engine, grid_engine, viz, outputs_dir
        )

        print_frame_report(i, len(bin_files), frame_name, summary)

        if images_dir:
            found = False
            for ext in (".png", ".jpg", ".jpeg"):
                img_src = os.path.join(images_dir, frame_name + ext)
                if os.path.exists(img_src):
                    dst = os.path.join(outputs_dir, "current_frame.jpg")
                    shutil.copy2(img_src, dst)
                    print(f"  {Fore.CYAN}[CAM] {os.path.basename(img_src)}{Style.RESET_ALL}")
                    found = True
                    break
            if not found:
                print(f"  {Fore.YELLOW}[WARN] No camera image for frame {frame_name}{Style.RESET_ALL}")

        if target_fps:
            elapsed   = time.time() - t_frame_start
            sleep_for = max(0.0, (1.0 / target_fps) - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

    print()
    print(Fore.GREEN + Style.BRIGHT + "  [DONE] Batch complete. Open the dashboard to view results." + Style.RESET_ALL)
    print()


if __name__ == "__main__":
    main()
