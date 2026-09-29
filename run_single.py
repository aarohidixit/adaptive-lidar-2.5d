import os
import shutil
import json
import time
import numpy as np

# Force UTF-8 on Windows
import sys
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from colorama import Fore, Style, init
init(autoreset=True)

from data_loader import load_kitti_bin
from segmentation import SegmentationEngine
from grid_engine import GridEngine
from visualizer import Visualizer
from metrics import MetricsTracker
from run_batch import print_frame_report

def main():
    frame_name = "0000000000"
    if len(sys.argv) > 1:
        frame_name = sys.argv[1].zfill(10)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    bin_file = os.path.join(base_dir, "data", f"{frame_name}.bin")
    img_file = os.path.join(base_dir, "image_data", f"{frame_name}.png")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    if not os.path.exists(bin_file):
        print(Fore.RED + f"[ERROR] Could not find {bin_file}")
        return

    print(Fore.CYAN + f"Processing single frame: {frame_name}")
    
    seg_engine = SegmentationEngine()
    grid_engine = GridEngine()
    viz = Visualizer(output_dir="outputs")
    
    metrics = MetricsTracker()
    metrics.set_frame_index(int(frame_name))
    metrics.start_timer()

    # Load and downsample slightly for speed
    points = load_kitti_bin(bin_file)
    sample_size = int(points.shape[0] * 0.3)
    indices = np.random.choice(points.shape[0], sample_size, replace=False)
    points = points[indices]

    # Run engines
    labels = seg_engine.segment(points)
    grid_cells, _, zone_stats = grid_engine.build_grid(points)
    
    # Generate Visuals
    viz.plot_bev(points, labels, filename="bev_map.png")
    viz.plot_front_view(points, labels, filename="front_view.png")
    
    # Metrics
    metrics.stop_timer()
    metrics.update(points, labels)
    metrics.update_zone_stats(zone_stats)
    metrics.calculate_memory_reduction(len(grid_cells))
    summary = metrics.get_summary()

    # Add missing keys expected by the report and dashboard
    ms = summary.get("processing_time_ms", 0)
    summary["fps"] = 1000.0 / ms if ms > 0 else 0.0
    summary["memory_delta_mb"] = 0.0

    # Print Report
    print_frame_report(1, 1, frame_name, summary)
    
    # Write JSON for dashboard
    json_path = os.path.join(outputs_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    # Copy Camera Image
    if os.path.exists(img_file):
        shutil.copy2(img_file, os.path.join(outputs_dir, "current_frame.jpg"))
        print(Fore.GREEN + f"Copied camera image for {frame_name}")

    print(Fore.GREEN + Style.BRIGHT + "\n[DONE] Frame generated. Check the dashboard!")

if __name__ == "__main__":
    main()
