import sys
import os
import json
import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Import project modules
from data_loader import load_kitti_bin
from segmentation import SegmentationEngine
from grid_engine import GridEngine
from visualizer import Visualizer
from metrics import MetricsTracker

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <path_to_bin_file> [--downsample]")
        sys.exit(1)
        
    bin_file = sys.argv[1]
    downsample = "--downsample" in sys.argv
    
    if not os.path.exists(bin_file):
        print(f"Error: File '{bin_file}' not found.")
        sys.exit(1)
        
    print(f"Starting pipeline for: {bin_file}")
    if downsample:
        print("Running in fast mode (30% point sample)")
    
    # Get initial memory
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
    else:
        mem_before = 0.0
        
    # Initialize modules
    seg_engine = SegmentationEngine()
    grid_engine = GridEngine()
    viz = Visualizer(output_dir="outputs")
    metrics = MetricsTracker()
    
    # Start timer
    metrics.start_timer()
    
    # Step 2: Load Data
    print("\n[1/4] Loading point cloud...")
    points = load_kitti_bin(bin_file)
    
    if downsample:
        num_points = points.shape[0]
        sample_size = int(num_points * 0.3)
        indices = np.random.choice(num_points, sample_size, replace=False)
        points = points[indices]
    
    # Step 3: Segmentation
    print("\n[2/4] Segmenting points...")
    labels = seg_engine.segment(points)
    
    # Step 4: Grid Engine
    print("\n[3/4] Building adaptive grid...")
    grid_cells, point_cell_mapping = grid_engine.build_grid(points)
    
    # Step 5: Visualizer
    print("\n[4/4] Generating BEV map...")
    out_path = viz.plot_bev(points, labels, filename="bev_map.png")
    
    # End timer and update metrics
    metrics.stop_timer()
    metrics.update(points, labels)
    mem_red = metrics.calculate_memory_reduction(len(grid_cells))
    summary = metrics.get_summary()
    
    # Get final memory
    if HAS_PSUTIL:
        mem_after = process.memory_info().rss / (1024 * 1024)
    else:
        mem_after = 0.0
        
    delta_mem = mem_after - mem_before
    
    processing_time_ms = summary['processing_time_ms']
    fps = 1000.0 / processing_time_ms if processing_time_ms > 0 else 0.0
    
    # Step 6: Print Metrics
    print("\n==============================")
    print("--- Pipeline Metrics ---")
    print(f"Estimated FPS:   {fps:.2f} FPS")
    print(f"Processing Time: {processing_time_ms:.2f} ms")
    
    print("\n--- Memory Comparison ---")
    if HAS_PSUTIL:
        print(f"Initial Memory:  {mem_before:.2f} MB")
        print(f"Final Memory:    {mem_after:.2f} MB")
        print(f"Delta:           {delta_mem:.2f} MB")
    else:
        print("Memory stats unavailable (psutil not installed).")
        
    print("\n--- Class Counts ---")
    print(f"Terrain:         {summary['class_counts']['terrain']}")
    print(f"Static Obstacle: {summary['class_counts']['static_obstacle']}")
    print(f"Vehicle:         {summary['class_counts']['vehicle']}")
    print(f"Pedestrian:      {summary['class_counts']['pedestrian']}")
    print(f"Unknown:         {summary['class_counts']['unknown']}")
    
    if summary['memory_reduction']:
        mr = summary['memory_reduction']
        print("\n--- Grid Memory Reduction ---")
        print(f"Uniform 5cm Grid (100mx100m): {mr['uniform_cells']} cells | {mr['uniform_mb']:.2f} MB")
        print(f"Our Adaptive Grid:            {mr['adaptive_cells']} cells | {mr['adaptive_mb']:.2f} MB")
        print(f"Memory Reduction:             {mr['reduction_pct']:.2f}%")
    print("==============================\n")
    
    # Step 7: Save Metrics JSON
    summary['fps'] = fps
    summary['memory_delta_mb'] = delta_mem
    
    outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    json_path = os.path.join(outputs_dir, "metrics.json")
    
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=4)
        
    # Step 8: Final Message
    print("Pipeline complete. Open dashboard to view results.")

if __name__ == "__main__":
    main()
