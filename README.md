# LiDAR 2.5D Foveated Adaptive Grid System

> **DRDO Smart India Hackathon 2026** — Problem Statement: Adaptive Resolution 2.5D LiDAR Mapping for Autonomous Vehicle Perception

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What Is This?

Modern autonomous vehicles carry **64-beam LiDAR sensors** that produce **~1.2 million 3D points per second**. Processing every point at full resolution requires enormous compute — impossible in real-time on embedded hardware.

This project implements a **foveated (variable-resolution) 2.5D grid engine** — the core algorithmic contribution required by the DRDO problem statement. It works like the human eye:

- **Near zone (0–10 m)**: Ultra-fine 5 cm cells — every pothole and kerb edge matters here
- **Mid zone (10–50 m)**: Medium 20 cm cells — approximate shape is enough
- **Far zone (50–100 m)**: Coarse 50 cm cells — only general awareness needed

Each grid cell stores **elevation (Z) + semantic label** — the "2.5D" representation. It is not a flat 2D map (loses height) and not full 3D (too heavy). It is the minimum-information-maximum-usefulness middle ground.

---

## The Core Problem (DRDO Statement)

| Approach | Memory (100 m range) | Detail | RT Capable? |
|----------|---------------------|--------|-------------|
| Full 3D point cloud | ~5 GB/s | Perfect | ❌ No |
| Uniform 2D grid (5 cm) | **488 MB** | High | ❌ No |
| Standard 2D map | ~10 MB | Zero height | ✅ Yes, useless |
| **Our Adaptive 2.5D Grid** | **~0.67 MB** | Zone-aware | ✅ **Yes** |

**Memory reduction: 99.9%** compared to a naive uniform 5 cm grid over 200×200 m.

---

## Architecture

```
                    ┌────────────────────────────────────────┐
                    │         LiDAR Sensor (KITTI)           │
                    │    121,000 points/frame @ 10 Hz        │
                    └──────────────┬─────────────────────────┘
                                   │  .bin file (x, y, z, intensity)
                                   ▼
                    ┌────────────────────────────────────────┐
                    │         data_loader.py                 │
                    │  Load & parse binary point cloud       │
                    └──────────────┬─────────────────────────┘
                                   │  np.ndarray (N, 4)
                        ┌──────────┴──────────┐
                        ▼                     ▼
          ┌─────────────────────┐   ┌──────────────────────────┐
          │  segmentation.py    │   │     grid_engine.py        │
          │                     │   │                           │
          │  1. RANSAC ground   │   │  Zone 1: 0–10m  @ 5cm    │
          │     plane fitting   │   │  Zone 2: 10–50m @ 20cm   │
          │  2. DBSCAN cluster  │   │  Zone 3: 50–100m @ 50cm  │
          │  3. Rule-based      │   │                           │
          │     classification  │   │  Returns: unique cells    │
          │     · Pedestrian    │   │  with [zone, cx, cy,      │
          │     · Vehicle       │   │   min_z, max_z, height]   │
          │     · Static obs    │   └──────────┬───────────────┘
          └──────────┬──────────┘              │
                     │ labels (N,)             │ grid_cells, zone_stats
                     └──────────┬─────────────┘
                                ▼
                    ┌────────────────────────────────────────┐
                    │         visualizer.py                  │
                    │  · plot_bev()  → bev_map.png           │
                    │  · plot_front_view() → current_frame.jpg│
                    └──────────────┬─────────────────────────┘
                                   │
                    ┌──────────────┴─────────────────────────┐
                    │         metrics.py                     │
                    │  · FPS, latency, class counts          │
                    │  · Memory reduction calculation        │
                    │  · Zone stats (zone1/2/3 cell counts)  │
                    │  → metrics.json, metrics_history.json  │
                    └──────────────┬─────────────────────────┘
                                   │
                    ┌──────────────┴─────────────────────────┐
                    │         dashboard/main.py (FastAPI)    │
                    │  GET /              → HTML dashboard   │
                    │  GET /api/metrics   → latest JSON      │
                    │  GET /api/metrics/history → last 60fr  │
                    │  GET /outputs/bev_map.png              │
                    │  GET /outputs/current_frame.jpg        │
                    └────────────────────────────────────────┘
```

---

## Directory Structure

```
lidar_2_5d/
├── data/
│   └── *.bin                   # KITTI LiDAR frames (binary point clouds)
├── outputs/
│   ├── bev_map.png             # Latest Bird's-Eye View map
│   ├── current_frame.jpg       # Latest front-view LiDAR projection
│   ├── metrics.json            # Latest frame metrics
│   └── metrics_history.json    # Rolling 60-frame history
├── dashboard/
│   ├── main.py                 # FastAPI web server
│   └── templates/
│       └── index.html          # Real-time dashboard UI
├── data_loader.py              # KITTI .bin loader
├── segmentation.py             # RANSAC + DBSCAN segmentation engine
├── grid_engine.py              # Foveated adaptive grid (core algorithm)
├── visualizer.py               # BEV + front-view matplotlib plots
├── metrics.py                  # Performance tracking & memory analysis
├── run_batch.py                # CLI batch processor with live terminal output
├── requirements.txt
└── README.md
```

---

## Algorithms In Depth

### 1. Foveated Grid Engine (`grid_engine.py`)

The central algorithmic contribution. Given a point `(x, y, z)`:

```python
distance = sqrt(x² + y²)

if   distance <  10m: cell_size = 0.05m   # Zone 1: 5 cm
elif distance <  50m: cell_size = 0.20m   # Zone 2: 20 cm
else:                 cell_size = 0.50m   # Zone 3: 50 cm

cell_x = floor(x / cell_size)
cell_y = floor(y / cell_size)
```

All points mapping to `(zone, cell_x, cell_y)` are grouped. Each cell stores:
- `min_z`, `max_z` — elevation range (the "2.5D")
- `height = max_z - min_z` — used for obstacle detection

Fully **vectorized with NumPy** — no Python loops over individual points.

#### Why This Saves 99.9% Memory

A uniform 5 cm grid over 200×200 m = `(200/0.05)² = 16,000,000` cells.  
Our adaptive grid for one real frame = `~22,000` unique populated cells.  
Ratio: `22,000 / 16,000,000 = 0.14%` — **99.86% reduction**.

### 2. Ground Segmentation — RANSAC (`segmentation.py`)

RANSAC (Random Sample Consensus) finds the dominant ground plane:

1. Randomly sample 3 points, fit a plane `ax + by + cz + d = 0`
2. Count inliers: points with distance to plane < 0.3 m
3. Repeat 50× — keep the best plane
4. Label all inliers as **terrain (drivable)**

This is robust to outliers (poles, cars) and requires no training data.

### 3. Object Classification — DBSCAN (`segmentation.py`)

Non-ground points are clustered with DBSCAN (`eps=0.5m, min_samples=10`), then each cluster is classified by bounding box geometry:

| Class | Height | Max dim | Min dim |
|-------|--------|---------|---------|
| Pedestrian | > 1.5 m | < 1.5 m | — |
| Vehicle | 1.0–3.0 m | 2.0–7.0 m | 1.2–3.0 m |
| Static Obstacle | everything else | | |

### 4. Front-View Projection (`visualizer.py`)

Since no camera images are available in the dataset, a **pseudo-camera view** is synthesized by projecting LiDAR into a cylindrical range image:

```
azimuth   = arctan2(y, x)      → horizontal pixel u
elevation = arctan2(z, sqrt(x²+y²)) → vertical pixel v
```

Each point is painted with its semantic class colour. This mimics what a front-facing camera would see.

---

## Getting Started

### Prerequisites

- Python 3.9+
- KITTI raw LiDAR data (`.bin` files) placed in `data/`

### Install

```powershell
# Create virtualenv
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### `requirements.txt`
```
numpy
scipy
scikit-learn
matplotlib
fastapi
uvicorn[standard]
psutil
colorama
```

### Run the Pipeline

```powershell
# Process all frames as fast as possible (with camera images)
python run_batch.py --downsample --images image_data

# Slow-mode for live demo (0.5 FPS = one frame every 2 seconds)
python run_batch.py --downsample --images image_data --fps 0.5

# Presentation Mode: Process EXACTLY ONE frame and freeze the dashboard
# (Perfect for explaining a specific frame to the judges without it moving)
python run_single.py 0
```

### Launch Dashboard

```powershell
cd dashboard
..\venv\Scripts\uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000** in a browser.

---

## Dashboard Features

| Panel | What It Shows |
|-------|--------------|
| **Performance** | Total points, latency (ms), estimated FPS + sparkline |
| **Class Breakdown** | Animated donut chart with % for each semantic class |
| **Memory Comparison** | Animated bars: uniform vs adaptive grid, "X% SAVED" badge |
| **Zone Grid Cells** | Cell counts for Zone 1 / Zone 2 / Zone 3 |
| **Camera View** | Front-view LiDAR cylindrical projection (pseudo-camera) |
| **LiDAR BEV Map** | Bird's-eye view with zone rings, class colours, legend |
| **FPS History** | SVG chart of last 60 frames' FPS |
| **Frame Log** | Live terminal feed: per-frame stats in colour |

---

## CLI Output (Coloured Terminal)

```
==============================================================
        LiDAR 2.5D Foveated Grid Pipeline
==============================================================
  Found 108 frame(s)
  Fast mode: 30% point sample

  Frame [1/108]  0000000000.bin  0.40 FPS  (2519 ms)
--------------------------------------------------------------
  CLASS BREAKDOWN
  Terrain            [###################.........]    25,514
  Static Obstacle    [#######.....................]     9,675
  Vehicle            [............................]       896

  ZONE GRID CELLS
  Zone 1 (0-10m)     [###############.............]    12,533
  Zone 2 (10-50m)    [###########.................]     9,371
  Zone 3 (50-100m)   [............................]       495
  Total unique cells       22,399
  Valid points             36,304

  MEMORY COMPARISON
         Uniform 5cm  ############################     488.3 MB
            Adaptive  #...........................        0.68 MB
        Memory saved  99.9% reduction
--------------------------------------------------------------
```

---

## Key Results

| Metric | Value |
|--------|-------|
| Uniform 5 cm grid memory | 488 MB |
| Adaptive grid memory | ~0.67 MB |
| **Memory reduction** | **99.9%** |
| Processing speed | 0.4–1.2 FPS (CPU, no GPU) |
| Zone 1 cells (0–10 m) | ~12,500 |
| Zone 2 cells (10–50 m) | ~9,000 |
| Zone 3 cells (50–100 m) | ~450 |
| Total unique cells / frame | ~22,000 |
| Points per frame (30% sample) | ~36,000 |

---

## Framing for SIH Judges

> "We implemented the **variable resolution foveated grid engine** — the core algorithmic contribution of the DRDO problem statement. The adaptive grid reduces memory footprint by **99.9%** versus a naive uniform 5 cm grid. Terrain segmentation uses RANSAC ground fitting; object classification uses geometry-based DBSCAN clustering — a baseline that the architecture explicitly supports replacing with PointNet++ or Sparse CNN inference."

### What We Did Not Do (and why)
- **No deep learning training** — requires days + GPU + labelled dataset. We use rule-based classification as a principled baseline.
- **No ROS integration** — out of scope for a 1-day hackathon build; the grid engine output format is ROS-compatible.
- **No sensor fusion** — camera images were not available in this KITTI split.

---

## Extending the System

| Extension | How |
|-----------|-----|
| Replace DBSCAN with PointNet++ | Swap `segment()` method, keep grid engine unchanged |
| Add camera fusion | Pass `(u,v)` camera projection into grid cells for RGB labelling |
| Export to ROS | Serialize `grid_cells` as `nav_msgs/OccupancyGrid` |
| Real-time LiDAR | Replace `data_loader.py` with a UDP socket / ROS subscriber |
| GPU acceleration | Replace NumPy grid ops with CuPy |

---

## Dataset

This project uses [KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php) — specifically the Velodyne HDL-64E LiDAR scans.

Each `.bin` file contains `N × 4` float32 values: `[x, y, z, intensity]`, where the coordinate system is vehicle-centered (x=forward, y=left, z=up).

---

## License

MIT License. See [LICENSE](LICENSE).

---

*Built for DRDO Smart India Hackathon 2026 — Problem: Adaptive Resolution 2.5D LiDAR Mapping for Autonomous Vehicle Perception*
