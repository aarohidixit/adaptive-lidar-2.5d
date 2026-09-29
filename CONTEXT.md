# PROJECT: Adaptive 2.5D LiDAR Mapping — SIH 2026
# Last Updated: 2026-09-29 05:22:57

## Project Structure
lidar_2_5d/
├── data/               # Put KITTI .bin files here
├── data_loader.py
├── grid_engine.py
├── segmentation.py
├── visualizer.py
├── metrics.py
├── dashboard/
│   ├── main.py         # FastAPI app
│   └── templates/
│       └── index.html
├── outputs/            # Saved plots and videos
├── CONTEXT.md
└── requirements.txt

## Grid Zones (DO NOT CHANGE)
Zone 1: 0–10m   → cell size 0.05m (5cm)
Zone 2: 10–50m  → cell size 0.20m (20cm)
Zone 3: 50–100m → cell size 0.50m (50cm)
Max range: 100m

## Segmentation Logic (DO NOT CHANGE)
- Ground detection: RANSAC plane fit, label points within 0.3m of ground as "terrain"
- Remaining points: DBSCAN clustering
  - Cluster height > 1.5m AND width < 1.5m → pedestrian
  - Cluster volume matches car-sized box → vehicle
  - Anything else → static obstacle

## Color Coding (DO NOT CHANGE)
- Terrain/drivable  → Green  (#00FF00)
- Static obstacle   → Red    (#FF0000)
- Vehicle           → Blue   (#0000FF)
- Pedestrian        → Yellow (#FFFF00)
- Unknown           → Grey   (#888888)

## Completed Modules
- [x] data_loader.py
- [x] grid_engine.py
- [x] segmentation.py
- [x] visualizer.py
- [x] metrics.py
- [x] dashboard/

## Known Issues / Blockers
[Write anything that broke or needs fixing here]

All tasks are complete! Ready to run.