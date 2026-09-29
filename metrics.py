import time


class MetricsTracker:
    def __init__(self):
        self.metrics = {
            'total_points': 0,
            'processing_time_ms': 0.0,
            'class_counts': {
                'unknown': 0,
                'terrain': 0,
                'static_obstacle': 0,
                'vehicle': 0,
                'pedestrian': 0
            },
            'memory_reduction': None,
            'zone_stats': None,
            'frame_index': 0,
        }
        self.start_time = 0

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        self.metrics['processing_time_ms'] = (time.time() - self.start_time) * 1000

    def update(self, points, labels):
        self.metrics['total_points'] = int(len(labels))
        self.metrics['class_counts']['unknown'] = int((labels == 0).sum())
        self.metrics['class_counts']['terrain'] = int((labels == 1).sum())
        self.metrics['class_counts']['static_obstacle'] = int((labels == 2).sum())
        self.metrics['class_counts']['vehicle'] = int((labels == 3).sum())
        self.metrics['class_counts']['pedestrian'] = int((labels == 4).sum())

    def update_zone_stats(self, zone_stats: dict):
        """Store zone breakdown from GridEngine."""
        self.metrics['zone_stats'] = zone_stats

    def calculate_memory_reduction(self, actual_cells, bytes_per_cell=32):
        """Calculate memory saved vs a naive uniform 5cm grid over 200×200m."""
        uniform_cells = (200 / 0.05) * (200 / 0.05)  # 200m x 200m @ 5cm
        uniform_mb = (uniform_cells * bytes_per_cell) / (1024 * 1024)
        adaptive_mb = (actual_cells * bytes_per_cell) / (1024 * 1024)
        reduction_pct = ((uniform_mb - adaptive_mb) / uniform_mb) * 100 if uniform_mb > 0 else 0.0

        result = {
            'uniform_cells': int(uniform_cells),
            'uniform_mb': uniform_mb,
            'adaptive_cells': int(actual_cells),
            'adaptive_mb': adaptive_mb,
            'reduction_pct': reduction_pct,
        }
        self.metrics['memory_reduction'] = result
        return result

    def set_frame_index(self, idx: int):
        self.metrics['frame_index'] = idx

    def get_summary(self) -> dict:
        return dict(self.metrics)
