import numpy as np

class GridEngine:
    """
    Constructs an adaptive 2.5D grid representation from LiDAR point cloud data.
    The grid resolution dynamically changes based on the distance from the origin (x=0, y=0).
    
    Zones:
      Zone 1: 0–10m   → cell size 0.05m  (5 cm)
      Zone 2: 10–50m  → cell size 0.20m  (20 cm)
      Zone 3: 50–100m → cell size 0.50m  (50 cm)
    """

    def __init__(self):
        self.zones = [
            {'id': 1, 'min_r': 0.0,  'max_r': 10.0,  'cell_size': 0.05},
            {'id': 2, 'min_r': 10.0, 'max_r': 50.0,  'cell_size': 0.20},
            {'id': 3, 'min_r': 50.0, 'max_r': 100.0, 'cell_size': 0.50},
        ]
        self.max_range = 100.0

    def build_grid(self, points):
        """
        Builds the 2.5D adaptive grid from point cloud data using vectorized operations.

        Args:
            points (np.ndarray): Shape (N, 4) with (x, y, z, intensity).

        Returns:
            grid_cells (list): List of dicts with cell info and statistics.
            point_cell_mapping (np.ndarray): Shape (N, 3) → [zone_id, cx, cy].
                                             -1 for out-of-range points.
            zone_stats (dict): {'zone1': int, 'zone2': int, 'zone3': int,
                                 'total': int, 'valid_points': int,
                                 'dropped': int}
        """
        num_points = points.shape[0]
        distances = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)

        point_cell_mapping = np.full((num_points, 3), -1, dtype=np.int32)
        valid_mask = distances <= self.max_range

        zone_counts = {1: 0, 2: 0, 3: 0}

        for zone_info in self.zones:
            z_id = zone_info['id']
            c_size = zone_info['cell_size']
            min_r = zone_info['min_r']
            max_r = zone_info['max_r']

            if z_id == 3:
                mask = (distances >= min_r) & (distances <= max_r)
            else:
                mask = (distances >= min_r) & (distances < max_r)

            point_cell_mapping[mask, 0] = z_id
            point_cell_mapping[mask, 1] = np.floor(points[mask, 0] / c_size).astype(np.int32)
            point_cell_mapping[mask, 2] = np.floor(points[mask, 1] / c_size).astype(np.int32)

        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            zone_stats = {'zone1': 0, 'zone2': 0, 'zone3': 0,
                          'total': 0, 'valid_points': 0, 'dropped': num_points}
            return [], point_cell_mapping, zone_stats

        valid_mappings = point_cell_mapping[valid_mask]
        valid_z = points[valid_mask, 2]

        unique_cells, inverse_indices = np.unique(valid_mappings, axis=0, return_inverse=True)

        sorted_idx = np.argsort(inverse_indices)
        sorted_valid_indices = valid_indices[sorted_idx]
        sorted_z = valid_z[sorted_idx]
        sorted_inverse = inverse_indices[sorted_idx]

        split_points = np.where(np.diff(sorted_inverse))[0] + 1

        grouped_indices = np.split(sorted_valid_indices, split_points)
        grouped_z = np.split(sorted_z, split_points)

        grid_cells = []
        for i, cell_info in enumerate(unique_cells):
            z_id, cx, cy = cell_info
            zone_counts[int(z_id)] += 1

            pts_idx = grouped_indices[i]
            z_vals = grouped_z[i]

            grid_cells.append({
                'zone_id': int(z_id),
                'cx': int(cx),
                'cy': int(cy),
                'point_indices': pts_idx,
                'min_z': float(np.min(z_vals)),
                'max_z': float(np.max(z_vals)),
                'height': float(np.max(z_vals) - np.min(z_vals))
            })

        zone_stats = {
            'zone1': zone_counts[1],
            'zone2': zone_counts[2],
            'zone3': zone_counts[3],
            'total': len(grid_cells),
            'valid_points': int(np.sum(valid_mask)),
            'dropped': int(num_points - np.sum(valid_mask)),
        }

        return grid_cells, point_cell_mapping, zone_stats


if __name__ == "__main__":
    pass
