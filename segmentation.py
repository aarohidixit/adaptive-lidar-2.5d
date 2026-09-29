import numpy as np
from sklearn.cluster import DBSCAN

class SegmentationEngine:
    def __init__(self):
        self.labels = {
            'unknown': 0,
            'terrain': 1,
            'static_obstacle': 2,
            'vehicle': 3,
            'pedestrian': 4
        }
        
    def fit_ground_ransac(self, points, iterations=100, threshold=0.3):
        """Fits a plane to the point cloud using RANSAC to find the ground."""
        best_inliers = []
        best_plane = None
        
        # We only need x, y, z for plane fitting
        coords = points[:, :3]
        n_points = coords.shape[0]
        
        if n_points < 3:
            return np.array([], dtype=np.int32)
            
        for _ in range(iterations):
            # Sample 3 random points
            idx = np.random.choice(n_points, 3, replace=False)
            p1, p2, p3 = coords[idx]
            
            # Plane equation: ax + by + cz + d = 0
            v1 = p2 - p1
            v2 = p3 - p1
            
            # Normal vector
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
                
            normal = normal / norm
            d = -np.dot(normal, p1)
            
            # Distance of all points to the plane
            distances = np.abs(np.dot(coords, normal) + d)
            
            inliers = np.where(distances < threshold)[0]
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (normal, d)
                
        return best_inliers

    def segment(self, points):
        """
        Segments the point cloud into terrain, pedestrian, vehicle, static obstacle.
        
        Args:
            points: (N, 4) or (N, 3) numpy array
            
        Returns:
            labels: (N,) numpy array of class IDs
        """
        n_points = points.shape[0]
        labels = np.zeros(n_points, dtype=np.int32) # Default 0 (unknown)
        
        if n_points == 0:
            return labels
            
        # 1. Ground Detection
        ground_inliers = self.fit_ground_ransac(points, iterations=50, threshold=0.3)
        labels[ground_inliers] = self.labels['terrain']
        
        # 2. Clustering remaining points
        non_ground_mask = labels == self.labels['unknown']
        non_ground_indices = np.where(non_ground_mask)[0]
        
        if len(non_ground_indices) == 0:
            return labels
            
        non_ground_points = points[non_ground_indices, :3]
        
        # DBSCAN clustering (eps=0.5m, min_samples=10)
        dbscan = DBSCAN(eps=0.5, min_samples=10, n_jobs=-1)
        cluster_labels = dbscan.fit_predict(non_ground_points)
        
        unique_clusters = set(cluster_labels)
        
        for c_id in unique_clusters:
            if c_id == -1:
                # Noise points -> static obstacle
                noise_idx = non_ground_indices[cluster_labels == -1]
                labels[noise_idx] = self.labels['static_obstacle']
                continue
                
            # Get points for this cluster
            c_mask = cluster_labels == c_id
            c_idx = non_ground_indices[c_mask]
            c_points = non_ground_points[c_mask]
            
            # Calculate bounding box dimensions
            min_x, max_x = np.min(c_points[:, 0]), np.max(c_points[:, 0])
            min_y, max_y = np.min(c_points[:, 1]), np.max(c_points[:, 1])
            min_z, max_z = np.min(c_points[:, 2]), np.max(c_points[:, 2])
            
            width = max_x - min_x
            length = max_y - min_y
            height = max_z - min_z
            
            # Use largest horizontal dimension as length and smallest as width
            h_max = max(width, length)
            h_min = min(width, length)
            
            volume = width * length * height
            
            # Rule-based classification
            # Pedestrian: height > 1.5m AND width < 1.5m
            if height > 1.5 and h_max < 1.5:
                labels[c_idx] = self.labels['pedestrian']
            # Vehicle: car-sized box (roughly 2.0-7.0m length, 1.2-3.0m width, 1.0-3.0m height)
            elif 2.0 <= h_max <= 7.0 and 1.2 <= h_min <= 3.0 and 1.0 <= height <= 3.0:
                labels[c_idx] = self.labels['vehicle']
            else:
                # Anything else -> static obstacle
                labels[c_idx] = self.labels['static_obstacle']
                
        return labels

if __name__ == "__main__":
    pass
