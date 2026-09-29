import numpy as np
import os

def load_kitti_bin(file_path):
    """
    Loads a KITTI .bin file containing LiDAR point cloud data.
    
    Args:
        file_path (str): Path to the .bin file.
        
    Returns:
        np.ndarray: A numpy array of shape (N, 4) containing x, y, z, and intensity.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # KITTI LiDAR data is stored as float32
    # The data consists of (x, y, z, intensity) for each point
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    
    # Calculate statistics
    num_points = points.shape[0]
    
    if num_points == 0:
        print("Warning: The provided file is empty.")
        return points

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    min_z, max_z = np.min(z), np.max(z)
    
    # Estimated range (max distance from origin in xy plane)
    distances = np.sqrt(x**2 + y**2)
    max_range = np.max(distances)
    
    print("--- LiDAR Point Cloud Statistics ---")
    print(f"Point count:   {num_points}")
    print(f"X range:       {min_x:.3f} to {max_x:.3f} m")
    print(f"Y range:       {min_y:.3f} to {max_y:.3f} m")
    print(f"Z range:       {min_z:.3f} to {max_z:.3f} m")
    print(f"Est. Range:    {max_range:.3f} m")
    print("------------------------------------")
    
    return points

if __name__ == "__main__":
    # Example usage:
    # points = load_kitti_bin("data/000000.bin")
    pass
