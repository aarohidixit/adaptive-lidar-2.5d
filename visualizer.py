import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Perceptually distinct, LIGHT-background-friendly palette
COLOR_MAP = {
    0: ('#94a3b8', 'Unknown'),        # slate
    1: ('#16a34a', 'Terrain'),        # green
    2: ('#ea580c', 'Static Obstacle'), # orange
    3: ('#0284c7', 'Vehicle'),         # blue
    4: ('#d97706', 'Pedestrian'),      # amber
}

ZONE_RADII  = [10, 50, 100]
ZONE_LABELS = ['Zone 1\n(0-10m, 5cm)', 'Zone 2\n(10-50m, 20cm)', 'Zone 3\n(50-100m, 50cm)']
ZONE_COLORS = ['#4f46e5', '#7c3aed', '#9333ea']


class Visualizer:
    def __init__(self, output_dir="outputs"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(base_dir, output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    # ── Bird's-Eye View ────────────────────────────────────────────────────
    def plot_bev(self, points, labels, filename="bev_map.png"):
        """
        Large, high-quality Bird's-Eye View with zone rings and legend.
        Light theme styling.
        """
        fig, ax = plt.subplots(figsize=(12, 12), facecolor='#ffffff')
        ax.set_facecolor('#f8fafc')

        # Zone boundary rings
        for i, r in enumerate(ZONE_RADII):
            circle = plt.Circle(
                (0, 0), r, color=ZONE_COLORS[i],
                fill=False, linestyle='--', linewidth=1.2, alpha=0.6
            )
            ax.add_patch(circle)
            angle_rad = np.deg2rad(38)
            tx = r * np.cos(angle_rad) + 1.2
            ty = r * np.sin(angle_rad) + 1.2
            ax.text(tx, ty, ZONE_LABELS[i],
                    fontsize=8, color=ZONE_COLORS[i], alpha=0.9,
                    fontfamily='monospace', linespacing=1.4)

        # Plot each class (ax.plot is much faster than ax.scatter for many points)
        for label_id, (hex_color, class_name) in COLOR_MAP.items():
            mask = labels == label_id
            if not np.any(mask):
                continue
            pts = points[mask]
            ax.plot(
                pts[:, 0], pts[:, 1],
                marker='.', linestyle='',
                color=hex_color, markersize=1.5, alpha=0.9,
                rasterized=True
            )

        # Ego vehicle marker
        ax.plot(0, 0, marker='D', markersize=10, color='#1e293b',
                markeredgecolor='#4f46e5', markeredgewidth=2.0, zorder=10)

        # Grid
        ax.grid(color='#cbd5e1', linestyle='--', linewidth=0.5, alpha=0.8)
        ax.set_xlim(-105, 105)
        ax.set_ylim(-105, 105)
        ax.set_aspect('equal')

        for spine in ax.spines.values():
            spine.set_edgecolor('#94a3b8')
        ax.tick_params(colors='#475569', labelsize=9)
        ax.xaxis.label.set_color('#334155')
        ax.yaxis.label.set_color('#334155')
        ax.set_xlabel('X (meters)', fontsize=10, labelpad=8, fontweight='bold')
        ax.set_ylabel('Y (meters)', fontsize=10, labelpad=8, fontweight='bold')
        ax.set_title(
            'Foveated Adaptive 2.5D LiDAR Grid - Bird\'s Eye View',
            color='#0f172a', fontsize=14, pad=16, fontweight='bold'
        )

        # Legend
        class_patches = [
            mpatches.Patch(color=hex_color, label=class_name)
            for _, (hex_color, class_name) in COLOR_MAP.items()
            if np.any(labels == _)
        ]
        class_patches.append(mpatches.Patch(color='#1e293b', label='Ego Vehicle'))
        ax.legend(
            handles=class_patches, loc='upper right', fontsize=9.5,
            framealpha=0.85, facecolor='#ffffff',
            edgecolor='#cbd5e1', labelcolor='#1e293b',
        )

        plt.tight_layout(pad=0.8)
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_path

    # ── Front-view (range image) — used as "camera view" ──────────────────
    def plot_front_view(self, points, labels, filename="front_view.png"):
        """
        Simulates a camera view by projecting LiDAR into a cylindrical
        range image (azimuth x elevation), coloured by semantic class.
        Light theme styling.
        """
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Only keep front 180 deg (x > 0) within 60 m
        dist_xy = np.sqrt(x**2 + y**2)
        front   = (x > 0) & (dist_xy < 60.0)
        if not np.any(front):
            front = dist_xy < 80.0  # fallback

        x, y, z, lbl = x[front], y[front], z[front], labels[front]
        dist = np.sqrt(x**2 + y**2 + z**2)

        # Spherical coords
        azimuth   = np.rad2deg(np.arctan2(y, x))      # -90..90
        elevation = np.rad2deg(np.arctan2(z, np.sqrt(x**2 + y**2)))  # ~-25..5

        # Canvas size
        W, H = 1280, 380
        az_min, az_max   = -60.0, 60.0
        el_min, el_max   = -25.0,  5.0

        u = ((azimuth   - az_min) / (az_max - az_min) * W).astype(int)
        v = ((1 - (elevation - el_min) / (el_max - el_min)) * H).astype(int)

        in_view = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, lbl_v, dist_v = u[in_view], v[in_view], lbl[in_view], dist[in_view]

        # Build RGBA image (light background)
        img = np.zeros((H, W, 4), dtype=np.uint8)
        img[:, :, :3] = 248  # #f8fafc
        img[:, :, 3] = 255

        order = np.argsort(-dist_v)
        u, v, lbl_v = u[order], v[order], lbl_v[order]

        palette_arr = np.array([
            [148, 163, 184, 255],   # 0: slate   – unknown
            [22,  163, 74,  255],   # 1: green   – terrain
            [234, 88,  12,  255],   # 2: orange  – obstacle
            [2,   132, 199, 255],   # 3: blue    – vehicle
            [217, 119, 6,   255],   # 4: amber   – pedestrian
        ], dtype=np.uint8)

        # Clip label indices safely
        lbl_v = np.clip(lbl_v.astype(int), 0, 4)
        
        # Vectorized rendering: set 3x3 blocks for visibility
        colors = palette_arr[lbl_v]
        
        # Instead of looping, map to a slightly downsampled 2D grid, or just fast scatter
        # For true speed, we can use 2D array assignment.
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                r = np.clip(v + dr, 0, H-1)
                c = np.clip(u + dc, 0, W-1)
                img[r, c] = colors

        fig, ax = plt.subplots(figsize=(14, 4.2), facecolor='#ffffff')
        ax.set_facecolor('#ffffff')
        ax.imshow(img, aspect='auto', interpolation='nearest',
                  extent=[az_min, az_max, el_min, el_max])

        ax.axhline(0, color='#94a3b8', linewidth=0.8, linestyle='--', alpha=0.8)

        ax.set_xlabel('Azimuth (deg)', fontsize=9, color='#334155', fontweight='bold')
        ax.set_ylabel('Elevation (deg)', fontsize=9, color='#334155', fontweight='bold')
        ax.set_title('Front-View LiDAR Projection (Pseudo-Camera)',
                     color='#0f172a', fontsize=11, pad=10, fontweight='bold')

        for spine in ax.spines.values():
            spine.set_edgecolor('#cbd5e1')
        ax.tick_params(colors='#475569', labelsize=8)

        plt.tight_layout(pad=0.4)
        out_path = os.path.join(self.output_dir, filename)
        plt.savefig(out_path, dpi=120, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return out_path
