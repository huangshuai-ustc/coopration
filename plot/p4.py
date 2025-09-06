import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Data preparation (triplet: count)
data = {
    (1, 1, 0): 1053, (1, 1, 2): 2803, (4, 0, 1): 76, (1, 2, 0): 505,
    (1, 0, 0): 219, (1, 0, 1): 10, (1, 2, 2): 205, (1, 0, 2): 239,
    (5, 1, 2): 88, (2, 1, 0): 20, (4, 2, 0): 334, (4, 1, 1): 43,
    (1, 1, 1): 104, (1, 2, 1): 186, (5, 2, 0): 53, (2, 0, 0): 113,
    (4, 2, 2): 127, (5, 0, 1): 20, (3, 1, 2): 30, (0, 2, 1): 9,
    (6, 1, 1): 45, (2, 0, 2): 21, (0, 0, 0): 13, (2, 1, 1): 8,
    (3, 1, 0): 22, (5, 1, 0): 1, (3, 0, 1): 9, (5, 2, 1): 18,
    (4, 1, 2): 7, (6, 1, 0): 2, (0, 0, 2): 5, (5, 0, 0): 4,
    (6, 2, 2): 4, (3, 2, 0): 13, (3, 2, 2): 1, (2, 0, 1): 6,
    (3, 1, 1): 1, (4, 2, 1): 1, (2, 2, 0): 4, (4, 0, 2): 1,
    (0, 2, 0): 1, (6, 2, 1): 1, (3, 0, 0): 2, (2, 1, 2): 1,
    (6, 2, 0): 1, (6, 1, 2): 1, (3, 2, 1): 1, (5, 2, 2): 2,
    (0, 1, 0): 1
}

# Extract coordinates and counts
x = [k[0] for k in data.keys()]
y = [k[1] for k in data.keys()]
z = [k[2] for k in data.keys()]
sizes = [v * 5 for v in data.values()]   # Scale sphere sizes
colors = np.log(np.array(list(data.values())) + 1)  # Log-transform for color mapping

# Create 3D plot
fig = plt.figure(figsize=(12, 10), dpi=600)
ax = fig.add_subplot(111, projection='3d')

# Plot scatter points as spheres
scatter = ax.scatter(
    x, y, z,
    s=sizes,          # Sphere sizes
    c=colors,         # Colors based on counts
    cmap='viridis',   # Colormap
    alpha=0.7,        # Transparency
    depthshade=True   # Depth shading
)

# Add colorbar (showing count range)
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Log(Count + 1)', rotation=270, labelpad=15)

# Annotate points with their counts (only for points above a threshold to avoid clutter)
threshold = 100  # Only label points with count > 100
for (xi, yi, zi), count in zip(zip(x, y, z), data.values()):
    if count > threshold:
        ax.text(
            xi, yi, zi,
            str(count),
            color='red', fontsize=8, ha='center', va='center'
        )

# Add global explanation (alternative to legend)
ax.text2D(
    0.05, 0.95,
    "Note: Numbers on spheres represent counts",
    transform=ax.transAxes,
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8)
)

# Set axis labels and title
ax.set_xlabel('X Coordinate', fontsize=12)
ax.set_ylabel('Y Coordinate', fontsize=12)
ax.set_zlabel('Z Coordinate', fontsize=12)
ax.set_title('3D Visualization of Triplet Categories with Counts', fontsize=14, pad=20)

# Adjust viewing angle
ax.view_init(elev=20, azim=45)
plt.tight_layout()

# Save figure (optional)
plt.savefig('3D_Triplet_Counts_Visualization.png', bbox_inches='tight', dpi=600)
plt.show()