import numpy as np
from plyfile import PlyData, PlyElement

input_ply = "/home/hayrap/repos/gl_image_to_3d/external/SuGaR/utils/pc_gaussian.ply"
output_ply = "gaussians_clean.ply"

ply = PlyData.read(input_ply)
v = ply['vertex'].data

xyz = np.stack([v['x'], v['y'], v['z']], axis=1)
opacity = v['opacity']

# scales (names may differ slightly)
scales = np.stack([v['scale_0'], v['scale_1'], v['scale_2']], axis=1)
scale_norm = np.linalg.norm(scales, axis=1)

# -----------------------
# Filters
# -----------------------

# 1️⃣ remove low opacity
mask_opacity = opacity > np.percentile(opacity, 20)

# 2️⃣ remove huge gaussians
mask_scale = scale_norm < np.percentile(scale_norm, 95)

# 3️⃣ remove spatial outliers
center = xyz.mean(axis=0)
dist = np.linalg.norm(xyz - center, axis=1)
mask_dist = dist < np.percentile(dist, 95)

mask = mask_opacity & mask_scale & mask_dist

print("Kept:", mask.sum(), "/", len(mask))

clean_vertices = v[mask]

PlyData([PlyElement.describe(clean_vertices, 'vertex')]).write(output_ply)