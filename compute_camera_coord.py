import numpy as np

eye    = np.array([-0.01, -0.33, 0.60])
# target = np.array([0.0, 0.0, 0.0])
target = np.array([0.0148, 0.0, 0.0])

forward = target - eye
forward /= np.linalg.norm(forward)

world_up = np.array([0.0, 0.0, 1.0])
right = np.cross(forward, world_up)
right /= np.linalg.norm(right)

up = np.cross(right, forward)

# OpenGL convention: camera looks down -Z, Y is up
R = np.column_stack([right, up, -forward])

# Convert rotation matrix to quaternion (w, x, y, z)
from scipy.spatial.transform import Rotation
q = Rotation.from_matrix(R).as_quat()  # returns (x, y, z, w)
w, x, y, z = q[3], q[0], q[1], q[2]
print(f"pos={tuple(eye)}, rot=({w:.4f}, {x:.4f}, {y:.4f}, {z:.4f})")