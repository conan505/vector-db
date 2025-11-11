import numpy as np

# Generate 5 random 3D vectors
vectors = np.random.rand(5, 3)
query = np.random.rand(1, 3)

# Euclidean distance (brute force)
distances = np.linalg.norm(vectors - query, axis=1)
print("vectors:", vectors)
print("Distances:", distances)
closest_idx = np.argmin(distances)
print("Closest vector:", vectors[closest_idx])