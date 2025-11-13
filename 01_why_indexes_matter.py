"""
Demonstration: Why do we need indexes?
Shows the curse of dimensionality and performance problems with brute force.
"""

import numpy as np
import time
from vectordb import VectorDB

print("=" * 70)
print("WHY DO WE NEED INDEXES?")
print("=" * 70)

print("\n📊 EXPERIMENT 1: The Curse of Dimensionality")
print("=" * 70)
print("\nAs dimensions increase, all distances become similar!")
print("This makes finding 'nearest' neighbors harder.\n")

for dims in [3, 10, 50, 100, 500, 1000]:
    # Generate random vectors
    n_vectors = 1000
    vectors = np.random.rand(n_vectors, dims)
    query = np.random.rand(1, dims)
    
    # Calculate all distances
    distances = np.linalg.norm(vectors - query, axis=1)
    
    # Statistics
    min_dist = distances.min()
    max_dist = distances.max()
    mean_dist = distances.mean()
    ratio = max_dist / min_dist
    
    print(f"{dims:4d}D: Min={min_dist:.2f}, Max={max_dist:.2f}, "
          f"Mean={mean_dist:.2f}, Ratio={ratio:.2f}")

print("\n💡 Key Insight:")
print("  - Low dimensions (3D): Ratio = ~12 (big difference between near/far)")
print("  - High dimensions (1000D): Ratio = ~1.1 (everything is similar distance!)")
print("  - This is the 'Curse of Dimensionality'")

print("\n" + "=" * 70)
print("📊 EXPERIMENT 2: Brute Force Performance")
print("=" * 70)
print("\nHow long does brute force search take as data grows?\n")

dimensions = 128  # Typical embedding size

for n_vectors in [100, 1000, 10000, 100000]:
    # Create database
    vectors = np.random.rand(n_vectors, dimensions)
    query = np.random.rand(1, dimensions)
    
    # Time the search
    start = time.time()
    distances = np.linalg.norm(vectors - query, axis=1)
    top_k = np.argsort(distances)[:10]
    elapsed = (time.time() - start) * 1000  # Convert to milliseconds
    
    print(f"{n_vectors:7,} vectors: {elapsed:6.2f} ms")

print("\n💡 Key Insight:")
print("  - Time grows linearly with data size (O(n))")
print("  - 100K vectors: ~100ms per query")
print("  - 1M vectors: ~1 second per query")
print("  - 10M vectors: ~10 seconds per query ❌ Too slow!")

print("\n" + "=" * 70)
print("📊 EXPERIMENT 3: Distance Distribution")
print("=" * 70)
print("\nLet's visualize how distances are distributed\n")

# Generate data
n_vectors = 10000
dimensions = 128
vectors = np.random.rand(n_vectors, dimensions)
query = np.random.rand(1, dimensions)

# Calculate distances
distances = np.linalg.norm(vectors - query, axis=1)

# Create histogram
hist, bins = np.histogram(distances, bins=20)
max_bar_width = 50

print("Distance Distribution:")
print("(Each * represents ~{} vectors)\n".format(n_vectors // max_bar_width))

for i in range(len(hist)):
    bar_width = int((hist[i] / hist.max()) * max_bar_width)
    bar = '*' * bar_width
    print(f"{bins[i]:5.2f} - {bins[i+1]:5.2f}: {bar} ({hist[i]})")

print("\n💡 Key Insight:")
print("  - Most vectors are at similar distances (bell curve)")
print("  - Hard to distinguish 'near' from 'far'")
print("  - Need smart algorithms to find true nearest neighbors")

print("\n" + "=" * 70)
print("📊 EXPERIMENT 4: What if we had an index?")
print("=" * 70)
print("\nSimulation: Index vs Brute Force\n")

n_vectors = 100000
dimensions = 128

print(f"Database: {n_vectors:,} vectors, {dimensions} dimensions")
print(f"Query: Find top-10 nearest neighbors\n")

# Brute force
vectors = np.random.rand(n_vectors, dimensions)
query = np.random.rand(1, dimensions)

start = time.time()
distances = np.linalg.norm(vectors - query, axis=1)
top_k = np.argsort(distances)[:10]
brute_force_time = (time.time() - start) * 1000

print(f"Brute Force:")
print(f"  - Checked: {n_vectors:,} vectors")
print(f"  - Time: {brute_force_time:.2f} ms")
print(f"  - Accuracy: 100% (exact)")

# Simulated index (IVF with 100 clusters)
n_clusters = 100
vectors_per_cluster = n_vectors // n_clusters
n_clusters_to_search = 5  # Search 5 nearest clusters
vectors_checked = vectors_per_cluster * n_clusters_to_search

# Simulate time (proportional to vectors checked)
index_time = brute_force_time * (vectors_checked / n_vectors)

print(f"\nWith IVF Index (100 clusters, search 5):")
print(f"  - Checked: {vectors_checked:,} vectors ({vectors_checked/n_vectors*100:.1f}%)")
print(f"  - Time: {index_time:.2f} ms")
print(f"  - Accuracy: ~95% (approximate)")
print(f"  - Speedup: {brute_force_time/index_time:.1f}x faster! 🚀")

# Simulated HNSW
hnsw_vectors_checked = int(np.log2(n_vectors) * 50)  # Logarithmic
hnsw_time = brute_force_time * (hnsw_vectors_checked / n_vectors)

print(f"\nWith HNSW Index:")
print(f"  - Checked: {hnsw_vectors_checked:,} vectors ({hnsw_vectors_checked/n_vectors*100:.1f}%)")
print(f"  - Time: {hnsw_time:.2f} ms")
print(f"  - Accuracy: ~98% (approximate)")
print(f"  - Speedup: {brute_force_time/hnsw_time:.1f}x faster! 🚀🚀")

print("\n" + "=" * 70)
print("📊 EXPERIMENT 5: Real-World Scale")
print("=" * 70)
print("\nWhat about production systems?\n")

scenarios = [
    ("Small startup", 10_000, 0.01),
    ("Medium company", 1_000_000, 1.0),
    ("Large company", 10_000_000, 10.0),
    ("Tech giant", 1_000_000_000, 1000.0),
]

print(f"{'Scenario':<20} {'Vectors':>15} {'Brute Force':>15} {'With Index':>15}")
print("-" * 70)

for name, n_vectors, brute_time in scenarios:
    # Assume HNSW gives 100x speedup
    index_time = brute_time / 100
    
    print(f"{name:<20} {n_vectors:>15,} {brute_time:>13.2f}s {index_time:>13.2f}s")

print("\n💡 Key Insight:")
print("  - Without indexes: Queries take seconds to minutes ❌")
print("  - With indexes: Queries take milliseconds ✅")
print("  - Indexes are ESSENTIAL for production systems!")

print("\n" + "=" * 70)
print("SUMMARY: Why Indexes Matter")
print("=" * 70)
print("""
1. 🎯 CURSE OF DIMENSIONALITY
   - High-dimensional spaces make all points equidistant
   - Hard to find true nearest neighbors
   - Need smart algorithms

2. ⚡ PERFORMANCE
   - Brute force: O(n) - linear time
   - With index: O(log n) - logarithmic time
   - 100x-1000x speedup possible!

3. 📈 SCALABILITY
   - Small data (< 10K): Brute force is fine
   - Medium data (10K-1M): Indexes help a lot
   - Large data (> 1M): Indexes are ESSENTIAL

4. 🎚️ TRADE-OFF
   - Exact search: 100% accurate, slow
   - Approximate search: 95-99% accurate, very fast
   - In practice, 95% is good enough!

Next: Let's build these indexes from scratch! 🚀
""")

