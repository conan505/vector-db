# IVF Index (Inverted File Index) - Theory

## What is IVF?

**IVF = Inverted File Index**

The core idea: **Divide and conquer using clustering**

Instead of searching through ALL vectors, we:
1. **Group similar vectors into clusters** (using k-means)
2. **At search time, only check nearby clusters**
3. **Result: 10-20x speedup!**

---

## Visual Explanation

### Without IVF (Brute Force):
```
Database with 1000 vectors:
┌─────────────────────────────┐
│ • • • • • • • • • • • • • • │
│ • • • • • • • • • • • • • • │
│ • • • • • • • • • • • • • • │
│ • • • • • • • • • • • • • • │
│ • • • • • • • • • • • • • • │
└─────────────────────────────┘

Query: ⭐
→ Check ALL 1000 vectors
→ Time: 10ms
```

### With IVF (10 clusters):
```
Database clustered into 10 groups:
┌──────┬──────┬──────┬──────┬──────┐
│ •••  │      │      │  ••• │      │
│ •••  │      │      │  ••• │      │
├──────┼──────┼──────┼──────┼──────┤
│      │  ••• │      │      │  ••• │
│      │  ••• │      │      │  ••• │
└──────┴──────┴──────┴──────┴──────┘
  C₁     C₂     C₃     C₄     C₅

Query: ⭐ (near C₁)
→ Find nearest cluster: C₁
→ Only check 100 vectors in C₁
→ Time: 1ms (10x faster!)
```

---

## The Algorithm

### Phase 1: Building the Index (One-time)

```
1. Choose number of clusters (k)
   Example: k = 100 for 100,000 vectors

2. Run k-means clustering
   - Initialize k random centroids
   - Assign each vector to nearest centroid
   - Update centroids as mean of assigned vectors
   - Repeat until convergence

3. Store the result
   - Centroids: [C₁, C₂, ..., Cₖ]
   - Inverted lists: {
       C₁: [v₁, v₅, v₇, ...],
       C₂: [v₂, v₃, v₉, ...],
       ...
     }
```

### Phase 2: Searching (Fast!)

```
1. Find nearest centroids to query
   Example: Find 3 nearest centroids
   
2. Search only those clusters
   - Get all vectors from those 3 clusters
   - Calculate distances to query
   - Return top-k results

3. Trade-off parameter: n_probe
   - n_probe = 1: Search 1 cluster (fastest, less accurate)
   - n_probe = 5: Search 5 clusters (slower, more accurate)
   - n_probe = k: Search all clusters (same as brute force)
```

---

## K-Means Clustering - Deep Dive

### The Algorithm

```python
# Pseudocode
def kmeans(vectors, k, max_iterations=100):
    # 1. Initialize centroids randomly
    centroids = random_sample(vectors, k)
    
    for iteration in range(max_iterations):
        # 2. Assign each vector to nearest centroid
        assignments = []
        for vector in vectors:
            nearest = argmin([distance(vector, c) for c in centroids])
            assignments.append(nearest)
        
        # 3. Update centroids as mean of assigned vectors
        new_centroids = []
        for cluster_id in range(k):
            cluster_vectors = vectors[assignments == cluster_id]
            new_centroids.append(mean(cluster_vectors))
        
        # 4. Check convergence
        if centroids == new_centroids:
            break
        centroids = new_centroids
    
    return centroids, assignments
```

### Visual Example (2D)

**Iteration 0 (Random initialization):**
```
┌─────────────┐
│ •  •  •     │
│  • •  •     │  C₁ = random
│   C₁        │  C₂ = random
│             │
│        C₂   │
│      •  •   │
│     •  •  • │
└─────────────┘
```

**Iteration 1 (Assign to nearest):**
```
┌─────────────┐
│ 1  1  1     │  Assign each point
│  1 1  1     │  to nearest centroid
│   C₁        │
│             │
│        C₂   │
│      2  2   │
│     2  2  2 │
└─────────────┘
```

**Iteration 2 (Update centroids):**
```
┌─────────────┐
│ 1  1  1     │  Move centroids to
│  1 1  1     │  center of their points
│    C₁'      │  C₁' = mean of all 1's
│             │  C₂' = mean of all 2's
│       C₂'   │
│      2  2   │
│     2  2  2 │
└─────────────┘
```

**Iteration 3 (Converged!):**
```
┌─────────────┐
│ 1  1  1     │  Centroids don't move
│  1 1  1     │  → Converged!
│   C₁        │
│             │
│      C₂     │
│      2  2   │
│     2  2  2 │
└─────────────┘
```

---

## Mathematical Details

### 1. Distance to Centroid

```
For vector v and centroid c:
d(v, c) = ||v - c||

Example:
v = [1, 2, 3]
c = [4, 5, 6]
d = √((1-4)² + (2-5)² + (3-6)²)
  = √(9 + 9 + 9)
  = √27 ≈ 5.196
```

### 2. Centroid Update

```
New centroid = mean of all assigned vectors

C_new = (1/n) × Σ(v_i) for all v_i in cluster

Example:
Cluster has 3 vectors:
v₁ = [1, 2, 3]
v₂ = [2, 3, 4]
v₃ = [3, 4, 5]

C = (v₁ + v₂ + v₃) / 3
  = ([1,2,3] + [2,3,4] + [3,4,5]) / 3
  = [6, 9, 12] / 3
  = [2, 3, 4]
```

### 3. Convergence Criterion

```
Stop when centroids don't change:
||C_new - C_old|| < threshold

Or after max iterations (e.g., 100)
```

---

## Time Complexity Analysis

### Building the Index

```
K-means:
- Iterations: I (typically 10-100)
- Per iteration:
  - Assign: O(n × k × d) where n=vectors, k=clusters, d=dimensions
  - Update: O(n × d)
- Total: O(I × n × k × d)

Example:
n = 100,000 vectors
k = 100 clusters
d = 128 dimensions
I = 20 iterations
→ ~2.5 billion operations (takes a few seconds)
```

### Searching

```
Without IVF:
- Check all vectors: O(n × d)
- Example: 100,000 × 128 = 12.8M operations

With IVF (n_probe clusters):
- Find nearest centroids: O(k × d)
- Search n_probe clusters: O((n/k) × n_probe × d)
- Total: O(k × d + (n/k) × n_probe × d)

Example (n_probe = 5):
- Find centroids: 100 × 128 = 12.8K operations
- Search clusters: (100,000/100) × 5 × 128 = 640K operations
- Total: ~650K operations (20x faster!)
```

---

## Space Complexity

```
Storage needed:
1. Centroids: k × d floats
2. Vectors: n × d floats (same as before)
3. Cluster assignments: n integers
4. Inverted lists: n pointers

Example:
n = 100,000 vectors
k = 100 clusters
d = 128 dimensions

Centroids: 100 × 128 × 4 bytes = 51 KB
Vectors: 100,000 × 128 × 4 bytes = 51 MB
Assignments: 100,000 × 4 bytes = 400 KB
Total: ~52 MB (minimal overhead!)
```

---

## Trade-offs

### Choosing k (number of clusters)

```
Too few clusters (k = 10):
✅ Fast to build
✅ Fast centroid search
❌ Large clusters → slow search
❌ Less accurate

Too many clusters (k = 10,000):
❌ Slow to build
❌ Slow centroid search
✅ Small clusters → fast search
✅ More accurate

Sweet spot: k = √n
- 1,000 vectors → k = 32
- 10,000 vectors → k = 100
- 100,000 vectors → k = 316
- 1,000,000 vectors → k = 1,000
```

### Choosing n_probe (clusters to search)

```
n_probe = 1:
✅ Fastest (only 1 cluster)
❌ Lowest accuracy (~70%)

n_probe = 5:
✅ Good balance
✅ Good accuracy (~90%)
⚖️ Medium speed

n_probe = 20:
⚖️ Slower
✅ High accuracy (~98%)

n_probe = k:
❌ Same as brute force
✅ 100% accuracy
```

---

## Accuracy vs Speed

```
┌─────────────────────────────────────┐
│                                     │
│  100% ┤                          •  │ Brute Force
│       │                       •     │
│       │                    •        │
│  95%  ┤                 •           │ IVF (n_probe=10)
│       │              •              │
│       │           •                 │
│  90%  ┤        •                    │ IVF (n_probe=5)
│       │     •                       │
│       │  •                          │
│  85%  ┤•                            │ IVF (n_probe=1)
│       │                             │
│       └─────────────────────────────┤
│         1x   5x   10x  20x  50x     │
│              Speedup                │
└─────────────────────────────────────┘
```

---

## Real-World Example

### Scenario: Document Search

```
Database: 1 million documents (1536D embeddings)
Query: "How to use Python for AI?"

Without IVF:
- Check 1,000,000 vectors
- Time: ~1 second per query
- Throughput: 1 QPS (query per second)

With IVF (k=1000, n_probe=5):
- Check 1,000 centroids: 0.1ms
- Search 5 clusters: ~5,000 vectors
- Time: ~5ms per query
- Throughput: 200 QPS
- Speedup: 200x faster!
- Accuracy: ~95% (good enough!)
```

---

## Summary

### IVF Index

**Algorithm:**
1. Cluster vectors using k-means
2. At search time, find nearest clusters
3. Only search those clusters

**Complexity:**
- Build: O(I × n × k × d)
- Search: O(k × d + (n/k) × n_probe × d)
- Space: O(n × d + k × d)

**Parameters:**
- `k`: Number of clusters (typically √n)
- `n_probe`: Clusters to search (1-20)

**Performance:**
- Speed: 10-50x faster than brute force
- Accuracy: 85-95% depending on n_probe
- Memory: Minimal overhead

**Best for:**
- Medium to large datasets (10K - 10M vectors)
- When 90-95% accuracy is acceptable
- Production systems with good balance

---

## Next: Implementation!

Now that you understand the theory, let's implement IVF from scratch! 🚀

