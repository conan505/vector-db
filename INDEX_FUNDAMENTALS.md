# Vector Database Indexes - Deep Dive 🚀

## Table of Contents
1. [What is an Index?](#what-is-an-index)
2. [Why Do We Need Indexes?](#why-do-we-need-indexes)
3. [The Fundamental Trade-off](#the-fundamental-trade-off)
4. [Types of Indexes](#types-of-indexes)
5. [Mathematical Foundations](#mathematical-foundations)

---

## What is an Index?

### Simple Analogy: Library vs Bookshelf

**Without Index (Brute Force):**
```
You want to find books about "Python"
→ Check EVERY book in the library (10,000 books)
→ Read each title one by one
→ Takes hours! ⏰
```

**With Index (Smart Search):**
```
You want to find books about "Python"
→ Look in the card catalog (organized by topic)
→ Go directly to "Programming > Python" section
→ Takes minutes! ⚡
```

### In Vector Databases:

**Without Index:**
```python
# Brute force - check ALL vectors
for vector in database:  # 1 million vectors
    distance = calculate_distance(query, vector)
# Time: O(n) where n = number of vectors
```

**With Index:**
```python
# Smart search - check only RELEVANT vectors
candidates = index.get_candidates(query)  # Only 100 vectors
for vector in candidates:
    distance = calculate_distance(query, vector)
# Time: O(log n) or O(√n) depending on index type
```

---

## Why Do We Need Indexes?

### The Scalability Problem

| Database Size | Brute Force Time | With Index Time | Speedup |
|---------------|------------------|-----------------|---------|
| 1,000 vectors | 1 ms | 0.5 ms | 2x |
| 10,000 vectors | 10 ms | 1 ms | 10x |
| 100,000 vectors | 100 ms | 2 ms | 50x |
| 1,000,000 vectors | 1,000 ms (1 sec) | 5 ms | **200x** |
| 10,000,000 vectors | 10,000 ms (10 sec) | 10 ms | **1000x** |

**Real-world example:**
- **Pinecone** (vector DB company): Handles billions of vectors
- **OpenAI embeddings**: 1536 dimensions
- **Without indexes**: Would take minutes per query ❌
- **With indexes**: Milliseconds per query ✅

---

## The Fundamental Trade-off

### Exact vs Approximate Search

```
┌─────────────────────────────────────────────┐
│                                             │
│  EXACT SEARCH          vs    APPROXIMATE   │
│                                             │
│  ✅ 100% accurate            ✅ Very fast   │
│  ❌ Slow for large data      ❌ ~95% accurate│
│                                             │
│  Use: Small datasets         Use: Large datasets│
│  Example: < 10,000 vectors   Example: > 100,000│
│                                             │
└─────────────────────────────────────────────┘
```

### The Accuracy-Speed Spectrum

```
Exact Search (Brute Force)
    ↓
    ├─ 100% recall, slowest
    │
Approximate Search (Indexes)
    ├─ IVF: 90-95% recall, 10x faster
    ├─ LSH: 85-90% recall, 50x faster
    └─ HNSW: 95-99% recall, 100x faster ⭐ (best!)
```

**Recall** = Percentage of true nearest neighbors found
- 100% recall = Found all correct neighbors (exact)
- 95% recall = Found 95 out of 100 correct neighbors (approximate)

---

## Types of Indexes

### Overview

| Index Type | Speed | Accuracy | Memory | Complexity | Best For |
|------------|-------|----------|--------|------------|----------|
| **Flat** | Slow | 100% | Low | Simple | < 10K vectors |
| **IVF** | Medium | 90-95% | Medium | Medium | 10K-1M vectors |
| **LSH** | Fast | 85-90% | Low | Medium | High dimensions |
| **HNSW** | Very Fast | 95-99% | High | Complex | Production systems |

---

## Mathematical Foundations

### 1. Distance Metrics (Review)

#### Euclidean Distance (L2)
```
d(p, q) = √(Σ(pᵢ - qᵢ)²)

Example:
p = [1, 2, 3]
q = [4, 5, 6]
d = √((1-4)² + (2-5)² + (3-6)²)
  = √(9 + 9 + 9)
  = √27 ≈ 5.196
```

**Properties:**
- Measures straight-line distance
- Sensitive to magnitude
- Triangle inequality: d(a,c) ≤ d(a,b) + d(b,c)

#### Cosine Similarity
```
similarity = (p · q) / (||p|| × ||q||)
distance = 1 - similarity

Example:
p = [1, 2, 3]
q = [2, 4, 6]  (same direction, different magnitude)
similarity = (1×2 + 2×4 + 3×6) / (√14 × √56)
           = 28 / 28 = 1.0 (identical direction!)
```

**Properties:**
- Measures angle, not distance
- Ignores magnitude
- Range: [-1, 1] for similarity, [0, 2] for distance

---

### 2. Curse of Dimensionality

**The Problem:** As dimensions increase, distances become meaningless!

```python
# Demonstration
import numpy as np

# Low dimensions (3D)
vectors_3d = np.random.rand(1000, 3)
query_3d = np.random.rand(1, 3)
distances_3d = np.linalg.norm(vectors_3d - query_3d, axis=1)

print(f"3D - Min: {distances_3d.min():.2f}, Max: {distances_3d.max():.2f}")
print(f"3D - Ratio: {distances_3d.max() / distances_3d.min():.2f}")
# Output: Min: 0.15, Max: 1.85, Ratio: 12.3

# High dimensions (1000D)
vectors_1000d = np.random.rand(1000, 1000)
query_1000d = np.random.rand(1, 1000)
distances_1000d = np.linalg.norm(vectors_1000d - query_1000d, axis=1)

print(f"1000D - Min: {distances_1000d.min():.2f}, Max: {distances_1000d.max():.2f}")
print(f"1000D - Ratio: {distances_1000d.max() / distances_1000d.min():.2f}")
# Output: Min: 12.5, Max: 14.2, Ratio: 1.14 (almost the same!)
```

**Key Insight:** In high dimensions, ALL points are roughly equidistant!
- Makes nearest neighbor search harder
- Indexes become more important

---

### 3. Space Partitioning

**Core Idea:** Divide space into regions, only search relevant regions

#### Example: 2D Space Partitioning

```
Without partitioning:          With partitioning:
┌─────────────────┐           ┌────────┬────────┐
│ • • • • • • • • │           │ • • • •│        │
│ • • • • • • • • │           │ • • • •│        │
│ • • • • • • • • │           ├────────┼────────┤
│ • • • • • • • • │           │        │ • • • •│
│ • • • • • • • • │           │        │ • • • •│
│ • • • • • • • • │           └────────┴────────┘
│ • • • • • • • • │           
│ • • • • • • • • │           Query in top-left?
└─────────────────┘           → Only search top-left region!
Query: Check all 64 points    Query: Check only 16 points
```

**Methods:**
1. **Grid-based** (simple, but doesn't scale)
2. **Tree-based** (KD-Tree, Ball Tree)
3. **Clustering-based** (IVF - what we'll build!)
4. **Hash-based** (LSH)
5. **Graph-based** (HNSW)

---

### 4. Clustering (K-Means)

**Used in IVF index** - Group similar vectors together

#### Algorithm:
```
1. Choose k cluster centers (randomly or smart initialization)
2. Assign each vector to nearest center
3. Recalculate centers as mean of assigned vectors
4. Repeat steps 2-3 until convergence
```

#### Visual Example (2D):
```
Initial (random):          After iteration 1:      Converged:
┌─────────────┐           ┌─────────────┐         ┌─────────────┐
│ •  •  •     │           │ •••         │         │ •••         │
│  • •  •     │           │ •••         │         │ •••         │
│   C₁        │           │  C₁         │         │  C₁         │
│             │           │             │         │             │
│        C₂   │           │        C₂   │         │       C₂    │
│      •  •   │           │       •••   │         │      •••    │
│     •  •  • │           │       •••   │         │      •••    │
└─────────────┘           └─────────────┘         └─────────────┘
```

#### Math:
```
Center update:
C_j = (1/|S_j|) × Σ(x_i) for all x_i in cluster j

Distance to center:
d(x, C_j) = ||x - C_j||
```

---

### 5. Hashing (LSH - Locality Sensitive Hashing)

**Core Idea:** Similar vectors hash to the same bucket

#### Random Projection:
```
High-dimensional space → Low-dimensional hash

Example:
Vector: [0.5, 0.3, 0.8, 0.2, ...]  (1000D)
         ↓ (random projection)
Hash:   [1, 0, 1, 1]  (4 bits)
```

#### How it works:
```python
# Random hyperplane
hyperplane = random_vector()

# Hash function
def hash_bit(vector, hyperplane):
    return 1 if dot(vector, hyperplane) > 0 else 0

# Multiple hyperplanes = multiple bits
hash_code = [hash_bit(v, h) for h in hyperplanes]
```

#### Visual (2D):
```
┌─────────────┐
│ •  •  │  •  │  Hyperplane divides space
│ •  •  │  •  │  Left side: hash = 0
│ •  •  │  •  │  Right side: hash = 1
├───────┼─────┤
│ •  •  │  •  │  Similar vectors likely
│ •  •  │  •  │  on same side!
└─────────────┘
```

---

### 6. Graphs (HNSW)

**Core Idea:** Navigate through a graph of connections

#### Small World Property:
```
"Six degrees of separation" - any two people connected by ~6 steps

In vectors:
- Each vector connected to nearest neighbors
- Can reach any vector in O(log n) hops
```

#### Hierarchical Layers:
```
Layer 2 (sparse):    •─────────•─────────•
                     │         │         │
Layer 1 (medium):    •───•───•─•───•───•─•───•
                     │ │ │ │ │ │ │ │ │ │ │ │
Layer 0 (dense):     •─•─•─•─•─•─•─•─•─•─•─•─•

Search: Start at top layer (fast jumps)
        → Move down layers (refine search)
        → Find exact neighbors at bottom
```

---

## Summary: Index Comparison

### Flat Index (Brute Force)
- **Algorithm:** Check every vector
- **Time:** O(n)
- **Space:** O(n)
- **Accuracy:** 100%
- **Use:** Baseline, small datasets

### IVF (Inverted File Index)
- **Algorithm:** Cluster vectors, search nearest clusters
- **Time:** O(√n) approximately
- **Space:** O(n + k) where k = clusters
- **Accuracy:** 90-95%
- **Use:** Medium datasets, good balance

### LSH (Locality Sensitive Hashing)
- **Algorithm:** Hash vectors, search same buckets
- **Time:** O(1) to O(log n)
- **Space:** O(n × h) where h = hash tables
- **Accuracy:** 85-90%
- **Use:** Very high dimensions, speed critical

### HNSW (Hierarchical Navigable Small World)
- **Algorithm:** Navigate graph with hierarchical layers
- **Time:** O(log n)
- **Space:** O(n × M) where M = connections per node
- **Accuracy:** 95-99%
- **Use:** Production systems, best overall performance

---

## Next Steps

Now that you understand the theory, we'll implement each index from scratch:

1. ✅ **Flat Index** - Understand the baseline
2. 🔨 **IVF Index** - Learn clustering-based search
3. 🔨 **LSH Index** - Learn hash-based search
4. 🔨 **HNSW Index** - Learn graph-based search (state-of-the-art!)
5. 📊 **Benchmark** - Compare all approaches

Ready to start building? Let's implement the Flat Index first as our baseline!

