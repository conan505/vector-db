# Core Concepts Explained 📚

Visual explanations of key concepts in vector databases.

---

## 1. Vector Norm (Length/Magnitude)

### What is it?
**Norm = The length of a vector** (distance from origin to the point)

### Formula:
```
||v|| = √(x² + y² + z² + ...)
```

### Visual Example (2D):
```
      y
      |
      |  * (3, 4)
      | /|
      |/ | 4
    3 |  |
      |__|_____ x
         3

Vector: [3, 4]
Norm: √(3² + 4²) = √25 = 5
```

### In Code:
```python
import numpy as np

vector = np.array([3, 4])
norm = np.linalg.norm(vector)
print(norm)  # Output: 5.0

# Manual calculation
norm_manual = np.sqrt(3**2 + 4**2)
print(norm_manual)  # Output: 5.0
```

### Multiple Vectors:
```python
vectors = np.array([
    [3, 4],      # norm = 5
    [1, 0],      # norm = 1
    [5, 12]      # norm = 13
])

# Calculate norm for each vector (axis=1 = across columns)
norms = np.linalg.norm(vectors, axis=1)
print(norms)  # Output: [5.0, 1.0, 13.0]
```

### Key Points:
- ✅ Norm measures "how far from origin"
- ✅ Always positive (it's a distance)
- ✅ Unit vector has norm = 1
- ✅ Used in normalization and distance calculations

---

## 2. Cosine Similarity

### What is it?
**Measures the angle between two vectors** (ignores magnitude)

### Formula:
```
cosine_similarity = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product
- ||A|| = norm of A
- ||B|| = norm of B
```

### Visual Example (2D):
```
      y
      |
      |  B [1, 1]
      | /
      |/_____ A [1, 0]
      |
      x

Angle between A and B = 45°
Cosine(45°) = 0.707

Even if B was [10, 10], cosine similarity would still be 0.707!
(Same direction, different length)
```

### Step-by-Step Calculation:

```python
import numpy as np

# Example vectors
A = np.array([1, 0, 0])  # Along x-axis
B = np.array([1, 1, 0])  # Diagonal in xy-plane

# Step 1: Calculate norms
norm_A = np.linalg.norm(A)  # = 1.0
norm_B = np.linalg.norm(B)  # = 1.414

# Step 2: Normalize (make length = 1)
A_normalized = A / norm_A  # = [1, 0, 0]
B_normalized = B / norm_B  # = [0.707, 0.707, 0]

# Step 3: Dot product of normalized vectors
cosine_sim = np.dot(A_normalized, B_normalized)
print(cosine_sim)  # = 0.707 (45° angle)
```

### Code from vectordb.py:
```python
def cosine_similarity(vectors, query):
    # Step 1 & 2: Normalize vectors
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    query_norm = query / np.linalg.norm(query)
    
    # Step 3: Dot product
    similarities = np.dot(vectors_norm, query_norm.T).flatten()
    
    # Convert to distance (smaller = more similar)
    return 1 - similarities
```

### Interpretation:
```
Cosine Similarity = 1.0  → Same direction (0° angle)
Cosine Similarity = 0.0  → Perpendicular (90° angle)
Cosine Similarity = -1.0 → Opposite directions (180° angle)
```

### Why Use It?
- ✅ Ignores magnitude (good for text: document length doesn't matter)
- ✅ Focuses on direction (semantic meaning)
- ✅ Range is [-1, 1] (easy to interpret)

### Example:
```python
vectors = np.array([
    [1, 0, 0],      # Along x-axis
    [0, 1, 0],      # Along y-axis
    [1, 1, 0],      # 45° from x-axis
    [10, 10, 0]     # Also 45° from x-axis (but longer!)
])

query = np.array([[1, 0, 0]])  # Along x-axis

# Cosine similarities:
# Vector 0: 1.0   (same direction)
# Vector 1: 0.0   (perpendicular)
# Vector 2: 0.707 (45° angle)
# Vector 3: 0.707 (same angle as Vector 2, despite being longer!)
```

---

## 3. Transpose (.T) and Flatten

### What is Transpose?
**Flip rows and columns**

### Visual:
```
Original (1×3):          Transposed (3×1):
┌──────────┐             ┌────┐
│ 10 20 30 │      .T     │ 10 │
└──────────┘      →      │ 20 │
                         │ 30 │
                         └────┘

1 row, 3 columns         3 rows, 1 column
```

### In Code:
```python
query = np.array([[10, 20, 30]])
print(query.shape)    # (1, 3)
print(query)          # [[10 20 30]]

query_T = query.T
print(query_T.shape)  # (3, 1)
print(query_T)        # [[10]
                      #  [20]
                      #  [30]]
```

### Why Do We Need .T for Dot Product?

**Matrix multiplication rule:**
```
(m × n) · (n × p) = (m × p)
     ↑       ↑
These must match!
```

**Example:**
```python
vectors = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])  # Shape: (3, 3)

query = np.array([[10, 20, 30]])  # Shape: (1, 3)

# This FAILS:
# np.dot(vectors, query)
# (3×3) · (1×3) = ERROR! (3 ≠ 1)

# This WORKS:
result = np.dot(vectors, query.T)
# (3×3) · (3×1) = (3×1) ✓
```

### Visual Matrix Multiplication:
```
vectors (3×3)        query.T (3×1)      result (3×1)
┌─────────┐          ┌────┐             ┌─────┐
│ 1  2  3 │          │ 10 │             │ 140 │
│ 4  5  6 │    ×     │ 20 │      =      │ 320 │
│ 7  8  9 │          │ 30 │             │ 500 │
└─────────┘          └────┘             └─────┘

Calculation for first row:
1×10 + 2×20 + 3×30 = 10 + 40 + 90 = 140
```

---

## 4. What is flatten()?

### Purpose:
**Convert any shape to 1D array**

### Visual:
```
Before flatten (3×1):    After flatten (3,):
┌─────┐                  [140, 320, 500]
│ 140 │
│ 320 │         →        Simple 1D array
│ 500 │
└─────┘
2D array with 1 column
```

### In Code:
```python
# 2D array (3 rows, 1 column)
result_2d = np.array([[140], [320], [500]])
print(result_2d.shape)  # (3, 1)
print(result_2d)        # [[140]
                        #  [320]
                        #  [500]]

# Flatten to 1D
result_1d = result_2d.flatten()
print(result_1d.shape)  # (3,)
print(result_1d)        # [140 320 500]
```

### Why Do We Need It?

**Without flatten:**
```python
result = np.dot(vectors, query.T)
# Shape: (3, 1)
# Value: [[140], [320], [500]]

# Awkward to use:
print(result[0])     # [140] (still an array!)
print(result[0][0])  # 140 (need double indexing)
```

**With flatten:**
```python
result = np.dot(vectors, query.T).flatten()
# Shape: (3,)
# Value: [140, 320, 500]

# Easy to use:
print(result[0])  # 140 (direct access)
np.argsort(result)  # Works directly
```

---

## 5. Complete Example: Putting It All Together

### The Line from vectordb.py:
```python
similarities = np.dot(vectors_norm, query_norm.T).flatten()
```

### Breaking It Down:

```python
import numpy as np

# Setup
vectors = np.array([
    [1, 2, 3],
    [4, 5, 6]
])  # Shape: (2, 3) - 2 vectors, 3 dimensions

query = np.array([[7, 8, 9]])  # Shape: (1, 3)

# Step 1: Normalize (for cosine similarity)
vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
query_norm = query / np.linalg.norm(query)

print("vectors_norm shape:", vectors_norm.shape)  # (2, 3)
print("query_norm shape:", query_norm.shape)      # (1, 3)

# Step 2: Transpose query
print("query_norm.T shape:", query_norm.T.shape)  # (3, 1)

# Step 3: Dot product
result = np.dot(vectors_norm, query_norm.T)
print("result shape:", result.shape)              # (2, 1)
print("result:", result)                          # [[0.974...], [0.998...]]

# Step 4: Flatten
similarities = result.flatten()
print("similarities shape:", similarities.shape)  # (2,)
print("similarities:", similarities)              # [0.974..., 0.998...]
```

### Visual Flow:
```
vectors_norm (2×3)   query_norm.T (3×1)   result (2×1)    flatten    similarities (2,)
┌─────────┐          ┌────┐               ┌──────┐                   ┌──────────┐
│ 0.27... │          │0.50│               │0.974 │         →         │ 0.974... │
│ 0.53... │    ×     │0.57│      =        │0.998 │                   │ 0.998... │
│ 0.80... │          │0.64│               └──────┘                   └──────────┘
└─────────┘          └────┘
```

---

## 6. Quick Reference

### Norm (Length):
```python
# Single vector
norm = np.linalg.norm(vector)

# Multiple vectors (each row is a vector)
norms = np.linalg.norm(vectors, axis=1)
```

### Cosine Similarity:
```python
# Normalize
v_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
q_norm = query / np.linalg.norm(query)

# Dot product
similarities = np.dot(v_norm, q_norm.T).flatten()
```

### Transpose:
```python
# Flip rows and columns
transposed = array.T
```

### Flatten:
```python
# Convert to 1D
flat = array.flatten()
```

---

## 7. Common Patterns

### Pattern 1: Distance Calculation
```python
# Euclidean distance
distances = np.linalg.norm(vectors - query, axis=1)
```

### Pattern 2: Cosine Similarity
```python
# Normalize + dot product
v_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
q_norm = query / np.linalg.norm(query)
similarities = np.dot(v_norm, q_norm.T).flatten()
```

### Pattern 3: Top-k Results
```python
# Get indices of k smallest values
top_k_indices = np.argsort(distances)[:k]
```

---

## 8. Visualization Tips

### Think of Vectors as Arrows:
```
3D Space:
        z
        |
        |    * Vector [2, 3, 6]
        |   /
        |  /
        | /
        |/_______ y
       /
      /
     x

- Origin: [0, 0, 0]
- Vector: Arrow from origin to point
- Norm: Length of arrow
- Cosine: Angle between arrows
```

### Normalized Vectors (Unit Vectors):
```
All have length = 1, but point in different directions

        y
        |
        | * [0, 1]
        |/
        |_____ * [1, 0]
       /
      /
     * [0, 0, 1]
    z
```

---

## Summary

| Concept | What It Does | Why We Use It |
|---------|--------------|---------------|
| **Norm** | Measures vector length | Distance calculations, normalization |
| **Cosine Similarity** | Measures angle between vectors | Text similarity (ignores length) |
| **Transpose (.T)** | Flips rows/columns | Matrix multiplication compatibility |
| **Flatten** | Converts to 1D array | Easier to work with results |

**Key Insight:** These operations work together to efficiently calculate similarity between many vectors at once using NumPy's optimized matrix operations!

