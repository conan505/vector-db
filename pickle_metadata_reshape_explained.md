# Pickle, Metadata, and Reshape Explained

## 1. What is Pickle? 🥒

**Pickle = Python's way to save objects to files**

Think of it like "freezing" a Python object and saving it to disk, then "thawing" it later.

### Example:

```python
import pickle
import numpy as np

# Create a complex object
my_database = {
    'vectors': np.array([[1, 2, 3], [4, 5, 6]]),
    'metadata': [{'name': 'doc1'}, {'name': 'doc2'}],
    'count': 2
}

# SAVE (pickle it)
with open('database.pkl', 'wb') as f:
    pickle.dump(my_database, f)
# ✅ Saved to disk!

# LOAD (unpickle it)
with open('database.pkl', 'rb') as f:
    loaded_database = pickle.load(f)
# ✅ Restored exactly as it was!
```

### In vectordb.py:

```python
def save(self, filepath):
    """Save the database to disk"""
    with open(filepath, 'wb') as f:
        pickle.dump({
            'vectors': self.vectors,
            'metadata': self.metadata,
            'ids': self.ids,
            'dimensions': self.dimensions,
            'metric': self.metric
        }, f)
```

### Why Use Pickle?

| Without Pickle | With Pickle |
|----------------|-------------|
| Rebuild database every time | Save once, load instantly |
| Re-insert all vectors | Everything preserved |
| Lose all data on exit | Persistent storage |

**Use Cases:**
- ✅ Save your VectorDB to disk
- ✅ Load it later without rebuilding
- ✅ Share databases between programs
- ✅ Backup your data

---

## 2. What is Metadata? 📝

**Metadata = Extra information about each vector**

Vectors are just numbers, but metadata gives them **meaning**!

### Example: Document Search

```python
# Just vectors (meaningless numbers)
vectors = np.array([
    [0.1, 0.2, 0.3, 0.4, ...],  # What is this?
    [0.5, 0.6, 0.7, 0.8, ...],  # What is this?
])

# With metadata (now it makes sense!)
metadata = [
    {
        'text': 'Python is great for AI',
        'author': 'Alice',
        'date': '2024-01-01',
        'category': 'programming'
    },
    {
        'text': 'I love machine learning',
        'author': 'Bob',
        'date': '2024-01-02',
        'category': 'AI'
    }
]
```

### When You Search:

```python
# Search for similar documents
results = db.search(query_vector, k=3)

# Without metadata:
# Result: [[0.1, 0.2, 0.3, ...]]  ← Just numbers!

# With metadata:
# Result: {
#   'text': 'Python is great for AI',
#   'author': 'Alice',
#   'similarity': 0.95
# }  ← Useful information!
```

### Real-World Examples:

| Use Case | Vector | Metadata |
|----------|--------|----------|
| **Document Search** | Text embedding | `{text, author, date, url}` |
| **Image Search** | Image embedding | `{filename, tags, size, date}` |
| **Product Recommendations** | Product features | `{name, price, category, rating}` |
| **Music Recommendations** | Audio features | `{title, artist, genre, duration}` |

### In vectordb.py:

```python
def insert(self, vector, metadata=None):
    """Insert a vector with optional metadata"""
    # Store the vector
    self.vectors = np.vstack([self.vectors, vector])
    
    # Store the metadata
    self.metadata.append(metadata)  # ← Keeps extra info!
    
    # Store the ID
    self.ids.append(self.next_id)
```

### Why Use Metadata?

**Without metadata:**
```python
results = db.search(query, k=3)
# Returns: [array([0.1, 0.2, 0.3]), ...]
# You: "What does this mean??" 🤷
```

**With metadata:**
```python
results = db.search(query, k=3)
# Returns: [
#   {'text': 'Python tutorial', 'author': 'Alice', 'distance': 0.12},
#   {'text': 'AI basics', 'author': 'Bob', 'distance': 0.15},
# ]
# You: "Perfect! I found what I need!" ✅
```

---

## 3. What is Reshape? `vector.reshape(1, -1)`

**Reshape = Change the dimensions of an array**

### The Problem:

```python
# Multiple vectors (2D array)
vectors = np.array([
    [1, 2, 3],  # Vector 0
    [4, 5, 6],  # Vector 1
    [7, 8, 9]   # Vector 2
])
print(vectors.shape)  # (3, 3) = 3 rows, 3 columns

# Single vector (1D array)
new_vector = np.array([10, 20, 30])
print(new_vector.shape)  # (3,) = just 3 numbers

# Try to add them:
np.vstack([vectors, new_vector])  # ❌ ERROR! Dimension mismatch!
```

### The Solution:

```python
# Reshape 1D to 2D
new_vector_2d = new_vector.reshape(1, -1)
print(new_vector_2d.shape)  # (1, 3) = 1 row, 3 columns

# Now it works!
np.vstack([vectors, new_vector_2d])  # ✅ Success!
```

### Visual Explanation:

```
1D Array (shape: (3,))
[10, 20, 30]  ← Just a list of numbers

After reshape(1, -1) → 2D Array (shape: (1, 3))
[[10, 20, 30]]  ← Now it's a row in a table!

┌──────────┐
│ 10 20 30 │  ← 1 row, 3 columns
└──────────┘
```

### What Does `reshape(1, -1)` Mean?

```python
vector.reshape(1, -1)
#              ↑  ↑
#              |  └─ -1 = "figure this out automatically"
#              └──── 1 = "I want 1 row"
```

**Examples:**

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# reshape(1, -1) = 1 row, auto columns
arr.reshape(1, -1)
# [[1, 2, 3, 4, 5, 6]]  → shape: (1, 6)

# reshape(2, -1) = 2 rows, auto columns
arr.reshape(2, -1)
# [[1, 2, 3],
#  [4, 5, 6]]  → shape: (2, 3)

# reshape(-1, 1) = auto rows, 1 column
arr.reshape(-1, 1)
# [[1],
#  [2],
#  [3],
#  [4],
#  [5],
#  [6]]  → shape: (6, 1)
```

### Why Do We Need This in vectordb.py?

```python
def insert(self, vector, metadata=None):
    vector = np.array(vector)
    
    # User might pass 1D: [1, 2, 3]
    if vector.ndim == 1:
        # Convert to 2D: [[1, 2, 3]]
        vector = vector.reshape(1, -1)
    
    # Now we can stack it with other vectors
    if self.vectors.size == 0:
        self.vectors = vector
    else:
        self.vectors = np.vstack([self.vectors, vector])
```

**Why?**
- Our storage format uses **rows** for vectors
- Each vector must be a **row** (2D with 1 row)
- `np.vstack()` requires consistent dimensions
- `reshape(1, -1)` ensures consistency

### Visual: Adding a Vector

```
Current database:
┌─────────┐
│ 1  2  3 │  ← Vector 0
│ 4  5  6 │  ← Vector 1
└─────────┘
Shape: (2, 3)

New vector (1D): [7, 8, 9]
Shape: (3,)
❌ Can't add directly!

After reshape(1, -1): [[7, 8, 9]]
Shape: (1, 3)
✅ Now it's a row!

Result after vstack:
┌─────────┐
│ 1  2  3 │
│ 4  5  6 │
│ 7  8  9 │  ← New vector added!
└─────────┘
Shape: (3, 3)
```

---

## Summary

### 1. Pickle 🥒
```python
# Save
with open('db.pkl', 'wb') as f:
    pickle.dump(database, f)

# Load
with open('db.pkl', 'rb') as f:
    database = pickle.load(f)
```
**Purpose:** Save/load Python objects to/from disk

---

### 2. Metadata 📝
```python
metadata = {
    'text': 'Python is great',
    'author': 'Alice',
    'date': '2024-01-01'
}
```
**Purpose:** Store extra information about each vector

---

### 3. Reshape 🔄
```python
vector_1d = np.array([1, 2, 3])        # Shape: (3,)
vector_2d = vector_1d.reshape(1, -1)   # Shape: (1, 3)
```
**Purpose:** Convert 1D array to 2D (1 row) for consistent storage

---

## Quick Reference

| Concept | What It Does | Why We Need It |
|---------|--------------|----------------|
| **Pickle** | Saves objects to files | Persistent storage |
| **Metadata** | Stores extra info | Gives meaning to vectors |
| **Reshape(1, -1)** | Converts 1D to 2D | Consistent dimensions |

---

## Real Example from vectordb.py

```python
class VectorDB:
    def __init__(self):
        self.vectors = np.array([])    # Store vectors
        self.metadata = []              # Store metadata
        self.ids = []                   # Store IDs
    
    def insert(self, vector, metadata=None):
        # Reshape if needed
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        
        # Add vector
        self.vectors = np.vstack([self.vectors, vector])
        
        # Add metadata
        self.metadata.append(metadata)
        
        # Add ID
        self.ids.append(self.next_id)
    
    def save(self, filepath):
        # Pickle everything
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectors': self.vectors,
                'metadata': self.metadata,
                'ids': self.ids
            }, f)
    
    def search(self, query, k=5):
        # Calculate distances
        distances = self._calculate_distance(query)
        
        # Get top-k indices
        top_k = np.argsort(distances)[:k]
        
        # Return results with metadata
        return [{
            'id': self.ids[i],
            'distance': distances[i],
            'metadata': self.metadata[i]  # ← Metadata makes it useful!
        } for i in top_k]
```

---

## Try It Yourself!

Run this code to see all three concepts in action:

```python
import numpy as np
import pickle

# Create vectors and metadata
vectors = np.array([[1, 2, 3], [4, 5, 6]])
metadata = [
    {'text': 'First document'},
    {'text': 'Second document'}
]

# Save with pickle
with open('test.pkl', 'wb') as f:
    pickle.dump({'vectors': vectors, 'metadata': metadata}, f)

# Load with pickle
with open('test.pkl', 'rb') as f:
    loaded = pickle.load(f)
    print("Loaded:", loaded)

# Reshape example
new_vector = np.array([7, 8, 9])
print("1D shape:", new_vector.shape)

new_vector_2d = new_vector.reshape(1, -1)
print("2D shape:", new_vector_2d.shape)

# Add to vectors
all_vectors = np.vstack([vectors, new_vector_2d])
print("Combined shape:", all_vectors.shape)
```

