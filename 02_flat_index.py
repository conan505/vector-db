"""
Flat Index (Brute Force) - The Baseline

This is the simplest index: no optimization, just check every vector.
We'll use this as a baseline to compare other indexes against.

Time Complexity: O(n) where n = number of vectors
Space Complexity: O(n * d) where d = dimensions
Accuracy: 100% (exact search)
"""

import numpy as np
import time
from typing import List, Tuple, Optional


class FlatIndex:
    """
    Flat Index - Brute force search through all vectors.
    
    This is the baseline index that provides exact nearest neighbor search
    by comparing the query against every vector in the database.
    
    Pros:
    - Simple to implement
    - 100% accurate (exact search)
    - No build time
    - Low memory overhead
    
    Cons:
    - Slow for large datasets (O(n) search time)
    - Doesn't scale beyond ~100K vectors
    """
    
    def __init__(self, dimensions: int, metric: str = 'euclidean'):
        """
        Initialize the Flat Index.
        
        Args:
            dimensions: Dimensionality of vectors
            metric: Distance metric ('euclidean' or 'cosine')
        """
        self.dimensions = dimensions
        self.metric = metric
        self.vectors = np.array([]).reshape(0, dimensions)
        self.ids = []
        self.next_id = 0
        
    def add(self, vectors: np.ndarray, ids: Optional[List[int]] = None):
        """
        Add vectors to the index.
        
        Args:
            vectors: Array of shape (n, dimensions)
            ids: Optional list of IDs for the vectors
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
            
        if vectors.shape[1] != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {vectors.shape[1]}")
        
        # Add vectors
        self.vectors = np.vstack([self.vectors, vectors]) if self.vectors.size > 0 else vectors
        
        # Add IDs
        if ids is None:
            ids = list(range(self.next_id, self.next_id + len(vectors)))
        self.ids.extend(ids)
        self.next_id = max(self.ids) + 1
        
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector of shape (dimensions,) or (1, dimensions)
            k: Number of nearest neighbors to return
            
        Returns:
            distances: Array of distances to k nearest neighbors
            indices: Array of indices of k nearest neighbors
        """
        if query.ndim == 1:
            query = query.reshape(1, -1)
            
        if self.vectors.size == 0:
            return np.array([]), np.array([])
        
        # Calculate distances to all vectors (BRUTE FORCE)
        if self.metric == 'euclidean':
            distances = self._euclidean_distance(query)
        elif self.metric == 'cosine':
            distances = self._cosine_distance(query)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Get top-k smallest distances
        k = min(k, len(distances))
        top_k_indices = np.argsort(distances)[:k]
        top_k_distances = distances[top_k_indices]
        
        return top_k_distances, top_k_indices
    
    def _euclidean_distance(self, query: np.ndarray) -> np.ndarray:
        """Calculate Euclidean distance to all vectors."""
        # ||a - b|| = sqrt(sum((a_i - b_i)^2))
        return np.linalg.norm(self.vectors - query, axis=1)
    
    def _cosine_distance(self, query: np.ndarray) -> np.ndarray:
        """Calculate cosine distance to all vectors."""
        # Normalize vectors
        vectors_norm = self.vectors / np.linalg.norm(self.vectors, axis=1, keepdims=True)
        query_norm = query / np.linalg.norm(query)
        
        # Cosine similarity
        similarities = np.dot(vectors_norm, query_norm.T).flatten()
        
        # Convert to distance (1 - similarity)
        return 1 - similarities
    
    def __len__(self):
        """Return number of vectors in the index."""
        return len(self.vectors)
    
    def __repr__(self):
        return f"FlatIndex(vectors={len(self)}, dimensions={self.dimensions}, metric={self.metric})"


def demo_flat_index():
    """Demonstrate the Flat Index."""
    print("=" * 70)
    print("FLAT INDEX DEMONSTRATION")
    print("=" * 70)
    
    # Create index
    print("\n1. Creating Flat Index")
    print("-" * 70)
    dimensions = 128
    index = FlatIndex(dimensions=dimensions, metric='euclidean')
    print(f"Created: {index}")
    
    # Add vectors
    print("\n2. Adding Vectors")
    print("-" * 70)
    n_vectors = 10000
    vectors = np.random.rand(n_vectors, dimensions)
    
    start = time.time()
    index.add(vectors)
    add_time = time.time() - start
    
    print(f"Added {n_vectors:,} vectors in {add_time:.3f} seconds")
    print(f"Index now contains: {len(index):,} vectors")
    
    # Search
    print("\n3. Searching")
    print("-" * 70)
    query = np.random.rand(dimensions)
    k = 10
    
    start = time.time()
    distances, indices = index.search(query, k=k)
    search_time = (time.time() - start) * 1000  # Convert to ms
    
    print(f"Query: {query[:5]}... (showing first 5 dimensions)")
    print(f"Found {k} nearest neighbors in {search_time:.2f} ms")
    print(f"\nTop-{k} Results:")
    for i, (dist, idx) in enumerate(zip(distances, indices), 1):
        print(f"  {i}. Index {idx}: distance = {dist:.4f}")
    
    # Benchmark different sizes
    print("\n4. Performance Benchmark")
    print("-" * 70)
    print(f"{'Vectors':>10} {'Search Time':>15} {'Vectors/ms':>15}")
    print("-" * 70)
    
    for n in [100, 1000, 10000, 50000, 100000]:
        # Create index with n vectors
        test_index = FlatIndex(dimensions=dimensions)
        test_vectors = np.random.rand(n, dimensions)
        test_index.add(test_vectors)
        
        # Time search
        query = np.random.rand(dimensions)
        start = time.time()
        test_index.search(query, k=10)
        elapsed = (time.time() - start) * 1000
        
        throughput = n / elapsed if elapsed > 0 else 0
        print(f"{n:>10,} {elapsed:>14.2f} ms {throughput:>14.0f}")
    
    print("\n💡 Observations:")
    print("  - Search time grows linearly with data size (O(n))")
    print("  - Simple and accurate (100% recall)")
    print("  - Good for small datasets (< 10K vectors)")
    print("  - Too slow for large datasets (> 100K vectors)")


def compare_metrics():
    """Compare Euclidean vs Cosine distance."""
    print("\n" + "=" * 70)
    print("COMPARING DISTANCE METRICS")
    print("=" * 70)
    
    dimensions = 128
    n_vectors = 1000
    
    # Create test data
    vectors = np.random.rand(n_vectors, dimensions)
    query = np.random.rand(dimensions)
    
    # Euclidean
    print("\n1. Euclidean Distance")
    print("-" * 70)
    index_euclidean = FlatIndex(dimensions=dimensions, metric='euclidean')
    index_euclidean.add(vectors)
    
    start = time.time()
    distances_euc, indices_euc = index_euclidean.search(query, k=5)
    time_euc = (time.time() - start) * 1000
    
    print(f"Search time: {time_euc:.2f} ms")
    print("Top-5 results:")
    for i, (dist, idx) in enumerate(zip(distances_euc, indices_euc), 1):
        print(f"  {i}. Index {idx}: distance = {dist:.4f}")
    
    # Cosine
    print("\n2. Cosine Distance")
    print("-" * 70)
    index_cosine = FlatIndex(dimensions=dimensions, metric='cosine')
    index_cosine.add(vectors)
    
    start = time.time()
    distances_cos, indices_cos = index_cosine.search(query, k=5)
    time_cos = (time.time() - start) * 1000
    
    print(f"Search time: {time_cos:.2f} ms")
    print("Top-5 results:")
    for i, (dist, idx) in enumerate(zip(distances_cos, indices_cos), 1):
        print(f"  {i}. Index {idx}: distance = {dist:.4f}")
    
    print("\n💡 Observations:")
    print("  - Euclidean: Measures absolute distance")
    print("  - Cosine: Measures angle (ignores magnitude)")
    print("  - Different metrics may return different results!")
    print("  - Choose based on your use case:")
    print("    • Euclidean: Spatial data, images")
    print("    • Cosine: Text embeddings, normalized data")


if __name__ == "__main__":
    demo_flat_index()
    compare_metrics()
    
    print("\n" + "=" * 70)
    print("SUMMARY: Flat Index")
    print("=" * 70)
    print("""
✅ PROS:
  - Simple to implement
  - 100% accurate (exact search)
  - No build/training time
  - Low memory overhead
  
❌ CONS:
  - O(n) search time (linear)
  - Doesn't scale to large datasets
  - Checks every single vector
  
📊 USE CASES:
  - Small datasets (< 10K vectors)
  - When 100% accuracy is required
  - As a baseline for comparison
  - Development and testing

    """)

