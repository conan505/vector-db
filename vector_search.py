import numpy as np


def euclidean_distance(vectors, query):
    """
    Calculate Euclidean distance (L2 norm).
    
    Formula: √((x₁-x₂)² + (y₁-y₂)² + (z₁-z₂)²)
    
    Use case: Geometric distance, good for spatial data
    Range: [0, ∞) - smaller is more similar
    """
    return np.linalg.norm(vectors - query, axis=1)


def cosine_similarity(vectors, query):
    """
    Calculate Cosine Similarity.
    
    Formula: (A · B) / (||A|| × ||B||)
    Measures the angle between vectors, ignoring magnitude.
    
    Use case: Text embeddings, when direction matters more than magnitude
    Range: [-1, 1] - larger is more similar (1 = identical direction)
    """
    # Normalize vectors and query to unit length
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    query_norm = query / np.linalg.norm(query)
    
    # Dot product of normalized vectors gives cosine similarity
    similarities = np.dot(vectors_norm, query_norm.T).flatten()
    
    # Convert to distance (smaller = more similar)
    # We return 1 - similarity so smaller values mean more similar
    return 1 - similarities


def dot_product_similarity(vectors, query):
    """
    Calculate Dot Product (inner product).
    
    Formula: A · B = a₁b₁ + a₂b₂ + a₃b₃
    
    Use case: When both magnitude and direction matter
    Range: (-∞, ∞) - larger is more similar
    """
    similarities = np.dot(vectors, query.T).flatten()
    
    # Convert to distance (smaller = more similar)
    return -similarities


def search_top_k(vectors, query, k=3, metric='euclidean'):
    """
    Find the top-k most similar vectors to the query.
    
    Args:
        vectors: numpy array of shape (n, d) - n vectors of d dimensions
        query: numpy array of shape (1, d) - query vector
        k: number of top results to return
        metric: distance metric to use ('euclidean', 'cosine', 'dot_product')
    
    Returns:
        indices: indices of top-k most similar vectors
        distances: distances/similarities of top-k vectors
    """
    # Calculate distances based on chosen metric
    if metric == 'euclidean':
        distances = euclidean_distance(vectors, query)
    elif metric == 'cosine':
        distances = cosine_similarity(vectors, query)
    elif metric == 'dot_product':
        distances = dot_product_similarity(vectors, query)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Get indices of k smallest distances (most similar)
    # argsort returns indices that would sort the array
    top_k_indices = np.argsort(distances)[:k]
    top_k_distances = distances[top_k_indices]
    
    return top_k_indices, top_k_distances


def main():
    print("=" * 70)
    print("VECTOR DATABASE - SIMILARITY SEARCH DEMO")
    print("=" * 70)
    
    # Generate sample data
    np.random.seed(42)  # For reproducibility
    n_vectors = 10
    dimensions = 3
    
    vectors = np.random.rand(n_vectors, dimensions)
    query = np.random.rand(1, dimensions)
    
    print(f"\nDataset: {n_vectors} vectors with {dimensions} dimensions")
    print(f"\nQuery vector: {query[0]}")
    print(f"\nAll vectors:")
    for i, vec in enumerate(vectors):
        print(f"  Vector {i}: {vec}")
    
    # Test different metrics
    metrics = ['euclidean', 'cosine', 'dot_product']
    k = 3
    
    print(f"\n{'=' * 70}")
    print(f"SEARCHING FOR TOP-{k} MOST SIMILAR VECTORS")
    print(f"{'=' * 70}")
    
    for metric in metrics:
        print(f"\n--- Using {metric.upper()} distance ---")
        
        indices, distances = search_top_k(vectors, query, k=k, metric=metric)
        
        for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
            print(f"  Rank {rank}: Vector {idx} - Distance: {dist:.4f}")
            print(f"           Values: {vectors[idx]}")
    
    # Demonstrate the difference between metrics
    print(f"\n{'=' * 70}")
    print("KEY INSIGHTS")
    print(f"{'=' * 70}")
    print("""
1. EUCLIDEAN DISTANCE:
   - Measures actual geometric distance
   - Sensitive to magnitude (vector length)
   - Good for: spatial data, when scale matters

2. COSINE SIMILARITY:
   - Measures angle between vectors
   - Ignores magnitude, only cares about direction
   - Good for: text embeddings, document similarity

3. DOT PRODUCT:
   - Combines both angle and magnitude
   - Larger values = more similar
   - Good for: recommendation systems, when both matter
    """)


if __name__ == "__main__":
    main()

