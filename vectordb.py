import numpy as np
import pickle
from typing import List, Tuple, Optional, Dict, Any


class VectorDB:
    """
    A simple vector database implementation from scratch.
    
    Features:
    - Insert vectors with metadata
    - Search for similar vectors (top-k)
    - Delete vectors by ID
    - Multiple distance metrics
    - Persistence (save/load to disk)
    """
    
    def __init__(self, dimensions: int, metric: str = 'euclidean'):
        """
        Initialize the vector database.
        
        Args:
            dimensions: Number of dimensions for vectors
            metric: Distance metric ('euclidean', 'cosine', 'dot_product')
        """
        self.dimensions = dimensions
        self.metric = metric
        
        # Storage
        self.vectors = np.array([]).reshape(0, dimensions)  # Shape: (0, dimensions)
        self.metadata = []  # List of metadata dicts
        self.ids = []  # List of vector IDs
        self.next_id = 0  # Auto-incrementing ID
        
        print(f"✓ VectorDB initialized: {dimensions}D vectors, {metric} metric")
    
    def insert(self, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Insert a vector into the database.
        
        Args:
            vector: numpy array of shape (dimensions,)
            metadata: optional dictionary with metadata (e.g., {"text": "hello", "category": "greeting"})
        
        Returns:
            id: unique ID assigned to this vector
        """
        # Validate dimensions
        if vector.shape != (self.dimensions,):
            raise ValueError(f"Vector must have {self.dimensions} dimensions, got {vector.shape}")
        
        # Add vector
        vector_2d = vector.reshape(1, -1)  # Convert to 2D: (1, dimensions)
        self.vectors = np.vstack([self.vectors, vector_2d])
        
        # Add metadata and ID
        self.metadata.append(metadata or {})
        self.ids.append(self.next_id)
        
        vector_id = self.next_id
        self.next_id += 1
        
        print(f"✓ Inserted vector ID={vector_id}, metadata={metadata}")
        return vector_id
    
    def insert_batch(self, vectors: np.ndarray, metadata_list: Optional[List[Dict[str, Any]]] = None) -> List[int]:
        """
        Insert multiple vectors at once (more efficient).
        
        Args:
            vectors: numpy array of shape (n, dimensions)
            metadata_list: optional list of metadata dicts
        
        Returns:
            ids: list of assigned IDs
        """
        n = vectors.shape[0]
        
        if metadata_list is None:
            metadata_list = [{}] * n
        
        if len(metadata_list) != n:
            raise ValueError(f"metadata_list length ({len(metadata_list)}) must match vectors ({n})")
        
        ids = []
        for vector, metadata in zip(vectors, metadata_list):
            vector_id = self.insert(vector, metadata)
            ids.append(vector_id)
        
        return ids
    
    def search(self, query: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for the k most similar vectors.
        
        Args:
            query: query vector of shape (dimensions,)
            k: number of results to return
        
        Returns:
            results: list of dicts with keys: 'id', 'distance', 'vector', 'metadata'
        """
        if len(self.vectors) == 0:
            return []
        
        if query.shape != (self.dimensions,):
            raise ValueError(f"Query must have {self.dimensions} dimensions, got {query.shape}")
        
        # Reshape query to 2D for broadcasting
        query_2d = query.reshape(1, -1)
        
        # Calculate distances
        distances = self._calculate_distances(self.vectors, query_2d)
        
        # Get top-k indices
        k = min(k, len(self.vectors))  # Don't request more than we have
        top_k_indices = np.argsort(distances)[:k]
        
        # Build results
        results = []
        for idx in top_k_indices:
            results.append({
                'id': self.ids[idx],
                'distance': float(distances[idx]),
                'vector': self.vectors[idx],
                'metadata': self.metadata[idx]
            })
        
        return results
    
    def delete(self, vector_id: int) -> bool:
        """
        Delete a vector by its ID.
        
        Args:
            vector_id: ID of the vector to delete
        
        Returns:
            success: True if deleted, False if not found
        """
        try:
            idx = self.ids.index(vector_id)
        except ValueError:
            print(f"✗ Vector ID={vector_id} not found")
            return False
        
        # Remove from all storage
        self.vectors = np.delete(self.vectors, idx, axis=0)
        del self.metadata[idx]
        del self.ids[idx]
        
        print(f"✓ Deleted vector ID={vector_id}")
        return True
    
    def _calculate_distances(self, vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Calculate distances based on the chosen metric."""
        if self.metric == 'euclidean':
            return np.linalg.norm(vectors - query, axis=1)
        
        elif self.metric == 'cosine':
            # Normalize vectors
            vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            query_norm = query / np.linalg.norm(query)
            similarities = np.dot(vectors_norm, query_norm.T).flatten()
            return 1 - similarities  # Convert to distance
        
        elif self.metric == 'dot_product':
            similarities = np.dot(vectors, query.T).flatten()
            return -similarities  # Convert to distance (negate so smaller = better)
        
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def save(self, filepath: str):
        """
        Save the database to disk.
        
        Args:
            filepath: path to save the database (e.g., 'my_db.pkl')
        """
        data = {
            'dimensions': self.dimensions,
            'metric': self.metric,
            'vectors': self.vectors,
            'metadata': self.metadata,
            'ids': self.ids,
            'next_id': self.next_id
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Database saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'VectorDB':
        """
        Load a database from disk.
        
        Args:
            filepath: path to the saved database
        
        Returns:
            db: loaded VectorDB instance
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create new instance
        db = cls(dimensions=data['dimensions'], metric=data['metric'])
        
        # Restore data
        db.vectors = data['vectors']
        db.metadata = data['metadata']
        db.ids = data['ids']
        db.next_id = data['next_id']
        
        print(f"✓ Database loaded from {filepath} ({len(db.ids)} vectors)")
        return db
    
    def __len__(self) -> int:
        """Return the number of vectors in the database."""
        return len(self.ids)
    
    def __repr__(self) -> str:
        """String representation of the database."""
        return f"VectorDB(vectors={len(self)}, dimensions={self.dimensions}, metric='{self.metric}')"


def demo():
    """Demonstrate VectorDB usage."""
    print("=" * 70)
    print("VECTOR DATABASE - CRUD OPERATIONS DEMO")
    print("=" * 70)
    
    # Create database
    print("\n1. CREATE DATABASE")
    db = VectorDB(dimensions=3, metric='cosine')
    
    # Insert vectors
    print("\n2. INSERT VECTORS")
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.3, 0.3, 0.3]
    ])
    
    metadata_list = [
        {"label": "x-axis", "category": "axis"},
        {"label": "y-axis", "category": "axis"},
        {"label": "z-axis", "category": "axis"},
        {"label": "xy-diagonal", "category": "diagonal"},
        {"label": "center", "category": "center"}
    ]
    
    ids = db.insert_batch(vectors, metadata_list)
    print(f"\nDatabase now has {len(db)} vectors")
    
    # Search
    print("\n3. SEARCH FOR SIMILAR VECTORS")
    query = np.array([0.6, 0.4, 0.0])
    print(f"Query: {query}")
    
    results = db.search(query, k=3)
    print(f"\nTop 3 results:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. ID={result['id']}, Distance={result['distance']:.4f}")
        print(f"     Vector: {result['vector']}")
        print(f"     Metadata: {result['metadata']}")
    
    # Delete
    print("\n4. DELETE A VECTOR")
    db.delete(ids[2])  # Delete z-axis vector
    print(f"Database now has {len(db)} vectors")
    
    # Save
    print("\n5. SAVE TO DISK")
    db.save('my_vector_db.pkl')
    
    # Load
    print("\n6. LOAD FROM DISK")
    db2 = VectorDB.load('my_vector_db.pkl')
    print(f"Loaded database: {db2}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    demo()

