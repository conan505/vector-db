# Vector Database from Scratch 🚀

A complete implementation of a vector database built from scratch in Python, designed for learning core AI concepts, vector search algorithms, and database design.

## 🎯 Project Goals

1. **Learn Python**: NumPy, OOP, type hints, and best practices
2. **Understand AI Concepts**: Embeddings, semantic search, and similarity metrics
3. **Master Vector Databases**: From basic search to production-ready features

## ✨ Features

- ✅ Multiple distance metrics (Euclidean, Cosine, Dot Product)
- ✅ Top-k similarity search
- ✅ CRUD operations (Create, Read, Update, Delete)
- ✅ Metadata storage and retrieval
- ✅ Persistence (save/load to disk)
- ✅ Batch operations
- ✅ Semantic search with pre-trained models
- ✅ Real-world AI integration

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/conan505/vector-db.git
cd vector-db

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy sentence-transformers
```

### Basic Usage

```python
from vectordb import VectorDB
import numpy as np

# Create database
db = VectorDB(dimensions=3, metric='cosine')

# Insert vectors
vector = np.array([1.0, 0.5, 0.3])
db.insert(vector, metadata={"label": "example"})

# Search
query = np.array([0.9, 0.6, 0.2])
results = db.search(query, k=5)

# Save
db.save('my_database.pkl')
```

### Semantic Search Example

```python
from sentence_transformers import SentenceTransformer
from vectordb import VectorDB

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create database
db = VectorDB(dimensions=384, metric='cosine')

# Add documents
documents = [
    "The cat sits on the mat",
    "Python is a programming language",
    "Machine learning uses neural networks"
]

for doc in documents:
    embedding = model.encode([doc])[0]
    db.insert(embedding, metadata={"text": doc})

# Search
query = "feline on a rug"
query_embedding = model.encode([query])[0]
results = db.search(query_embedding, k=3)

for result in results:
    print(f"Match: {result['metadata']['text']}")
    print(f"Similarity: {1 - result['distance']:.2%}\n")
```

## 📁 Project Structure

```
vector-db/
├── main.py                    # Original simple example
├── vector_search.py           # Distance metrics comparison
├── vectordb.py                # Core VectorDB class
├── semantic_search.py         # Real-world semantic search demo
├── LEARNING_PATH.md          # Detailed learning roadmap
└── README.md                 # This file
```

## 🎓 What You'll Learn

### Core Concepts
- **Embeddings**: Converting data to numerical vectors
- **Distance Metrics**: Measuring similarity between vectors
- **Vector Search**: Finding nearest neighbors efficiently
- **Semantic Search**: Meaning-based search (not just keywords)

### Python Skills
- NumPy array operations and broadcasting
- Object-oriented programming
- Type hints and documentation
- File I/O and serialization
- Working with ML libraries

### AI/ML Integration
- Using pre-trained embedding models
- Sentence transformers
- Real-world applications (RAG, semantic search)

## 🏃 Running the Examples

### 1. Basic Vector Search
```bash
python vector_search.py
```
Demonstrates different distance metrics and their use cases.

### 2. VectorDB Class Demo
```bash
python vectordb.py
```
Shows CRUD operations and persistence.

### 3. Semantic Search
```bash
python semantic_search.py
```
Real-world example with text embeddings and semantic matching.

## 📊 Performance

Current implementation uses **brute force search** (O(n) complexity):
- Simple and accurate
- Good for small to medium datasets (< 100k vectors)
- Educational foundation for understanding vector search

**Future optimizations** (see LEARNING_PATH.md):
- HNSW indexing for O(log n) search
- Product Quantization for memory efficiency
- GPU acceleration for large-scale operations


## 🛠️ API Reference

### VectorDB Class

#### `__init__(dimensions: int, metric: str = 'euclidean')`
Initialize a new vector database.

**Parameters:**
- `dimensions`: Number of dimensions for vectors
- `metric`: Distance metric ('euclidean', 'cosine', 'dot_product')

#### `insert(vector: np.ndarray, metadata: dict = None) -> int`
Insert a single vector.

**Returns:** Unique ID for the inserted vector

#### `insert_batch(vectors: np.ndarray, metadata_list: list = None) -> list`
Insert multiple vectors efficiently.

#### `search(query: np.ndarray, k: int = 5) -> list`
Search for k most similar vectors.

**Returns:** List of dicts with keys: 'id', 'distance', 'vector', 'metadata'

#### `delete(vector_id: int) -> bool`
Delete a vector by ID.

#### `save(filepath: str)`
Save database to disk.

#### `load(filepath: str) -> VectorDB`
Load database from disk (class method).

## 🎯 Next Steps

See [LEARNING_PATH.md](LEARNING_PATH.md) for:
- Advanced indexing techniques (HNSW, IVF)
- Production features (filtering, hybrid search)
- RAG (Retrieval Augmented Generation)
- Scaling to millions of vectors
- Project ideas and challenges

## 📚 Resources

### Papers
- [HNSW: Efficient and robust approximate nearest neighbor search](https://arxiv.org/abs/1603.09320)
- [FAISS: Billion-scale similarity search with GPUs](https://arxiv.org/abs/1702.08734)
- [RAG: Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401)

### Libraries
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook's vector search
- [Annoy](https://github.com/spotify/annoy) - Spotify's ANN library
- [ChromaDB](https://www.trychroma.com/) - Simple vector database
- [Pinecone](https://www.pinecone.io/) - Managed vector database

## 🤝 Contributing

This is a learning project! Feel free to:
- Add new features
- Improve documentation
- Create examples
- Optimize performance

## 📝 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built as a learning project to understand:
- Vector databases (Pinecone, Weaviate, Qdrant)
- Embedding models (Sentence Transformers)
- NumPy and scientific computing
- AI/ML application development

---

**Happy Learning! 🚀**

For questions or suggestions, open an issue on GitHub.
