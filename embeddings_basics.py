from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["I love AI", "Machine learning is fun", "Quantum physics rocks"]
embeddings = model.encode(sentences)  # Shape: (3, 384)
print(embeddings.shape)
print(embeddings)