"""
Semantic Search Demo using VectorDB

This demonstrates how vector databases are used in real AI applications:
- Convert text to embeddings (vectors) using a pre-trained model
- Store embeddings in our custom VectorDB
- Search for semantically similar text (not just keyword matching!)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from vectordb import VectorDB


def main():
    print("=" * 70)
    print("SEMANTIC SEARCH ENGINE - Real-world Vector DB Application")
    print("=" * 70)
    
    # Step 1: Load a pre-trained embedding model
    print("\n📥 Loading embedding model (this may take a moment)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model
    print(f"✓ Model loaded! Embedding dimension: {model.get_sentence_embedding_dimension()}")
    
    # Step 2: Create our vector database
    print("\n🗄️  Creating vector database...")
    db = VectorDB(dimensions=384, metric='cosine')  # This model produces 384D vectors
    
    # Step 3: Add documents to the database
    print("\n📝 Adding documents to database...")
    documents = [
        "The cat sits on the mat",
        "A dog plays in the park",
        "Python is a programming language",
        "Machine learning uses neural networks",
        "The kitten sleeps on the carpet",
        "JavaScript is used for web development",
        "Deep learning is a subset of AI",
        "The puppy runs in the garden",
        "Rust is a systems programming language",
        "Natural language processing analyzes text"
    ]
    
    # Generate embeddings for all documents
    print("   Converting text to vectors...")
    embeddings = model.encode(documents)
    print(f"   Generated {len(embeddings)} embeddings of shape {embeddings[0].shape}")
    
    # Insert into database with metadata
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        db.insert(embedding, metadata={"text": doc, "doc_id": i})
    
    print(f"\n✓ Database ready with {len(db)} documents")
    
    # Step 4: Perform semantic searches
    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH EXAMPLES")
    print("=" * 70)
    
    queries = [
        "feline on a rug",           # Similar to "cat on mat" but different words!
        "coding in Python",          # Programming related
        "AI and neural nets",        # Machine learning related
    ]
    
    for query_text in queries:
        print(f"\n🔍 Query: '{query_text}'")
        print("-" * 70)
        
        # Convert query to embedding
        query_embedding = model.encode([query_text])[0]
        
        # Search database
        results = db.search(query_embedding, k=3)
        
        print("Top 3 most similar documents:")
        for rank, result in enumerate(results, 1):
            similarity = 1 - result['distance']  # Convert distance back to similarity
            print(f"\n  {rank}. Similarity: {similarity:.4f} ({similarity*100:.1f}%)")
            print(f"     Text: \"{result['metadata']['text']}\"")
    
    # Step 5: Demonstrate why this is "semantic" search
    print("\n" + "=" * 70)
    print("WHY IS THIS 'SEMANTIC' SEARCH?")
    print("=" * 70)
    print("""
Traditional keyword search would fail here:
- Query: "feline on a rug"
- Would NOT match: "The cat sits on the mat"
  (No shared words!)

But semantic search understands MEANING:
- "feline" ≈ "cat" (both are animals)
- "rug" ≈ "mat" (both are floor coverings)
- The sentence structure is similar

This is the power of embeddings + vector databases!
    """)
    
    # Step 6: Show the vector space
    print("=" * 70)
    print("UNDERSTANDING THE VECTOR SPACE")
    print("=" * 70)
    print(f"""
Each document is represented as a {db.dimensions}-dimensional vector.
Similar documents have vectors pointing in similar directions.

Example vectors (first 5 dimensions):
""")
    
    for i in range(min(3, len(documents))):
        print(f"  Doc {i}: {embeddings[i][:5]} ...")
        print(f"         \"{documents[i]}\"")
    
    # Step 7: Save the database
    print("\n💾 Saving database to disk...")
    db.save('semantic_search_db.pkl')
    print("✓ Database saved! You can load it later with VectorDB.load()")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print("""
What you learned:
1. ✓ Text → Embeddings (using sentence-transformers)
2. ✓ Store embeddings in VectorDB
3. ✓ Semantic search (meaning-based, not keyword-based)
4. ✓ Real-world AI application

Next steps to explore:
- Try different embedding models (larger = more accurate)
- Add more documents (scale to thousands!)
- Implement filtering (e.g., search only "programming" docs)
- Build a RAG system (Retrieval Augmented Generation with LLMs)
    """)


if __name__ == "__main__":
    main()

