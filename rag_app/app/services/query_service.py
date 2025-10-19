# Logic to handle searching the vector store against user queries

from .ingest_service import VECTOR_STORE # shared in-memory vector DB

## Driver Function ##
def query_service_main(query: str):
    """
    Returns comprehensive answer to user query by searching VECTOR_STORE
    """

    # Check if VECTOR_STORE is empty
    if not VECTOR_STORE:
        return {"answer": "Vector store is empty. Please upload PDFs first."}
    
    # Detect the intent of the query
    eval_query(query)

    # Provide a default greeting if query
    if not eval_query(query):
        return "Hello from the RAG App! To query the knowledge base, please provide a specific question."

    else:
        # transform_query(query)
        print("Query requires knowledge base search.")

        # Embed the question
        query_embedding = embed_query(query)
    
        # Perform semantic search
        # retrieved_chunks = combined_search(query, query_embedding)
        retrieved_chunks = semantic_search(query_embedding)

        '''
        # Post-process results
        final_results = post_process_results(results)

        # Generate final answer
        '''

        return retrieved_chunks


## Helper Functions ##
def eval_query(query_text: str) -> bool:
    from app.main import MISTRAL_CLIENT, LANGUAGE_MODEL
    """Determine intent of query, whether to trigger kb search."""

    system_prompt = (
        "You are a classifier that decides whether a user's query "
        "requires information retrieval from a knowledge base (RAG search). "
        "Return only 'YES' or 'NO'.\n\n"
        "Examples:\n"
        "User: 'Hello there!' → NO\n"
        "User: 'What are the latest NHTSA vehicle recall statistics?' → YES\n"
        "User: 'How are you?' → NO\n"
        "User: 'What is the definition of ammortized.' → YES\n"
    )

    chat_response = MISTRAL_CLIENT.chat.complete(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_text},
        ],
    )

    print("Response from intent model:", chat_response)

    decision = chat_response.choices[0].message.content

    # Debug/logging
    print(f"Query intent decision: {decision} for query: {query_text}")

    # Return True if model said "YES"
    return decision.startswith("Y")

def transform_query(text: str) -> str:
    """A simple transformation function to test the query service."""
    return text.lower()

def embed_query(text: str):
    from app.main import MISTRAL_CLIENT, EMBEDDING_MODEL
    """Generate embedding vector for a single text string."""
    response = MISTRAL_CLIENT.embeddings.create(model=EMBEDDING_MODEL, inputs=[text])
    return response.data[0].embedding  # 1024-dim vector

def normalize(vec):
    """Return L2-normalized vector."""
    norm = sum(x**2 for x in vec) ** 0.5
    return [x / norm for x in vec] if norm > 0 else vec

def semantic_search(query_embedding, top_k: int = 3, similarity_threshold: float = 0.7):
    """Perform semantic search over VECTOR_STORE with normalized embeddings."""
    
    def cosine_similarity(a, b):
        dot_product = sum(x * y for x, y in zip(a, b))
        return max(0.0, dot_product)  # clip negatives to 0

    query_embedding = normalize(query_embedding)
    similarities = []

    for record in VECTOR_STORE:
        chunk_embedding = normalize(record["embedding"])
        similarity = cosine_similarity(query_embedding, chunk_embedding)
        if similarity >= similarity_threshold:
            similarities.append({
                "chunk": record["chunk"],
                "source_file": record["source_file"],
                "similarity": similarity
            })

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    return similarities[:top_k]


# TODO: Work on this more
def keyword_search(query_text: str, top_k: int = 3):
    query_words = set(query_text.lower().split())
    results = []

    for record in VECTOR_STORE:
        chunk_words = set(record["chunk"].lower().split())
        common_words = query_words & chunk_words
        if common_words:
            score = len(common_words) / len(query_words)
            results.append({
                "chunk": record["chunk"],
                "source_file": record["source_file"],
                "score": score
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def combined_search(
    query_text: str,
    query_embedding,
    top_k: int = 3,
    sim_threshold: float = 0.7,
    semantic_weight: float = 1.0,
    keyword_boost: float = 0.2  # only additive, never subtracts
):
    """
    Combine semantic and keyword search results.
    Keyword matches only boost the semantic score.
    """

    semantic_results = semantic_search(
        query_embedding, top_k=top_k, similarity_threshold=sim_threshold
    )
    keyword_results = keyword_search(query_text, top_k=top_k)

    combined = {}

    # Apply semantic score first
    for r in semantic_results:
        combined[r["chunk"]] = {
            "chunk": r["chunk"],
            "source_file": r["source_file"],
            "score": r["similarity"] * semantic_weight
        }

    # Apply keyword boost (only additive)
    for r in keyword_results:
        if r["chunk"] in combined:
            combined[r["chunk"]]["score"] += r["score"] * keyword_boost
        else:
            # If semantic search didn't pick this chunk, we can optionally include it
            # with only keyword score boost
            combined[r["chunk"]] = {
                "chunk": r["chunk"],
                "source_file": r["source_file"],
                "score": r["score"] * keyword_boost
            }

    # Sort descending by combined score
    final_results = sorted(combined.values(), key=lambda x: x["score"], reverse=True)
    return final_results[:top_k]



def post_process_results(results):
    """Post-process the results from the vector store search."""
    return results

def generate_final_answer(results):
    """Generate final answer from the top results with LLM."""
    answer = "Based on the documents, here are the top relevant chunks:\n"
    for idx, item in enumerate(results["top_chunks"], 1):
        answer += f"{idx}. {item['text']} (Similarity: {item['similarity']:.4f})\n"
    return answer