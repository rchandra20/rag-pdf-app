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

    # Provide a default greeting if kb search not needed
    if not eval_query(query):
        return {"answer": "Hello from the RAG App! To query the knowledge base, please provide a specific question."}

    else:
        print("Query requires knowledge base search.")

        # Transform and expand the query for better retrieval
        expanded_query = transform_query(query)
        
        # Embed the expanded question
        query_embedding = embed_query(expanded_query)
    
        # Perform hybrid search (semantic + keyword) for better comprehensive answers
        retrieved_chunks = hybrid_search(
            query, 
            query_embedding, 
            top_k=5, 
            sim_threshold=0.6,
            semantic_weight=0.8,
            keyword_boost=0.2
        )

        # Post-process results to improve ranking
        final_results = post_process_results(retrieved_chunks)

        # Generate comprehensive final answer
        comprehensive_answer = generate_final_answer(query, final_results)

        return comprehensive_answer


## Helper Functions ##
def eval_query(query_text: str) -> bool:
    from app.main import MISTRAL_CLIENT, LANGUAGE_MODEL
    """Determine intent of query, whether to trigger kb search."""

    system_prompt = (
        "You are a classifier that decides whether a user's query "
        "requires information retrieval from a knowledge base (RAG search). "
        "Return only 'YES' or 'NO'.\n\n"
        "Classify as YES if the query:\n"
        "- Asks for factual information, definitions, or explanations\n"
        "- Requests specific data, statistics, or details\n"
        "- Seeks analysis, comparisons, or summaries\n"
        "- Asks 'what', 'how', 'why', 'when', 'where' questions about topics\n"
        "- Requests examples, procedures, or methodologies\n\n"
        "Classify as NO if the query:\n"
        "- Is a simple greeting or casual conversation\n"
        "- Asks about the AI's capabilities or identity\n"
        "- Is a command to the system (like 'help' or 'status')\n"
        "- Contains only pleasantries or small talk\n\n"
        "Examples:\n"
        "User: 'Hello there!' → NO\n"
        "User: 'What are the latest NHTSA vehicle recall statistics?' → YES\n"
        "User: 'How are you?' → NO\n"
        "User: 'What is the definition of amortized?' → YES\n"
        "User: 'Can you help me?' → NO\n"
        "User: 'Explain the process of photosynthesis' → YES\n"
        "User: 'Thank you' → NO\n"
        "User: 'What are the benefits of renewable energy?' → YES\n"
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
    """Transform and expand the query for better retrieval."""
    from app.main import MISTRAL_CLIENT, LANGUAGE_MODEL
    
    # Simple preprocessing
    processed_text = text.lower().strip()
    
    # Query expansion for better retrieval
    expansion_prompt = f"""Given the user query: "{text}"

Generate 2-3 related search terms or phrases that would help find relevant information in a knowledge base. These should be:
- Synonyms or alternative phrasings
- Related concepts or topics
- Broader or narrower terms
- Technical terms that might appear in documents

Return only the additional terms, separated by commas. Do not repeat the original query."""

    try:
        chat_response = MISTRAL_CLIENT.chat.complete(
            model=LANGUAGE_MODEL,
            messages=[
                {"role": "user", "content": expansion_prompt},
            ],
        )
        
        expanded_terms = chat_response.choices[0].message.content.strip()
        print(f"Query expansion: {expanded_terms}")
        
        # Combine original query with expanded terms
        combined_query = f"{text} {expanded_terms}"
        return combined_query
        
    except Exception as e:
        print(f"Query expansion failed: {e}")
        return processed_text

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

# TODO: improve this
def hybrid_search(
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
    """Merge and re-rank the results from the vector store search."""
    
    if not results:
        return results
    
    # Remove duplicates based on chunk content
    seen_chunks = set()
    unique_results = []
    
    for result in results:
        chunk_content = result['chunk'].strip()
        if chunk_content not in seen_chunks:
            seen_chunks.add(chunk_content)
            unique_results.append(result)
    
    # Re-rank by similarity score (they should already be sorted, but ensure it)
    unique_results.sort(key=lambda x: x.get('similarity', x.get('score', 0)), reverse=True)
    
    # Filter out very low-quality results
    filtered_results = [
        result for result in unique_results 
        if result.get('similarity', result.get('score', 0)) > 0.3
    ]
    
    return filtered_results

def generate_final_answer(query_text, results):
    """Generate comprehensive answer from the top RAG search results with LLM."""
    from app.main import MISTRAL_CLIENT, LANGUAGE_MODEL
    import json
    import os
    from datetime import datetime

    if not results:
        return "I couldn't find relevant information to answer your question in the knowledge base."

    # Prepare context from retrieved chunks
    context_chunks = []
    sources = []
    
    for i, result in enumerate(results, 1):
        context_chunks.append(f"[Source {i}] {result['chunk']}")
        sources.append(result.get('source_file', 'Unknown source'))
    
    context = "\n\n".join(context_chunks)
    source_list = ", ".join(set(sources))

    system_prompt = """You are an expert AI assistant that provides comprehensive, accurate, and well-structured answers based on retrieved information from a knowledge base. Your role is to:

1. **Synthesize Information**: Combine and analyze information from multiple sources to create a coherent, comprehensive answer
2. **Provide Context**: Give background information and explain concepts when helpful
3. **Structure Your Response**: Organize your answer logically with clear sections, bullet points, or numbered lists when appropriate
4. **Cite Sources**: Always reference which sources support your claims using [Source X] format
5. **Be Thorough**: Address all aspects of the question and provide relevant details
6. **Maintain Accuracy**: Only use information from the provided context, and clearly state when information is incomplete
7. **Be Helpful**: Provide actionable insights, examples, or recommendations when relevant

**Response Guidelines:**
- Start with a direct answer to the question
- Provide detailed explanations with supporting evidence
- Use clear, professional language
- Include relevant examples or details from the sources
- If the context doesn't fully answer the question, acknowledge limitations
- End with source citations

**Available Sources:** {sources}

**Context from Knowledge Base:**
{context}

Now, provide a comprehensive answer to the user's question based on the retrieved information."""

    user_prompt = f"Question: {query_text}\n\nPlease provide a comprehensive answer based on the retrieved information above."

    chat_response = MISTRAL_CLIENT.chat.complete(
        model=LANGUAGE_MODEL,
        messages=[
            {"role": "system", "content": system_prompt.format(sources=source_list, context=context)},
            {"role": "user", "content": user_prompt},
        ],
    )

    print("Response from generation model:", chat_response)

    generated_answer = chat_response.choices[0].message.content
    
    # Log query and answer to JSON file
    query_log = {
        "timestamp": datetime.now().isoformat(),
        "query": query_text,
        "answer": generated_answer,
        "sources": sources,
        "retrieved_chunks": len(results)
    }
    
    try:
        log_file = "query_log.json"
        # Load existing logs or create new list
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        # Append new log entry
        logs.append(query_log)
        
        # Save back to file
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
            
        print(f"Query logged to {log_file}")
    except Exception as e:
        print(f"Error logging query: {e}")
    
    return {
        "answer": generated_answer,
        "sources": sources,
        "retrieved_chunks": len(results)
    }