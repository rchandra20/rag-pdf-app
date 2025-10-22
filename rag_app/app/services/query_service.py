# Logic to handle searching the vector store against user queries

from .ingest_service import VECTOR_STORE # custom vector store
import json
import os
from datetime import datetime

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
            5, # top_k
            0.7, # Semantic weight in score
             0.3 # Keyword weight in score
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
        return text

def embed_query(text: str):
    from app.main import MISTRAL_CLIENT, EMBEDDING_MODEL
    """Generate embedding vector for a single text string."""
    response = MISTRAL_CLIENT.embeddings.create(model=EMBEDDING_MODEL, inputs=[text])
    return response.data[0].embedding

def normalize(vec):
    """Return L2-normalized vector."""
    norm = sum(x**2 for x in vec) ** 0.5
    return [x / norm for x in vec] if norm > 0 else vec

def semantic_search(query_embedding, top_k, similarity_threshold: float = 0.7):
    """Perform semantic search over VECTOR_STORE with normalized embeddings."""
    
    def cosine_similarity(a, b):
        dot_product = sum(x * y for x, y in zip(a, b))
        return max(0.0, dot_product)  # clip negatives to 0

    #  Normalize to guarantee consistent similarity scores between 0 and 1
    query_embedding = normalize(query_embedding)
    similarities = []

    for record in VECTOR_STORE:
        chunk_embedding = normalize(record["embedding"])
        similarity = cosine_similarity(query_embedding, chunk_embedding)

        # Ensure chunk meets similarity threshold
        if similarity >= similarity_threshold:
            similarities.append({
                "chunk": record["chunk"],
                "source_file": record["source_file"],
                "score": similarity
            })

    similarities.sort(key=lambda x: x["score"], reverse=True)
    return similarities[:top_k]

def keyword_search(query_text: str, top_k):
    """Perform keyword search over VECTOR_STORE."""
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

def hybrid_search(
    query_text: str,
    query_embedding,
    top_k,
    semantic_weight,
    keyword_weight
):
    """
    Simple hybrid search that combines semantic and keyword results with weighted scores.
    """
    # Get more results from each method to have better candidates
    semantic_results = semantic_search(query_embedding, top_k=top_k*2, similarity_threshold=0.5)
    keyword_results = keyword_search(query_text, top_k=top_k*2)
    
    # Create a combined score for each unique chunk
    chunk_scores = {}
    
    # Add semantic scores
    for result in semantic_results:
        chunk_key = result["chunk"]
        chunk_scores[chunk_key] = {
            "chunk": result["chunk"],
            "source_file": result["source_file"],
            "semantic_score": result["score"],
            "keyword_score": 0.0,
            "combined_score": result["score"] * semantic_weight
        }
    
    # Add keyword scores and combine
    for result in keyword_results:
        chunk_key = result["chunk"]
        if chunk_key in chunk_scores:
            # Chunk exists from semantic search - add keyword score
            chunk_scores[chunk_key]["keyword_score"] = result["score"]
            chunk_scores[chunk_key]["combined_score"] += result["score"] * keyword_weight
        else:
            # Chunk only found by keyword search
            chunk_scores[chunk_key] = {
                "chunk": result["chunk"],
                "source_file": result["source_file"],
                "semantic_score": 0.0,
                "keyword_score": result["score"],
                "combined_score": result["score"] * keyword_weight
            }
    
    # Sort by combined score and return top_k
    sorted_results = sorted(chunk_scores.values(), key=lambda x: x["combined_score"], reverse=True)
    
    # Simplify output format
    final_results = []
    for result in sorted_results[:top_k]:
        final_results.append({
            "chunk": result["chunk"],
            "source_file": result["source_file"],
            "score": result["combined_score"]
        })
    
    return final_results

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
    
    # Merge adjacent chunks from same source
    merged_results = []
    current_group = []
    current_source = None
    
    for result in unique_results:
        source = result['source_file']
        
        # If same source, add to current group
        if source == current_source:
            current_group.append(result)
        else:
            # Process previous group if exists
            if current_group:
                if len(current_group) > 1:
                    # Merge multiple chunks
                    combined_text = " ".join([chunk['chunk'] for chunk in current_group])
                    max_score = max([chunk.get('similarity', chunk.get('score', 0)) for chunk in current_group])
                    merged_results.append({
                        'chunk': combined_text,
                        'source_file': current_group[0]['source_file'],
                        'score': max_score
                    })
                else:
                    # Single chunk, keep as is
                    merged_results.append(current_group[0])
            
            # Start new group
            current_group = [result]
            current_source = source
    
    # Process last group
    if current_group:
        if len(current_group) > 1:
            # Merge multiple chunks
            combined_text = " ".join([chunk['chunk'] for chunk in current_group])
            max_score = max([chunk.get('similarity', chunk.get('score', 0)) for chunk in current_group])
            merged_results.append({
                'chunk': combined_text,
                'source_file': current_group[0]['source_file'],
                'score': max_score
            })
        else:
            # Single chunk, keep as is
            merged_results.append(current_group[0])
    
    # Re-rank by similarity score
    merged_results.sort(key=lambda x: x.get('similarity', x.get('score', 0)), reverse=True)
    
    # Filter out very low-quality results
    filtered_results = [
        result for result in merged_results 
        if result.get('similarity', result.get('score', 0)) > 0.3
    ]
    
    return filtered_results 

def generate_final_answer(query_text, results):
    """Generate comprehensive answer from the top RAG search results with LLM."""
    from app.main import MISTRAL_CLIENT, LANGUAGE_MODEL

    if not results:
        return "I couldn't find relevant information to answer your question in the knowledge base."

    # Prepare context from retrieved chunks
    context_chunks = []
    sources = []

    print(f"Final Chunk Results: {results}")
    
    for result in results:
        source_file = result.get('source_file', 'Unknown source')
        context_chunks.append(f"[{source_file}] {result['chunk']}")
        sources.append(source_file)
    
    context = "\n\n".join(context_chunks)
    source_list = ", ".join(set(sources))

    system_prompt = """You are an expert AI assistant that provides concise, accurate answers based on retrieved information from a knowledge base. Your role is to:

1. **Be Concise**: Provide direct, clear answers without unnecessary elaboration
2. **Stay Focused**: Answer only what was asked, avoid tangents
3. **Be Accurate**: Only use information from the provided context
4. **Cite Sources**: Reference sources using [filename.pdf] format when making claims
5. **Be Complete**: Include all essential information to fully answer the question

**Response Guidelines:**
- Start with a direct, brief answer
- Use bullet points or numbered lists only when the information naturally fits that format
- Keep sentences clear and concise
- If the context doesn't fully answer the question, acknowledge limitations briefly
- End with source citations

**Available Sources:** {sources}

**Context from Knowledge Base:**
{context}

Now, provide a concise answer to the user's question based on the retrieved information."""

    user_prompt = f"Question: {query_text}\n\nPlease provide a concise answer based on the retrieved information above."

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
        "retrieved_chunks": results
    }