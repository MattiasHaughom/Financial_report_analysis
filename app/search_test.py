import logging
from app.database.vector_store import VectorStore
import pandas as pd

def test_search_functions():
    vector_store = VectorStore()
    query = "Revenue Growth Operating Margin"

    # Perform keyword search
    keyword_results = vector_store.keyword_search(
        query=query,
        limit=5,
        table_name='reports',
        return_dataframe=True
    )
    logging.info("Keyword Search Results:")
    logging.info(keyword_results)

    # Perform semantic search
    semantic_results = vector_store.semantic_search(
        query=query,
        limit=5,
        table_name='reports',
        return_dataframe=True
    )
    logging.info("Semantic Search Results:")
    logging.info(semantic_results)

    # Perform hybrid search
    hybrid_results = vector_store.hybrid_search(
        query=query,
        keyword_k=5,
        semantic_k=5,
        rerank=True,
        top_n=5,
        table_name='reports'
    )
    logging.info("Hybrid Search Results:")
    logging.info(hybrid_results)

    # Serialize hybrid search results
    contents = hybrid_results['content']
    
    logging.info("Hybrid Search Results Serialized:")
    logging.info(contents)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_search_functions()