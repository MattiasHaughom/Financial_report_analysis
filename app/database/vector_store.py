import logging
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple, Union

import cohere
import pandas as pd
import psycopg
from ..config.settings import get_settings
from ..services.llm_factory import LLMFactory
from pydantic import BaseModel
from openai import OpenAI
from ..services.synthesizer import Keywords
from timescale_vector import client
import nltk
from nltk.corpus import wordnet
from typing import List

# Ensure the necessary NLTK data is downloaded
nltk.download('punkt_tab')
nltk.download('wordnet')


def expand_with_synonyms(keywords: List[str]) -> List[str]:
    """Expand keywords with synonyms using WordNet."""
    expanded_keywords = set(keywords)
    for keyword in keywords:
        for syn in wordnet.synsets(keyword):
            for lemma in syn.lemmas():
                expanded_keywords.add(lemma.name())
    return list(expanded_keywords)

class VectorStore:
    """A class for managing vector operations and database interactions."""

    def __init__(self):
        """Initialize the VectorStore with settings, OpenAI client, and Timescale Vector client."""
        self.settings = get_settings()
        self.openai_client = OpenAI(api_key=self.settings.openai.api_key)
        self.embedding_model = self.settings.openai.embedding_model
        self.cohere_client = cohere.ClientV2(api_key=self.settings.cohere.api_key)
        self.vector_settings = self.settings.vector_store
        self.vec_client = client.Sync(
            self.settings.database.service_url,
            self.vector_settings.reports_table,
            self.vector_settings.embedding_dimensions,
            time_partition_interval=self.vector_settings.time_partition_interval,
        )

    def create_keyword_search_index(self):
        """Create a GIN index for keyword search if it doesn't exist."""
        index_name = f"idx_{self.vector_settings.table_name}_contents_gin"
        create_index_sql = f"""
        CREATE INDEX IF NOT EXISTS {index_name}
        ON {self.vector_settings.table_name} USING gin(to_tsvector('english', contents));
        """
        try:
            with psycopg.connect(self.settings.database.service_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(create_index_sql)
                    conn.commit()
                    logging.info(f"GIN index '{index_name}' created or already exists.")
        except Exception as e:
            logging.error(f"Error while creating GIN index: {str(e)}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for the given text.

        Args:
            text: The input text to generate an embedding for.

        Returns:
            A list of floats representing the embedding.
        """
        text = text.replace("\n", " ")
        start_time = time.time()
        embedding = (
            self.openai_client.embeddings.create(
                input=[text],
                model=self.embedding_model,
            )
            .data[0]
            .embedding
        )
        elapsed_time = time.time() - start_time
        logging.info(f"Embedding generated in {elapsed_time:.3f} seconds")
        return embedding

    def create_tables(self) -> None:
        """Create the necessary tablesin the database"""
        self.vec_client.create_tables()

    def create_index(self) -> None:
        """Create the StreamingDiskANN index to spseed up similarity search"""
        self.vec_client.create_embedding_index(client.DiskAnnIndex())

    def drop_index(self) -> None:
        """Drop the StreamingDiskANN index in the database"""
        self.vec_client.drop_embedding_index()

    def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, contents, embedding
        """
        records = df.to_records(index=False)
        self.vec_client.upsert(list(records))
        logging.info(
            f"Inserted {len(df)} records into {self.vector_settings.table_name}"
        )

    def semantic_search(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        predicates: Optional[client.Predicates] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Query the vector database for similar embeddings based on input text.

        More info:
            https://github.com/timescale/docs/blob/latest/ai/python-interface-for-pgvector-and-timescale-vector.md

        Args:
            query: The input text to search for.
            limit: The maximum number of results to return.
            metadata_filter: A dictionary or list of dictionaries for equality-based metadata filtering.
            predicates: A Predicates object for complex metadata filtering.
                - Predicates objects are defined by the name of the metadata key, an operator, and a value.
                - Operators: ==, !=, >, >=, <, <=
                - & is used to combine multiple predicates with AND operator.
                - | is used to combine multiple predicates with OR operator.
            time_range: A tuple of (start_date, end_date) to filter results by time.
            return_dataframe: Whether to return results as a DataFrame (default: True).

        Returns:
            Either a list of tuples or a pandas DataFrame containing the search results.

        Basic Examples:
            Basic search:
                vector_store.semantic_search("What are your shipping options?")
            Search with metadata filter:
                vector_store.semantic_search("Shipping options", metadata_filter={"category": "Shipping"})
        
        Predicates Examples:
            Search with predicates:
                vector_store.semantic_search("Pricing", predicates=client.Predicates("price", ">", 100))
            Search with complex combined predicates:
                complex_pred = (client.Predicates("category", "==", "Electronics") & client.Predicates("price", "<", 1000)) | \
                               (client.Predicates("category", "==", "Books") & client.Predicates("rating", ">=", 4.5))
                vector_store.semantic_search("High-quality products", predicates=complex_pred)
        
        Time-based filtering:
            Search with time range:
                vector_store.semantic_search("Recent updates", time_range=(datetime(2024, 1, 1), datetime(2024, 1, 31)))
        """
        query_embedding = self.get_embedding(query)

        start_time = time.time()

        search_args = {
            "limit": limit,
        }

        if metadata_filter:
            search_args["filter"] = metadata_filter

        if predicates:
            search_args["predicates"] = predicates

        if time_range:
            start_date, end_date = time_range
            search_args["uuid_time_filter"] = client.UUIDTimeRange(start_date, end_date)

        results = self.vec_client.search(query_embedding, **search_args)
        elapsed_time = time.time() - start_time

        self._log_search_time("Vector", elapsed_time)

        if return_dataframe:
            return self._create_dataframe_from_results(results)
        else:
            return results

    def _create_dataframe_from_results(
        self,
        results: List[Tuple[Any, ...]],
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from the search results.

        Args:
            results: A list of tuples containing the search results.

        Returns:
            A pandas DataFrame containing the formatted search results.
        """
        # Convert results to DataFrame
        df = pd.DataFrame(
            results, columns=["id", "metadata", "content", "embedding", "distance"]
        )

        # Expand metadata column
        df = pd.concat(
            [df.drop(["metadata"], axis=1), df["metadata"].apply(pd.Series)], axis=1
        )

        # Convert id to string for better readability
        df["id"] = df["id"].astype(str)

        return df

    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
    ) -> None:
        """Delete records from the vector database.

        Args:
            ids (List[str], optional): A list of record IDs to delete.
            metadata_filter (dict, optional): A dictionary of metadata key-value pairs to filter records for deletion.
            delete_all (bool, optional): A boolean flag to delete all records.

        Raises:
            ValueError: If no deletion criteria are provided or if multiple criteria are provided.

        Examples:
            Delete by IDs:
                vector_store.delete(ids=["8ab544ae-766a-11ef-81cb-decf757b836d"])

            Delete by metadata filter:
                vector_store.delete(metadata_filter={"category": "Shipping"})

            Delete all records:
                vector_store.delete(delete_all=True)
        """
        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError(
                "Provide exactly one of: ids, metadata_filter, or delete_all"
            )

        if delete_all:
            self.vec_client.delete_all()
            logging.info(f"Deleted all records from {self.vector_settings.table_name}")
        elif ids:
            self.vec_client.delete_by_ids(ids)
            logging.info(
                f"Deleted {len(ids)} records from {self.vector_settings.table_name}"
            )
        elif metadata_filter:
            self.vec_client.delete_by_metadata(metadata_filter)
            logging.info(
                f"Deleted records matching metadata filter from {self.vector_settings.table_name}"
            )

    def _log_search_time(self, search_type: str, elapsed_time: float) -> None:
        """
        Log the time taken for a search operation.

        Args:
            search_type: The type of search performed (e.g., 'Vector', 'Keyword').
            elapsed_time: The time taken for the search operation in seconds.
        """
        logging.info(f"{search_type} search completed in {elapsed_time:.3f} seconds")

    def extract_keywords_with_llm(self, query: str) -> List[str]:
        """Extract keywords from the query using an LLM."""
        # Set the OpenAI API key
        client = OpenAI()
        OpenAI.api_key = self.settings.openai.api_key

        # Define the prompt for keyword extraction
        message = [
            {
            "role": "system",
             "content": "You are a financial analyst. Your task is to extract the two most important keywords from the given query, and return them as a list of strings."},
            {
            "role": "user",
            "content": f"""Extract the most important keywords from the following query: '{query}'
            Only return only two keywords, no other text.
            """
            }
        ]

        # Call the OpenAI API to get the keywords
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=message,
            response_format=Keywords
        )
        
        keywords = response.choices[0].message.parsed.content
        # Extract keywords from the response
        if isinstance(keywords, list) and all(isinstance(i, list) for i in keywords):
            keywords = [item for sublist in keywords for item in sublist]

        #keywords = response.choices[0].message.parsed.content
        return keywords

    def keyword_search(
        self, query: str, limit: int = 5, return_dataframe: bool = True
    ) -> Union[List[Tuple[str, str, float]], pd.DataFrame]:
        """
        Perform a keyword search on the contents of the vector store.

        Args:
            query: The search query string.
            limit: The maximum number of results to return. Defaults to 5.
            return_dataframe: Whether to return results as a DataFrame. Defaults to True.

        Returns:
            Either a list of tuples (id, contents, rank) or a pandas DataFrame containing the search results.

        Example:
            results = vector_store.keyword_search("shipping options")
        """
        # Extract keywords using LLM
        keywords = self.extract_keywords_with_llm(query)
        logging.info(f"Extracted keywords: {keywords}")

        # Expand keywords with synonyms
        expanded_keywords = expand_with_synonyms(keywords)
        logging.info(f"Expanded keywords: {expanded_keywords}")

        # Construct a search query with expanded keywords
        search_query = ' | '.join(expanded_keywords)
        logging.info(f"Search query: {search_query}")

        search_sql = f"""
        SELECT id, contents, ts_rank_cd(to_tsvector('english', contents), query) as rank
        FROM {self.vector_settings.table_name}, to_tsquery('english', %s) query
        WHERE to_tsvector('english', contents) @@ query
        ORDER BY rank DESC
        LIMIT %s
        """

        start_time = time.time()

        with psycopg.connect(self.settings.database.service_url) as conn:
            with conn.cursor() as cur:
                cur.execute(search_sql, (search_query, limit))
                results = cur.fetchall()

        elapsed_time = time.time() - start_time
        self._log_search_time("Keyword", elapsed_time)

        if return_dataframe:
            df = pd.DataFrame(results, columns=["id", "content", "rank"])
            df["id"] = df["id"].astype(str)
            return df
        else:
            return results

    def hybrid_search(
        self,
        query: str,
        keyword_k: int = 5,
        semantic_k: int = 5,
        rerank: bool = False,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """
        Perform a hybrid search combining keyword and semantic search results,
        with optional reranking using Cohere.

        Args:
            query: The search query string.
            keyword_k: The number of results to return from keyword search. Defaults to 5.
            semantic_k: The number of results to return from semantic search. Defaults to 5.
            rerank: Whether to apply Cohere reranking. Defaults to True.
            top_n: The number of top results to return after reranking. Defaults to 5.

        Returns:
            A pandas DataFrame containing the combined search results with a 'search_type' column.

        Example:
            results = vector_store.hybrid_search("shipping options", keyword_k=3, semantic_k=3, rerank=True, top_n=5)
        """
        # Perform keyword search
        keyword_results = self.keyword_search(
            query, limit=keyword_k, return_dataframe=True
        )
        keyword_results["search_type"] = "keyword"
        keyword_results = keyword_results[["id", "content", "search_type"]]

        # Perform semantic search
        semantic_results = self.semantic_search(
            query, limit=semantic_k, return_dataframe=True
        )
        semantic_results["search_type"] = "semantic"
        semantic_results = semantic_results[["id", "content", "search_type"]]

        # Combine results
        combined_results = pd.concat(
            [keyword_results, semantic_results], ignore_index=True
        )

        # Remove duplicates, keeping the first occurrence (which maintains the original order)
        combined_results = combined_results.drop_duplicates(subset=["id"], keep="first")

        if rerank:
            return self._rerank_results(query, combined_results, top_n)

        return combined_results

    def _rerank_results(
        self, query: str, combined_results: pd.DataFrame, top_n: int
    ) -> pd.DataFrame:
        """
        Rerank the combined search results using Cohere.

        Args:
            query: The original search query.
            combined_results: DataFrame containing the combined keyword and semantic search results.
            top_n: The number of top results to return after reranking.

        Returns:
            A pandas DataFrame containing the reranked results.
        """
        rerank_results = self.cohere_client.v2.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=combined_results["content"].tolist(),
            top_n=top_n,
            return_documents=True,
        )

        reranked_df = pd.DataFrame(
            [
                {
                    "id": combined_results.iloc[result.index]["id"],
                    "content": result.document,
                    "search_type": combined_results.iloc[result.index]["search_type"],
                    "relevance_score": result.relevance_score,
                }
                for result in rerank_results.results
            ]
        )

        return reranked_df.sort_values("relevance_score", ascending=False)

    def upsert_report(self, report_data: dict) -> None:
        """
        Insert or update a report record in the database.

        Args:
            report_data: A dictionary containing the report data.
        """
        # Extract data
        company_id = self.get_or_create_company(report_data['issuerSign'])
        report_id = report_data['id']
        report_date = report_data['publishedTime']
        raw_text = report_data['raw_text']
        embedding = self.get_embedding(raw_text)

        # Insert into reports table
        with psycopg.connect(self.settings.database.service_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.vector_settings.reports_table} (id, company_id, report_date, raw_text, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                    company_id = EXCLUDED.company_id,
                    report_date = EXCLUDED.report_date,
                    raw_text = EXCLUDED.raw_text,
                    embedding = EXCLUDED.embedding;
                    """,
                    (report_id, company_id, report_date, raw_text, embedding)
                )
                conn.commit()

    def get_or_create_company(self, issuer_sign: str) -> int:
        """
        Get or create a company record in the database.

        Args:
            issuer_sign: The company ticker symbol.

        Returns:
            The company ID.
        """
        with psycopg.connect(self.settings.database.service_url) as conn:
            with conn.cursor() as cur:
                # Check if company exists
                cur.execute(
                    f"SELECT id FROM {self.vector_settings.company_table} WHERE name = %s",
                    (issuer_sign,)
                )
                result = cur.fetchone()

                if result:
                    return result[0]

                # Insert new company
                cur.execute(
                    f"INSERT INTO {self.vector_settings.company_table} (name) VALUES (%s) RETURNING id",
                    (issuer_sign,)
                )
                company_id = cur.fetchone()[0]
                conn.commit()
                return company_id

    def upsert_analysis(self, report_id: int, analysis_text: str, summary_text: str) -> None:
        """
        Insert or update an analysis record in the database.

        Args:
            report_id: The ID of the report being analyzed.
            analysis_text: The detailed AI analysis text.
            summary_text: The AI-generated summary text.
        """
        # Generate embeddings for analysis and summary texts
        analysis_embedding = self.get_embedding(analysis_text)
        summary_embedding = self.get_embedding(summary_text)

        # Insert or update the analysis record in the database
        with psycopg.connect(self.settings.database.service_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    INSERT INTO {self.vector_settings.analysis_table} (report_id, analysis_text, embedding_analysis, summary_text, embedding_summary)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (report_id) DO UPDATE SET
                    analysis_text = EXCLUDED.analysis_text,
                    embedding_analysis = EXCLUDED.embedding_analysis,
                    summary_text = EXCLUDED.summary_text,
                    embedding_summary = EXCLUDED.embedding_summary;
                    """,
                    (report_id, analysis_text, analysis_embedding, summary_text, summary_embedding)
                )
                conn.commit()