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
from timescale_vector import client
from typing import List
import uuid
import json



def expand_with_synonyms(keywords: List[str]) -> List[str]:
    """Expand keywords with synonyms using WordNet."""
    import nltk
    from nltk.corpus import wordnet

    nltk.download('wordnet')
    nltk.download('omw-1.4')

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

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.settings.openai.api_key)
        self.embedding_model = self.settings.openai.embedding_model

        # Initialize Cohere client
        self.cohere_client = cohere.ClientV2(api_key=self.settings.cohere.api_key)

        # Store vector settings
        self.vector_settings = self.settings.vector_store

        # Initialize Timescale Vector clients for reports and analysis
        self.reports_vec_client = client.Sync(
            self.settings.database.service_url,
            "reports",
            self.vector_settings.embedding_dimensions,
            time_partition_interval=self.vector_settings.time_partition_interval
        )

        self.analysis_vec_client = client.Sync(
            self.settings.database.service_url,
            "analysis",
            self.vector_settings.embedding_dimensions,
            time_partition_interval=self.vector_settings.time_partition_interval
        )

        # Initialize or create the company table separately
        self.create_company_table()



#------------------------------------------------------------------------------------------------
# Embedding
#------------------------------------------------------------------------------------------------

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


#------------------------------------------------------------------------------------------------
# Get existing message IDs
#------------------------------------------------------------------------------------------------

    def get_existing_message_ids(self, message_ids: List[int]) -> List[int]:
        """
        Retrieve existing message IDs from the database to filter out already processed reports.
        """
        sql = """
        SELECT (metadata->>'messageId')::integer AS messageId
        FROM reports
        WHERE (metadata->>'messageId')::integer = ANY(%s);
        """
        try:
            with psycopg.connect(self.settings.database.service_url) as conn:
                df = pd.read_sql(sql, conn, params=(message_ids,))
                existing_ids = df['messageId'].tolist()
                return existing_ids
        except Exception as e:
            logging.error(f"Error retrieving existing message IDs: {e}")
            return []

#------------------------------------------------------------------------------------------------
# Index operations
#------------------------------------------------------------------------------------------------
    def create_reports_table(self) -> None:
        """Create the necessary tablesin the database"""
        self.reports_vec_client.create_tables()

    def create_analysis_table(self) -> None:
        """Create the necessary tablesin the database"""
        self.analysis_vec_client.create_tables()

    def create_reports_index(self) -> None:
        """Create the DiskANN indexes for the reports table."""
        self.reports_vec_client.create_embedding_index(client.DiskAnnIndex())
        logging.info(f"DiskANN index created for reports table.")

    def create_analysis_index(self) -> None:
        """Create the DiskANN indexes for the analysis table."""
        self.analysis_vec_client.create_embedding_index(client.DiskAnnIndex())
        logging.info(f"DiskANN index created for analysis table.")

    def drop_index_reports(self) -> None:
        """Drop the StreamingDiskANN index in the database"""
        self.reports_vec_client.drop_embedding_index()
    
    def drop_index_analysis(self) -> None:
        """Drop the StreamingDiskANN index in the database"""
        self.analysis_vec_client.drop_embedding_index()

    def create_reports_keyword_search_index(self):
            """Create a GIN index for keyword search if it doesn't exist."""
            index_name = f"idx_reports_contents_gin"
            create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON reports USING gin(to_tsvector('english', contents));
            """
            try:
                with psycopg.connect(self.settings.database.service_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute(create_index_sql)
                        conn.commit()
                        logging.info(f"GIN index '{index_name}' created or already exists.")
            except Exception as e:
                logging.error(f"Error while creating GIN index: {str(e)}")

    def create_analysis_keyword_search_index(self):
            """Create a GIN index for keyword search if it doesn't exist."""
            index_name = f"idx_analysis_contents_gin"
            create_index_sql = f"""
            CREATE INDEX IF NOT EXISTS {index_name}
            ON analysis USING gin(to_tsvector('english', contents));
            """
            try:
                with psycopg.connect(self.settings.database.service_url) as conn:
                    with conn.cursor() as cur:
                        cur.execute(create_index_sql)
                        conn.commit()
                        logging.info(f"GIN index '{index_name}' created or already exists.")
            except Exception as e:
                logging.error(f"Error while creating GIN index: {str(e)}")


#------------------------------------------------------------------------------------------------
# Upsert methods
#------------------------------------------------------------------------------------------------

    def upsert_reports(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, contents, embedding
        """
        records = df.to_records(index=False)
        self.reports_vec_client.upsert(list(records))
        print(self.vector_settings.reports_table)
        logging.info(
            f"Inserted {len(df)} records into {self.vector_settings.reports_table}"
        )

    def upsert_analysis(self, df: pd.DataFrame) -> None:
        """
        Insert or update records in the database from a pandas DataFrame.

        Args:
            df: A pandas DataFrame containing the data to insert or update.
                Expected columns: id, metadata, contents, embedding
        """
        records = df.to_records(index=False)
        self.analysis_vec_client.upsert(list(records))
        logging.info(
            f"Inserted {len(df)} records into {self.vector_settings.analysis_table}"
        )

    def create_company_table(self):
        """Create the company table """
        with psycopg.connect(self.settings.database.service_url) as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                    CREATE TABLE IF NOT EXISTS {self.vector_settings.company_table} (
                        id SERIAL PRIMARY KEY,
                        isin TEXT UNIQUE,
                        name TEXT,
                        symbol TEXT,
                        industry TEXT,
                        sector TEXT
                    );
                """)
                conn.commit()
                logging.info(f"Company table '{self.vector_settings.company_table}' is ready.")


    def create_sector_metrics_table(self):
        """Create the sector metrics table."""
        with psycopg.connect(self.settings.database.service_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sector_metrics (
                        sector_name TEXT PRIMARY KEY,
                        financial_metrics TEXT[],
                        operational_metrics TEXT[]
                    );
                """)
                conn.commit()
                logging.info("Sector metrics table is ready.")


#------------------------------------------------------------------------------------------------
# Get metadata
#------------------------------------------------------------------------------------------------
    
    def get_report_metadata(self, report_id: str) -> Tuple[str, str]:
        """
        Retrieve company_id and report_date for a given report_id.

        Args:
            report_id: The report ID.

        Returns:
            A tuple of (company_id, report_date).
        """
        query = f"""
        SELECT 
            metadata->>'company_id' AS company_id, 
            metadata->>'report_date' AS report_date
        FROM {self.vector_settings.reports_table}
        WHERE metadata->>'report_id' = %s
        LIMIT 1;
        """
        with psycopg.connect(self.settings.database.service_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (report_id,))
                result = cur.fetchone()
                if result:
                    return result[0], result[1]
                else:
                    raise ValueError(f"Report ID {report_id} not found.")

    def get_reports_metadata(self, report_ids: List[str]) -> pd.DataFrame:
        """
        Retrieve company_id and report_date for multiple report_ids.

        Args:
            report_ids: List of report IDs.

        Returns:
            A pandas DataFrame with columns: report_id, company_id, report_date
        """
        sql = f"""
        SELECT 
            metadata->>'report_id' AS report_id,
            metadata->>'company_id' AS company_id, 
            metadata->>'report_date' AS report_date
        FROM {self.vector_settings.reports_table}
        WHERE metadata->>'report_id' = ANY(%s);
        """
        with psycopg.connect(self.settings.database.service_url) as conn:
            df = pd.read_sql(sql, conn, params=(report_ids,))
        return df

#------------------------------------------------------------------------------------------------
# Semantic search
#------------------------------------------------------------------------------------------------
    def semantic_search(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        table_name: str = 'reports',
        return_dataframe: bool = True,
    ) -> Union[List[dict], pd.DataFrame]:
        """Perform semantic search on the specified table."""
        query_embedding = self.get_embedding(query)
        vec_client = self.reports_vec_client if table_name == 'reports' else self.analysis_vec_client

        # Convert metadata_filter to predicates if provided
        predicates = None
        if metadata_filter:
            # Initialize predicates with the first condition
            first_key, first_value = next(iter(metadata_filter.items()))
            predicates = client.Predicates(first_key, "==", first_value)
            # Add additional conditions
            for key, value in list(metadata_filter.items())[1:]:
                predicates &= client.Predicates(key, "==", value)

        # Adjust search arguments to match the expected parameters
        search_args = {
            "limit": limit,
            "predicates": predicates
        }

        results = vec_client.search(query_embedding, **search_args)
        logging.info(f"Semantic search on '{table_name}' completed.")

        if return_dataframe:
            return self._create_dataframe_from_results(results)
        else:
            return results

#------------------------------------------------------------------------------------------------
# Keyword search
#-----------------------------------------------------------------------------------------------
    
    def keyword_search(
        self,
        query: str,
        limit: int = 5,
        table_name: str = 'reports',
        metadata_filter: dict = None,
        return_dataframe: bool = True,
    ) -> Union[pd.DataFrame, List[Tuple[str, str, float]]]:
        """
        Perform a keyword search on the contents of the specified table.
        """
        search_query = ' | '.join(query.split())
        logging.info(f"Search query: {search_query}")

        table = self.vector_settings.reports_table if table_name == 'reports' else "analysis"

        filter_sql = ""
        params = [search_query]

        if metadata_filter:
            filter_conditions = []
            for key, value in metadata_filter.items():
                filter_conditions.append(f"metadata ->> '{key}' = %s")
                params.append(value)
            filter_sql = ' AND '.join(filter_conditions)

        search_sql = f"""
        SELECT id, contents, ts_rank_cd(to_tsvector('english', contents), query) as rank
        FROM {table}, to_tsquery('english', %s) query
        WHERE to_tsvector('english', contents) @@ query
        {f'AND {filter_sql}' if filter_sql else ''}
        ORDER BY rank DESC
        LIMIT %s
        """

        params.append(limit)

        start_time = time.time()

        with psycopg.connect(self.settings.database.service_url) as conn:
            with conn.cursor() as cur:
                cur.execute(search_sql, params)
                results = cur.fetchall()

        elapsed_time = time.time() - start_time
        self._log_search_time("Keyword", elapsed_time)

        if return_dataframe:
            df = pd.DataFrame(results, columns=["id", "content", "rank"])
            df["id"] = df["id"].astype(str)
            return df
        else:
            return results

#------------------------------------------------------------------------------------------------
# Hybrid search
#------------------------------------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        keyword_k: int = 5,
        semantic_k: int = 5,
        rerank: bool = False,
        top_n: int = 5,
        table_name: str = 'reports',
        metadata_filter: dict = None,
    ) -> pd.DataFrame:
        """
        Perform a hybrid search combining keyword and semantic search results,
        with optional reranking using Cohere.
        """
        # Perform keyword search
        keyword_results = self.keyword_search(
            query,
            limit=keyword_k,
            table_name=table_name,
            metadata_filter=metadata_filter,
            return_dataframe=True
        )
        keyword_results["search_type"] = "keyword"
        keyword_results = keyword_results[["id", "content", "search_type"]]

        # Perform semantic search
        semantic_results = self.semantic_search(
            query,
            limit=semantic_k,
            table_name=table_name,
            metadata_filter=metadata_filter,
            return_dataframe=True
        )
        semantic_results["search_type"] = "semantic"
        semantic_results = semantic_results[["id", "content", "search_type"]]

        # Combine results
        combined_results = pd.concat(
            [keyword_results, semantic_results], ignore_index=True
        )

        # Remove duplicates
        combined_results = combined_results.drop_duplicates(subset=["id"], keep="first")

        if rerank:
            if combined_results.empty:
                logging.warning("No results to rerank.")
                return combined_results
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
    
#------------------------------------------------------------------------------------------------
# Search helper functions
#------------------------------------------------------------------------------------------------

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


    def _log_search_time(self, search_type: str, elapsed_time: float) -> None:
        """
        Log the time taken for a search operation.

        Args:
            search_type: The type of search performed (e.g., 'Vector', 'Keyword').
            elapsed_time: The time taken for the search operation in seconds.
        """
        logging.info(f"{search_type} search completed in {elapsed_time:.3f} seconds")


#------------------------------------------------------------------------------------------------
# Delete
#------------------------------------------------------------------------------------------------

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
            logging.info(f"Deleted all records from {self.vector_settings.reports_table}")
        elif ids:
            self.vec_client.delete_by_ids(ids)
            logging.info(
                f"Deleted {len(ids)} records from {self.vector_settings.reports_table}"
            )
        elif metadata_filter:
            self.vec_client.delete_by_metadata(metadata_filter)
            logging.info(
                f"Deleted records matching metadata filter from {self.vector_settings.reports_table}"
            )


#------------------------------------------------------------------------------------------------
# Analysis 
#------------------------------------------------------------------------------------------------

    def get_reports_to_analyze(self):
        # Fetch unique doc_ids from reports that are not yet analyzed
        sql = """
        SELECT DISTINCT metadata->>'doc_id' AS doc_id
        FROM reports
        WHERE (metadata->>'doc_id') IS NOT NULL
        AND metadata->>'doc_id' NOT IN (
            SELECT metadata->>'report_id' FROM analysis
        )
        """
        with psycopg.connect(self.settings.database.service_url) as conn:
            df = pd.read_sql(sql, conn)
        return df['doc_id'].tolist()

    def get_company_sector(self, company_id: str) -> str:
        # Fetch sector from 'company' table
        sql = """
        SELECT sector FROM company WHERE name = %s
        """
        with psycopg.connect(self.settings.database.service_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (company_id,))
                result = cur.fetchone()
                return result[0] if result else 'Other'

    def get_sector_metrics(self, sector: str) -> dict[str, List[str]]:
        # Fetch sector metrics from 'sector_metrics' table
        sql = """
        SELECT financial_metrics, operational_metrics FROM sector_metrics WHERE sector_name = %s
        """
        with psycopg.connect(self.settings.database.service_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (sector,))
                result = cur.fetchone()
                if result:
                    financial_metrics, operational_metrics = result
                    return {
                        'financial': financial_metrics,
                        'operational': operational_metrics
                    }
                else:
                    return {}

    def get_company_data(self, symbol: str) -> Optional[dict]:
        """Fetch company data from the company table using symbol."""
        sql = f"""
        SELECT isin, name, symbol, industry, sector
        FROM {self.vector_settings.company_table}
        WHERE symbol = %s;
        """
        with psycopg.connect(self.settings.database.service_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (symbol,))
                row = cur.fetchone()
                if row:
                    columns = ['isin', 'name', 'symbol', 'industry', 'sector']
                    return dict(zip(columns, row))
                else:
                    return None

    def get_sector_metrics(self, sector: str) -> dict[str, List[str]]:
        """Fetch sector metrics from the sector_metrics table."""
        sql = """
        SELECT financial_metrics, operational_metrics
        FROM sector_metrics
        WHERE sector_name = %s;
        """
        with psycopg.connect(self.settings.database.service_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (sector,))
                result = cur.fetchone()
                if result:
                    financial_metrics, operational_metrics = result
                    return {
                        'financial': financial_metrics,
                        'operational': operational_metrics
                    }
                else:
                    return {}

    def get_report_chunks(self, doc_id: str) -> List[dict]:
        # Fetch all chunks for the given doc_id
        sql = """
        SELECT id, contents, metadata
        FROM reports
        WHERE metadata->>'doc_id' = %s
        """
        with psycopg.connect(self.settings.database.service_url) as conn:
            with conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                cur.execute(sql, (doc_id,))
                results = cur.fetchall()
        return results