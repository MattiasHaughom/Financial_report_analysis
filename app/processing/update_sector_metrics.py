import os
import psycopg
import logging

from ..config.settings import get_settings
from ..database.vector_store import VectorStore

def load_sector_metrics(file_path: str) -> dict:
    """Load sector metrics from a text file."""
    with open(file_path, 'r') as file:
        content = file.read()
    # Evaluate the content to convert it into a dictionary
    sector_metrics = eval(content)
    return sector_metrics

def update_sector_metrics():
    settings = get_settings()
    vector_store = VectorStore()

    # Ensure the sector_metrics table is created
    vector_store.create_sector_metrics_table()

    # Get the path to the sector_metrics.txt file
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sector_metrics_path = os.path.join(project_root, 'downloads', 'processed', 'sector', 'sector_metrics.txt')

    # Load sector metrics from the file
    SECTOR_METRICS = load_sector_metrics(sector_metrics_path)

    # Establish database connection
    with psycopg.connect(settings.database.service_url) as conn:
        with conn.cursor() as cur:
            # Delete existing data
            cur.execute("DELETE FROM sector_metrics;")
            logging.info("Deleted existing data from sector_metrics table.")

            # Prepare insert query
            insert_query = """
                INSERT INTO sector_metrics (sector_name, financial_metrics, operational_metrics)
                VALUES (%s, %s, %s)
                ON CONFLICT (sector_name) DO UPDATE SET
                    financial_metrics = EXCLUDED.financial_metrics,
                    operational_metrics = EXCLUDED.operational_metrics;
            """

            # Use SECTOR_METRICS to insert data
            for sector, metrics in SECTOR_METRICS.items():
                financial_metrics = metrics['financial_metrics']
                operational_metrics = metrics['operational_metrics']
                
                # Insert or update logic here
                cur.execute(insert_query, (sector, financial_metrics, operational_metrics))
            conn.commit()
            logging.info(f"Upserted metrics for {len(SECTOR_METRICS)} sectors.")

def main():
    update_sector_metrics()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()