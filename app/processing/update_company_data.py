import os
import pandas as pd
import psycopg
import logging

from ..config.settings import get_settings
from ..database.vector_store import VectorStore

def upsert_company_data():
    settings = get_settings()
    vector_store = VectorStore()

    # Ensure the company table is created
    vector_store.create_company_table()

    # Get the path to the company_data.csv file
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    company_data_path = os.path.join(project_root, 'downloads', 'processed', 'company', 'company_data.csv')

    # Read the CSV file
    df = pd.read_csv(company_data_path)

    # Establish database connection
    with psycopg.connect(settings.database.service_url) as conn:
        with conn.cursor() as cur:
            # Delete existing data
            cur.execute(f"DELETE FROM {settings.vector_store.company_table};")
            logging.info("Deleted existing data from company table.")

            # Prepare insert query
            insert_query = f"""
                INSERT INTO {settings.vector_store.company_table} (isin, name, symbol, industry, sector)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (isin) DO UPDATE SET
                    name = EXCLUDED.name,
                    symbol = EXCLUDED.symbol,
                    industry = EXCLUDED.industry,
                    sector = EXCLUDED.sector;
            """

            # Insert data row by row
            for idx, row in df.iterrows():
                data_tuple = (row['ISIN'], row['Name'], row['Symbol'], row['Industry'], row['Sector'])
                cur.execute(insert_query, data_tuple)
            conn.commit()
            logging.info(f"Upserted {len(df)} records into company table.")

def main():
    upsert_company_data()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 