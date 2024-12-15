import os
import pdfplumber
from PyPDF2 import PdfWriter, PdfReader
from datetime import datetime
import pandas as pd
import logging
import json
from timescale_vector.client import uuid_from_time
import nltk
import pypdf
from pathlib import Path
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm

from ..database.vector_store import VectorStore

vector_store = VectorStore()

def load_reports_metadata(metadata_dir: str) -> pd.DataFrame:
    """Load and process reports metadata from JSON files."""
    import glob

    # Load all JSON files in the metadata directory
    json_files = glob.glob(os.path.join(metadata_dir, '*.json'))
    reports_list = []

    for json_file in json_files:
        with open(json_file, 'r') as file:
            reports = json.load(file)
            for report in reports:
                # Check if the report has attachments
                attachments = report.get('attachments', [])
                for attachment in attachments:
                    report_entry = {
                        'messageId': report['messageId'],
                        'attachmentId': attachment['attachmentId'],
                        'attachmentName': attachment['attachmentName'],
                        'issuerSign': report['issuerSign'],
                        'publishedTime': report['publishedTime'],
                        'downloadedFiles': report.get('downloadedFiles', [])
                    }
                    reports_list.append(report_entry)

    # Convert to DataFrame
    reports_df = pd.DataFrame(reports_list)
    return reports_df


def extract_relevant_pages(pdf_path: str, keywords: List[str]) -> List[int]:
    """Extract pages from a PDF that contain specific keywords."""
    relevant_pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                # Check if any of the keywords are in the page text
                if any(keyword.lower() in text.lower() for keyword in keywords):
                    relevant_pages.append(i)
    
    return relevant_pages


def save_extracted_pages(pdf_path: str, relevant_pages: List[int], output_path: str) -> None:
    """Save extracted pages to a new PDF file."""
    writer = PdfWriter()
    
    # Use PdfReader to read the original PDF
    reader = PdfReader(pdf_path)
    
    for page_number in relevant_pages:
        # Get the page from the PdfReader
        page = reader.pages[page_number]
        writer.add_page(page)

    with open(output_path, 'wb') as out_file:
        writer.write(out_file)
    logging.info(f"Saved processed PDF to {output_path}")


def process_pdfs(
    reports_df: pd.DataFrame,
    raw_pdf_dir: str,
    processed_reports_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> pd.DataFrame:
    """Process reports to extract text and prepare data for upsert."""
    from nltk.tokenize import sent_tokenize

    data = []
    keywords = ["highlight", 
                "summary", 
                "key figures", 
                "hovedtall", 
                "nøkkeltall",
                "oppsummering"
                ]

    # Ensure NLTK data is downloaded
    nltk.download('punkt', quiet=True)

    for idx, row in reports_df.iterrows():
        attachment_name = row['attachmentName']
        file_path = os.path.join(raw_pdf_dir, attachment_name)

        if not os.path.isfile(file_path):
            logging.warning(f"File not found: {file_path}")
            continue  # Skip if the file does not exist

        # Extract relevant pages
        relevant_pages = extract_relevant_pages(file_path, keywords)

        if not relevant_pages:
            logging.info(f"No relevant pages found in {attachment_name}")
            continue  # Skip if no relevant pages found

        # Save extracted pages to a new PDF
        output_pdf_path = os.path.join(processed_reports_dir, f"processed_{attachment_name}")
        save_extracted_pages(file_path, relevant_pages, output_pdf_path)
        logging.info(f"Saved processed PDF to {output_pdf_path}")
        
    # Get PDF files
    # Ensure output_pdf_path is a Path object
    processed_reports_dir = Path(processed_reports_dir)

    # Use glob to find all PDF files in the directory
    pdf_files = list(processed_reports_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {processed_reports_dir}")

    documents = []
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            documents.extend(docs)
            logging.info(f"Successfully loaded {pdf_file.name}")
        except Exception as e:
            logging.error(f"Error loading {pdf_file.name}: {str(e)}")
            continue

    # Split documents into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = splitter.split_documents(documents)
    logging.info(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")

    # Convert to DataFrame
    data: List[Dict[str, Any]] = []
    for i, chunk in enumerate(chunked_docs):
        source_file = Path(chunk.metadata.get("source", "unknown"))
        data.append({
            "chunk_id": i,
            "doc_id": f"{source_file.stem}_{source_file.stat().st_mtime_ns}",  # More unique ID
            'messageId': row['messageId'],
            'attachmentId': row['attachmentId'],
            'attachmentName': row['attachmentName'],
            'issuerSign': row['issuerSign'],
            'publishedTime': row['publishedTime'],
            'embedding_created_at': datetime.utcnow().isoformat() + 'Z',
            "content": chunk.page_content,
            "page": chunk.metadata.get("page", 0),  # Additional metadata
            "total_pages": len(documents)  # Additional metadata
        })
    
    df = pd.DataFrame(data)
    logging.info(f"Created DataFrame with {len(df)} rows")
    return df


def prepare_record(row: pd.Series):
    """
    Prepare a record for insertion into the vector store.
    
    Args:
        row (pd.Series): A row from the DataFrame containing document information
        
    Returns:
        dict: Prepared record with metadata and embedding, or None if document exists
    """
    doc_id = row["doc_id"]
    chunk_id = row["chunk_id"]
    
    try:
        # Check if the document or chunk already exists
        #if is_document_exists(doc_id, chunk_id):
        #    logging.info(f"Skipping existing document: {doc_id}, chunk: {chunk_id}")
        #    return None

        content = row["content"]
        embedding = vector_store.get_embedding(content)
        
        return {
            "id": str(uuid_from_time(datetime.now())),
            "metadata": {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                'messageId': row['messageId'],
                'attachmentId': row['attachmentId'],
                'attachmentName': row['attachmentName'],
                'issuerSign': row['issuerSign'],
                'publishedTime': row['publishedTime'],
                'downloadedFiles': row.get('downloadedFiles', []),
                "created_at": datetime.now().isoformat(),
                "page": row.get("page", 0),  # Include additional metadata
                "total_pages": row.get("total_pages", 0),
            },
            "contents": content,
            "embedding": embedding,
        }
    except Exception as e:
        logging.error(f"Error preparing record for doc_id: {doc_id}, chunk: {chunk_id}: {str(e)}")
        return None


def main():
    """Main function to process reports."""
    # Initialize VectorStore
    #vector_store.create_reports_table()
    #vector_store.create_reports_index()  # DiskAnnIndex
    vector_store.create_keyword_search_indexes()  # GIN Index

    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    # Set up directory paths
    downloads_dir = os.path.join(project_root, 'downloads', 'raw')       # Raw PDFs
    processed_reports_dir = os.path.join(project_root, 'downloads', 'processed', 'reports')  # Processed PDFs
    metadata_dir = os.path.join(project_root, 'downloads', 'processed', 'metadata')        # Metadata JSON files

    # Ensure directories exist
    os.makedirs(downloads_dir, exist_ok=True)
    os.makedirs(processed_reports_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    try:
        # Load reports metadata
        reports_df = load_reports_metadata(metadata_dir)
        logging.info(f"Loaded metadata for {len(reports_df)} attachments.")

        if reports_df.empty:
            logging.warning("No reports found in metadata.")
            return

        # Extract text, metadata and return a DataFrame
        df = process_pdfs(
            reports_df,
            downloads_dir,
            processed_reports_dir,
            chunk_size=int(os.getenv('CHUNK_SIZE', '500')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '100'))
        )

        # Apply the preparation to each row
        records_df = df.apply(prepare_record, axis=1)
        records_df.to_csv(f'{project_root}/final_output.csv', index=False)

        # Drop any failed records (None values) and convert to DataFrame
        records_df = pd.DataFrame([r for r in records_df if r is not None])
        vector_store.upsert_reports(records_df)
        logging.info(f"Successfully upserted {len(records_df)} records.")
        
    except Exception as e:
        logging.error(f"An error occurred during report processing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('process_reports.log'),
            logging.StreamHandler()
        ]
    )
    main()