import os
import pdfplumber
from PyPDF2 import PdfWriter, PdfReader

def extract_relevant_pages(pdf_path, keywords):
    """Extract pages from a PDF that contain specific keywords."""
    relevant_pages = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                # Check if any of the keywords are in the page text
                if any(keyword.lower() in text.lower() for keyword in keywords):
                    relevant_pages.append((i, text))
    
    return relevant_pages

def save_extracted_pages(pdf_path, relevant_pages, output_path):
    """Save extracted pages to a new PDF file."""
    writer = PdfWriter()
    
    # Use PdfReader to read the original PDF
    reader = PdfReader(pdf_path)
    
    for page_number, _ in relevant_pages:
        # Get the page from the PdfReader
        page = reader.pages[page_number]
        writer.add_page(page)

    with open(output_path, 'wb') as out_file:
        writer.write(out_file)

def process_reports(directory):
    """Process all PDF reports in the given directory."""
    keywords = ["highlight", "summary", "key figures", "hovedtall", "nøkkeltall", "oppsummering"]
    
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            relevant_pages = extract_relevant_pages(pdf_path, keywords)
            
            if relevant_pages:
                output_path = os.path.join(directory, f"extracted_{filename}")
                save_extracted_pages(pdf_path, relevant_pages, output_path)
                print(f"Extracted relevant pages from {filename} to {output_path}")
                
                # Delete the original PDF file
                os.remove(pdf_path)
                print(f"Deleted original file: {filename}")

# Example usage
process_reports('downloads')